import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { type ImageContent } from "@earendil-works/pi-ai";
import { type ImageRef } from "./protocol.js";

const DATA_URI_RE = /^data:([^;,]+);base64,(.+)$/s;
const IMAGE_ALIAS_RE = /^(?:(?:the|an?)\s+)?(?:(uploaded|input|original|source|current|latest|cropped|enhanced)\s+)?(?:image|photo|picture)(?:[_\s-]?(\d+))?(?:\.[a-z0-9]+)?$/i;

function imageNames(ref: ImageRef): Set<string> {
  const names = new Set([ref.id.toLowerCase(), ref.uri.toLowerCase()]);
  try {
    const pathname = ref.uri.startsWith("file://") ? fileURLToPath(ref.uri) : new URL(ref.uri).pathname;
    const filename = pathname.split("/").filter(Boolean).at(-1);
    if (filename) {
      names.add(filename.toLowerCase());
      names.add(filename.replace(/\.[^.]+$/, "").toLowerCase());
    }
  } catch {
    // artifact:// identifiers and data URIs are already covered by id/uri.
  }
  return names;
}

/** Resolve the natural image aliases models commonly emit at tool boundaries. */
export function resolveImageReference(reference: unknown, images: readonly ImageRef[]): string {
  if (typeof reference !== "string" || !reference.trim()) return String(reference ?? "");
  const value = reference.trim();
  if (/^(?:https?:\/\/|file:\/\/|artifact:\/\/|data:)/i.test(value)) return value;
  if (images.length === 0) return value;

  const lowered = value.toLowerCase();
  const exact = images.find((image) => imageNames(image).has(lowered));
  if (exact) return exact.uri;

  const alias = IMAGE_ALIAS_RE.exec(value);
  if (alias?.[2] !== undefined) {
    const index = Number(alias[2]);
    if (Number.isInteger(index) && index >= 0 && index < images.length) return images[index]?.uri ?? value;
  }
  if (alias?.[1] === "latest" || alias?.[1] === "cropped" || alias?.[1] === "enhanced") {
    return images.at(-1)?.uri ?? value;
  }
  if (images.length === 1) return images[0]?.uri ?? value;
  return value;
}

function enforceLimit(bytes: Uint8Array, maxBytes: number, source: string): void {
  if (bytes.byteLength > maxBytes) {
    throw new Error(`Image exceeds ${maxBytes} bytes: ${source}`);
  }
}

function asImageContent(bytes: Uint8Array, mimeType: string): ImageContent {
  return { type: "image", data: Buffer.from(bytes).toString("base64"), mimeType };
}

export async function resolveImage(ref: ImageRef, maxBytes: number, signal?: AbortSignal): Promise<ImageContent> {
  const dataMatch = DATA_URI_RE.exec(ref.uri);
  if (dataMatch) {
    const mimeType = dataMatch[1];
    const payload = dataMatch[2];
    if (!mimeType || !payload) throw new Error(`Invalid data URI for image ${ref.id}`);
    const bytes = Buffer.from(payload, "base64");
    enforceLimit(bytes, maxBytes, ref.id);
    return asImageContent(bytes, ref.mimeType ?? mimeType);
  }

  if (ref.uri.startsWith("file://")) {
    const bytes = await readFile(fileURLToPath(ref.uri));
    enforceLimit(bytes, maxBytes, ref.uri);
    return asImageContent(bytes, ref.mimeType ?? "application/octet-stream");
  }

  const url = new URL(ref.uri);
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error(`Unsupported image URI protocol: ${url.protocol}`);
  }
  const response = await fetch(url, signal ? { signal } : undefined);
  if (!response.ok) throw new Error(`Image download failed (${response.status}): ${ref.uri}`);
  const declaredLength = Number(response.headers.get("content-length") ?? 0);
  if (declaredLength > maxBytes) throw new Error(`Image exceeds ${maxBytes} bytes: ${ref.uri}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  enforceLimit(bytes, maxBytes, ref.uri);
  const contentType = response.headers.get("content-type")?.split(";", 1)[0];
  return asImageContent(bytes, ref.mimeType ?? contentType ?? "application/octet-stream");
}
