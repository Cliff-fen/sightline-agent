import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { type ImageContent } from "@earendil-works/pi-ai";
import { type ImageRef } from "./protocol.js";

const DATA_URI_RE = /^data:([^;,]+);base64,(.+)$/s;

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
