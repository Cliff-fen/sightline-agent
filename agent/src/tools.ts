import { ToolGateway } from "./tool-gateway.js";
import { ToolRegistry } from "./tool-registry.js";
import { createId, type ToolCall, type ToolResult } from "./protocol.js";

const descriptions = {
  web_search: "Search the web and return ranked evidence snippets.",
  text_search: "Search the web for factual evidence. Use for external facts and current information.",
  visit: "Read a webpage and return its extracted text for evidence.",
  image_search: "Find visual matches and source metadata for an image.",
  crop: "Create a new image artifact for a rectangular crop.",
  layout_parsing: "Extract structured text and layout from a document image.",
  sharpen: "Enhance local image detail when the image is blurry.",
  super_resolution: "Upscale a low-resolution image before inspection.",
  perspective_correct: "Correct a mild perspective or rotation distortion.",
};
const schemas: Record<string, Record<string, unknown>> = {
  web_search: { type: "object", properties: { query: { type: "string" }, topK: { type: "integer", minimum: 1, maximum: 20 } }, required: ["query"] },
  text_search: { type: "object", properties: { query: { type: "string" }, topK: { type: "integer", minimum: 1, maximum: 10 } }, required: ["query"] },
  visit: { type: "object", properties: { url: { type: "string", format: "uri" } }, required: ["url"] },
  image_search: { type: "object", properties: { image: { type: "string", description: "Public image URL or image artifact URI" } }, required: ["image"] },
  crop: { type: "object", properties: { image: { type: "string" }, x: { type: "integer" }, y: { type: "integer" }, width: { type: "integer", minimum: 1 }, height: { type: "integer", minimum: 1 } }, required: ["image", "x", "y", "width", "height"] },
  layout_parsing: { type: "object", properties: { image: { type: "string" }, orientation: { type: "boolean" } }, required: ["image"] },
  sharpen: { type: "object", properties: { image: { type: "string" }, amount: { type: "number", minimum: 0, maximum: 5 } }, required: ["image"] },
  super_resolution: { type: "object", properties: { image: { type: "string" }, scale: { type: "number", minimum: 1.1, maximum: 4 } }, required: ["image"] },
  perspective_correct: { type: "object", properties: { image: { type: "string" }, angle: { type: "number", minimum: -20, maximum: 20 } }, required: ["image"] },
};

export function registerSearchTools(registry: ToolRegistry, gateway: ToolGateway): void {
  for (const [name, inputSchema] of Object.entries(schemas)) {
    registry.register({ name, description: descriptions[name as keyof typeof descriptions], inputSchema,
      execute: (call, signal) => gateway.execute(call, signal) });
  }
}
export function normalizeToolResult(value: unknown, call: ToolCall): ToolResult {
  if (!value || typeof value !== "object") return { callId: call.id, name: call.name, ok: false, content: [{ type: "text", text: "Invalid tool result" }], durationMs: 0 };
  return value as ToolResult;
}
export function newToolCall(name: string, arguments_: Record<string, unknown>): ToolCall { return { id: createId("call"), name, arguments: arguments_ }; }
