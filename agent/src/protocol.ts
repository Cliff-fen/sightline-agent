/** Stable wire contracts shared by the Agent runtime, tools and UI. */

export type Role = "system" | "user" | "assistant" | "tool";
export type ImageRef = {
  kind: "image";
  id: string;
  uri: string;
  mimeType?: string;
  width?: number;
  height?: number;
};
export type ContentPart =
  | { type: "text"; text: string }
  | { type: "image"; image: ImageRef };
export type ChatMessage = {
  id: string;
  role: Role;
  content: ContentPart[];
  createdAt: string;
  toolCallId?: string;
  name?: string;
};
export type ToolCall = { id: string; name: string; arguments: Record<string, unknown> };
export type ToolResult = {
  callId: string;
  name: string;
  ok: boolean;
  content: ContentPart[];
  error?: { code: string; message: string; retryable: boolean };
  artifacts?: ImageRef[];
  durationMs: number;
};
export type AgentEvent =
  | { type: "run_started"; runId: string; at: string }
  | { type: "message_delta"; runId: string; delta: string; at: string }
  | { type: "tool_started"; runId: string; call: ToolCall; at: string }
  | { type: "tool_finished"; runId: string; result: ToolResult; at: string }
  | { type: "run_finished"; runId: string; reason: string; at: string }
  | { type: "run_failed"; runId: string; error: string; at: string };
export type RunRequest = { input: string | ContentPart[]; systemPrompt?: string };
export type RunResponse = {
  runId: string;
  output: string;
  model: { profile: string; id: string };
  events: AgentEvent[];
};
export type ToolDefinition = { name: string; description: string; inputSchema: Record<string, unknown> };
export type ToolHandler = (call: ToolCall, signal: AbortSignal) => Promise<ToolResult>;

export function now(): string { return new Date().toISOString(); }
export function createId(prefix: string): string { return `${prefix}_${crypto.randomUUID()}`; }
export function textContent(text: string): ContentPart[] { return [{ type: "text", text }]; }
export function assertRunRequest(value: unknown): RunRequest {
  if (!value || typeof value !== "object") throw new Error("Request body must be an object");
  const request = value as Partial<RunRequest>;
  if (typeof request.input !== "string" && !Array.isArray(request.input)) {
    throw new Error("input must be a string or content-part array");
  }
  if (Array.isArray(request.input)) {
    for (const part of request.input) {
      if (!part || typeof part !== "object") throw new Error("Each content part must be an object");
      const candidate = part as Partial<ContentPart>;
      if (candidate.type === "text" && typeof candidate.text === "string") continue;
      if (candidate.type === "image" && candidate.image?.kind === "image"
        && typeof candidate.image.id === "string" && typeof candidate.image.uri === "string") continue;
      throw new Error("Invalid content part");
    }
  }
  if (request.systemPrompt !== undefined && typeof request.systemPrompt !== "string") {
    throw new Error("systemPrompt must be a string");
  }
  return request as RunRequest;
}
export function assertToolCall(value: unknown): ToolCall {
  if (!value || typeof value !== "object") throw new Error("Invalid tool call");
  const call = value as Partial<ToolCall>;
  if (typeof call.id !== "string" || typeof call.name !== "string") throw new Error("Tool call requires id and name");
  if (!call.arguments || typeof call.arguments !== "object") throw new Error("Tool call arguments must be an object");
  return call as ToolCall;
}
