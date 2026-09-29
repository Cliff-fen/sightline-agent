import { Agent, type AgentTool } from "@earendil-works/pi-agent-core";
import { type Api, type Model, type Models } from "@earendil-works/pi-ai";
import { type TSchema } from "typebox";
import { resolveImage, resolveImageReference } from "./image-input.js";
import {
  type AgentEvent,
  type ContentPart,
  type ImageRef,
  type ToolCall,
  type ToolResult,
  createId,
  now,
} from "./protocol.js";
import { ToolRegistry } from "./tool-registry.js";

export function lastAssistantText(messages: readonly unknown[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (!message || typeof message !== "object") continue;
    const candidate = message as { role?: string; content?: unknown };
    if (candidate.role !== "assistant" || !Array.isArray(candidate.content)) continue;
    return candidate.content.flatMap((part) => {
      if (!part || typeof part !== "object") return [];
      const content = part as { type?: string; text?: string };
      return content.type === "text" && typeof content.text === "string" ? [content.text] : [];
    }).join("");
  }
  return "";
}

export type RuntimeOptions = {
  model: Model<Api>;
  models: Models;
  systemPrompt: string;
  tools: ToolRegistry;
  maxTurns?: number;
  maxImageBytes?: number;
};
/** Product-facing runtime. Pi owns the loop; this class owns our contracts and telemetry. */
export class SearchAgentRuntime {
  private readonly agent: Agent;
  private readonly runId = createId("run");
  private readonly listeners = new Set<(event: AgentEvent) => void>();
  private activeImages: ImageRef[] = [];
  private turnCount = 0;
  constructor(private readonly options: RuntimeOptions) {
    this.agent = new Agent({
      initialState: { systemPrompt: options.systemPrompt, model: options.model, tools: this.piTools(options.tools) },
      streamFn: options.models.streamSimple.bind(options.models),
      // The data-collection protocol records one action, then its observation, per turn.
      toolExecution: "sequential",
      prepareNextTurn: () => {
        this.turnCount += 1;
        if (this.turnCount >= (this.options.maxTurns ?? 20)) this.agent.abort();
        return {};
      },
    });
    this.agent.subscribe((event: unknown) => this.forwardPiEvent(event));
  }
  get id(): string { return this.runId; }
  onEvent(listener: (event: AgentEvent) => void): () => void { this.listeners.add(listener); return () => this.listeners.delete(listener); }
  async prompt(content: string | ContentPart[]): Promise<string> {
    this.emit({ type: "run_started", runId: this.runId, at: now() });
    const parts = typeof content === "string" ? [{ type: "text" as const, text: content }] : content;
    const text = parts.filter((part) => part.type === "text").map((part) => part.text).join("\n");
    let output = "";
    const removeCapture = this.onEvent((event) => {
      if (event.type === "message_delta") output += event.delta;
    });
    try {
      const images = await Promise.all(
        parts.filter((part) => part.type === "image").map((part) => resolveImage(part.image, this.options.maxImageBytes ?? 20 * 1024 * 1024)),
      );
      this.activeImages = parts.filter((part) => part.type === "image").map((part) => part.image);
      if (images.length > 0 && !this.options.model.input.includes("image")) {
        throw new Error(`Model ${this.options.model.id} does not declare image input support`);
      }
      await this.agent.prompt(text, images);
      const finalText = lastAssistantText(this.agent.state.messages);
      if (!output && finalText) this.emit({ type: "message_delta", runId: this.runId, delta: finalText, at: now() });
      this.emit({ type: "run_finished", runId: this.runId, reason: "completed", at: now() });
      return finalText || output;
    }
    catch (error) { const message = error instanceof Error ? error.message : String(error); this.emit({ type: "run_failed", runId: this.runId, error: message, at: now() }); throw error; }
    finally { removeCapture(); }
  }
  abort(): void { this.agent.abort(); }
  private piTools(registry: ToolRegistry): AgentTool[] {
    return registry.definitions().map((definition) => ({
      name: definition.name,
      label: definition.name,
      description: definition.description,
      parameters: definition.inputSchema as TSchema,
      execute: async (toolCallId: string, params: Record<string, unknown>, signal?: AbortSignal) => {
        const arguments_ = "image" in params
          ? { ...params, image: resolveImageReference(params.image, this.activeImages) }
          : params;
        const result = await registry.execute({ id: toolCallId, name: definition.name, arguments: arguments_ }, signal ?? new AbortController().signal);
        for (const artifact of result.artifacts ?? []) {
          if (artifact.kind === "image" && !this.activeImages.some((image) => image.uri === artifact.uri)) {
            this.activeImages.push(artifact);
          }
        }
        if (!result.ok) throw new Error(result.error?.message ?? "Tool execution failed");
        return { content: result.content as never[], details: result };
      },
    })) as AgentTool[];
  }
  private emit(event: AgentEvent): void { for (const listener of this.listeners) listener(event); }
  private forwardPiEvent(event: unknown): void {
    if (!event || typeof event !== "object") return;
    const value = event as { type?: string; [key: string]: unknown };
    if (value.type === "message_update") {
      const update = value.assistantMessageEvent as { type?: string; delta?: string } | undefined;
      if (update?.type === "text_delta" && update.delta) this.emit({ type: "message_delta", runId: this.runId, delta: update.delta, at: now() });
    }
    if (value.type === "tool_execution_start") {
      const id = value.toolCallId;
      const name = value.toolName;
      const args = value.args;
      if (typeof id === "string" && typeof name === "string" && args && typeof args === "object") {
        const call: ToolCall = { id, name, arguments: args as Record<string, unknown> };
        this.emit({ type: "tool_started", runId: this.runId, call, at: now() });
      }
    }
    if (value.type === "tool_execution_end") {
      const id = value.toolCallId;
      const name = value.toolName;
      if (typeof id !== "string" || typeof name !== "string") return;
      const piResult = value.result as { content?: unknown; details?: unknown } | undefined;
      const details = piResult?.details;
      const structured = details && typeof details === "object"
        && "callId" in details && "name" in details && "ok" in details;
      const result: ToolResult = structured
        ? details as ToolResult
        : {
            callId: id,
            name,
            ok: value.isError !== true,
            content: Array.isArray(piResult?.content)
              ? piResult.content.flatMap((part) => {
                  if (!part || typeof part !== "object") return [];
                  const candidate = part as { type?: string; text?: string };
                  return candidate.type === "text" && typeof candidate.text === "string"
                    ? [{ type: "text" as const, text: candidate.text }]
                    : [];
                })
              : [],
            ...(value.isError === true ? {
              error: {
                code: "TOOL_EXECUTION_ERROR",
                message: Array.isArray(piResult?.content)
                  ? piResult.content.flatMap((part) => {
                      if (!part || typeof part !== "object") return [];
                      const candidate = part as { type?: string; text?: string };
                      return candidate.type === "text" && typeof candidate.text === "string" ? [candidate.text] : [];
                    }).join("\n") || "Tool execution failed."
                  : "Tool execution failed.",
                retryable: true,
              },
            } : {}),
            durationMs: 0,
          };
      this.emit({ type: "tool_finished", runId: this.runId, result, at: now() });
    }
  }
}
