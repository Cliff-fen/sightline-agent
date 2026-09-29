import { type ToolCall, type ToolDefinition, type ToolHandler, type ToolResult, textContent } from "./protocol.js";
export type RegisteredTool = ToolDefinition & { execute: ToolHandler };

export class ToolRegistry {
  private readonly tools = new Map<string, RegisteredTool>();
  register(tool: RegisteredTool): void {
    if (this.tools.has(tool.name)) throw new Error(`Duplicate tool: ${tool.name}`);
    this.tools.set(tool.name, tool);
  }
  get(name: string): RegisteredTool {
    const tool = this.tools.get(name);
    if (!tool) throw new Error(`Unknown tool: ${name}`);
    return tool;
  }
  definitions(): ToolDefinition[] {
    return [...this.tools.values()].map(({ execute: _execute, ...definition }) => definition);
  }
  async execute(call: ToolCall, signal: AbortSignal): Promise<ToolResult> {
    const started = performance.now();
    try {
      const result = await this.get(call.name).execute(call, signal);
      return { ...result, durationMs: Math.round(performance.now() - started) };
    } catch (error) {
      return {
        callId: call.id, name: call.name, ok: false, content: textContent("Tool execution failed."),
        error: { code: "TOOL_EXECUTION_ERROR", message: error instanceof Error ? error.message : String(error), retryable: true },
        durationMs: Math.round(performance.now() - started),
      };
    }
  }
}
export function result(call: ToolCall, text: string): ToolResult {
  return { callId: call.id, name: call.name, ok: true, content: textContent(text), durationMs: 0 };
}
export function failed(call: ToolCall, code: string, message: string): ToolResult {
  return { callId: call.id, name: call.name, ok: false, content: textContent(message), error: { code, message, retryable: false }, durationMs: 0 };
}
