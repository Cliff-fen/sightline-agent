import { type ToolCall, type ToolResult } from "./protocol.js";
import { failed } from "./tool-registry.js";
export type ToolGatewayOptions = { baseUrl: string; timeoutMs?: number };

/** HTTP adapter; the Agent never needs to know where a tool is deployed. */
export class ToolGateway {
  private readonly timeoutMs: number;
  constructor(private readonly options: ToolGatewayOptions) { this.timeoutMs = options.timeoutMs ?? 30_000; }
  async execute(call: ToolCall, signal: AbortSignal): Promise<ToolResult> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    const onAbort = () => controller.abort();
    signal.addEventListener("abort", onAbort, { once: true });
    try {
      const response = await fetch(`${this.options.baseUrl}/v1/tools/${call.name}`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ callId: call.id, arguments: call.arguments }), signal: controller.signal,
      });
      const body = (await response.json()) as ToolResult;
      if (!response.ok) return failed(call, "TOOL_GATEWAY_ERROR", body.error?.message ?? response.statusText);
      return body;
    } catch (error) {
      return failed(call, "TOOL_GATEWAY_UNAVAILABLE", error instanceof Error ? error.message : String(error));
    } finally {
      clearTimeout(timer); signal.removeEventListener("abort", onAbort);
    }
  }
}
