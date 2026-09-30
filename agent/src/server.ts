import http, { type IncomingMessage, type ServerResponse } from "node:http";
import { AGENT_CONTRACT } from "./contract.js";
import { loadRuntimeConfig } from "./model-config.js";
import { assertRunRequest, type RunResponse } from "./protocol.js";
import { SearchAgentRuntime } from "./runtime.js";
import { ToolGateway } from "./tool-gateway.js";
import { ToolRegistry } from "./tool-registry.js";
import { registerSearchTools } from "./tools.js";

const port = Number(process.env.PORT ?? 8080);
const runtimeConfig = loadRuntimeConfig();
const gateway = new ToolGateway({
  baseUrl: runtimeConfig.toolGatewayUrl,
  timeoutMs: runtimeConfig.toolTimeoutMs,
});
const registry = new ToolRegistry();
registerSearchTools(registry, gateway);

const defaultSystemPrompt = AGENT_CONTRACT.systemPrompt;

function json(response: ServerResponse, status: number, body: unknown): void {
  response.writeHead(status, { "content-type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(body));
}

async function readJson(request: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of request) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    size += buffer.length;
    if (size > 32 * 1024 * 1024) throw new Error("Request body exceeds 32 MiB");
    chunks.push(buffer);
  }
  if (chunks.length === 0) throw new Error("Request body is required");
  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

async function runAgent(request: IncomingMessage, response: ServerResponse): Promise<void> {
  if (!runtimeConfig.configured) {
    json(response, 503, {
      error: "model_not_configured",
      message: `Set ${runtimeConfig.apiKeyEnv} before using ${runtimeConfig.profile}.`,
    });
    return;
  }

  let body;
  try {
    body = assertRunRequest(await readJson(request));
  } catch (error) {
    json(response, 400, { error: "invalid_request", message: error instanceof Error ? error.message : String(error) });
    return;
  }

  const runtime = new SearchAgentRuntime({
    models: runtimeConfig.models,
    model: runtimeConfig.model,
    systemPrompt: body.systemPrompt ?? defaultSystemPrompt,
    tools: registry,
    maxTurns: runtimeConfig.maxTurns,
    maxImageBytes: runtimeConfig.maxImageBytes,
  });
  const events: RunResponse["events"] = [];
  runtime.onEvent((event) => events.push(event));

  try {
    const output = await runtime.prompt(body.input);
    json(response, 200, {
      runId: runtime.id,
      output,
      model: { profile: runtimeConfig.profile, id: runtimeConfig.model.id },
      events,
    } satisfies RunResponse);
  } catch (error) {
    json(response, 502, {
      error: "agent_run_failed",
      runId: runtime.id,
      message: error instanceof Error ? error.message : String(error),
      events,
    });
  }
}

const server = http.createServer((request, response) => {
  if (request.method === "GET" && request.url === "/health") {
    json(response, 200, {
      ok: true,
      configured: runtimeConfig.configured,
      model: {
        profile: runtimeConfig.profile,
        provider: runtimeConfig.model.provider,
        id: runtimeConfig.model.id,
        api: runtimeConfig.model.api,
        input: runtimeConfig.model.input,
      },
      tools: registry.definitions().map((tool) => tool.name),
    });
    return;
  }
  if (request.method === "GET" && request.url === "/v1/tools") {
    json(response, 200, { tools: registry.definitions() });
    return;
  }
  if (request.method === "POST" && request.url === "/v1/runs") {
    void runAgent(request, response);
    return;
  }
  json(response, 404, { error: "not_found" });
});

server.listen(port, () => {
  console.log(`Sightline Agent runtime listening on http://127.0.0.1:${port}`);
});
