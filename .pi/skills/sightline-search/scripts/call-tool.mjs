#!/usr/bin/env node

const [tool, rawArguments] = process.argv.slice(2);
if (!tool || !rawArguments) {
  console.error("usage: call-tool.mjs <tool-name> '<json-arguments>'");
  process.exit(2);
}

let args;
try {
  args = JSON.parse(rawArguments);
} catch (error) {
  console.error(`invalid JSON arguments: ${error.message}`);
  process.exit(2);
}

const baseUrl = (process.env.TOOL_GATEWAY_URL ?? "http://127.0.0.1:8090").replace(/\/$/, "");
const response = await fetch(`${baseUrl}/v1/tools/${encodeURIComponent(tool)}`, {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ callId: `skill-${crypto.randomUUID()}`, arguments: args }),
  signal: AbortSignal.timeout(90_000),
});
const text = await response.text();
let payload;
try {
  payload = JSON.parse(text);
} catch {
  console.error(`gateway returned non-JSON HTTP ${response.status}: ${text.slice(0, 500)}`);
  process.exit(1);
}
console.log(JSON.stringify(payload, null, 2));
if (!response.ok || payload.ok !== true) process.exit(1);
