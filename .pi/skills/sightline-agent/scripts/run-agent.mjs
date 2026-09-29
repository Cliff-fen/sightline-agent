#!/usr/bin/env node

const [prompt, imageUri] = process.argv.slice(2);
if (!prompt) {
  console.error("usage: run-agent.mjs <prompt> [image-uri]");
  process.exit(2);
}

const input = [{ type: "text", text: prompt }];
if (imageUri) {
  input.push({ type: "image", image: { kind: "image", id: `skill-${crypto.randomUUID()}`, uri: imageUri } });
}
const baseUrl = (process.env.SIGHTLINE_AGENT_URL ?? "http://127.0.0.1:8080").replace(/\/$/, "");
const response = await fetch(`${baseUrl}/v1/runs`, {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ input }),
  signal: AbortSignal.timeout(10 * 60_000),
});
const text = await response.text();
let payload;
try {
  payload = JSON.parse(text);
} catch {
  console.error(`agent returned non-JSON HTTP ${response.status}: ${text.slice(0, 500)}`);
  process.exit(1);
}
console.log(JSON.stringify(payload, null, 2));
if (!response.ok) process.exit(1);
