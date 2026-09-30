import test from "node:test";
import assert from "node:assert/strict";
import { loadRuntimeConfig } from "../src/model-config.js";

test("local is the default multimodal profile", () => {
  const config = loadRuntimeConfig({});
  assert.equal(config.profile, "local");
  assert.equal(config.model.id, "local-model");
  assert.deepEqual(config.model.input, ["text", "image"]);
  assert.equal(config.model.maxTokens, 4096);
  assert.equal(config.maxTurns, 50);
});

test("local profile accepts an arbitrary model name and endpoint", () => {
  const config = loadRuntimeConfig({ MODEL_PROFILE: "local", MODEL_NAME: "my-vlm", MODEL_BASE_URL: "http://model:8000/v1" });
  assert.equal(config.model.id, "my-vlm");
  assert.equal(config.model.baseUrl, "http://model:8000/v1");
});

test("explicit model name supports arbitrary local checkpoints", () => {
  const config = loadRuntimeConfig({ MODEL_PROFILE: "local", MODEL_NAME: "org/multimodal-model" });
  assert.equal(config.model.id, "org/multimodal-model");
});

test("Claude uses Pi's Anthropic Messages provider", () => {
  const config = loadRuntimeConfig({ MODEL_PROFILE: "claude", ANTHROPIC_API_KEY: "test-key" });
  assert.equal(config.model.provider, "anthropic");
  assert.equal(config.model.id, "claude-sonnet-5");
  assert.equal(config.model.api, "anthropic-messages");
  assert.equal(config.configured, true);
});

test("OpenAI uses Pi's Responses provider", () => {
  const config = loadRuntimeConfig({ MODEL_PROFILE: "openai", MODEL_BASE_URL: "https://relay.example/v1", MODEL_NAME: "test-model", OPENAI_API_KEY: "test-key" });
  assert.equal(config.model.provider, "openai-relay");
  assert.equal(config.model.id, "test-model");
  assert.equal(config.model.api, "openai-responses");
  assert.deepEqual(config.model.input, ["text", "image"]);
});

test("DeepSeek uses the official text model by default", () => {
  const config = loadRuntimeConfig({ MODEL_PROFILE: "deepseek", DEEPSEEK_API_KEY: "test-key" });
  assert.equal(config.model.provider, "deepseek");
  assert.equal(config.model.id, "deepseek-chat");
  assert.equal(config.model.api, "openai-completions");
  assert.deepEqual(config.model.input, ["text"]);
});

test("a multimodal DeepSeek-compatible relay must opt into image input", () => {
  const config = loadRuntimeConfig({
    MODEL_PROFILE: "deepseek",
    MODEL_NAME: "custom-vision-model",
    MODEL_BASE_URL: "https://relay.example/v1",
    MODEL_INPUT: "text,image",
    DEEPSEEK_API_KEY: "test-key",
  });
  assert.deepEqual(config.model.input, ["text", "image"]);
});

test("remote profiles require their provider-specific API keys", () => {
  assert.equal(loadRuntimeConfig({ MODEL_PROFILE: "claude", MODEL_API_KEY: "wrong-variable" }).configured, false);
  assert.equal(loadRuntimeConfig({ MODEL_PROFILE: "deepseek" }).configured, false);
  assert.equal(loadRuntimeConfig({ MODEL_PROFILE: "openai" }).configured, false);
});

test("invalid profiles and non-http endpoints fail fast", () => {
  assert.throws(() => loadRuntimeConfig({ MODEL_PROFILE: "70b" }), /MODEL_PROFILE/);
  assert.throws(() => loadRuntimeConfig({ MODEL_BASE_URL: "file:///tmp/model" }), /http or https/);
});

test("tool timeout is configurable and defaults to five minutes", () => {
  assert.equal(loadRuntimeConfig({}).toolTimeoutMs, 300_000);
  assert.equal(loadRuntimeConfig({ TOOL_TIMEOUT_MS: "45000" }).toolTimeoutMs, 45_000);
  assert.throws(() => loadRuntimeConfig({ TOOL_TIMEOUT_MS: "0" }), /positive integer/);
});
