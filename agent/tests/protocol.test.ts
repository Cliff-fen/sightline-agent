import test from "node:test";
import assert from "node:assert/strict";
import { assertRunRequest, assertToolCall, createId, textContent } from "../src/protocol.js";
test("tool calls have a stable id and typed arguments", () => {
  const call = assertToolCall({ id: createId("call"), name: "web_search", arguments: { query: "multimodal agents" } });
  assert.equal(call.name, "web_search"); assert.deepEqual(call.arguments, { query: "multimodal agents" });
});
test("text content is represented as a content part", () => { assert.deepEqual(textContent("hello"), [{ type: "text", text: "hello" }]); });
test("malformed tool calls are rejected", () => { assert.throws(() => assertToolCall({ name: "web_search" })); });
test("run requests accept text and image references", () => {
  const request = assertRunRequest({ input: [
    { type: "text", text: "What is shown?" },
    { type: "image", image: { kind: "image", id: "img_1", uri: "https://example.com/image.jpg" } },
  ] });
  assert.equal(Array.isArray(request.input), true);
});
test("run requests reject malformed content parts", () => {
  assert.throws(() => assertRunRequest({ input: [{ type: "image", image: { uri: "missing fields" } }] }), /Invalid content part/);
});
