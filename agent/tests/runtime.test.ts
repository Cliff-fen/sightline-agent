import test from "node:test";
import assert from "node:assert/strict";
import { lastAssistantText } from "../src/runtime.js";

test("final assistant text is recovered when a provider emits no deltas", () => {
  const messages = [
    { role: "user", content: "question" },
    { role: "assistant", content: [{ type: "toolCall", name: "web_search" }] },
    { role: "toolResult", content: [{ type: "text", text: "evidence" }] },
    { role: "assistant", content: [{ type: "text", text: "Grounded answer." }, { type: "thinking", thinking: "hidden" }] },
  ];
  assert.equal(lastAssistantText(messages), "Grounded answer.");
});

test("final assistant text ignores tool-only messages", () => {
  assert.equal(lastAssistantText([{ role: "assistant", content: [{ type: "toolCall", name: "crop" }] }]), "");
});
