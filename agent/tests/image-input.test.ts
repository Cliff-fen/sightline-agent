import test from "node:test";
import assert from "node:assert/strict";
import { resolveImage } from "../src/image-input.js";

test("data URI images are converted to model image content", async () => {
  const image = await resolveImage({ kind: "image", id: "one", uri: "data:image/png;base64,aGVsbG8=" }, 100);
  assert.deepEqual(image, { type: "image", data: "aGVsbG8=", mimeType: "image/png" });
});

test("image size limit is enforced before model submission", async () => {
  await assert.rejects(
    resolveImage({ kind: "image", id: "large", uri: "data:image/png;base64,aGVsbG8=" }, 2),
    /exceeds 2 bytes/,
  );
});
