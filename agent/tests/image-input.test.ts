import test from "node:test";
import assert from "node:assert/strict";
import { resolveImage, resolveImageReference } from "../src/image-input.js";

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

test("natural aliases resolve to the only input image", () => {
  const image = { kind: "image" as const, id: "document", uri: "file:///tmp/document.png", mimeType: "image/png" };
  assert.equal(resolveImageReference("uploaded image", [image]), image.uri);
  assert.equal(resolveImageReference("original photo", [image]), image.uri);
  assert.equal(resolveImageReference("document.png", [image]), image.uri);
});

test("indexed inputs and the latest artifact resolve deterministically", () => {
  const images = [
    { kind: "image" as const, id: "first", uri: "file:///tmp/first.png" },
    { kind: "image" as const, id: "crop", uri: "artifact://crop.png" },
  ];
  assert.equal(resolveImageReference("image_0", images), images[0]?.uri);
  assert.equal(resolveImageReference("img_1", images), images[0]?.uri);
  assert.equal(resolveImageReference("img_2", images), images[1]?.uri);
  assert.equal(resolveImageReference("latest image", images), images[1]?.uri);
  assert.equal(resolveImageReference("artifact://explicit.png", images), "artifact://explicit.png");
});
