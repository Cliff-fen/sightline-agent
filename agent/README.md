# Online agent

The runtime uses Pi for provider messages, conversation state, and tool dispatch.
Sightline adds the multimodal request contract,
bounded image ingestion, an HTTP service surface, and a tool gateway.

| Path | Purpose |
| --- | --- |
| `src/runtime.ts` | Creates one Pi session per run and streams structured events |
| `src/tool-registry.ts` | Registers the model-visible tools and JSON schemas |
| `src/tool-gateway.ts` | Calls the Python gateway and preserves structured failures |
| `src/image-input.ts` | Validates URLs, local files, data URIs, and image limits |
| `services/tool_gateway.py` | Search, page reading, image search, crop, and OCR implementations |

The model does not receive API credentials and cannot perform arbitrary network
requests. Search credentials stay in the gateway process. Local images are
validated and bounded before they reach the provider.

Run `npm test` for the protocol/runtime tests and see the repository root README
for full setup and API examples.
