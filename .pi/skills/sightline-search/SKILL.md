---
name: sightline-search
description: Query Sightline's web, page-reading, reverse-image, crop, and document-layout tools through its HTTP gateway. Use when a Pi session needs grounded search or visual evidence without running the full Sightline agent.
license: MIT
---

# Sightline Search

Requires Node.js 22+ and a running Sightline tool gateway.

Use `scripts/call-tool.mjs` rather than making network requests in model text. The script reads `TOOL_GATEWAY_URL`, defaulting to `http://127.0.0.1:8090`.

Choose the narrowest tool that answers the evidence need:

- `web_search` or `text_search` for ranked factual sources.
- `visit` after search when the source body is needed.
- `image_search` for a public image URL or a Sightline artifact URI.
- `crop` to create a focused visual artifact, then pass its artifact URI to another visual tool.
- `layout_parsing` for OCR and reading order.

```bash
node .pi/skills/sightline-search/scripts/call-tool.mjs \
  web_search '{"query":"Qwen3-VL technical report","topK":5}'
```

Treat `ok: false` as a failed observation. Report the structured error or try a materially different evidence route; never cite failed or empty output as evidence. Keep image artifacts and signed URLs out of public logs.
