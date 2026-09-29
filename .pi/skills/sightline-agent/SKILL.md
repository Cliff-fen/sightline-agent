---
name: sightline-agent
description: Run a complete grounded multimodal investigation through the Sightline Pi agent service. Use when a task needs the agent to inspect images, select tools, gather evidence over multiple turns, and return one final answer.
license: MIT
---

# Sightline Agent

Requires Node.js 22+ and a running Sightline agent HTTP service.

Delegate the whole investigation to the running service with `scripts/run-agent.mjs`. Do not reproduce its tool loop in the calling Pi session: Sightline's Pi runtime owns tool selection, parallel execution, turn state, limits, and provider message conversion.

```bash
node .pi/skills/sightline-agent/scripts/run-agent.mjs \
  "Identify this venue and support the answer with web evidence." \
  "https://example.com/photo.jpg"
```

`SIGHTLINE_AGENT_URL` defaults to `http://127.0.0.1:8080`. The optional second argument may be an HTTP URL, data URI, file URL, or Sightline artifact URI accepted by the service.

Use the returned final answer as an investigation result, not as an infallible fact. If the request fails, preserve the HTTP error and agent telemetry; do not silently replace it with an ungrounded answer.
