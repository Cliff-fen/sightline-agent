# Sightline Architecture

## Boundaries

1. `agent-runtime`: Pi `Agent` owns the stateful model/tool loop, argument validation and sequential tool execution.
2. `model-providers`: Pi `Models` resolves credentials and translates messages for Anthropic Messages, OpenAI Responses, or OpenAI-compatible endpoints.
3. `protocol`: JSON contracts for messages, images, tool calls, results and traces.
4. `services`: search and vision tools run behind HTTP and can be deployed independently.
5. `evaluation`: fixed-set evaluation and trajectory replay consume stored events.

Python SFT and GRPO remain a separate training boundary. Any trained checkpoint is consumed through an OpenAI-compatible model gateway; training internals are not imported by the online agent, and checkpoints are deliberately excluded from the repository.

## Request flow

```text
User -> HTTP /v1/runs -> Pi Agent -> Pi Models -> selected model API
                    -> tool call -> tool gateway -> search/OCR/crop
                    <- tool result + artifacts
                    <- model answer + event stream
```

Images are referenced by `ImageRef` rather than copied into every message. A tool can return a new artifact and the next model turn receives its URI.

Before a model request, the runtime resolves each `ImageRef`, enforces a byte limit, and hands Pi normalized image content. The provider then serializes it for Anthropic Messages, OpenAI Responses, or an OpenAI-compatible endpoint.

The application does not parse XML tool tags or implement its own ReAct loop. Tool schemas are passed to Pi as TypeBox-compatible JSON Schema. Pi validates arguments, executes one action per turn, appends the tool result, and decides whether another model turn is needed.

Public image URLs can go directly to reverse-image search. Local files and crop artifacts may use temporary object staging when a search provider requires a public URL. Staged objects use short-lived access and are removed after the request.

The `local` profile accepts any user-supplied multimodal OpenAI-compatible endpoint. Remote profiles change only the Pi provider and model configuration. Text-only providers require a compatible visual endpoint before image input can be enabled. No checkpoint or model weight is distributed by this repository.

Training follows SFT initialization with tool-interactive GRPO. Answer correctness is scored by deterministic normalized match and token F1. A reviewer is optional and limited to classifying fatal execution failures; it does not decide factual correctness. Reward signals remain proxy metrics and should be audited against an independently labeled evaluation set.

## Reliability rules

- Every tool call has an id, timeout and structured error.
- Tool errors are observable and do not become silent empty strings.
- Every run emits start/finish/failure events.
- The UI consumes events, not framework-specific internal objects.
- Public traces must be scrubbed of API keys, private paths and base64 payloads.
