# Supervised fine-tuning

This stage teaches the model the shared message schema: user observations,
structured tool calls, tool results, image artifacts, and final answers.
The example profile uses assistant-only loss and gradient checkpointing. It
does not prescribe an adapter or distributed strategy; set `use_lora`,
DeepSpeed, batch, and process options in a private deployment profile when
needed. The vision tower can be frozen independently.

```bash
accelerate launch sft/train.py --config sft/configs/production.yaml
```
