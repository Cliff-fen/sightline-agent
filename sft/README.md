# Supervised fine-tuning

This stage teaches the model the shared message schema: user observations,
structured tool calls, tool results, image artifacts, and final answers.
Vision tokens are never truncated (`max_length: null`).

```bash
accelerate launch sft/train.py --config sft/configs/production.yaml
```

The production profile uses per-device batch 2 with two accumulation steps.
On eight GPUs the effective optimizer batch is `8 x 2 x 2 = 32`.
