# Group-relative policy optimization

The policy interacts with a fresh `SearchEnvironment` for every rollout. The
environment exposes the same HTTP tools as the online Pi runtime and can feed
cropped images back into the next model turn.

```bash
accelerate launch rl/train.py --config rl/configs/production.yaml
```

The answer score is deterministic. An optional reviewer only classifies fatal
execution failures and is never used as the source of factual correctness.
