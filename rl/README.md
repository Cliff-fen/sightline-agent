# Group-relative policy optimization

The policy interacts with a fresh `SearchEnvironment` for every rollout. The
environment exposes the same HTTP tools as the online Pi runtime and can feed
cropped images back into the next model turn.

The example profile leaves adapter and distributed-training choices open. Each
prompt produces a configurable group of sampled trajectories. Runtime limits
are loaded from the shared Agent contract so online inference and training do
not silently diverge.

The scalar reward is:

```text
format * (0.8 * accuracy + 0.2 * query_utility)
```

`accuracy` is a binary semantic judgment over the question, reference answer,
and final report. `query_utility` is a 0-1 judgment over the search sequence and
its observations. The format term is the mean validity of model turns, with
failed tool steps receiving zero. Both judgments are required; a failed Judge
request falls back to exact match for accuracy and zero for query utility.

The Judge implementation is backend-agnostic. Configure any compatible
endpoint without committing credentials:

```bash
export JUDGE_API_BASE_URL=http://127.0.0.1:9000/v1
export JUDGE_API_KEY=
export JUDGE_MODEL=<model-id>
```

The query Judge inherits these values unless `QUERY_JUDGE_API_BASE_URL`,
`QUERY_JUDGE_API_KEY`, or `QUERY_JUDGE_MODEL` is set. Then launch:

```bash
accelerate launch rl/train.py --config rl/configs/production.yaml
```

The complete correctness and query-utility prompts are defined in
[`rewards.py`](rewards.py) and are part of the public evaluation contract. No
particular Judge model is required by the repository.
