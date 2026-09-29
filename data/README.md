# Data contract

`build_datasets.py` converts trajectory records into two explicit contracts:

- SFT: `messages`, `images`, and `tools`.
- GRPO: `prompt`, `images`, `answer`, and optional metadata.

Images stay as paths in JSONL and are decoded by `datasets.Image` at load time.
The converter rejects unmatched tool results and incomplete tool calls instead
of silently training on broken trajectories.

## Generate grounded data

The generation package is an explicit, restartable pipeline:

1. Sample a legal English Wikipedia seed and a 2-4 hop hyperlink path. It
   removes disambiguation/list pages, namespace pages, cycles, and high-degree
   hub nodes.
2. Retrieve representative Wikimedia Commons images and optionally rank them
   with CLIP. The image is anchored to the source node, not the answer node.
3. Use the configured model relay to construct a canonical question, rewrite
   intermediate entities into non-leaking descriptions, and audit answer
   invariance, uniqueness, visual grounding, tool necessity, and non-triviality.
   A fixed path and image receive up to three terminal-fact candidates, each
   with up to three independently audited rewrites. Tool-free or predictable
   answers are retired before the next terminal fact is tried. All failed
   candidates remain in the audit record.
4. Optionally create a small degraded-image subset and require a restoration
   tool in its metadata.
5. Apply two independent difficulty gates: a strict tool-free attempt filter
   and a single reverse-image-search filter. Samples that pass both are sent
   to the running Agent service for five independent tool trajectories by
   default. The same relay performs answer and process rejection checks; failed
   samples remain in `audit.jsonl` with their rejection reason and never enter
   training data.

This builder intentionally contains only the Wikimedia path-sampling stream.
It does not mix in external VQA corpora; additional sources can be added later
as separate adapters without changing this data contract.

All intermediate files are JSONL so a failed API request can be retried from
the last completed stage. Credentials are read from environment variables and
are never written to the generated records.

Install the stage dependencies and run a one-sample connectivity smoke test:

```bash
conda run -n sightline-gen python -m pip install -r data/requirements-generation.txt
set -a; source /path/to/private/model.env; set +a
conda run --no-capture-output -n sightline-gen python -m data.generation.pipeline \
  --output-dir /tmp/sightline-generation-smoke \
  --samples 1 --skip-clip --no-enhancement
```

For quality data, omit `--skip-clip`, add `--generate-trajectories`, and use a
real `--agent-url`. The checked-in default threshold is `0.28`, the image
candidate count is 8, and trajectory rejection sampling uses 5 candidates per
accepted item. Adjust these only after inspecting the audit distribution.
