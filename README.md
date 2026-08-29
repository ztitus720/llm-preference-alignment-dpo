# Small-Scale LLM Preference Alignment — SFT + DPO

A minimal, runnable pipeline for fine-tuning a small open-source instruct model
(default: `Qwen/Qwen2.5-1.5B-Instruct`) with LoRA + Direct Preference
Optimization (DPO), and for measuring whether the result is actually better —
on held-out data, with no API key and no LLM judge required.

Everything runs end-to-end on a single free-tier GPU (Colab T4, Kaggle T4x2).
`run_dpo_colab.ipynb` does the whole thing in one "Run all".

## Why this project

Most of the "verifier / reward model / RLVR" vocabulary in current alignment
work maps onto a fairly small set of practical skills: building a preference
dataset, running SFT and DPO/PPO-style training, and quantifying whether the
resulting model is actually better. This project is a minimal but complete
version of that loop — not a toy that prints "training complete", but one that
produces a before/after comparison you can report a real number for.

## Pipeline

```
scripts/build_preference_data.py   →  data/preference_pairs.jsonl
scripts/train_dpo.py               →  ./dpo-out  (LoRA adapter + heldout_pairs.jsonl)
scripts/evaluate.py                →  ./eval_metrics.json + ./eval_results.jsonl
```

### 1. Build a preference dataset

Fastest path — sample pairs from an existing open preference dataset, to get the
pipeline working end to end before investing time in your own data:

```bash
python scripts/build_preference_data.py --source hf --n 300
```

More interesting path — self-sample a small model twice per prompt at different
temperatures and use a simple rubric (swap for an LLM judge or your own
checklist) to label chosen/rejected:

```bash
python scripts/build_preference_data.py --source synthetic --n 100 \
    --model Qwen/Qwen2.5-1.5B-Instruct --out data/synthetic_pairs.jsonl
```

Either way the output is one `{"prompt", "chosen", "rejected"}` object per line.
Conversational datasets that carry the prompt inside the message list rather
than in a `prompt` column are handled — the user turn is recovered from the
chosen/rejected messages instead of silently becoming `None`.

### 2. Train

```bash
pip install -r requirements.txt
python scripts/train_dpo.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 1
```

`--eval_frac` (default 0.15) holds out a slice of the pairs, writes them to
`dpo-out/heldout_pairs.jsonl`, and reports `eval_rewards/accuracies` during
training. Precision is chosen from the actual GPU: bf16 only where the card
supports it, fp16 on Turing cards like the T4, with the LoRA master weights kept
in fp32 so long fp16 runs don't drift to NaN. LoRA config, batch size, and
gradient accumulation are CLI flags — see `--help`.

On a free Colab T4 with ~300 pairs at `max_length=512`, one epoch takes roughly
20-30 minutes. Drop to `Qwen2.5-0.5B-Instruct` if you are short on memory.

### 3. Evaluate

```bash
python scripts/evaluate.py --base Qwen/Qwen2.5-1.5B-Instruct --tuned ./dpo-out \
    --pairs ./dpo-out/heldout_pairs.jsonl
```

For every held-out pair the script computes the log-likelihood of the chosen and
the rejected response under the base model and under the tuned policy (the same
weights with the adapter toggled off, so no second copy of the model is loaded),
and reports:

- **preference accuracy** — how often the model ranks chosen above rejected,
  before and after DPO. Reported both as a raw sum of log-probabilities and
  length-normalised, because the raw version quietly rewards whichever response
  is shorter.
- **DPO implicit-reward accuracy and margin** — `r(y|x) = β·(log p_tuned −
  log p_base)`, the quantity DPO actually optimises, with accuracy being the
  fraction of pairs where `r(chosen) > r(rejected)`. This is computed
  independently of TRL and should agree with the `eval_rewards/accuracies` the
  trainer reported.

It also prints side-by-side generations on five fixed prompts, so the difference
is inspectable and not only a statistic.

## Results

_Filled in by the last cell of `run_dpo_colab.ipynb` after a run._

## Design notes

- **DPO over PPO** for a project this size: no separate reward model to train and
  no RL loop to stabilise — DPO optimises directly on preference pairs, with the
  frozen base model acting as the reference policy.
- **β (0.1 by default)** controls how far the policy is allowed to move from that
  reference. Lower β lets the policy drift further and overfit the preference
  set; higher β keeps it close and blunts the effect.
- **LoRA on q/k/v/o only** (r=16): enough capacity to shift preferences without
  touching the MLP blocks, which is what keeps a 1.5B run inside 16GB.
- **Rubric design is where most of the signal in preference data comes from.**
  The heuristic in `build_preference_data.py` (length target + clean ending) is
  deliberately crude and is the first thing worth replacing — the labels set the
  ceiling on everything downstream.
- **Length is the standard confound** in preference evaluation, which is why the
  accuracy is reported both ways and the mean chosen/rejected lengths are printed
  alongside.
