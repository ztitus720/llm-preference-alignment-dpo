# Small-Scale LLM Preference Alignment — SFT + DPO

A minimal, runnable pipeline for fine-tuning a small open-source instruct
model (default: `Qwen/Qwen2.5-1.5B-Instruct`) with LoRA + Direct Preference
Optimization (DPO), built to run end-to-end on a single free-tier GPU
(Colab T4, Kaggle T4x2).

**Status: scaffolded and verified, not yet trained.** All three scripts
were written against the current `trl`/`peft`/`transformers` APIs and their
training-loop wiring (`LoraConfig` → `DPOConfig` → `DPOTrainer`) was smoke-
tested end-to-end on a tiny locally-built model to confirm the code runs
without errors. This environment has no GPU and no access to
huggingface.co, so the actual training run — and the real before/after
numbers — needs to happen on your own GPU (Colab is enough). Once you have
those numbers, swap them into the "Results" section below and into your
resume bullet.

## Why this project

Most of the "Verifier / Reward Model / RLVR" language in job postings maps
to a fairly small set of practical skills: building a preference dataset,
running SFT and DPO/PPO-style training, and quantifying whether the
resulting model is actually better. This project is a minimal but complete
version of that loop — not a toy that only prints "training complete",
but one that produces a before/after comparison you can actually report a
number for.

## Pipeline

```
scripts/build_preference_data.py   →  data/preference_pairs.jsonl
scripts/train_dpo.py               →  ./dpo-out (LoRA adapter)
scripts/evaluate.py                →  ./eval_results.jsonl (base vs. tuned, side by side)
```

### 1. Build a preference dataset

Fastest path — sample pairs from an existing open preference dataset
(already in chosen/rejected form), to get the pipeline working end to end
before investing time in your own data:

```bash
python scripts/build_preference_data.py --source hf --n 300
```

More interesting path for a portfolio — self-sample a small model twice per
prompt at different temperatures and use a simple heuristic (swap for an
LLM judge or your own rubric) to label chosen/rejected:

```bash
python scripts/build_preference_data.py --source synthetic --n 100 \
    --model Qwen/Qwen2.5-1.5B-Instruct
```

Either way, `data/preference_pairs.jsonl` ends up with one
`{"prompt", "chosen", "rejected"}` object per line.

### 2. Train

```bash
pip install -r requirements.txt
python scripts/train_dpo.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 1
```

On a free Colab T4 with ~300 pairs at `max_length=512`, this should take on
the order of 15-30 minutes for 1 epoch. Drop to `Qwen2.5-0.5B-Instruct` if
you're short on GPU memory or time. LoRA config, batch size, and gradient
accumulation are all CLI flags — see `--help`.

### 3. Evaluate

```bash
python scripts/evaluate.py --base Qwen/Qwen2.5-1.5B-Instruct --tuned ./dpo-out
```

Prints base vs. DPO-tuned outputs side by side on 5 held-out prompts and
saves them to `eval_results.jsonl`. For a number worth quoting on a resume,
score each pair (yourself, or with an LLM judge) and report a win-rate —
"the DPO-tuned model was preferred in X/N held-out prompts" is the DPO
analogue of the AUROC numbers used in the semantic-entropy thesis project.

## Results

_Not yet run in this environment (no GPU here). Fill in after training:_

- Preference data source: `hf` / `synthetic`, N pairs
- Training time / epochs / final loss
- Win-rate on held-out prompts (tuned vs. base), and who/what judged it
- 2-3 example prompts where the difference is clearest

## What to say about this in an interview

Once trained: describe the preference-data construction choice (existing
dataset vs. self-sampled + rubric), why DPO over PPO for a project this
size (no separate reward model or RL loop to stabilize — directly
optimizes on preference pairs), what beta/KL-penalty controls, and what the
win-rate evaluation showed. If you self-sampled the data, the "rubric
design" part is worth emphasizing — it's the closest thing here to the
"Rubric RL" / "Verifier design" language in job postings.
