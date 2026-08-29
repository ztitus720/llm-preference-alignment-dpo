"""
Build a small preference dataset (chosen/rejected pairs) for DPO training.

Two modes:
  --source hf         Sample N pairs from an existing open preference dataset
                      (default: trl-lib/ultrafeedback_binarized). Fastest way
                      to get the pipeline working end to end.
  --source synthetic  Self-sample one model twice per prompt at different
                      temperatures and label chosen/rejected with a rubric.
                      This is the version worth doing for the data-synthesis
                      story; the rubric is deliberately simple and is the part
                      you should replace with a real judge.

Output: data/preference_pairs.jsonl, one {"prompt", "chosen", "rejected"} per
line, directly loadable by train_dpo.py.

Usage:
    python build_preference_data.py --source hf --n 300
    python build_preference_data.py --source synthetic --n 100 --model Qwen/Qwen2.5-1.5B-Instruct
"""
import argparse
import json
import random
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "preference_pairs.jsonl"


def _last_assistant_text(x):
    """Conversational columns are message lists; plain ones are already strings."""
    if isinstance(x, list) and x:
        last = x[-1]
        return last["content"] if isinstance(last, dict) else str(last)
    return x if isinstance(x, str) else None


def _first_user_text(x):
    if isinstance(x, list):
        for msg in x:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg["content"]
    return None


def from_hf(n: int, dataset_name: str, seed: int, split: str = "train") -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    pairs, skipped = [], 0
    for row in ds:
        chosen_msgs, rejected_msgs = row.get("chosen"), row.get("rejected")

        # trl-lib/ultrafeedback_binarized ships NO "prompt" column — the prompt
        # is the user turn inside the chosen/rejected message lists. Reading
        # row["prompt"] there yields None for every row, which silently poisons
        # the whole dataset, so fall back to the first user message.
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            prompt = _first_user_text(chosen_msgs) or _first_user_text(rejected_msgs)

        chosen = _last_assistant_text(chosen_msgs)
        rejected = _last_assistant_text(rejected_msgs)

        if not (prompt and chosen and rejected) or chosen == rejected:
            skipped += 1
            continue
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    if skipped:
        print(f"[warn] skipped {skipped} rows with missing/degenerate fields")
    if not pairs:
        raise SystemExit(
            f"No usable pairs from {dataset_name}. Columns seen: {ds.column_names}"
        )
    return pairs


def from_synthetic(n: int, model_name: str, seed: int) -> list[dict]:
    """
    Generate chosen/rejected pairs by sampling the SAME model twice per prompt
    (low vs. high temperature), then scoring both with a simple rubric.

    The rubric is a starting point, not a reward model. Swap `score_response`
    for an LLM judge or a rubric checklist before trusting the labels.
    """
    import torch

    from common import chat_encode, load_model, load_tokenizer, pick_device

    random.seed(seed)

    PROMPTS = [
        "Explain what a hash table is and why it's fast, in 3-4 sentences.",
        "What's a common mistake beginners make when learning Python?",
        "Summarize the plot of Romeo and Juliet in two sentences.",
        "Give me a polite way to decline a meeting invite.",
        "What is the difference between TCP and UDP?",
        "Suggest three tips for writing a clear commit message.",
        "Explain overfitting in machine learning to a beginner.",
        "What should I consider before choosing a database for a new project?",
        "Write a short, friendly reminder email about a deadline.",
        "Explain what an API rate limit is and why services use it.",
    ]
    prompts = (PROMPTS * ((n // len(PROMPTS)) + 1))[:n]

    device = pick_device()
    tok = load_tokenizer(model_name)
    model = load_model(model_name, device=device)
    model.eval()

    def generate(prompt: str, temperature: float) -> str:
        enc = chat_encode(tok, prompt, device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=120,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-4),
                top_p=0.95,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def score_response(text: str) -> float:
        # Placeholder rubric: penalise near-empty or rambling answers, reward
        # answers that end on proper punctuation (crude "wasn't cut off" proxy).
        length_score = -abs(len(text.split()) - 60) / 60
        ends_clean = 1.0 if text.rstrip().endswith((".", "!", "?")) else -0.5
        return length_score + ends_clean

    pairs = []
    for i, prompt in enumerate(prompts, 1):
        a = generate(prompt, temperature=0.2)
        b = generate(prompt, temperature=1.0)
        if not a or not b or a == b:
            continue
        chosen, rejected = (a, b) if score_response(a) >= score_response(b) else (b, a)
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        if i % 10 == 0:
            print(f"[synthetic] {i}/{len(prompts)} prompts sampled")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "synthetic"], default="hf")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--dataset", default="trl-lib/ultrafeedback_binarized",
                    help="HF dataset to sample from when --source hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="Model to self-sample from when --source synthetic")
    ap.add_argument("--split", default="train",
                    help='dataset split; HuggingFaceH4/ultrafeedback_binarized uses "train_prefs"')
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    if args.source == "hf":
        pairs = from_hf(args.n, args.dataset, args.seed, args.split)
    else:
        pairs = from_synthetic(args.n, args.model, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(pairs)} preference pairs to {out_path}")


if __name__ == "__main__":
    main()
