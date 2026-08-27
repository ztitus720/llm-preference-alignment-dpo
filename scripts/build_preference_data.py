"""
Build a small preference dataset (chosen/rejected pairs) for DPO training.

Two modes:
  --source hf        Sample N pairs from an existing open preference dataset
                      (default: trl-lib/ultrafeedback_binarized), already in
                      chosen/rejected form. Fastest way to get a working
                      pipeline end to end before investing in your own data.
  --source synthetic  Build your own pairs from a list of prompts, using one
                      model (or two decoding settings of the same model) to
                      produce a "better" and a "worse" answer, which you then
                      keep or discard after a quick manual/rule-based check.
                      This is the version worth doing if you want the
                      "high-quality data synthesis" story for interviews,
                      not just "I ran a script on someone else's dataset".

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


def from_hf(n: int, dataset_name: str, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    pairs = []
    for row in ds:
        # trl-lib/ultrafeedback_binarized stores chosen/rejected as chat-style
        # message lists; normalise to plain (prompt, chosen, rejected) strings.
        prompt = row.get("prompt")
        chosen = row.get("chosen")
        rejected = row.get("rejected")

        def last_text(x):
            if isinstance(x, list):
                return x[-1]["content"] if x and isinstance(x[-1], dict) else str(x[-1])
            return x

        pairs.append({
            "prompt": prompt if isinstance(prompt, str) else last_text(prompt),
            "chosen": last_text(chosen),
            "rejected": last_text(rejected),
        })
    return pairs


def from_synthetic(n: int, model_name: str, seed: int) -> list[dict]:
    """
    Generates chosen/rejected pairs by sampling the SAME model twice per
    prompt with different decoding settings (greedy/low-temperature vs.
    high-temperature), then using a simple length + keyword heuristic as a
    stand-in "rubric" to decide which is chosen vs rejected.

    This is intentionally simple — it is a starting point, not a finished
    reward model. Swap `score_response` for something better (an LLM judge,
    a rubric checklist, human labels) before trusting the labels for a real
    write-up.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    model.eval()

    def generate(prompt: str, temperature: float) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=120,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-4),
                top_p=0.95,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()

    def score_response(text: str) -> float:
        # Placeholder rubric: penalise near-empty or wildly long answers,
        # reward answers that end with proper punctuation (crude proxy for
        # "didn't get cut off / ramble"). Replace this with a real judge.
        length_score = -abs(len(text.split()) - 60) / 60
        ends_clean = 1.0 if text.rstrip().endswith((".", "!", "?")) else -0.5
        return length_score + ends_clean

    pairs = []
    for prompt in prompts:
        a = generate(prompt, temperature=0.2)
        b = generate(prompt, temperature=1.0)
        if a == b:
            continue
        (chosen, rejected) = (a, b) if score_response(a) >= score_response(b) else (b, a)
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hf", "synthetic"], default="hf")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--dataset", default="trl-lib/ultrafeedback_binarized",
                     help="HF dataset to sample from when --source hf")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                     help="Model to self-sample from when --source synthetic")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.source == "hf":
        pairs = from_hf(args.n, args.dataset, args.seed)
    else:
        pairs = from_synthetic(args.n, args.model, args.seed)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(pairs)} preference pairs to {OUT_PATH}")


if __name__ == "__main__":
    main()
