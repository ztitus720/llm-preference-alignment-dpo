"""
Quantified base-vs-DPO comparison on held-out preference pairs.

Two things happen here:

1. A NUMBER. For every held-out pair the script computes the log-likelihood of
   the chosen and the rejected response under the base model and under the
   LoRA/DPO policy, and reports:
     - preference accuracy  : how often the model ranks chosen above rejected
                              (base vs. tuned — the movement is the result)
     - DPO implicit-reward accuracy and margin
                              r(y|x) = beta * (log p_tuned - log p_base), the
                              quantity DPO actually optimises; accuracy is the
                              fraction of pairs with r(chosen) > r(rejected).
   No API key and no LLM judge needed, so the number is reproducible by anyone
   who clones the repo.

2. A LOOK. Side-by-side generations on a fixed prompt set, so the difference is
   inspectable rather than just a statistic.

The adapter is toggled with `disable_adapter()` instead of loading a second
copy of the model, which keeps this inside a free-tier T4's memory.

Usage:
    python evaluate.py --base Qwen/Qwen2.5-1.5B-Instruct --tuned ./dpo-out \
        --pairs ./dpo-out/heldout_pairs.jsonl
"""
import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel

from common import chat_encode, load_model, load_tokenizer, pick_device

HELD_OUT_PROMPTS = [
    "How do I write a good README for a small open-source project?",
    "Explain the difference between a list and a tuple in Python.",
    "What's a respectful way to ask a colleague to speed up a late deliverable?",
    "Give me a one-paragraph explanation of gradient descent for a non-ML audience.",
    "What are the trade-offs of using a NoSQL database instead of a SQL one?",
]


def sequence_logprob(model, tok, prompt: str, response: str, device: str,
                     max_length: int = 512):
    """
    (sum log p, n_tokens) over the response tokens only, prompt masked out.

    The token count is returned so accuracy can also be reported length-
    normalised: a raw sum-of-logprobs comparison quietly rewards whichever
    response is shorter, and on some preference sets that alone reproduces most
    of the "accuracy".
    """
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )["input_ids"][0]
    resp_ids = tok(response, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if len(resp_ids) == 0:
        return float("nan"), 0

    ids = torch.cat([prompt_ids, resp_ids])[:max_length].unsqueeze(0).to(device)
    n_resp = min(len(resp_ids), max_length - len(prompt_ids))
    if n_resp <= 0:
        return float("nan"), 0

    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits.float()

    # predict token t from position t-1
    logprobs = F.log_softmax(logits[0, :-1], dim=-1)
    targets = ids[0, 1:]
    token_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_lp[-n_resp:].sum().item(), int(n_resp)


def score_pairs(model, tok, pairs, device, beta: float, max_length: int):
    """Log-likelihoods of chosen/rejected with the adapter on and off."""
    rows = []
    for i, p in enumerate(pairs, 1):
        row = {"prompt": p["prompt"]}
        for tag, ctx in (("tuned", nullcontext()), ("base", model.disable_adapter())):
            with ctx:
                for side in ("chosen", "rejected"):
                    lp, n_tok = sequence_logprob(
                        model, tok, p["prompt"], p[side], device, max_length
                    )
                    row[f"{tag}_{side}"] = lp
                    row[f"n_{side}"] = n_tok
        row["r_chosen"] = beta * (row["tuned_chosen"] - row["base_chosen"])
        row["r_rejected"] = beta * (row["tuned_rejected"] - row["base_rejected"])
        rows.append(row)
        if i % 10 == 0:
            print(f"[score] {i}/{len(pairs)} pairs")
    return rows


def summarise(rows) -> dict:
    ok = [r for r in rows if not any(r[k] != r[k] for k in  # NaN check
                                     ("tuned_chosen", "tuned_rejected",
                                      "base_chosen", "base_rejected"))]
    n = len(ok)
    if n == 0:
        raise SystemExit("No scorable pairs.")
    mean = lambda xs: sum(xs) / len(xs)
    norm = lambda r, tag, side: r[f"{tag}_{side}"] / max(r[f"n_{side}"], 1)
    return {
        "n_pairs": n,
        "base_preference_accuracy": mean([r["base_chosen"] > r["base_rejected"] for r in ok]),
        "tuned_preference_accuracy": mean([r["tuned_chosen"] > r["tuned_rejected"] for r in ok]),
        "base_preference_accuracy_len_norm":
            mean([norm(r, "base", "chosen") > norm(r, "base", "rejected") for r in ok]),
        "tuned_preference_accuracy_len_norm":
            mean([norm(r, "tuned", "chosen") > norm(r, "tuned", "rejected") for r in ok]),
        "dpo_implicit_reward_accuracy": mean([r["r_chosen"] > r["r_rejected"] for r in ok]),
        "dpo_implicit_reward_margin": mean([r["r_chosen"] - r["r_rejected"] for r in ok]),
        "mean_len_chosen": mean([r["n_chosen"] for r in ok]),
        "mean_len_rejected": mean([r["n_rejected"] for r in ok]),
    }


def generate(tok, model, prompt: str, device: str) -> str:
    enc = chat_encode(tok, prompt, device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=180, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tuned", default="./dpo-out", help="Path to the LoRA adapter dir")
    ap.add_argument("--pairs", default="./dpo-out/heldout_pairs.jsonl")
    ap.add_argument("--beta", type=float, default=0.1, help="must match training beta")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--out", default="./eval_results.jsonl")
    ap.add_argument("--metrics_out", default="./eval_metrics.json")
    ap.add_argument("--skip_generation", action="store_true")
    args = ap.parse_args()

    device = pick_device()
    tok = load_tokenizer(args.base)
    model = PeftModel.from_pretrained(load_model(args.base, device=device), args.tuned)
    model.eval()

    metrics = None
    pairs_path = Path(args.pairs)
    if pairs_path.exists():
        pairs = [json.loads(l) for l in pairs_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"[eval] scoring {len(pairs)} held-out pairs on {device}")
        rows = score_pairs(model, tok, pairs, device, args.beta, args.max_length)
        metrics = summarise(rows)
        Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    else:
        print(f"[warn] {pairs_path} not found — skipping the quantified part.")

    results = []
    if not args.skip_generation:
        for prompt in HELD_OUT_PROMPTS:
            with model.disable_adapter():
                base_out = generate(tok, model, prompt, device)
            tuned_out = generate(tok, model, prompt, device)
            results.append({"prompt": prompt, "base": base_out, "dpo_tuned": tuned_out})
            print("=" * 80)
            print("PROMPT:", prompt)
            print("-- base --\n", base_out)
            print("-- dpo_tuned --\n", tuned_out)
        with open(args.out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if metrics:
        print("\n" + "=" * 70)
        print("PASTE THIS INTO THE README'S RESULTS SECTION")
        print("=" * 70)
        print(f"- Held-out pairs: {metrics['n_pairs']}")
        print(f"- Preference accuracy (sum logprob): base "
              f"{metrics['base_preference_accuracy']:.1%} -> DPO-tuned "
              f"{metrics['tuned_preference_accuracy']:.1%}")
        print(f"- Preference accuracy (length-normalised): base "
              f"{metrics['base_preference_accuracy_len_norm']:.1%} -> DPO-tuned "
              f"{metrics['tuned_preference_accuracy_len_norm']:.1%}")
        print(f"- Mean response length: chosen {metrics['mean_len_chosen']:.0f} tok / "
              f"rejected {metrics['mean_len_rejected']:.0f} tok")
        print(f"- DPO implicit-reward accuracy: {metrics['dpo_implicit_reward_accuracy']:.1%}")
        print(f"- Mean implicit-reward margin: {metrics['dpo_implicit_reward_margin']:.4f}")


if __name__ == "__main__":
    main()
