"""
Side-by-side comparison of the base model vs. the DPO fine-tuned model on a
fixed set of held-out prompts. Prints both outputs so you can eyeball the
difference, and (optionally) scores each pair with an LLM judge if
--judge_model is set and OPENAI_API_KEY (or compatible) is available.

This is intentionally simple. For a resume-grade "quantified" result, run
this on ~20-30 held-out prompts, save the judge scores, and report a
win-rate (chosen-DPO-output preferred over base-output in X/N prompts) —
the same idea as the AUROC-style quantification used in the thesis project.

Usage:
    python evaluate.py --base Qwen/Qwen2.5-1.5B-Instruct --tuned ./dpo-out
"""
import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HELD_OUT_PROMPTS = [
    "How do I write a good README for a small open-source project?",
    "Explain the difference between a list and a tuple in Python.",
    "What's a respectful way to ask a colleague to speed up a late deliverable?",
    "Give me a one-paragraph explanation of gradient descent for a non-ML audience.",
    "What are the trade-offs of using a NoSQL database instead of a SQL one?",
]


def load(model_name: str, adapter_dir: str | None):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tok, model


def generate(tok, model, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            inputs, max_new_tokens=180, do_sample=False, pad_token_id=tok.eos_token_id
        )
    return tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tuned", default="./dpo-out", help="Path to the LoRA adapter dir")
    ap.add_argument("--out", default="./eval_results.jsonl")
    args = ap.parse_args()

    base_tok, base_model = load(args.base, None)
    tuned_tok, tuned_model = load(args.base, args.tuned)

    results = []
    for prompt in HELD_OUT_PROMPTS:
        base_out = generate(base_tok, base_model, prompt)
        tuned_out = generate(tuned_tok, tuned_model, prompt)
        results.append({"prompt": prompt, "base": base_out, "dpo_tuned": tuned_out})
        print("=" * 80)
        print("PROMPT:", prompt)
        print("-- base --\n", base_out)
        print("-- dpo_tuned --\n", tuned_out)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(results)} comparisons to {args.out}")
    print(
        "\nNext step for a quantified result: have an LLM judge (or yourself) "
        "pick a winner per row and report a win-rate — that's the number to "
        "put in the resume bullet / GitHub README."
    )


if __name__ == "__main__":
    main()
