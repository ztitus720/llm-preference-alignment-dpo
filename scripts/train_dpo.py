"""
LoRA + DPO fine-tuning on a small open-source instruct model.

Reads data/preference_pairs.jsonl (built by build_preference_data.py), applies
LoRA to a base instruct model, and trains it with TRL's DPOTrainer. Designed to
run on a single free-tier GPU (Colab T4 / Kaggle T4x2) with a 0.5B-1.5B model.

A slice of the pairs is held out as an eval split so that training reports
`eval_rewards/accuracies` — the fraction of unseen pairs where the DPO implicit
reward ranks the preferred answer above the rejected one. That is the number
worth quoting, not the training loss.

Usage:
    python train_dpo.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 1
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

from common import dtype_flags, load_model, load_tokenizer, pick_device, pick_dtype

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "preference_pairs.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--output_dir", default="./dpo-out")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1, help="DPO KL-penalty strength")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--eval_frac", type=float, default=0.15,
                    help="fraction of pairs held out to measure preference accuracy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default=str(DATA_PATH))
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"{data_path} not found — run build_preference_data.py first.")

    device = pick_device()
    dtype = pick_dtype(device)
    print(f"[setup] device={device} dtype={dtype} "
          f"gpu={torch.cuda.get_device_name(0) if device == 'cuda' else 'n/a'}")

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    dataset = dataset.shuffle(seed=args.seed)
    n_eval = max(1, int(len(dataset) * args.eval_frac)) if args.eval_frac > 0 else 0
    eval_dataset = dataset.select(range(n_eval)) if n_eval else None
    train_dataset = dataset.select(range(n_eval, len(dataset)))
    print(f"[data] {len(train_dataset)} train pairs / {n_eval} held-out pairs")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if eval_dataset is not None:
        # evaluate.py scores exactly these pairs, so the reported number always
        # refers to data the adapter never saw.
        with (out_dir / "heldout_pairs.jsonl").open("w", encoding="utf-8") as f:
            for row in eval_dataset:
                f.write(json.dumps({k: row[k] for k in ("prompt", "chosen", "rejected")},
                                   ensure_ascii=False) + "\n")

    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, dtype=dtype, device=device)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    training_args = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        logging_steps=5,
        save_strategy="no",
        eval_strategy="epoch" if n_eval else "no",
        seed=args.seed,
        report_to="none",
        **dtype_flags(dtype),
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    if dtype is torch.float16:
        # Keep the LoRA master weights in fp32. Pure-fp16 adapter parameters
        # with an fp16 GradScaler is the usual reason a T4 run drifts to NaN
        # loss partway through; the base weights stay fp16, so memory is
        # essentially unchanged.
        n_cast = 0
        for _, param in trainer.model.named_parameters():
            if param.requires_grad and param.dtype is torch.float16:
                param.data = param.data.float()
                n_cast += 1
        print(f"[setup] cast {n_cast} trainable LoRA tensors to fp32 for stability")

    train_out = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics = dict(train_out.metrics)
    if eval_dataset is not None:
        metrics.update(trainer.evaluate())

    out_json = Path(args.output_dir) / "train_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"Saved LoRA adapter to {args.output_dir}")
    for key in ("train_runtime", "train_loss",
                "eval_loss", "eval_rewards/accuracies", "eval_rewards/margins"):
        if key in metrics:
            print(f"  {key:28s} {metrics[key]}")
    print(f"Full metrics: {out_json}")


if __name__ == "__main__":
    main()
