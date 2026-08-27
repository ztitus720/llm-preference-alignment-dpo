"""
LoRA + DPO fine-tuning on a small open-source instruct model.

Reads data/preference_pairs.jsonl (built by build_preference_data.py),
applies LoRA to a base instruct model, and trains it with TRL's DPOTrainer.
Designed to run on a single free-tier GPU (Colab T4 / Kaggle T4x2) with a
0.5B-1.5B model; scale `model_name` up if you have more GPU memory.

Usage:
    python train_dpo.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 1
"""
import argparse
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "preference_pairs.jsonl"


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
    ap.add_argument("--bf16", action="store_true", default=True)
    args = ap.parse_args()

    if not DATA_PATH.exists():
        raise SystemExit(
            f"{DATA_PATH} not found — run build_preference_data.py first."
        )

    dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")

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
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        logging_steps=5,
        save_strategy="epoch",
        bf16=args.bf16,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved LoRA-adapted model to {args.output_dir}")


if __name__ == "__main__":
    main()
