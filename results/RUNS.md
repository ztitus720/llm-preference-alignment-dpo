# Run log

Both runs: single Colab free-tier Tesla T4 (compute capability 7.5, fp16),
`Qwen/Qwen2.5-1.5B-Instruct`, LoRA r=16 / alpha=32 on q,k,v,o, beta 0.1,
batch 2 x grad_accum 8, seed 42.
Data: `trl-lib/ultrafeedback_binarized`, 299 usable pairs (1 row skipped),
eval_frac 0.15 -> 255 train / 44 held out. Date: 2026-08-30.

| | Run A | Run B |
|---|---|---|
| epochs | 1 | 3 |
| lr | 5e-6 | 5e-5 |
| train max_length | 512 | 768 |
| eval max_length | 512 | 1024 |
| optimiser steps | 15 | 48 |
| adapter dir | `dpo-out` | `dpo-out-b` |

Commands:

```bash
# Run A
python scripts/train_dpo.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --epochs 1 --batch_size 2 --grad_accum 8 --max_length 512 --beta 0.1 \
    --output_dir ./dpo-out
python scripts/evaluate.py --base Qwen/Qwen2.5-1.5B-Instruct \
    --tuned ./dpo-out --pairs ./dpo-out/heldout_pairs.jsonl --beta 0.1

# Run B
python scripts/train_dpo.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --epochs 3 --lr 5e-5 --batch_size 2 --grad_accum 8 --max_length 768 \
    --beta 0.1 --output_dir ./dpo-out-b
python scripts/evaluate.py --base Qwen/Qwen2.5-1.5B-Instruct \
    --tuned ./dpo-out-b --pairs ./dpo-out-b/heldout_pairs.jsonl \
    --beta 0.1 --max_length 1024 --skip_generation
```

Environment fix required on the current Colab image, before either run:

```bash
pip uninstall -y torchao   # peft 0.20.0 rejects the preinstalled torchao 0.10.0
```

See the README's Results section for what the numbers mean.
