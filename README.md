# 小规模 LLM 偏好对齐 — SFT + DPO

**中文在前，英文全文见下半部分 → [English version](#english-version)**

一条最小可运行的偏好对齐管线：在一个小体量开源 instruct 模型（默认
`Qwen/Qwen2.5-1.5B-Instruct`）上用 LoRA + DPO（Direct Preference Optimization，
直接偏好优化）做微调，并且**在留出数据上量化它到底有没有变好**——不需要 API key，
不需要 LLM 当裁判。

全流程可以在一张免费 GPU（Colab T4 / Kaggle T4x2）上跑完，`run_dpo_colab.ipynb`
一次「全部运行」即可。

## 为什么做这个项目

当前对齐方向的一堆词汇——verifier、reward model、RLVR——落到工程上其实就是一小组
可动手的技能：构造偏好数据集、跑 SFT 与 DPO/PPO 式训练、量化结果到底有没有变好。
这个仓库是那条闭环的最小完整版：不是一个打印「training complete」就结束的玩具，而是
能产出一个可以被追问的 before/after 对比数字。

结果本身是一个**阴性结果**，而它为什么是阴性的，是这个仓库里最有信息量的部分。
见 [实验结果](#实验结果)。

## 管线

```
scripts/build_preference_data.py   →  data/preference_pairs.jsonl
scripts/train_dpo.py               →  ./dpo-out  (LoRA 适配器 + heldout_pairs.jsonl)
scripts/evaluate.py                →  ./eval_metrics.json + ./eval_results.jsonl
```

### 1. 构造偏好数据集

最快的路径——从现成的开源偏好数据集里采样，先把管线端到端跑通，再考虑自己造数据：

```bash
python scripts/build_preference_data.py --source hf --n 300
```

更有意思的路径——让同一个模型对同一个问题在两个温度下各采样一次，再用一条打分规则
（rubric，可替换成 LLM judge 或你自己的检查清单）标出 chosen / rejected：

```bash
python scripts/build_preference_data.py --source synthetic --n 100 \
    --model Qwen/Qwen2.5-1.5B-Instruct --out data/synthetic_pairs.jsonl
```

两条路径的输出都是每行一个 `{"prompt", "chosen", "rejected"}`。对于把 prompt 藏在
消息列表里、而不是单独放一列的对话式数据集，脚本会从 chosen/rejected 的 user 轮里
把它捞出来，而不是让它悄悄变成 `None`。

### 2. 训练

```bash
pip install -r requirements.txt
python scripts/train_dpo.py --model Qwen/Qwen2.5-1.5B-Instruct --epochs 1
```

`--eval_frac`（默认 0.15）会切出一部分数据作为留出集，写进
`dpo-out/heldout_pairs.jsonl`，并在训练过程中报告 `eval_rewards/accuracies`。
精度按显卡实际能力选择：只有真正支持 bf16 的卡才用 bf16，T4 这类 Turing 卡走 fp16，
同时把 LoRA 的可训练参数保留在 fp32，避免长时间 fp16 训练漂成 NaN。LoRA 配置、
batch size、梯度累积都是命令行参数，见 `--help`。

在免费 Colab T4 上实测：299 对数据、`max_length=512` 跑 1 个 epoch 是 **15 步优化、
3.2 分钟**；`max_length=768` 跑 3 个 epoch 是 **48 步、13.8 分钟**。显存吃紧就换成
`Qwen2.5-0.5B-Instruct`。

### 3. 评测

```bash
python scripts/evaluate.py --base Qwen/Qwen2.5-1.5B-Instruct --tuned ./dpo-out \
    --pairs ./dpo-out/heldout_pairs.jsonl
```

对每一对留出数据，脚本分别计算 chosen 与 rejected 在 base 模型和微调后策略下的
对数似然（同一份权重，把适配器开关一下即可，不用在显存里装第二份模型），并报告：

- **偏好准确率（preference accuracy）**——模型把 chosen 排在 rejected 前面的比例，
  微调前后各一个。同时给出原始版和长度归一化版，因为直接把每个 token 的对数概率
  加起来会偏袒更短的回答。
- **DPO 隐式奖励准确率与 margin**——`r(y|x) = β·(log p_微调后 − log p_原始)`，也就是
  DPO 真正在优化的那个量；准确率是 `r(chosen) > r(rejected)` 的比例。这一项独立于
  TRL 计算，应当与训练器报告的 `eval_rewards/accuracies` 相互印证。

脚本还会在 5 个固定 prompt 上打印微调前后的对照生成，让差异可以被肉眼检查，而不只是
一个统计量。

## 实验结果

两组实验，均在同一张 Colab 免费 Tesla T4 上完成，2026-08-30。数据：
`trl-lib/ultrafeedback_binarized`，299 对可用样本，15% 留出。留出集在训练开始前就写进
`dpo-out*/heldout_pairs.jsonl`，适配器从未见过它。

| | A 组 · 原始默认参数 | B 组 · 调参后 |
|---|---|---|
| epochs / lr / max_length | 1 / 5e-6 / 512 | 3 / 5e-5 / 768 |
| 优化步数 | 15 | 48 |
| 训练耗时 | 3.2 分钟 | 13.8 分钟 |
| 最终训练 loss | 0.6926 | 0.5648 |
| TRL `eval_rewards/accuracies` | 0.600 | 0.714 |
| TRL `eval_rewards/margins` | 0.0035 | 0.0958 |

`evaluate.py` 在同一批留出数据上的独立复算（base 列由关闭适配器得到）：

| | A 组 (n=40) | B 组 (n=43) |
|---|---|---|
| 偏好准确率（对数概率求和）base → 微调后 | 45.0% → 45.0% | 41.9% → 41.9% |
| 偏好准确率（长度归一化）base → 微调后 | 45.0% → 45.0% | 41.9% → 41.9% |
| DPO 隐式奖励准确率 | 42.5% | 67.4% |
| 平均隐式奖励 margin | −0.0016 | +0.1017 |
| 平均回答长度 chosen / rejected | 208 / 208 token | 261 / 254 token |

### 这些数字在说什么

**A 组什么都没发生，原因就是那组默认参数。** 255 对训练数据、batch 2 × 梯度累积 8，
等于全程只有 **15 步优化**，学习率还是 `5e-6`——那是全参数微调的量级，比 LoRA 适配器
需要的低一个数量级。训练 loss 停在 0.6926，而 DPO 的起点就是 `ln 2 = 0.6931`；隐式奖励
margin 是 −0.0016；更关键的是，隐式奖励准确率的两个独立估计（TRL 的 0.600、本仓库的
0.425）之差超过了 40 个样本所允许的 ±0.08 标准误。**这个「对不上」本身就是结论：
这里没有信号，只有噪声。** 另外 A 组两侧的平均留出长度都恰好是 208 token，说明每条
回答都撞上了 `max_length 512` 的截断上限，分数是算在被削掉尾巴的前缀上的。

**B 组把 DPO 在优化的那个量推动了——而且只推动了它。** 隐式奖励准确率
42.5% → 67.4%，margin 涨了约 65 倍，TRL 自己的数字（0.714）也终于和本仓库的
（0.674）落在同一个噪声区间内。但是偏好准确率——微调后的模型是否给 chosen 更高的
对数似然——**纹丝不动，41.9% → 41.9%**。

这个落差是结果，不是 bug。隐式奖励是一个**相对量**：`β·(log p_微调后 − log p_原始)`；
β = 0.1 时 margin 0.10 大约相当于 **1 个 nat** 的对数概率。而绝对偏好排序比较的是两条
约 260 token 回答的**总**对数似然，两者相差数百个 nat。往正确方向挪 1 个 nat，几乎
永远翻不动这个排序。训练确实按 DPO 的目标把策略推动了、推动的幅度也正是这个目标
要求的幅度，但这个幅度远不足以重排 base 模型的似然偏好。**任何在这个规模上宣称
「偏好准确率提升了」的说法，都该先问一句：说的是相对指标还是绝对指标。**

另外两点值得留意：

- 两组的 base 偏好准确率都**低于 50%**（45.0%、41.9%）。按模型自身的似然，原始 Qwen
  反而略微更偏爱 ultrafeedback 标为 *rejected* 的那一条。那里的「chosen」是裁判给的
  质量标签，不是似然排序，所以在这个数据集上绝对偏好准确率本身就是一个弱代理指标。
- 长度归一化在两组里都没有改变结论——数字精确到小数点后一位都与原始版相同。它仍然
  保留在报告里，因为两侧长度差距大时这个混淆是真实存在的；这次只是恰好不差。

### 复现这些数字

`run_dpo_colab.ipynb` 端到端跑的是 A 组。B 组只需给 `train_dpo.py` 传
`--epochs 3 --lr 5e-5 --max_length 768`，给 `evaluate.py` 传 `--max_length 1024`。

当前 Colab 镜像上还需要一个环境修复，notebook 的安装 cell 已经包含它；如果你不用
notebook、直接跑脚本，记得自己先执行一次。`peft` 0.20.0 拒绝与预装的 `torchao`
0.10.0 共存——

```
ImportError: Found an incompatible version of torchao. Found version 0.10.0,
but only versions above 0.16.0 are supported
```

这个异常在 `get_peft_model` 内部抛出，训练在启动阶段就死掉。训练前执行
`pip uninstall -y torchao` 即可，这条管线本身不使用 torchao。

原始 metrics 与完整复现命令见 `results/`。

## 设计取舍

- **在这个规模上选 DPO 而不是 PPO**：不需要单独训一个奖励模型，也不需要稳住一个 RL
  循环——DPO 直接在成对偏好数据上优化，冻结的 base 模型天然充当参考策略。
- **β（默认 0.1）** 控制策略被允许偏离参考模型多远。β 越小，策略跑得越远、越容易在
  偏好集上过拟合；β 越大，策略被拽得越紧、效果越弱。
- **默认的 `lr 5e-6` 对 LoRA 是错的**，上面的 A 组就是它的样子：loss 从来没离开
  `ln 2`，margin 与零无异。LoRA 适配器需要的量级在 1e-5 – 1e-4；默认值保留下来，只是
  为了让这个失败可以和 B 组并排复现。
- **LoRA 只挂在 q/k/v/o 上**（r=16）：足够改变偏好，又不动 MLP 模块——这是 1.5B 能塞进
  16GB 的原因。
- **偏好数据里大部分信号来自 rubric 的设计。** `build_preference_data.py` 里那条启发式
  （长度接近目标 + 结尾标点完整）故意做得很粗，是最该被替换掉的一环——标注质量决定了
  下游一切的天花板。
- **长度是偏好评测里的标准混淆项**，所以准确率两种口径都报，并同时打印 chosen /
  rejected 的平均长度。也要盯着截断上限：当两侧平均长度完全一样（如 A 组）时，那不是
  「一样长」，那是「都被截断了」。

## 已知局限

- 一个模型（1.5B）、一个数据集、约 300 对样本、一个随机种子。这里的任何结论都不是关于
  DPO 的一般性论断，只是关于这一组配置的论断。
- 留出集只有 40–44 对，单个准确率的标准误约 ±8 个百分点。小于这个幅度的差异不是差异。
- 没有对生成质量做人工或 LLM 裁判评测；`eval_results.jsonl` 里的对照输出是给人看的，
  不是评分。
- **自采样 + rubric 那条数据路线尚未完成对比实验。** 现有脚本的 prompt 池只有 10 个
  固定问题（重复凑到 n 对），据此训练后留出集只剩十几对，标准误约 ±13 个百分点——在
  做这个对比之前，需要先把 prompt 池扩到上百个不同问题。

---

# English version

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

As it turned out, the number the loop produced was a negative one, and the
reason it was negative is the most informative thing in this repo. See
[Results](#results).

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

Measured on a free Colab T4 with 299 pairs: one epoch at `max_length=512` is
15 optimiser steps and takes **3.2 minutes**; three epochs at `max_length=768`
is 48 steps and takes **13.8 minutes**. Drop to `Qwen2.5-0.5B-Instruct` if you
are short on memory.

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

Two runs, both on one Colab free-tier Tesla T4, 2026-08-30. Data:
`trl-lib/ultrafeedback_binarized`, 299 usable pairs, 15% held out. The held-out
split is written to `dpo-out*/heldout_pairs.jsonl` before training starts, so
the adapter never sees it.

| | Run A — shipped defaults | Run B — retuned |
|---|---|---|
| epochs / lr / max_length | 1 / 5e-6 / 512 | 3 / 5e-5 / 768 |
| optimiser steps | 15 | 48 |
| train runtime | 3.2 min | 13.8 min |
| final train loss | 0.6926 | 0.5648 |
| TRL `eval_rewards/accuracies` | 0.600 | 0.714 |
| TRL `eval_rewards/margins` | 0.0035 | 0.0958 |

Independent scoring by `evaluate.py` on the same held-out pairs, with the
adapter toggled off for the base column:

| | Run A (n=40) | Run B (n=43) |
|---|---|---|
| preference accuracy, sum log-prob — base → tuned | 45.0% → 45.0% | 41.9% → 41.9% |
| preference accuracy, length-normalised — base → tuned | 45.0% → 45.0% | 41.9% → 41.9% |
| DPO implicit-reward accuracy | 42.5% | 67.4% |
| mean implicit-reward margin | −0.0016 | +0.1017 |
| mean response length, chosen / rejected | 208 / 208 tok | 261 / 254 tok |

### What these numbers say

**Run A did nothing, and the defaults are the reason.** 255 training pairs at
batch 2 × grad-accum 8 is 15 optimiser steps, run at `lr 5e-6` — a
full-fine-tuning learning rate, an order of magnitude below what a LoRA adapter
needs. Training loss ends at 0.6926 against a `ln 2 = 0.6931` starting point,
the implicit-reward margin is −0.0016, and the two independent estimates of
implicit-reward accuracy (TRL's 0.600, this repo's 0.425) disagree by more than
the ±0.08 standard error a 40-pair sample allows. That disagreement *is* the
finding: there is no signal here, only noise. Run A's mean held-out lengths are
also both exactly 208 tokens, i.e. every response hit the `max_length 512`
truncation cap, so those scores were computed on prefixes.

**Run B moved the quantity DPO optimises — and only that quantity.**
Implicit-reward accuracy goes 42.5% → 67.4%, the margin grows by ~65×, and TRL's
own figure (0.714) now agrees with this repo's (0.674) inside the noise. But
preference accuracy — does the tuned model give the chosen response a higher
log-likelihood than the rejected one — is **unchanged, 41.9% → 41.9%**.

That gap is the result, not a bug. The implicit reward is a *relative* quantity,
`β·(log p_tuned − log p_base)`; a margin of 0.10 at β = 0.1 is about one nat of
log-probability. Absolute preference ranking is decided by the total
log-likelihood of two ~260-token responses, which differ by hundreds of nats.
One nat of movement in the right direction almost never flips that ordering.
DPO moved the policy exactly as far as its objective asks, and that distance is
far too small to reorder the base model's likelihood preferences. Anyone
quoting "preference accuracy improved" off a run this size should check whether
they are quoting the relative metric or the absolute one.

Two further observations worth keeping:

- Base preference accuracy is **below 50%** in both runs (45.0%, 41.9%). Under
  its own likelihood the base model prefers ultrafeedback's *rejected* response
  slightly more often than the chosen one. "Chosen" there is a judge's quality
  label, not a likelihood ordering, so absolute preference accuracy is a weak
  proxy for alignment quality on this dataset.
- Length normalisation changed nothing in either run — identical to the raw
  numbers at one decimal place. It stays in the report because the confound is
  real whenever the two sides differ in length; here they did not.

### Reproducing these numbers

`run_dpo_colab.ipynb` runs Run A end to end. For Run B, pass
`--epochs 3 --lr 5e-5 --max_length 768` to `train_dpo.py` and `--max_length
1024` to `evaluate.py`.

One environment fix is needed on current Colab images. The notebook's install
cell already applies it; if you run the scripts outside the notebook, apply it
yourself. `peft` 0.20.0 refuses to load against the preinstalled `torchao`
0.10.0 —

```
ImportError: Found an incompatible version of torchao. Found version 0.10.0,
but only versions above 0.16.0 are supported
```

— raised inside `get_peft_model`, so training dies at startup. `pip uninstall -y
torchao` before training clears it; nothing in this pipeline uses torchao.

Raw metrics and the exact commands are in `results/`.

## Design notes

- **DPO over PPO** for a project this size: no separate reward model to train and
  no RL loop to stabilise — DPO optimises directly on preference pairs, with the
  frozen base model acting as the reference policy.
- **β (0.1 by default)** controls how far the policy is allowed to move from that
  reference. Lower β lets the policy drift further and overfit the preference
  set; higher β keeps it close and blunts the effect.
- **The default `lr 5e-6` is wrong for LoRA** and Run A above is what that looks
  like: a loss that never leaves `ln 2` and a margin indistinguishable from
  zero. LoRA adapters want something in the 1e-5 – 1e-4 range; the default is
  kept only so the failure stays reproducible next to Run B.
- **LoRA on q/k/v/o only** (r=16): enough capacity to shift preferences without
  touching the MLP blocks, which is what keeps a 1.5B run inside 16GB.
- **Rubric design is where most of the signal in preference data comes from.**
  The heuristic in `build_preference_data.py` (length target + clean ending) is
  deliberately crude and is the first thing worth replacing — the labels set the
  ceiling on everything downstream.
- **Length is the standard confound** in preference evaluation, which is why the
  accuracy is reported both ways and the mean chosen/rejected lengths are printed
  alongside. Watch the truncation cap too: when both sides report the same mean
  length, as in Run A, they are being clipped rather than compared.

## Known limitations

- One model (1.5B), one dataset, ~300 pairs, one seed. Nothing here should be
  read as a general claim about DPO — it is a claim about this configuration.
- The held-out set is 40–44 pairs, so a single accuracy figure carries a ±8pp
  standard error. Differences smaller than that are not differences.
- No human or LLM-judge evaluation of generation quality; the side-by-side
  outputs in `eval_results.jsonl` are for inspection, not scoring.
- **The self-sampled / rubric data route has not been compared yet.** The prompt
  pool in the current script is 10 fixed questions repeated up to `n`, which
  leaves a held-out set in the teens and a ±13pp standard error. That comparison
  needs a prompt pool in the hundreds before it can say anything.
