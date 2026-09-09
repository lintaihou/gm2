# Graph Machine: Towards Better Pretraining via Edges

This repository contains the code and [calculation and evaluation results](outputs/results.csv) behind the tables in [Graph Machine: Towards Better Pretraining via Edges](https://arxiv.org/abs/2609.02881), by Lintai Hou; [checkpoints](https://huggingface.co/lintaihou/gm2) and [training curves](https://api.wandb.ai/links/lintaihou/ccz3laev) are available separately.

**Contents:** [Configuration](#model-and-training-configuration) · [Model presets](#model-presets) · [Implementation details](#implementation-details) · [Installation](#installation) · [Data](#data) · [Training and evaluation](#training-and-evaluation) · [Repository layout](#repository-layout) · [Citation](#citation) · [Licenses](#licenses-and-acknowledgments)

## Model and training configuration

The model backbone shapes and many shared design choices follow [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base): pre-RMSNorm, Q/K normalization, bias-free projections, SwiGLU, tied embeddings, and RoPE in dense attention. All models, including the baseline, are trained from scratch with the same training recipe. Shared settings are defined in [configs/default.yaml](configs/default.yaml) and the implementation; preset-specific values appear below.

| Setting | Value |
| --- | --- |
| **Architecture** | |
| Layers | Qwen3: 28 dense; GM: 21 sparse and 7 dense, arranged as `[S, S, D, S] × 7`. |
| Width and vocabulary | Hidden size 1,024; vocabulary size 151,936. |
| Attention | 16 query heads, 8 KV heads, head dimension 128; causal attention. |
| Normalization | Pre-RMSNorm residual blocks, per-head Q/K RMSNorm in both sparse and dense layers, and final RMSNorm; epsilon `1e-6`, statistics computed in FP32. |
| MLP | SwiGLU; intermediate size 3,072 (3× hidden size). |
| Bias and dropout | No linear biases or dropout. |
| Position encoding | RoPE in dense attention only, with base `1e6`; omitted from SEA. |
| Embeddings and initialization | Tied input/output embeddings; embedding and linear weights initialized with normal standard deviation `0.02`; RMSNorm scales initialized to one. |
| Edges | 8 input-refresh edges; realignment enabled; stored-edge count, referral count, sparsity, and dense refresh vary by preset. |
| Temperatures | Shifted softplus, $f(x)=\operatorname{softplus}(x+\log(e-1))$, so $f(0)=1$. |
| **Training** | |
| Objective | Next-token cross-entropy. |
| Sequence length | 4,096, including one prepended padding token. |
| Data loading | Incomplete sequences and incomplete microbatches are dropped. |
| Training budget | 30,000 optimizer steps; approximately 15.7B content tokens. |
| Effective batch | 128 sequences per step; microbatch size × accumulation steps shown below. |
| Optimizer | Fused AdamW; learning rate `1e-3`, betas `(0.9, 0.95)`, epsilon `1e-8`. |
| Schedule | 1,000-step linear warmup, then linear decay to zero. |
| Regularization | Weight decay `0.1` on parameters with at least 2 dimensions, zero otherwise; gradient norm clipping at `1.0`. |
| Precision | FP32 parameters with BF16 `torch.amp.autocast`; FP32 matmul precision set to `"high"`. |
| Compilation | `torch.compile` enabled; custom Triton kernel for sparsification. |
| Seed | `0` for Python, NumPy, and PyTorch. |

### Model presets

| Preset | Stored edges `K` | Referrals `R` | Sparsities `(s1, s2, s3, s4)` | Dense-refresh edges | Microbatch × accumulation |
| --- | ---: | ---: | --- | ---: | --- |
| `Qwen3` | — | — | — | — | 4 × 32 |
| `Theia-K24` | 24 | 0 | (2, 4, 4, 2) | 0 | 4 × 32 |
| `Theia-K32-R2` | 32 | 2 | (2, 4, 4, 2) | 0 | 4 × 32 |
| `Theia-K24-R3` | 24 | 3 | (2, 4, 4, 2) | 0 | 4 × 32 |
| `Theia-K24-R4` | 24 | 4 | (2, 4, 4, 2) | 0 | 2 × 64 |
| `Theia-K16-R4-S` | 16 | 4 | (2, 4, 4, 2) | 8 | 2 × 64 |
| `Theia-K16-R6` | 16 | 6 | (2, 4, 4, 2) | 0 | 2 × 64 |
| `Hyperion-K32-R1` | 32 | 1 | (4, 4, 4, 4) | 0 | 4 × 32 |
| `Hyperion-K24-R3` | 24 | 3 | (4, 4, 4, 4) | 0 | 2 × 64 |
| `Hyperion-K16-R3` | 16 | 3 | (4, 4, 4, 4) | 0 | 4 × 32 |
| `Hyperion-K16-R3-S` | 16 | 3 | (4, 4, 4, 4) | 8 | 2 × 64 |
| `Hyperion-K16-R4` | 16 | 4 | (4, 4, 4, 4) | 0 | 2 × 64 |

`R` is the referral count per sparse layer; `S` denotes eight dense-refresh edges. The sparsity values control stored edges, the 2 referral inputs, and attention retrieval, respectively: Theia retrieves 2 positions per KV head, Hyperion 4.

## Implementation details

Some of the behaviors below are arbitrary implementation artifacts or minor bugs that are not deliberate design choices and have not been ablated. I will add a few of these into v2 of the paper.

- **Layer scheduling:** Sparse and dense layers are evenly interleaved, prioritizing sparse layers on ties. A sparse layer performs no referrals if it is first in the schedule; otherwise, it performs `n_referrals` referrals. Input edges are refreshed only after a previous referral, and dense edges only after a previous dense layer.
- **Output-temperature scaling:** SER scales uncoalesced sparse-coordinate contributions before sparsification. SEA sparsifies and coalesces first, then applies output-temperature scaling through attention logits.
- **Stored-edge updates:** SEA retains input-temperature-scaled weights in the next edge state. Without realignment, all stored weights are updated; with realignment and more stored edges than query heads, unreplaced edges retain their scaled weights.
- **Refresh edges:** These are appended after input-temperature scaling, so their weights bypass it, equivalently using an input temperature of one.
- **Zero-weight slots in SEA:** SEA clamps weights to $10^{-8}$ before taking logarithms, so zero-weight slots are not strictly masked. Their multiplicative edge contribution relative to unit weight is $10^{-8\tau}$, where $\tau$ is the output-edge temperature. This is negligible near $\tau=1$ but can increase as the learned temperature becomes small.

## Installation

Use Linux with an NVIDIA GPU. Install a suitable CUDA-enabled PyTorch build using the [PyTorch installation instructions](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -r requirements.txt
```

PyTorch and Triton are unpinned to allow compatible environment choices. Run all commands from the repository root.

The original training run used PyTorch `2.9.1+cu130` (CUDA 13.0) and Triton `3.5.1`.

For RunPod, filter instances for CUDA 13.0, select an H100 SXM, and use this Docker image:

```text
runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster
```

## Data

Datasets live in `data/<dataset>/train.bin` and `test.bin` as raw `uint32` token IDs. Select a directory with `dataset=NAME`. The Qwen3-0.6B-Base tokenizer is downloaded on first use.

I provide a bundled smoke subset for [smoke runs](#smoke-run), so you can check the environment without preparing the full dataset.

Preparing the full training data takes a long time and produces approximately 84 GB of token files; additional disk space is required for downloads and cached intermediate data. Run:

```bash
python -m src.dataset
```

This processes [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)'s `sample-100BT` subset: shuffle seed 42, 20 million training documents, and 20,000 test documents. It appends EOS after each document and writes to `data/fineweb-edu-100bt-20m-20k/`.

The training iterator does not restart, so provide enough data for the requested run.

## Training and evaluation

### Smoke run

Use the smoke run to check that training, checkpoint saving, and evaluation work in your environment.

The binaries are included in `data/fineweb-edu-10bt-smoke/`. Derived from `sample-10BT`, they total approximately 23 MB: 10 training steps' worth of data and one effective training batch's worth of test data at the preset settings.

```bash
WANDB_MODE=disabled python -m src.launch \
  --preset Hyperion-K16-R3 --smoke test=true
```

This trains for 10 steps with one warmup step, saves only the final checkpoint, and evaluates. `--smoke` alone only trains; `test=true` adds evaluation. Model backbone shapes and sequence length stay unchanged. Increasing steps or batch/sequence sizes may require more data.

Initial compilation can take a few minutes and may emit warnings; add `--no-compile` to disable `torch.compile`.

### Full runs

```bash
# Train
python -m src.launch --preset Hyperion-K16-R3

# Evaluate a checkpoint
python -m src.launch --preset Hyperion-K16-R3 \
  --eval path/to/checkpoint.pth
```

The paper's [checkpoints are available on Hugging Face](https://huggingface.co/lintaihou/gm2). Download a checkpoint and pass its local path to `--eval`. Evaluation uses all complete batches in `test.bin`. Supply the checkpoint's preset and architecture overrides; add `--smoke` to use the smoke test data.

### Arguments and overrides

| Argument | Effect |
| --- | --- |
| `--preset NAME` | Required preset from the table above. |
| `--eval CHECKPOINT` | Load weights and evaluate without training. |
| `--smoke` | Use the smoke dataset and short-run settings; with `--eval`, only evaluation runs. |
| `--no-compile` | Disable `torch.compile`; custom Triton kernels still run. |
| `key=value ...` | Override existing configuration keys last, after presets and flags. |

For example:

```bash
python -m src.launch --preset Hyperion-K16-R3 \
  lr=5e-4 n_steps=1000 n_warmup_steps=100 group=my-experiment
```

### Compilation compatibility

Some Theia configurations encounter Triton lowering errors during `torch.compile` on some PyTorch versions. In `src/model/modules.py`, switch `scale = scale_softmax` to `scale = scale_power`, adding `scale_power` to the `.functions` import. Alternatively, use `--no-compile`.

### Logging and checkpoints

Losses are logged to Weights & Biases. Use `wandb login` for online logging, `WANDB_MODE=offline` for local logs, or `WANDB_MODE=disabled` to turn logging off ([W&B documentation](https://docs.wandb.ai/models/track/environment-variables)).

By default, checkpoints are saved every 7,500 steps and at the final step:

```text
outputs/checkpoints/<group>/<timestamp>-<name>/models/<name>-<step>.pth
```

Checkpoints contain model weights only; optimizer, scheduler, and training progress are not restored. The trainer does not update `outputs/results.csv`.

## Repository layout

| Path | Contents |
| --- | --- |
| `configs/default.yaml` | Model and training defaults. |
| `src/launch.py`, `src/train.py` | Model presets, CLI, training, and evaluation. |
| `src/dataset.py` | Data preprocessing and loader. |
| `src/model/` | `core.py`: model assembly; `modules.py`: layers; `functions.py`: kernels and helpers. |
| `data/<dataset>/` | Token binaries and data notices. |
| `outputs/results.csv` | Calculation and evaluation results for the paper's tables. |

## Citation

If you use this work, please cite:

```bibtex
@misc{hou2026gm2,
  title         = {Graph Machine: Towards Better Pretraining via Edges},
  author        = {Hou, Lintai},
  year          = {2026},
  eprint        = {2609.02881},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.02881}
}
```

Citation metadata is also available in [CITATION.cff](CITATION.cff).

## Licenses and acknowledgments

- **Code and configurations:** [Apache 2.0](LICENSE), copyright 2026 Lintai Hou.
- **RoPE helpers:** adapted from Sebastian Raschka's [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch); see [upstream license and attribution](LICENSE-LLMs-from-scratch.txt).
- **Qwen3:** backbone design and tokenizer from [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base), released under [Apache 2.0](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/LICENSE).
- **Data:** contains information from [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), by Anton Lozhkov, Loubna Ben Allal, Leandro von Werra, and Thomas Wolf (2024), made available under [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) and subject to [Common Crawl's Terms of Use](https://commoncrawl.org/terms-of-use). See the [smoke subset notice](data/fineweb-edu-10bt-smoke/LICENSE.txt). The software license does not cover the data or grant rights to underlying web content.
- **Paper:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as specified on [arXiv](https://arxiv.org/abs/2609.02881).
