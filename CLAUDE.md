# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

HRM (Hierarchical Reasoning Model) is a recurrent neural network architecture for solving complex reasoning puzzles (ARC-AGI, Sudoku, Mazes). It operates in a single forward pass without Chain-of-Thought supervision, using two nested recurrent modules and a learned halting mechanism (ACT - Adaptive Computation Time).

## Setup

Requires CUDA, PyTorch, and FlashAttention (FA3 for Hopper GPUs, FA2 for Ampere and earlier).

```bash
pip install -r requirements.txt
wandb login
git submodule update --init --recursive  # for ARC dataset raw data
```

Set `DISABLE_COMPILE=1` to skip `torch.compile` (useful for debugging).

## Common Commands

### Dataset preparation (run from repo root)
```bash
# ARC-1
python dataset/build_arc_dataset.py

# ARC-2
python dataset/build_arc_dataset.py --dataset-dirs dataset/raw-data/ARC-AGI-2/data --output-dir data/arc-2-aug-1000

# Sudoku (1000-sample subset)
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000

# Maze
python dataset/build_maze_dataset.py
```

### Training
```bash
# Single GPU
OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0

# Multi-GPU (8 GPUs)
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py
```

Config flags are passed as Hydra overrides (e.g., `data_path=...`, `epochs=...`). The base config is `config/cfg_pretrain.yaml`, which defaults to the `hrm_v1` arch defined in `config/arch/hrm_v1.yaml`.

### Evaluation
```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 evaluate.py checkpoint=<CHECKPOINT_PATH>
```

The `evaluate.py` reads `all_config.yaml` from the checkpoint directory to reconstruct the full `PretrainConfig`. After running, use `arc_eval.ipynb` for ARC-specific analysis.

Metrics to watch: `eval/exact_accuracy` in W&B.

## Architecture

### Model class loading

Models and loss heads are referenced by string identifiers of the form `module.path@ClassName` (e.g., `hrm.hrm_act_v1@HierarchicalReasoningModel_ACTV1`). `utils/functions.py:load_model_class` resolves these by importing from the `models.` package prefix. Adding a new model means placing it in `models/`, then referencing it with this pattern in a YAML config.

### HRM forward pass (`models/hrm/hrm_act_v1.py`)

The model has two interleaved recurrent modules:
- **H_level** (high-level, slow): processes abstract plans
- **L_level** (low-level, fast): processes fine-grained computations

Per ACT step, the inner loop runs `H_cycles × L_cycles` iterations with gradients disabled for all but the final step (1-step-grad trick for training stability). The sequence includes prepended **puzzle embeddings** (`puzzle_emb_len` tokens) before the actual token sequence; the LM head output slices these off.

The outer `HierarchicalReasoningModel_ACTV1` wrapper manages the ACT halting loop:
- During **training**: halting is decided by a learned Q-head (`q_halt_logits > q_continue_logits`) with epsilon-exploration. A bootstrapped Q-continue target is computed (similar to PQN, no replay buffer).
- During **evaluation**: always runs `halt_max_steps` steps (fixed computation for batching consistency).

The carry state (`HierarchicalReasoningModel_ACTV1Carry`) persists `z_H`, `z_L` tensors, current step count, and halted flags across ACT steps. Newly halted sequences get their carry reset at the start of the next step.

### Loss (`models/losses.py:ACTLossHead`)

Wraps the model. Computes:
1. `lm_loss` — token-level cross-entropy (using `stablemax_cross_entropy` by default, a numerically stable alternative to softmax)
2. `q_halt_loss` — binary cross-entropy predicting whether the current sequence is correct
3. `q_continue_loss` — bootstrapped Q-learning loss for continue decisions

Metrics emitted per batch: `count`, `accuracy`, `exact_accuracy`, `q_halt_accuracy`, `steps`, and the three loss components.

### Puzzle embeddings (`models/sparse_embedding.py`)

Each puzzle has a learned dense embedding (`CastedSparseEmbedding`) updated with a custom `CastedSparseEmbeddingSignSGD_Distributed` optimizer using SignSGD. This is separate from the main AdamATan2 optimizer because embeddings are sparse (only a few puzzles per batch) and SignSGD approximates Adam for very sparse gradients. The main weights buffer is updated in-place using scatter; only updated IDs are gathered across ranks.

### Dataset format

Datasets are stored as NumPy arrays under `data/<dataset-name>/train/` and `data/<dataset-name>/test/`. Each split contains per-set files named `<set_name>__<field>.npy`:
- `inputs`, `labels` — shape `[N, seq_len]`
- `puzzle_identifiers` — shape `[num_puzzles]`, indices into global identifier table
- `puzzle_indices` — CSR-style offsets: examples `[puzzle_indices[i], puzzle_indices[i+1])` belong to puzzle `i`
- `group_indices` — CSR-style offsets grouping puzzles into augmentation groups

Plus `dataset.json` with `PuzzleDatasetMetadata`. During training, one puzzle is sampled per group per step; during eval, all examples are iterated in order.

### Training loop (`pretrain.py`)

Uses Hydra for config. Key design notes:
- `global_batch_size` is fixed at model init time (used to size `puzzle_emb.local_weights`); changing it requires recreating the model.
- Distributed training uses `torchrun`; gradients are manually all-reduced before optimizer step.
- Two schedulers run in parallel: cosine-with-warmup for both the main optimizer and the puzzle embedding optimizer, with separate `lr`/`puzzle_emb_lr`.
- Checkpoints save only model weights (`torch.save(model.state_dict(), ...)`); optimizer state is not saved.
- The model is always compiled with `torch.compile(dynamic=False)` unless `DISABLE_COMPILE` is set.

## Key Conventions

- All forward passes use `bfloat16` by default (`forward_dtype` config field); weights are stored in float32.
- `CastedLinear` and `CastedEmbedding` cast to the input tensor's dtype on every forward call.
- `IGNORE_LABEL_ID = -100` (matching PyTorch's cross-entropy default) marks padding in labels.
- Dataset builders must be run from the repo root (they import `from common import ...` relative to `dataset/`).
- The `arc_eval.ipynb` notebook expects `all_preds` tensors saved by `evaluate.py`.
