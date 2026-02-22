# nanoGPT Baseline (FineWeb10B)

This is a pure non-character NanoGPT baseline setup for FineWeb10B.

## Run

From `baselines/nanogpt/`:

```bash
python train.py config/train_fineweb10B.py
```

DDP example:

```bash
torchrun --standalone --nproc_per_node=4 train.py config/train_fineweb10B.py
```

## Data

Download shard files into `baselines/nanogpt/data/fineweb10B/`:

```bash
python data/fineweb10B/download.py 9
```
