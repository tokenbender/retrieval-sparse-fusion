import glob
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig

out_dir = "out"
eval_interval = 200
log_interval = 10
eval_iters = 100
max_iters = 2000

batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
bias = False

learning_rate = 3e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

warmup_iters = 200
lr_decay_iters = 2000
min_lr = 6e-5

gradient_accumulation_steps = 1
seed = 1337

dataset = "fineweb10B"
data_dir = None

dtype = "bfloat16"
device = None
compile_model = False

backend = "nccl"

exec(open(os.path.join(os.path.dirname(__file__), "configurator.py")).read())


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


ddp = int(os.environ.get("RANK", -1)) != -1
ddp_local_rank = 0
if ddp:
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", ddp_local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend=backend, device_id=device)
    dist.barrier()
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = (
    "cuda"
    if "cuda" in str(device)
    else ("mps" if "mps" in str(device) else "cpu")
)

ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]

if device_type == "cpu":
    ctx = nullcontext()
    scaler = None
else:
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == "float16"))

if data_dir is None:
    data_dir = os.path.join(os.path.dirname(__file__), "data", dataset)

if dataset != "fineweb10B":
    raise ValueError(f"unknown dataset: {dataset}")

FINEWEB_MAGIC = 20240520
FINEWEB_VERSION = 1
HEADER_SIZE = 256


def load_fineweb_shard(path):
    header = torch.from_file(str(path), shared=False, size=HEADER_SIZE, dtype=torch.int32)
    assert header[0].item() == FINEWEB_MAGIC, f"bad magic in {path}"
    assert header[1].item() == FINEWEB_VERSION, f"bad version in {path}"
    num_tokens = int(header[2].item())
    with open(path, "rb") as f:
        f.seek(HEADER_SIZE * 4)
        buf = np.frombuffer(f.read(num_tokens * 2), dtype=np.uint16)
        tokens = torch.from_numpy(buf.astype(np.int64))
    return tokens


train_shards = sorted(glob.glob(os.path.join(data_dir, "fineweb_train_*.bin")))
val_shards = sorted(glob.glob(os.path.join(data_dir, "fineweb_val_*.bin")))
assert len(train_shards) > 0, f"no train shards found in {data_dir}"
assert len(val_shards) > 0, f"no val shards found in {data_dir}"

if master_process:
    print(f"Found {len(train_shards)} train shards, {len(val_shards)} val shards")

train_data = torch.cat([load_fineweb_shard(s) for s in train_shards])
val_data = torch.cat([load_fineweb_shard(s) for s in val_shards])
vocab_size = 50304

if master_process:
    print(f"Train tokens: {len(train_data):,}, Val tokens: {len(val_data):,}")


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


model_config = GPTConfig(
    block_size=block_size,
    vocab_size=vocab_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=dropout,
    bias=bias,
)
model = GPT(model_config)
model.to(device)

if compile_model:
    if master_process:
        print("Compiling model...")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model
optimizer = raw_model.configure_optimizers(
    weight_decay=weight_decay,
    learning_rate=learning_rate,
    betas=(beta1, beta2),
    device_type=device_type,
)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            with ctx:
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


iter_num = 0
best_val_loss = 1e9
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size

if master_process:
    print(f"Training on {device}, dtype={dtype}, DDP={ddp}")
    print(f"tokens per iteration: {tokens_per_iter:,}")
    print(f"model params: {sum(p.numel() for p in raw_model.parameters()):,}")

while iter_num <= max_iters:
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        lr_scale = param_group.get("lr_scale", 1.0)
        param_group["lr"] = lr * lr_scale

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"iter {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            os.makedirs(out_dir, exist_ok=True)
            ckpt = {
                "model": raw_model.state_dict(),
                "config": model_config.__dict__,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            }
            torch.save(ckpt, os.path.join(out_dir, "ckpt.pt"))

    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    loss = None

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        x, y = get_batch("train")
        with ctx:
            _, loss = model(x, y)
            loss = loss / gradient_accumulation_steps
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    if grad_clip != 0.0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    dt = time.time() - t0
    if iter_num % log_interval == 0 and master_process:
        if loss is None:
            raise RuntimeError("loss is None after training step")
        loss_item = loss.item() * gradient_accumulation_steps
        tok_per_s = tokens_per_iter / dt
        print(f"iter {iter_num}: loss {loss_item:.4f}, lr {lr:.2e}, time {dt * 1000:.0f}ms, tok/s {tok_per_s:.0f}")

    iter_num += 1

if ddp:
    dist.destroy_process_group()
