import math
import time
from dataclasses import asdict

import matplotlib.pyplot as plt
import torch

from data import vocab_size, get_batch, decode
from model import GPT, GPTConfig

######################
# Config
######################

block_size = 128      # GPT-1: 512. long enough for the fine-tuning task's sentence pairs
batch_size = 32       # GPT-1: 64
n_embd = 64           # GPT-1: 768
n_head = 4            # GPT-1: 12
n_layer = 4           # GPT-1: 12
dropout = 0.1         # GPT-1: 0.1 on embeddings, attention weights and residual branches

max_lr = 1e-3         # GPT-1: 2.5e-4, small models tolerate bigger steps
warmup_steps = 500    # GPT-1: 2000
max_steps = 5000
weight_decay = 0.01   # GPT-1: 0.01, not applied to biases or LayerNorm gains
grad_clip = 1.0       # GPT-1's released code clipped at 1.0 (not stated in the paper)

eval_interval = 250
eval_iters = 100
ckpt_path = "gpt1_pretrained.pt"

torch.manual_seed(1337)

##########################
# Learning rate schedule
#########################

def get_lr(step):
    # linear warmup from 0, then cosine down to 0
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))

######################
# Evaluation
######################

@torch.no_grad()
def estimate_loss(model):
    model.eval()  # dropout off
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, block_size, batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

######################
# Model and optimizer
######################

cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer,
                n_head=n_head, n_embd=n_embd, dropout=dropout)
model = GPT(cfg)

# matrices (Linear weights, embeddings) get weight decay; biases and LayerNorm params don't
# named_parameters() lists the tied embedding once, so it isn't added twice
decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW([
    {"params": decay, "weight_decay": weight_decay},
    {"params": no_decay, "weight_decay": 0.0},
], lr=max_lr)

n_params = sum(p.numel() for p in model.parameters())

print("=" * 54)
print("GPT-1 pre-training (scaled down)")
print("=" * 54)
print(f"parameters      {n_params:,}")
print(f"layers x heads  {n_layer} x {n_head}, width {n_embd}")
print(f"context         {block_size} characters")
print(f"dropout         {dropout}")
print(f"lr              warmup {warmup_steps} steps to {max_lr}, cosine to 0")
print(f"expected start  {math.log(vocab_size):.4f}")
print()
print(f"{'step':>6} | {'train':>7} | {'val':>7} | {'gap':>6} | {'lr':>8} | {'time':>7}")
print("-" * 56)

######################
# Training
######################

step_log, train_log, val_log, batch_log, lr_log = [], [], [], [], []
t0 = time.time()

for step in range(max_steps + 1):
    lr = get_lr(step)
    for group in optimizer.param_groups:
        group["lr"] = lr
    lr_log.append(lr)

    if step % eval_interval == 0:
        losses = estimate_loss(model)
        gap = losses["val"] - losses["train"]
        print(f"{step:>6} | {losses['train']:>7.4f} | {losses['val']:>7.4f} | "
              f"{gap:>6.3f} | {lr:>8.2e} | {time.time() - t0:>6.1f}s")
        step_log.append(step)
        train_log.append(losses["train"])
        val_log.append(losses["val"])

    if step == max_steps:
        break  # lr is 0 here, nothing left to learn

    x, y = get_batch("train", block_size, batch_size)
    _, loss = model(x, y)
    batch_log.append(loss.item())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # rescale all gradients together if their combined norm is above grad_clip
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

print("-" * 56)
print(f"final val loss  {val_log[-1]:.4f}")

torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, ckpt_path)
print(f"checkpoint saved to {ckpt_path}")

######################
# Plots
######################

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax1.plot(batch_log, color="lightgray", linewidth=0.8, label="single batch (with dropout)")
ax1.plot(step_log, train_log, "o-", markersize=3, label="train (averaged, no dropout)")
ax1.plot(step_log, val_log, "s-", markersize=3, label="val (averaged)")
ax1.axhline(math.log(vocab_size), color="red", linestyle="--", linewidth=1, label="uniform guess")
ax1.set_ylabel("cross-entropy loss")
ax1.set_title("GPT-1 pre-training (scaled down)")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(lr_log, color="black")
ax2.set_xlabel("step")
ax2.set_ylabel("learning rate")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("pretrain_loss.png", dpi=120)
print("plot saved to pretrain_loss.png")

######################
# Quick sample
######################

print()
print("=" * 54)
start = torch.zeros((1, 1), dtype=torch.long)
model.eval()
print(decode(model.generate(start, max_new_tokens=300)[0].tolist()))
print("=" * 54)