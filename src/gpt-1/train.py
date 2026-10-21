import math
import time

import matplotlib.pyplot as plt
import torch

from data import vocab_size, get_batch, decode
from model import AttentionLM

######################
# Config
######################

block_size = 8
batch_size = 32
n_embd = 32           # size of each token's vector; GPT-1 used 768
learning_rate = 1e-3  # lower than the bigram, attention is less forgiving
max_steps = 5000
eval_interval = 250   # denser than before so the curve isn't a straight line
eval_iters = 200

bigram_val = 2.4888   # from phase 1

torch.manual_seed(1337)

######################
# Evaluation
######################

@torch.no_grad()
def estimate_loss(model):
    model.eval()
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
# Training
######################

model = AttentionLM(vocab_size, n_embd, block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
n_params = sum(p.numel() for p in model.parameters())
expected = math.log(vocab_size)

print("=" * 50)
print("Single-head attention model")
print("=" * 50)
print(f"parameters      {n_params:,}")
print(f"context         {block_size} characters")
print(f"expected start  {expected:.4f}")
print(f"bigram val      {bigram_val:.4f}  (to beat)")
print()
print(f"{'step':>6} | {'train':>7} | {'val':>7} | {'gap':>6} | {'time':>6}")
print("-" * 46)

step_log, train_log, val_log, batch_log = [], [], [], []
t0 = time.time()

for step in range(max_steps + 1):
    if step % eval_interval == 0:
        losses = estimate_loss(model)
        gap = losses["val"] - losses["train"]
        print(f"{step:>6} | {losses['train']:>7.4f} | {losses['val']:>7.4f} | "
              f"{gap:>6.3f} | {time.time() - t0:>5.1f}s")
        step_log.append(step)
        train_log.append(losses["train"])
        val_log.append(losses["val"])

    x, y = get_batch("train", block_size, batch_size)
    _, loss = model(x, y)
    batch_log.append(loss.item())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("-" * 46)
print(f"final val loss  {val_log[-1]:.4f}  (bigram: {bigram_val:.4f})")

######################
# Loss curve
######################

plt.figure(figsize=(9, 5))
plt.plot(batch_log, color="lightgray", linewidth=0.8, label="single batch (noisy)")
plt.plot(step_log, train_log, "o-", markersize=3, label="train (averaged)")
plt.plot(step_log, val_log, "s-", markersize=3, label="val (averaged)")
plt.axhline(bigram_val, color="green", linestyle="--", linewidth=1, label=f"bigram val = {bigram_val:.2f}")
plt.axhline(expected, color="red", linestyle="--", linewidth=1, label=f"uniform guess = {expected:.2f}")
plt.xlabel("step")
plt.ylabel("cross-entropy loss")
plt.title("Single-head attention loss curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("attention_loss.png", dpi=120)
print("loss curve saved to attention_loss.png")

######################
# Generation
######################

print()
print("=" * 50)
print("Sample (300 characters)")
print("=" * 50)
start = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(start, max_new_tokens=300)[0].tolist()))
print("=" * 50)