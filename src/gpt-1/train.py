import math
import time

import matplotlib.pyplot as plt
import torch

from data import vocab_size, get_batch, decode
from model import TransformerLM

######################
# Config
######################

block_size = 32       # up from 8, the model can now use longer context
batch_size = 32
n_embd = 64           # GPT-1: 768
n_head = 4            # GPT-1: 12, head size here = 64 / 4 = 16 (GPT-1: 64)
n_layer = 4           # GPT-1: 12
learning_rate = 1e-3  # if loss spikes or goes nan, drop to 5e-4 (post-LN is touchy)
max_steps = 5000
eval_interval = 250
eval_iters = 200

baselines = {"bigram": 2.4888, "1 head": 2.4012}  # val losses from earlier phases

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

model = TransformerLM(vocab_size, n_embd, n_head, n_layer, block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
n_params = sum(p.numel() for p in model.parameters())
expected = math.log(vocab_size)

print("=" * 50)
print("Transformer (post-LN blocks)")
print("=" * 50)
print(f"parameters      {n_params:,}")
print(f"layers x heads  {n_layer} x {n_head}, width {n_embd}")
print(f"context         {block_size} characters")
print(f"expected start  {expected:.4f}")
for name, val in baselines.items():
    print(f"{name + ' val':<15} {val:.4f}")
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
print(f"final val loss  {val_log[-1]:.4f}")

######################
# Loss curve
######################

plt.figure(figsize=(9, 5))
plt.plot(batch_log, color="lightgray", linewidth=0.8, label="single batch (noisy)")
plt.plot(step_log, train_log, "o-", markersize=3, label="train (averaged)")
plt.plot(step_log, val_log, "s-", markersize=3, label="val (averaged)")
for (name, val), color in zip(baselines.items(), ["green", "purple"]):
    plt.axhline(val, color=color, linestyle="--", linewidth=1, label=f"{name} val = {val:.2f}")
plt.axhline(expected, color="red", linestyle="--", linewidth=1, label=f"uniform guess = {expected:.2f}")
plt.xlabel("step")
plt.ylabel("cross-entropy loss")
plt.title("Transformer loss curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("transformer_loss.png", dpi=120)
print("loss curve saved to transformer_loss.png")

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