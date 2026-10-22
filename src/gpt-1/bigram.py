import math
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import vocab_size, get_batch, decode
from paths import RESULTS_DIR, save_metrics

######################
# Config
######################

block_size = 8        # context length; GPT-1 used 512
batch_size = 32
learning_rate = 1e-2  # high on purpose, a single table is easy to train; GPT-1 used 2.5e-4
max_steps = 3000
eval_interval = 500
eval_iters = 200      # batches averaged per eval, one batch is too noisy

torch.manual_seed(1337)

######################
# Model
######################

class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # row i = scores for the character that comes after character i
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.table(idx)  # (B, T, vocab_size)
        if targets is None:
            return logits, None

        # cross_entropy wants (N, C) and (N,), so flatten batch and time together
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Write text one character at a time, feeding each output back in."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # only the last position predicts the future
            probs = F.softmax(logits, dim=-1)
            # sample instead of argmax, argmax gets stuck in repetitive loops
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

######################
# Evaluation
######################

@torch.no_grad()
def estimate_loss(model):
    model.eval()  # matters once there's dropout (GPT-1 has it)
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, block_size, batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()  # easy to forget, dropout would stay off for the rest of training
    return out

######################
# Training
######################

model = Bigram(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
n_params = sum(p.numel() for p in model.parameters())

# untrained model ~ uniform guess over vocab, so loss should start near -log(1/V)
expected = math.log(vocab_size)

print("=" * 50)
print("Bigram model")
print("=" * 50)
print(f"parameters      {n_params:,}")
print(f"vocab size      {vocab_size}")
print(f"expected start  {expected:.4f}  (log of vocab size)")
print()
print(f"{'step':>6} | {'train':>7} | {'val':>7} | {'gap':>6} | {'time':>6}")
print("-" * 46)

step_log, train_log, val_log, batch_log = [], [], [], []
t0 = time.time()

for step in range(max_steps + 1):
    if step % eval_interval == 0:
        losses = estimate_loss(model)
        gap = losses["val"] - losses["train"]  # growing gap = overfitting
        print(f"{step:>6} | {losses['train']:>7.4f} | {losses['val']:>7.4f} | "
              f"{gap:>6.3f} | {time.time() - t0:>5.1f}s")
        step_log.append(step)
        train_log.append(losses["train"])
        val_log.append(losses["val"])

    x, y = get_batch("train", block_size, batch_size)
    _, loss = model(x, y)
    batch_log.append(loss.item())
    optimizer.zero_grad(set_to_none=True)  # grads accumulate by default
    loss.backward()
    optimizer.step()

elapsed = time.time() - t0
print("-" * 46)
print(f"final val loss  {val_log[-1]:.4f}  (baseline to beat)")

save_metrics("bigram", {
    "params": n_params, "block_size": block_size, "steps": max_steps, "time_s": elapsed,
    "final_train": train_log[-1], "final_val": val_log[-1],
    "eval_steps": step_log, "train": train_log, "val": val_log,
})

######################
# Loss curve
######################

plt.figure(figsize=(9, 5))
plt.plot(batch_log, color="lightgray", linewidth=0.8, label="single batch (noisy)")
plt.plot(step_log, train_log, "o-", label="train (averaged)")
plt.plot(step_log, val_log, "s-", label="val (averaged)")
plt.axhline(expected, color="red", linestyle="--", linewidth=1, label=f"uniform guess = {expected:.2f}")
plt.xlabel("step")
plt.ylabel("cross-entropy loss")
plt.title("Bigram loss curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "bigram_loss.png", dpi=120)
print("loss curve saved to results/bigram_loss.png")

######################
# Generation
######################

print()
print("=" * 50)
print("Sample (300 characters)")
print("=" * 50)
start = torch.zeros((1, 1), dtype=torch.long)  # id 0 is "\n"
print(decode(model.generate(start, max_new_tokens=300)[0].tolist()))
print("=" * 50)
