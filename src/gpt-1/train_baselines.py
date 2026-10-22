import argparse
import math
import time

import matplotlib.pyplot as plt
import torch

from data import vocab_size, get_batch, decode
from model import AttentionLM, TransformerLM, GPT, GPTConfig
from paths import RESULTS_DIR, save_metrics, load_metrics

# phases 2 to 4: same loop for all, constant lr, no dropout
PRESETS = {
    "attention": dict(block_size=8, n_embd=32, n_head=1, n_layer=1,
                      title="Single-head attention", compare=["bigram"]),
    "transformer": dict(block_size=32, n_embd=64, n_head=4, n_layer=4,
                        title="Transformer (post-LN blocks)", compare=["bigram", "attention"]),
    "gpt": dict(block_size=32, n_embd=64, n_head=4, n_layer=4,
                title="GPT-1 architecture, no regularization", compare=["bigram", "attention", "transformer"]),
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=PRESETS, required=True)
parser.add_argument("--block_size", type=int, default=None, help="override the preset context length")
parser.add_argument("--seed", type=int, default=1337)
args = parser.parse_args()

######################
# Config
######################

preset = PRESETS[args.model]
block_size = args.block_size or preset["block_size"]
batch_size = 32
n_embd, n_head, n_layer = preset["n_embd"], preset["n_head"], preset["n_layer"]
learning_rate = 1e-3
max_steps = 5000
eval_interval = 250
eval_iters = 200

# runs that differ from the preset get their own name, so nothing is overwritten
tag = args.model
if args.block_size:
    tag += f"_ctx{block_size}"
if args.seed != 1337:
    tag += f"_seed{args.seed}"

torch.manual_seed(args.seed)

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
# Model
######################

if args.model == "attention":
    model = AttentionLM(vocab_size, n_embd, block_size)
elif args.model == "transformer":
    model = TransformerLM(vocab_size, n_embd, n_head, n_layer, block_size)
else:
    cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer,
                    n_head=n_head, n_embd=n_embd, dropout=0.0)
    model = GPT(cfg)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
n_params = sum(p.numel() for p in model.parameters())
expected = math.log(vocab_size)

baselines = {}
for name in preset["compare"]:
    m = load_metrics(name)
    if m is not None:
        baselines[name] = m["final_val"]

print("=" * 50)
print(f"{preset['title']}  [{tag}]")
print("=" * 50)
print(f"parameters      {n_params:,}")
print(f"layers x heads  {n_layer} x {n_head}, width {n_embd}")
print(f"context         {block_size} characters")
print(f"seed            {args.seed}")
print(f"expected start  {expected:.4f}")
for name, val in baselines.items():
    print(f"{name + ' val':<15} {val:.4f}")
print()
print(f"{'step':>6} | {'train':>7} | {'val':>7} | {'gap':>6} | {'time':>6}")
print("-" * 46)

######################
# Training
######################

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

elapsed = time.time() - t0
print("-" * 46)
print(f"final val loss  {val_log[-1]:.4f}")

save_metrics(tag, {
    "params": n_params, "block_size": block_size, "seed": args.seed, "steps": max_steps, "time_s": elapsed,
    "final_train": train_log[-1], "final_val": val_log[-1],
    "eval_steps": step_log, "train": train_log, "val": val_log,
})

######################
# Loss curve
######################

plt.figure(figsize=(9, 5))
plt.plot(batch_log, color="lightgray", linewidth=0.8, label="single batch (noisy)")
plt.plot(step_log, train_log, "o-", markersize=3, label="train (averaged)")
plt.plot(step_log, val_log, "s-", markersize=3, label="val (averaged)")
for (name, val), color in zip(baselines.items(), ["green", "purple", "orange"]):
    plt.axhline(val, color=color, linestyle="--", linewidth=1, label=f"{name} val = {val:.2f}")
plt.axhline(expected, color="red", linestyle="--", linewidth=1, label=f"uniform guess = {expected:.2f}")
plt.xlabel("step")
plt.ylabel("cross-entropy loss")
plt.title(f"{preset['title']} loss curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / f"{tag}_loss.png", dpi=120)
print(f"loss curve saved to results/{tag}_loss.png")

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
