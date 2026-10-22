import argparse
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import text, encode
from model import GPT, GPTConfig
from paths import CKPT_DIR, save_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--aux", type=float, default=0.5, help="lambda for the auxiliary LM loss (paper: 0.5)")
parser.add_argument("--scratch", action="store_true", help="skip pre-trained weights, for comparison")
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--n_train", type=int, default=8000)
parser.add_argument("--n_val", type=int, default=1000)
parser.add_argument("--lr", type=float, default=2.5e-4)
parser.add_argument("--seed", type=int, default=1337)
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)

batch_size = 32      # paper: 32
weight_decay = 0.01
grad_clip = 1.0
max_line = 60        # characters kept per line, so a pair fits in block_size

######################
# Task: does line B come right after line A?
######################

def build_pairs(chunk, n):
    speeches = chunk.split("\n\n")
    positives, all_lines = [], []
    for speech in speeches:
        # drop speaker labels ("JULIET:") and very short lines
        lines = [l for l in speech.split("\n") if len(l) >= 10 and not l.endswith(":")]
        all_lines.extend(lines)
        positives.extend(zip(lines, lines[1:]))

    random.shuffle(positives)
    positives = positives[:n // 2]
    pairs = [(a, b, 1) for a, b in positives]
    # negatives reuse the same first lines, so line A alone gives nothing away
    pairs += [(a, random.choice(all_lines), 0) for a, _ in positives]
    random.shuffle(pairs)
    return pairs

split = int(0.9 * len(text))  # same split point as pre-training
train_pairs = build_pairs(text[:split], args.n_train)
val_pairs = build_pairs(text[split:], args.n_val)

######################
# Input transformation (paper, Figure 1)
######################

ckpt = torch.load(CKPT_DIR / "gpt1_pretrained.pt", map_location="cpu")
cfg = GPTConfig(**ckpt["cfg"])

# three new tokens, placed right after the 65 characters
START, DELIM, EXTRACT = cfg.vocab_size, cfg.vocab_size + 1, cfg.vocab_size + 2

def encode_pair(a, b):
    return [START] + encode(a[:max_line]) + [DELIM] + encode(b[:max_line]) + [EXTRACT]

def to_tensors(pairs):
    seqs = [encode_pair(a, b) for a, b, _ in pairs]
    T = max(len(s) for s in seqs)
    x = torch.zeros(len(seqs), T, dtype=torch.long)       # pad with 0
    y = torch.full((len(seqs), T), -1, dtype=torch.long)  # -1 = ignored by the LM loss
    for i, s in enumerate(seqs):
        s = torch.tensor(s)
        x[i, :len(s)] = s
        y[i, :len(s) - 1] = s[1:]
    ext = torch.tensor([len(s) - 1 for s in seqs])  # where each EXTRACT token sits
    labels = torch.tensor([label for _, _, label in pairs])
    return x, y, ext, labels

train = to_tensors(train_pairs)
val = to_tensors(val_pairs)
assert train[0].size(1) <= cfg.block_size and val[0].size(1) <= cfg.block_size

######################
# Model
######################

def add_tokens(gpt, n_new):
    # grow the tied embedding: old rows kept, new rows initialized like the rest
    old = gpt.token_emb.weight.data
    new = torch.empty(old.size(0) + n_new, old.size(1))
    nn.init.normal_(new, mean=0.0, std=0.02)
    new[:old.size(0)] = old
    gpt.token_emb = nn.Embedding.from_pretrained(new, freeze=False)
    gpt.lm_head = nn.Linear(old.size(1), new.size(0), bias=False)
    gpt.lm_head.weight = gpt.token_emb.weight
    gpt.cfg.vocab_size += n_new


class GPTClassifier(nn.Module):
    def __init__(self, gpt, n_classes):
        super().__init__()
        self.gpt = gpt
        self.drop = nn.Dropout(0.1)  # paper: dropout 0.1 on the classifier
        self.head = nn.Linear(gpt.cfg.n_embd, n_classes)  # W_y, the only new weights besides 3 embeddings
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, ext):
        h = self.gpt.hidden(x)                      # (B, T, C)
        h_ext = h[torch.arange(x.size(0)), ext]     # (B, C), hidden state at EXTRACT
        cls_logits = self.head(self.drop(h_ext))    # (B, n_classes)
        lm_logits = self.gpt.lm_head(h)             # (B, T, vocab), for the auxiliary loss
        return cls_logits, lm_logits


gpt = GPT(cfg)
if not args.scratch:
    gpt.load_state_dict(ckpt["model"])
add_tokens(gpt, 3)
model = GPTClassifier(gpt, n_classes=2)

decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW([
    {"params": decay, "weight_decay": weight_decay},
    {"params": no_decay, "weight_decay": 0.0},
], lr=args.lr)

######################
# Schedule (paper: warmup over 0.2% of training, then linear decay)
######################

steps_per_epoch = math.ceil(len(train_pairs) / batch_size)
total_steps = args.epochs * steps_per_epoch
warmup = max(1, int(0.002 * total_steps))

def get_lr(step):
    if step < warmup:
        return args.lr * (step + 1) / warmup
    return args.lr * max(0.0, (total_steps - step) / (total_steps - warmup))

######################
# Evaluation
######################

@torch.no_grad()
def evaluate(data):
    model.eval()
    x, y, ext, labels = data
    correct, lm_total = 0, 0.0
    n_batches = 0
    for i in range(0, len(labels), batch_size):
        cls_logits, lm_logits = model(x[i:i + batch_size], ext[i:i + batch_size])
        correct += (cls_logits.argmax(-1) == labels[i:i + batch_size]).sum().item()
        lm_total += F.cross_entropy(lm_logits.reshape(-1, lm_logits.size(-1)),
                                    y[i:i + batch_size].reshape(-1), ignore_index=-1).item()
        n_batches += 1
    model.train()
    return correct / len(labels), lm_total / n_batches

######################
# Training
######################

print("=" * 54)
print("GPT-1 fine-tuning: does line B follow line A?")
print("=" * 54)
print(f"weights         {'random (no pre-training)' if args.scratch else 'pre-trained'}")
print(f"aux lambda      {args.aux}")
print(f"train / val     {len(train_pairs)} / {len(val_pairs)} pairs, 50% positive")
print(f"steps           {total_steps} ({args.epochs} epochs), warmup {warmup}")
before_acc, _ = evaluate(val)
print(f"before training val acc {before_acc:.3f}  (chance 0.500)")
print()
print(f"{'epoch':>5} | {'cls loss':>8} | {'lm loss':>7} | {'val acc':>7} | {'val lm':>6} | {'time':>6}")
print("-" * 54)

x_all, y_all, ext_all, labels_all = train
history = []
step = 0
t0 = time.time()

for epoch in range(1, args.epochs + 1):
    perm = torch.randperm(len(labels_all))
    cls_sum, lm_sum, n_batches = 0.0, 0.0, 0

    for i in range(0, len(perm), batch_size):
        idx = perm[i:i + batch_size]
        x, y, ext, labels = x_all[idx], y_all[idx], ext_all[idx], labels_all[idx]

        lr = get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        cls_logits, lm_logits = model(x, ext)
        loss_cls = F.cross_entropy(cls_logits, labels)                                    # L2
        loss_lm = F.cross_entropy(lm_logits.reshape(-1, lm_logits.size(-1)),
                                  y.reshape(-1), ignore_index=-1)                         # L1
        loss = loss_cls + args.aux * loss_lm                                              # L3 = L2 + lambda * L1

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        cls_sum += loss_cls.item()
        lm_sum += loss_lm.item()
        n_batches += 1
        step += 1

    acc, val_lm = evaluate(val)
    history.append({"epoch": epoch, "cls_loss": cls_sum / n_batches, "lm_loss": lm_sum / n_batches,
                    "val_acc": acc, "val_lm": val_lm})
    print(f"{epoch:>5} | {cls_sum / n_batches:>8.4f} | {lm_sum / n_batches:>7.4f} | "
          f"{acc:>7.3f} | {val_lm:>6.3f} | {time.time() - t0:>5.1f}s")

print("-" * 54)
print(f"final val accuracy {acc:.3f}")

tag = (f"finetune_{'scratch' if args.scratch else 'pretrained'}_aux{args.aux:g}"
       f"_ep{args.epochs}_lr{args.lr:g}_n{args.n_train}_seed{args.seed}")
save_metrics(tag, {"args": vars(args), "before_acc": before_acc, "history": history,
                   "final_val_acc": acc, "time_s": time.time() - t0})
print(f"metrics saved to results/metrics/{tag}.json")
