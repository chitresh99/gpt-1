import torch

from model import GPT, GPTConfig, MultiHeadAttention, CausalSelfAttention

torch.manual_seed(0)

######################
# 1. Batched heads == looped heads
######################

cfg = GPTConfig(vocab_size=65, block_size=16, n_layer=1, n_head=4, n_embd=32, dropout=0.0)
C = cfg.n_embd

loop = MultiHeadAttention(cfg.n_embd, cfg.n_head, cfg.block_size)
fast = CausalSelfAttention(cfg)

# copy the looped version's weights into the combined layout
# c_attn rows are [all q heads | all k heads | all v heads], head i at rows i*hs:(i+1)*hs
with torch.no_grad():
    for j, name in enumerate(["query", "key", "value"]):
        w = torch.cat([getattr(h, name).weight for h in loop.heads], dim=0)
        b = torch.cat([getattr(h, name).bias for h in loop.heads], dim=0)
        fast.c_attn.weight[j * C:(j + 1) * C] = w
        fast.c_attn.bias[j * C:(j + 1) * C] = b
    fast.c_proj.weight.copy_(loop.proj.weight)
    fast.c_proj.bias.copy_(loop.proj.bias)

x = torch.randn(2, 10, C)
diff = (loop(x) - fast(x)).abs().max().item()
print(f"1. max difference, looped vs batched heads: {diff:.2e}  (should be ~1e-7 or smaller)")

######################
# 2. Weight tying
######################

model = GPT(cfg)
same = model.lm_head.weight is model.token_emb.weight
print(f"2. lm_head shares token_emb weights: {same}")

######################
# 3. Initialization
######################

w = model.blocks[0].mlp.c_fc.weight
print(f"3. std of a Linear weight after init: {w.std().item():.4f}  (should be ~0.02)")

######################
# 4. Parameter counts
######################

def count(m):
    # parameters() yields a tied tensor only once, so it isn't double counted
    return sum(p.numel() for p in m.parameters())

small = GPT(GPTConfig(vocab_size=65, block_size=32, n_layer=4, n_head=4, n_embd=64, dropout=0.0))
print(f"4. our CPU config:        {count(small):,}")

# 40478 is the vocab size usually quoted for GPT-1's BPE (unverified here)
full = GPT(GPTConfig(vocab_size=40478))
print(f"   full GPT-1 sizes:      {count(full):,}")
