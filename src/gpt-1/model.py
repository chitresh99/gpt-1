from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

################################
# Self attention (one head)
################################

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size)
        self.key = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        # a buffer moves with the model (.to(device), state_dict) but isn't trained
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)  # (B, T, hs)
        k = self.key(x)    # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        # softmax(q k^T / sqrt(hs)) v
        att = q @ k.transpose(-2, -1) * k.size(-1) ** -0.5  # (B, T, T)
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # causal: no peeking ahead
        att = F.softmax(att, dim=-1)
        return att @ v  # (B, T, hs)

######################
# Multi head attention
######################

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        # split the width between heads so total cost matches one big head
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        return self.proj(out)  # lets heads' outputs mix before re-entering the residual stream

######################
# Feed-forward
######################

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        # 4x expansion as in the paper (768 -> 3072 -> 768)
        # tanh-approximate GELU is the exact variant GPT-1's code used
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

##############################################
# Transformer block (post-LN, as in GPT-1)
##############################################

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        self.ln1 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # post-LN: normalize after the residual add
        # GPT-2 and later use pre-LN instead: x = x + attn(ln(x))
        x = self.ln1(x + self.attn(x))  # communicate: tokens read from each other
        x = self.ln2(x + self.mlp(x))   # compute: each token processes what it gathered
        return x

##########################
# Embeddings + one head
##########################

class AttentionLM(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        # learned positions, as in GPT-1 (the original Transformer used fixed sinusoids)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.head = Head(n_embd, n_embd, block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)                                   # (B, T, C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))      # (T, C), broadcast over B
        x = tok + pos
        x = self.head(x)
        logits = self.lm_head(x)                                    # (B, T, vocab_size)

        if targets is None:
            return logits, None
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # pos_emb only has block_size rows, so never feed in more than that
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

#####################################
# Model: stacked Transformer blocks
#####################################

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        # no final LayerNorm: with post-LN, the last block's output is already normalized
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(tok + pos)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

######################
# GPT-1
######################

@dataclass
class GPTConfig:
    vocab_size: int
    # defaults are GPT-1's sizes
    block_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        # q, k, v for every head in one matmul, instead of 3 x n_head small ones
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape
        hs = C // self.n_head

        q, k, v = self.c_attn(x).split(C, dim=2)  # each (B, T, C)
        # (B, T, C) -> (B, nh, T, hs): heads become a batch dimension
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        att = q @ k.transpose(-2, -1) * hs ** -0.5  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, hs)

        # back to (B, T, C); contiguous() because view needs memory in order after transpose
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(self.gelu(self.c_fc(x))))


class GPTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = CausalSelfAttention(cfg)
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)

    def forward(self, x):
        x = self.ln_1(x + self.attn(x))  # post-LN
        x = self.ln_2(x + self.mlp(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.Sequential(*[GPTBlock(cfg) for _ in range(cfg.n_layer)])
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        # weight tying: the output layer reuses the input embedding matrix
        self.lm_head.weight = self.token_emb.weight

    def _init_weights(self, module):
        # paper: N(0, 0.02). LayerNorm keeps its default (weight 1, bias 0)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence of {T} is longer than block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx