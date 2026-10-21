import torch
import torch.nn as nn
import torch.nn.functional as F

############################
# self attention (one head)
############################


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
        k = self.key(x)  # (B, T, hs)
        v = self.value(x)  # (B, T, hs)

        # softmax(q k^T / sqrt(hs)) v
        att = q @ k.transpose(-2, -1) * k.size(-1) ** -0.5  # (B, T, T)
        att = att.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # causal: no peeking ahead
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
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)
        return self.proj(
            out
        )  # lets heads' outputs mix before re-entering the residual stream


######################
# Feed forward
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


#############################################
# Transformer block (post-LN, as in GPT-1)
#############################################


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
        x = self.ln2(x + self.mlp(x))  # compute: each token processes what it gathered
        return x


######################
# Embeddings + One head
######################


class AttentionLM(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        # learned positions, as in gpt-1 (the original transformer used fixed sinusoids)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.head = Head(n_embd, n_embd, block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)  # (B, T, C)
        pos = self.pos_emb(
            torch.arange(T, device=idx.device)
        )  # (T, C), broadcast over B
        x = tok + pos
        x = self.head(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            return logits, None
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # pos_emb only has block_size rows, so never feed in more than that
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


###################################
# stacked transformer blocks
###################################


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
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
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
