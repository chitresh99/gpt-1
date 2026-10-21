import torch
import torch.nn as nn
import torch.nn.functional as F

#############################
# Self attention
#############################


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


###########################
# Model: embeddings + one head
###########################


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
