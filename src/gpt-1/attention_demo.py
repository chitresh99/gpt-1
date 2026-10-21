import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
T = 4  # four positions in a sequence

######################
# 1. Averaging the past with a matrix multiply
######################

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # one number per position, shape (T, 1)

tril = torch.tril(torch.ones(T, T))
weights = tril / tril.sum(dim=1, keepdim=True)  # each row sums to 1

print("1. averaging weights (row t = how much position t takes from each position)")
print(weights)
print("result: running mean of the past")
print(weights @ x)  # 1.0, 1.5, 2.0, 2.5

######################
# 2. The same weights from masking + softmax
######################

scores = torch.zeros(T, T)
scores = scores.masked_fill(
    tril == 0, float("-inf")
)  # exp(-inf) = 0, so the future gets no weight
weights = F.softmax(scores, dim=-1)

print("\n2. masked softmax of all-equal scores (same as step 1)")
print(weights)

######################
# 3. Scores that depend on the data: one head of self-attention
######################

C = 8  # features per position
head_size = 4
x = torch.randn(T, C)

query = nn.Linear(C, head_size)
key = nn.Linear(C, head_size)
value = nn.Linear(C, head_size)

q, k, v = query(x), key(x), value(x)  # each (T, head_size)
scores = q @ k.T * head_size**-0.5  # (T, T), how well each query matches each key
scores = scores.masked_fill(tril == 0, float("-inf"))
weights = F.softmax(scores, dim=-1)
out = weights @ v  # (T, head_size)

print("\n3. learned attention weights (uneven now, the data decides)")
print(weights.detach().round(decimals=2))
print("output shape:", tuple(out.shape))

######################
# 4. Why divide by sqrt(head_size)
######################

d = 64
q = torch.randn(10000, d)
k = torch.randn(10000, d)
dots = (q * k).sum(dim=-1)

print("\n4. variance of q.k for random unit-variance vectors, d =", d)
print(f"raw:    {dots.var().item():6.2f}   (grows with d)")
print(f"scaled: {(dots * d ** -0.5).var().item():6.2f}   (stays near 1)")

s = torch.tensor([1.0, 2.0, 3.0])
print("softmax of small scores:", F.softmax(s, dim=-1).round(decimals=3))
print("softmax of same x 8:    ", F.softmax(s * 8, dim=-1).round(decimals=3))
