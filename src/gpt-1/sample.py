import argparse

import torch

from data import encode, decode, stoi
from model import GPT, GPTConfig

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="\n")
parser.add_argument("--tokens", type=int, default=500)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=None)
parser.add_argument("--seed", type=int, default=1337)
args = parser.parse_args()

unknown = sorted(set(c for c in args.prompt if c not in stoi))
if unknown:
    raise SystemExit(f"prompt has characters the model never saw: {unknown}")

torch.manual_seed(args.seed)

ckpt = torch.load("gpt1_pretrained.pt", map_location="cpu")
model = GPT(GPTConfig(**ckpt["cfg"]))
model.load_state_dict(ckpt["model"])
model.eval()  # dropout off for generation

idx = torch.tensor([encode(args.prompt)], dtype=torch.long)
out = model.generate(idx, args.tokens, temperature=args.temperature, top_k=args.top_k)
print(decode(out[0].tolist()))