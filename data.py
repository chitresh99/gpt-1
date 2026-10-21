import os
import urllib.request

import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.txt")

if not os.path.exists(PATH):
    urllib.request.urlretrieve(URL, PATH)

with open(PATH, encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}


def encode(s):
    return [stoi[c] for c in s]


def decode(ids):
    return "".join(itos[i] for i in ids)


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # the first 90% is for training, the last 10% is held back.
train_data = data[:n]
val_data = data[n:]


def get_batch(split, block_size, batch_size):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    # y is x shifted one to the right: y[t] is the answer for x[:t+1]
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x, y


if __name__ == "__main__":
    print("characters:", len(text))
    print("vocab size:", vocab_size)
    print("vocab:", repr("".join(chars)))
    print(encode("hello"), "->", decode(encode("hello")))

    torch.manual_seed(1337)
    x, y = get_batch("train", block_size=8, batch_size=4)
    print("x:\n", x)
    print("y:\n", y)