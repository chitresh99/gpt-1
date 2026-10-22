# Implementing Radford et al. (2018): GPT-1

Paper: [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

## Quick start

```bash
uv sync
./run_all.sh
```

`run_all.sh` reruns every experiment in this README in order and saves each run's full output to `results/logs/`. Individual scripts can also be run on their own from any folder, for example:

```bash
python3 src/gpt-1/train.py
python3 src/gpt-1/sample.py --prompt "ROMEO:" --temperature 0.8 --top_k 10
python3 src/gpt-1/finetune.py --aux 0
```

`sample.py` and `finetune.py` need the checkpoint written by `train.py`.

## Goal

Implement GPT-1 from scratch in PyTorch: a decoder-only Transformer pre-trained as a language model, then fine-tuned on a labeled task using the paper's input transformation and auxiliary language modeling loss. The model is built up in phases, from a bigram baseline to the full architecture, so the contribution of each component can be measured. The goal was a correct small-scale implementation that runs on a CPU, where every component is checked against the paper, not a reproduction of the paper's benchmark results.

## Data

### Pre-training

- **Dataset:** Tiny Shakespeare, 1,115,394 characters.
- **Tokenizer:** character level. 65 unique characters: newline, space, `!$&',-.3:;?`, A to Z and a to z.
- **Split:** the first 90% for training, the last 10% for validation. The split is at the end rather than random, so the two sets never share passages.
- **Batches:** random contiguous chunks of `block_size` characters. The target `y` is the input `x` shifted one position to the right, so a chunk of length `block_size` holds `block_size` next-character prediction examples.

### Fine-tuning task

The paper's benchmarks need BPE, real sentences and large downloads, so the fine-tuning task is built from the same Shakespeare text:

**Does line B come directly after line A?**

- **Positive (label 1):** two consecutive lines from the same speech.
- **Negative (label 0):** the same line A with a random line B. Reusing line A means the first line alone gives nothing away.
- Speaker labels (lines ending in `:`) and lines shorter than 10 characters are dropped. Each line is cut to 60 characters.
- Training pairs come from the first 90% of the text and validation pairs from the last 10%, which the model never saw during pre-training.
- 8,000 training pairs and 1,000 validation pairs by default, 50% positive. Chance accuracy is 50%.

This is a sentence-pair task, the same input shape as the paper's textual entailment tasks, so it uses the paper's input transformation with a delimiter.

## What follows the paper

All items below were checked against the paper text.

| Component | Paper | This implementation |
|---|---|---|
| Pre-training objective | Maximize the sum of log P(u_i given the previous k tokens) (Section 3.1, Eq. 1) | Same, next-character cross-entropy |
| Input | h_0 = U W_e + W_p, token plus position embeddings (Eq. 2) | Same |
| Output | P(u) = softmax(h_n W_e^T), output uses the token embedding matrix (Eq. 2) | Same, weight tying |
| Architecture | 12-layer decoder-only Transformer with masked self-attention | Same structure, 4 layers |
| Width | 768-dim states, 12 heads, 3072-dim feed-forward | 64-dim, 4 heads, 256-dim feed-forward (same 4x ratio) |
| Activation | GELU | Same (tanh approximation) |
| Position embeddings | Learned, not sinusoidal | Same |
| Initialization | N(0, 0.02), justified by the heavy use of LayerNorm | Same |
| Dropout | 0.1 on residual, embedding and attention | Same |
| Optimizer | Adam | AdamW |
| Weight decay | Modified L2 from Loshchilov and Hutter, w = 0.01 on all non-bias, non-gain weights | Same, via AdamW parameter groups |
| Learning rate schedule | Linear warmup from 0, then cosine annealing to 0 | Same shape, scaled (see Deviations) |
| Fine-tuning head | P(y given x) = softmax(h_l^m W_y), read at the last token (Eq. 3) | Same |
| Fine-tuning loss | L3 = L2 + lambda * L1, lambda = 0.5 (Eq. 4 and 5) | Same |
| New parameters at fine-tuning | Only W_y and the embeddings of the added tokens | Same |
| Input transformation | Randomly initialized start and end tokens; for entailment, premise, delimiter, hypothesis (Section 3.3) | `<start> A <delim> B <extract>` |
| Fine-tuning schedule | Warmup over 0.2% of training, then linear decay | Same |
| Fine-tuning settings | Batch 32, 3 epochs, classifier dropout 0.1 | Same by default |

## Deviations

| | Paper | This implementation | Reason |
|---|---|---|---|
| Scale | 12 layers, 768 width, 12 heads | 4 layers, 64 width, 4 heads, 212,288 parameters | CPU |
| Data | BooksCorpus, over 7,000 unpublished books | Tiny Shakespeare, about 1.1M characters | CPU |
| Tokenizer | BPE with 40,000 merges | Characters, vocabulary of 65 | Keeps the tokenizer trivial; the model code does not depend on it |
| Context | 512 tokens | 128 characters | CPU; long enough for the fine-tuning pairs (at most 123 tokens) |
| Batch size | 64 | 32 | CPU |
| Training length | 100 epochs | 5,000 steps | CPU |
| Max learning rate | 2.5e-4 | 1e-3 | A 200K-parameter model tolerates larger steps; 1e-3 was already stable in the earlier phases |
| Warmup | 2,000 updates | 500 steps (10% of training) | Scaled to the much shorter run |
| Fine-tuning learning rate | 6.25e-5 | 2.5e-4 by default | Pre-training used 4x the paper's rate, so fine-tuning was scaled by the same factor |
| Fine-tuning task | 12 benchmarks (NLI, QA, similarity, classification) | One sentence-pair task built from Shakespeare | Benchmarks need BPE and real data |
| Gradient clipping | Not mentioned | Global norm clipped to 1.0 | Cheap protection against rare large gradients |
| End token name | `<e>` (end token) | `<extract>` | Same role: the classifier reads the hidden state at this token |
| Hardware | GPUs | CPU only (Intel Core i5-1155G7, 4 cores, 8 threads) | Compute |

## Not stated in the paper

These choices are not specified by the paper text. Each is either inherited from the original Transformer or a decision made here.

- **Post-LN.** LayerNorm is applied after each residual add: `x = LN(x + sublayer(x))`. The paper says the model "largely follows the original transformer work", which used post-LN, but does not state the placement directly. GPT-2 and later models moved to pre-LN.
- **No final LayerNorm before the output layer.** With post-LN, every block already ends in a LayerNorm.
- **Biases on the query, key and value projections.**
- **The tanh approximation of GELU.**
- **Vocabulary size of 40,478 for the full-size count.** The paper gives 40,000 merges, not a vocabulary size. 40,478 is the figure commonly used for GPT-1's released vocabulary and is not verified here.
- **117M parameters.** The paper does not give a parameter count. This code builds a 116,534,784-parameter model at the paper's sizes, assuming the 40,478 vocabulary.

## Code decisions

- **The model is built in phases,** from a bigram table to the full GPT-1, and every phase stays runnable. Each step can be compared against the one before it.
- **Character-level tokens.** The tokenizer is a lookup table. BPE would add a separate topic without changing the model.
- **Batched multi-head attention.** One linear layer computes queries, keys and values for all heads, which are then reshaped into a head dimension. `checks.py` shows it gives the same output as running the heads one at a time (maximum difference 0.0).
- **Layer names follow OpenAI's code** (`c_attn`, `c_proj`, `c_fc`, `ln_1`, `ln_2`), to make side-by-side reading easier.
- **Weight decay groups are chosen by dimension.** Matrices (2D) get weight decay; biases and LayerNorm parameters (1D) do not.
- **Growing the tied embedding for fine-tuning.** The three new tokens get new rows initialized like the rest, the old 65 rows are kept, and the output layer is re-tied to the new matrix. Without the re-tie, the output layer would still point at the old 65-row matrix and the auxiliary loss would break.
- **Right padding with ignored targets.** Pairs of different lengths are padded at the end. Causal attention means no real position can see the padding, and padded targets are set to -1, which the loss skips.
- **Every run saves its numbers** to `results/metrics/` as JSON, and plots read their baseline lines from there instead of hard-coded values.
- **Seed 1337 everywhere.** Every run that existed before `run_all.sh` was written reproduced its earlier results exactly when rerun on the same machine. The context ablation and runs 41, 50 and 51 were first run with `run_all.sh`.

## Folder structure

```
gpt-1/
  README.md
  pyproject.toml, uv.lock
  run_all.sh              reruns every experiment, saves logs
  data/input.txt          downloaded on first run
  checkpoints/
    gpt1_pretrained.pt    written by train.py
  results/
    *.png                 loss curves
    metrics/*.json        final numbers and curves per run
    logs/*.txt            full terminal output per run
  src/gpt-1/
    paths.py              locates the repo folders from any working directory
    data.py               data loading, tokenizer, batching
    bigram.py             Phase 1 baseline
    attention_demo.py     attention built up from averaging, with toy numbers
    model.py              every model, from one attention head to GPT-1
    checks.py             correctness checks for the GPT-1 model
    train_baselines.py    Phases 2 to 4, and the context ablation
    train.py              pre-training with GPT-1's recipe
    sample.py             text generation with temperature and top-k
    finetune.py           fine-tuning with the input transformation and auxiliary loss
```

## Pipeline overview

1. **Data.** Download Tiny Shakespeare, map characters to integers, split 90/10.
2. **Phase 1, bigram.** A 65 x 65 table predicting the next character from the current one. Sets the baseline.
3. **Phase 2, one attention head.** Token and position embeddings, one head of causal self-attention, an output layer.
4. **Phase 3, Transformer blocks.** Multi-head attention, a GELU feed-forward network, residual connections and post-LN, stacked 4 deep.
5. **Phase 4, GPT-1 architecture.** Batched attention, weight tying, N(0, 0.02) initialization, a config with the paper's sizes as defaults.
6. **Phase 5, pre-training.** Dropout, weight decay, gradient clipping, warmup and cosine decay. Saves a checkpoint.
7. **Phase 6, generation.** Temperature and top-k sampling from the checkpoint.
8. **Phase 7, fine-tuning.** Add three special tokens, transform each pair into one sequence, read the hidden state at the last token, train with L3 = L2 + lambda * L1.

## Checks

| Check | Result | What it verifies |
|---|---|---|
| Starting loss vs log(65) = 4.1744 | 4.1775 in pre-training | Initialization gives a near-uniform first prediction |
| Batched vs looped attention heads | Maximum difference 0.0 | The reshaped multi-head attention is exactly equivalent |
| Weight tying | `lm_head.weight is token_emb.weight` is True | Output and input share one matrix, not copies |
| Initialization | Weight standard deviation 0.0200 | N(0, 0.02) is applied |
| Parameter count | 206,144 (context 32) and 212,288 (context 128) | Matches the hand calculation below |
| Full-size count | 116,534,784 | The code builds a model of GPT-1's size at the paper's settings |
| Attention scaling | Variance of q.k for d = 64: 63.29 raw, 0.99 scaled | Dividing by sqrt(d) keeps scores at unit variance |

### Parameter count

Pre-training configuration: 4 layers, width 64, 4 heads, context 128, vocabulary 65.

| Part | Calculation | Parameters |
|---|---|---|
| Token embeddings (tied with output) | 65 * 64 | 4,160 |
| Position embeddings | 128 * 64 | 8,192 |
| Attention: combined q, k, v | 64 * 192 + 192 | 12,480 |
| Attention: output projection | 64 * 64 + 64 | 4,160 |
| Feed-forward: first layer | 64 * 256 + 256 | 16,640 |
| Feed-forward: second layer | 256 * 64 + 64 | 16,448 |
| Two LayerNorms | 2 * (64 + 64) | 256 |
| One block | sum of the five rows above | 49,984 |
| Four blocks | 4 * 49,984 | 199,936 |
| Output layer | tied with the token embeddings | 0 |
| Total | 4,160 + 8,192 + 199,936 | 212,288 |

Fine-tuning adds 192 parameters for the three new token embeddings (3 * 64) and 130 for the classifier (64 * 2 + 2).

At the paper's sizes with a 40,478-token vocabulary, the token embeddings hold 31,087,104 parameters, the position embeddings 393,216 and the 12 blocks 85,054,464, for 116,534,784 in total.

## Issues found during implementation

| Issue | Symptom | Fix |
|---|---|---|
| Evaluation every 500 steps | The averaged loss curve drew a straight line from step 0 to 500, far above the real curve shown by the per-batch loss | Evaluate every 250 steps |
| One training script edited in every phase | Results from Phases 2 to 4 could no longer be reproduced | `train_baselines.py` reproduces each phase from a preset; the reruns matched the original numbers exactly |
| Scripts used bare file names | Moving the checkpoint and plots into their own folders broke loading | `paths.py` resolves the repo folders from the script's own location |
| Replacing the embedding to add special tokens | Would leave the output layer tied to the old 65-row matrix | Re-tie `lm_head.weight` after resizing |

## Results

All runs on CPU with seed 1337. Times are wall-clock from the `run_all.sh` run and depend on what else the machine was doing.

### Pre-training: building up the architecture

| Phase | Model | Params | Context | Final train | Final val | Val perplexity | Time |
|---|---|---|---|---|---|---|---|
| 1 | Bigram | 4,225 | 1 | 2.4660 | 2.4888 | 12.05 | 5.9s |
| 2 | Single-head attention | 7,649 | 8 | 2.3839 | 2.4012 | 11.04 | 23.6s |
| 3 | Transformer, 4 x 4, width 64 | 208,833 | 8 | 1.8278 | 1.9649 | 7.13 | 160.5s |
| 3 | Transformer, 4 x 4, width 64 | 210,369 | 32 | 1.5435 | 1.7351 | 5.67 | 293.6s |
| 4 | GPT-1 architecture, no regularization | 206,144 | 32 | 1.6031 | 1.7707 | 5.87 | 164.8s |
| 5 | GPT-1 with the paper's training recipe | 212,288 | 128 | 1.7142 | 1.8654 | 6.46 | 1128.8s |

Perplexity is exp(val loss). Phases 1 to 4 use a constant learning rate of 1e-2 (bigram) or 1e-3, no dropout and 5,000 steps (3,000 for the bigram). Phase 5 uses dropout 0.1, weight decay 0.01, gradient clipping and warmup with cosine decay.

### Starting loss

| Model | Step 0 loss | Expected (log 65) |
|---|---|---|
| Bigram | 4.7305 | 4.1744 |
| Transformer | 4.3190 | 4.1744 |
| GPT-1 architecture | 4.1923 | 4.1744 |
| GPT-1 pre-training | 4.1775 | 4.1744 |

### Fine-tuning

| Run | Weights | lambda | Epochs | lr | Train pairs | Untrained-head acc | Final val acc | Final train cls loss | Final val LM loss |
|---|---|---|---|---|---|---|---|---|---|
| 30 | Pre-trained | 0.5 | 3 | 2.5e-4 | 8,000 | 53.7% | 56.1% | 0.6869 | 1.866 |
| 31 | Pre-trained | 0 | 3 | 2.5e-4 | 8,000 | 53.7% | 54.7% | 0.6860 | 2.502 |
| 32 | From scratch | 0.5 | 3 | 2.5e-4 | 8,000 | 50.2% | 53.2% | 0.6873 | 2.717 |
| 40 | Pre-trained | 0.5 | 10 | 1e-3 | 8,000 | 53.7% | 58.0% | 0.6591 | 1.789 |
| 41 | Pre-trained | 0 | 10 | 1e-3 | 8,000 | 53.7% | 56.2% | 0.6741 | 4.363 |
| 42 | From scratch | 0.5 | 10 | 1e-3 | 8,000 | 50.2% | 58.8% | 0.6563 | 2.155 |
| 50 | Pre-trained | 0.5 | 10 | 1e-3 | 1,000 | 53.4% | 52.8% | 0.6081 | 1.898 |
| 51 | From scratch | 0.5 | 10 | 1e-3 | 1,000 | 50.8% | 52.5% | 0.6677 | 2.596 |

The run numbers match the log files in `results/logs/`. Validation uses 1,000 pairs in every run.

### Validation LM loss during fine-tuning without the auxiliary loss

| Epoch | Run 40 (lambda 0.5) | Run 41 (lambda 0) |
|---|---|---|
| 1 | 1.863 | 4.073 |
| 2 | 1.846 | 5.189 |
| 3 | 1.836 | 4.822 |
| 10 | 1.789 | 4.363 |

Pre-training ended at a validation loss of 1.8654.

### Samples

Phase 3 Transformer (context 32), end of training:

```
KING HENRY VI:
Shwark, by true ank thy other husband, and indeed and that we may mean to cust part less'd
Rominals one of vuled and ressued,
Which face my heart ignon--
I pare you!

JULIET:
Ho.
```

Pre-trained GPT-1, `sample.py --temperature 0.8 --top_k 10`, seed 1337:

```
Fram shall much a beard that may thou day.

SCORIOLANUS:
Are mead, sir, and be spooks son the shim;
The did the well the arte staret the sears of this
Is cale and the he done, my the may day; and this she
as the sand her are son fears for the stain,
```

### Plots

| File | Contents |
|---|---|
| `results/bigram_loss.png` | Phase 1 |
| `results/attention_loss.png` | Phase 2 |
| `results/transformer_loss.png` | Phase 3, context 32 |
| `results/transformer_ctx8_loss.png` | Phase 3, context 8 |
| `results/gpt_loss.png` | Phase 4 |
| `results/pretrain_loss.png` | Phase 5, loss and learning rate schedule |

## Analysis

### How much noise is in the fine-tuning numbers

The standard error of an accuracy near 50% on 1,000 examples is sqrt(0.5 * 0.5 / 1000), about 0.016. Differences of roughly 3 percentage points are within normal noise (95% interval). The untrained-head accuracy shows this directly: the pre-trained model with a random classifier scored 53.7% on one validation set and 53.4% on the other, from noise alone. Each result is a single run with one seed.

### Architecture and context both help, architecture more

The context ablation separates the Phase 3 improvement. At a context of 8, replacing one attention head with four Transformer blocks lowered val loss from 2.4012 to 1.9649, a drop of 0.436. At the same architecture, raising the context from 8 to 32 lowered it from 1.9649 to 1.7351, a drop of 0.230. The architecture change also raised the parameter count from 7,649 to 208,833, so it mixes better structure with more capacity.

### GPT-1's initialization works as described

The starting loss moved closer to log(65) at each change: 4.7305 for the bigram, whose embeddings start with standard deviation 1, 4.3190 for the Transformer with PyTorch's default initialization, and 4.1923 with N(0, 0.02). Small weights give small first logits and a near-uniform first prediction.

### Matching GPT-1 did not improve this small model's language modeling

The Phase 4 model, with weight tying and GPT-1's initialization, ended at 1.7707 against 1.7351 for the untied Phase 3 model. It started lower but learned more slowly, and was still improving at step 5,000. The full training recipe in Phase 5 ended higher again, at 1.8654, despite a context four times longer. Two likely causes, neither tested: dropout removes capacity from a model whose limit is capacity rather than overfitting (the final gap was 0.151, close to Phase 4's 0.168), and the cosine schedule spends its last third at small learning rates (val loss moved from 1.8910 to 1.8654 after step 3,500). The recipe was tuned for a 117M-parameter model on BooksCorpus.

### The auxiliary loss prevents catastrophic forgetting

This is the clearest result. Without the auxiliary loss, the model's language modeling degraded within one epoch of fine-tuning. At the paper-style settings (runs 30 and 31), validation LM loss rose from the pre-trained 1.8654 to 2.736 after one epoch, against 1.893 with the auxiliary loss. At the higher learning rate (runs 40 and 41) it reached 5.189 after two epochs, above log(68) = 4.2195, the loss of a uniform guess over the 68 tokens. With the auxiliary loss, LM loss kept improving to 1.789. Task accuracy was not measurably different (58.0% against 56.2%, inside the noise band). The LM loss here is measured on the sentence pairs, including the special tokens, so it is not the same measurement as pre-training validation loss, but it is close enough for this comparison.

### Pre-training did not transfer to the task at this scale

At 3 epochs (runs 30 and 32), the classifier barely learned anything: every final training classification loss was about 0.687, against log(2) = 0.6931 for a 50/50 guess. At 10 epochs with a higher learning rate, it did learn: run 40 reached 58.0%, about 5 standard errors above chance. But the matched run without pre-training (run 42) reached 58.8%, and its training loss fell at the same pace. With 1,000 training pairs (runs 50 and 51), both models stayed at chance, 52.8% and 52.5%. The pre-trained model drove its training loss lower (0.608 against 0.668) without any gain on validation, so it memorized the small set faster without generalizing.

A likely explanation, not tested: part of this task can be solved from surface cues, such as how a line's ending punctuation fits the next line, which a small model can learn from labeled pairs alone. Getting past those cues needs an understanding of meaning that neither model has at 200K parameters, so the pre-trained model's better language knowledge has nothing to add.

### How this compares with the paper

The paper's own ablations are consistent with parts of these results:

- **Auxiliary loss.** In Table 5, the average score is 74.7 with the auxiliary loss and 75.0 without. The paper reports that it helps on the NLI tasks and QQP, and that larger datasets benefit while smaller ones do not. Here too, the auxiliary loss had a large effect on language modeling but no measurable effect on task accuracy.
- **Pre-training.** Removing pre-training cost 14.8% on average (74.7 against 59.9). The smallest gap among the paper's tasks was on RTE, 2,490 training examples: 56.0% with pre-training and 53.8% without. That is a far larger model on a different task, so it proves nothing about this one, but it shows that even in the paper, a small, hard sentence-pair task stayed close to chance and gained little from pre-training.

The paper's perplexity of 18.4 is per BPE token on BooksCorpus, and the perplexities here are per character on Shakespeare. They cannot be compared.

## Glossary

**Language model.** A model that assigns a probability to the next token given the tokens before it. GPT-1's pre-training trains exactly this.

**Decoder-only Transformer.** A stack of Transformer blocks with causal self-attention and no encoder. Each position can only see itself and earlier positions.

**Self-attention.** Each position computes a query, key and value. Scores are query-key dot products scaled by 1 / sqrt(d), the future is masked, softmax turns each row into weights, and the output is the weighted sum of values: softmax(Q K^T / sqrt(d) + M) V.

**Causal mask.** Setting the scores for future positions to negative infinity before softmax, so they get exactly zero weight.

**Multi-head attention.** Several attention heads in parallel, each with its own projections and a slice of the width, concatenated and mixed by an output projection.

**Residual connection.** Adding a sublayer's output to its input, x + f(x). Gives gradients a direct path through deep stacks.

**LayerNorm.** Normalizes each position's vector to mean 0 and variance 1 across its features, then applies a learned scale and shift.

**Post-LN and pre-LN.** Post-LN applies LayerNorm after the residual add, LN(x + f(x)), as in the original Transformer and GPT-1. Pre-LN applies it before the sublayer, x + f(LN(x)), as in GPT-2 onward, which keeps the residual path clean and trains more stably in deep models.

**GELU.** x * Phi(x), where Phi is the standard normal CDF. A smooth alternative to ReLU.

**Weight tying.** Using the token embedding matrix as the output layer's weights, so the score for each token is the dot product between the model's output and that token's embedding.

**Dropout.** During training, zeroing each value with a fixed probability and rescaling the rest. Discourages reliance on any single feature.

**Weight decay (decoupled).** Shrinking weights toward zero by a fixed fraction each step, applied directly rather than through the gradient, so Adam's per-weight scaling does not weaken it.

**Warmup and cosine decay.** The learning rate rises linearly from 0 to its peak, then follows half a cosine down to 0.

**Gradient clipping.** If the norm of all gradients together exceeds a threshold, they are scaled down to that norm.

**Cross-entropy.** The loss -log(p), where p is the probability given to the correct token, averaged over positions. For a uniform guess over V tokens it equals log(V).

**Perplexity.** exp(cross-entropy). The effective number of equally likely choices the model is deciding between.

**Temperature.** Dividing the logits by tau before softmax. Below 1 sharpens the distribution, above 1 flattens it.

**Top-k sampling.** Keeping only the k highest-scoring tokens and sampling among them.

**Fine-tuning.** Continuing to train a pre-trained model on a labeled task, with a small new output layer.

**Input transformation.** Turning a structured input, such as a sentence pair, into one token sequence with start, delimiter and end tokens, so the pre-trained model can read it without architectural changes.

**Auxiliary loss.** Keeping the language modeling loss on during fine-tuning, weighted by lambda: L3 = L2 + lambda * L1.

**Catastrophic forgetting.** A model losing what it learned earlier while training on something new.

**Ablation.** Changing one component at a time to measure its contribution.

## Lessons

- **Check the starting loss.** log(vocabulary size) is a free test of initialization, and it tracked every change to initialization in this project.
- **Change one thing at a time.** Phase 3 changed the architecture and the context together, and only the later ablation could say how much each contributed.
- **Know the noise level before comparing runs.** With 1,000 validation examples, differences under about 3 points mean nothing, and an untrained classifier already scored 53.7%.
- **Look at the loss, not only the accuracy.** The fine-tuning accuracies hid that the classifiers were at chance-level loss, and the LM loss column showed the forgetting effect that accuracy could not.
- **A recipe is tuned for a scale.** GPT-1's dropout and schedule were designed for 117M parameters, and on 200K parameters they made language modeling worse.
- **Keep every phase runnable.** Editing one training script in place made earlier results unreproducible until the phases were split into presets.
- **A negative result is still a result.** Pre-training did not transfer here, and the analysis says why that is consistent with the scale rather than a bug.

## Open items

- Repeat runs with other seeds, to measure run-to-run variance.
- A BPE tokenizer, as in the paper.
- A real benchmark task, at a model size where transfer can show.
- Ablations of dropout and the learning rate schedule on their own, to test why Phase 5 ended higher than Phase 4.

## Citations

```bibtex
@techreport{radford2018improving,
    title       = {Improving Language Understanding by Generative Pre-Training},
    author      = {Alec Radford and Karthik Narasimhan and Tim Salimans and Ilya Sutskever},
    institution = {OpenAI},
    year        = {2018}
}

@inproceedings{vaswani2017attention,
    title     = {Attention Is All You Need},
    author    = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
    booktitle = {Advances in Neural Information Processing Systems},
    year      = {2017}
}

@misc{loshchilov2017fixing,
    title         = {Fixing Weight Decay Regularization in Adam},
    author        = {Ilya Loshchilov and Frank Hutter},
    year          = {2017},
    eprint        = {1711.05101},
    archivePrefix = {arXiv}
}

@misc{hendrycks2016gelu,
    title         = {Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units},
    author        = {Dan Hendrycks and Kevin Gimpel},
    year          = {2016},
    eprint        = {1606.08415},
    archivePrefix = {arXiv}
}
```

Tiny Shakespeare is taken from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) repository. The phase-by-phase teaching order follows his [ng-video-lecture](https://github.com/karpathy/ng-video-lecture), and the module layout follows [minGPT](https://github.com/karpathy/minGPT).