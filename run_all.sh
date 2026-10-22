#!/usr/bin/env bash
# reruns every experiment in order. 
# each run's terminal output is saved to results/logs/.

set -euo pipefail

cd "$(dirname "$0")"
SRC=src/gpt-1
LOGS=results/logs
mkdir -p "$LOGS"

run() {
    local name=$1; shift
    echo ">>> $name"
    python3 "$@" 2>&1 | tee "$LOGS/$name.txt"
}

# setup checks
run 00_data             $SRC/data.py
run 01_attention_demo   $SRC/attention_demo.py
run 02_checks           $SRC/checks.py

# phases 1 to 4: building up the architecture
run 10_bigram           $SRC/bigram.py
run 11_attention        $SRC/train_baselines.py --model attention
run 12_transformer      $SRC/train_baselines.py --model transformer
run 13_transformer_ctx8 $SRC/train_baselines.py --model transformer --block_size 8
run 14_gpt              $SRC/train_baselines.py --model gpt

# pre-training with GPT-1's recipe
run 20_pretrain         $SRC/train.py
run 21_sample           $SRC/sample.py --temperature 0.8 --top_k 10

# fine-tuning, paper-style settings (3 epochs)
run 30_ft_pre_aux0.5    $SRC/finetune.py
run 31_ft_pre_aux0      $SRC/finetune.py --aux 0
run 32_ft_scratch       $SRC/finetune.py --scratch

# fine-tuning, longer (10 epochs, lr 1e-3)
run 40_ft_pre_aux0.5_long  $SRC/finetune.py --epochs 10 --lr 1e-3
run 41_ft_pre_aux0_long    $SRC/finetune.py --epochs 10 --lr 1e-3 --aux 0
run 42_ft_scratch_long     $SRC/finetune.py --epochs 10 --lr 1e-3 --scratch

# fine-tuning, low data (1000 labeled pairs)
run 50_ft_pre_n1000        $SRC/finetune.py --epochs 10 --lr 1e-3 --n_train 1000
run 51_ft_scratch_n1000    $SRC/finetune.py --epochs 10 --lr 1e-3 --n_train 1000 --scratch

echo "all runs done, logs in $LOGS, metrics in results/metrics"