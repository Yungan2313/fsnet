#!/bin/bash

# 參數設定
i=1
n=1
bsz=1
pred_len=30
method="fsnet"

# 自訂資料集
data_name="2454"
data_file="2454.csv"
root="./data/"

CUDA_VISIBLE_DEVICES=0 python -u main.py \
  --method $method \
  --data $data_name \
  --data_path $data_file \
  --features M \
  --seq_len 180 \
  --label_len 0 \
  --pred_len $pred_len \
  --root_path $root \
  --n_inner $n \
  --test_bsz $bsz \
  --train_epochs 6 \
  --learning_rate 1e-3 \
  --online_learning full \
  --itr $i \
  --des 'exp'
