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

#要修改input 維度記得修改 => enc_in

python -u main.py \
  --method $method \
  --data $data_name \
  --data_path $data_file \
  --target close \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --e_layers 6 \
  --d_layers 3 \
  --features S \
  --seq_len 180 \
  --label_len 15 \
  --pred_len $pred_len \
  --root_path $root \
  --n_inner $n \
  --test_bsz $bsz \
  --train_epochs 60 \
  --patience 20 \
  --learning_rate 1e-4 \
  --online_learning full \
  --itr $i \
  --des 'exp'
