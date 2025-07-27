#!/bin/bash

# 參數設定
i=1
n=3
bsz=1
pred_len=15
method="fsnet"

# 自訂資料集
data_name="3045"
data_file="3045.csv"
root="./data/"

#要修改input 維度記得修改 => enc_in
#--tickers '1216,2303,2308,2317,2330,2357,2412,2603,2881,2882,2884,2886,2891,3711'
#--pretrained './checkpoints/fsnet_2454_pl15_olfull_optadam_tb1_2025_07_24_18_44_ee89/checkpoint.pth'

python -u main.py \
  --method $method \
  --data $data_name \
  --data_path $data_file \
  --target 'close' \
  --enc_in 15 \
  --dec_in 15 \
  --c_out 15 \
  --features M \
  --seq_len 60 \
  --label_len 0 \
  --pred_len $pred_len \
  --root_path $root \
  --n_inner $n \
  --test_bsz $bsz \
  --train_epochs 60 \
  --patience 20 \
  --learning_rate 1e-4 \
  --online_learning full \
  --itr $i \
  --des 'Exp'\
  --inverse \
  --loss_mode 'diff' \
  --pretrained './checkpoints/fsnet_2454_pl15_olfull_optadam_tb1_2025_07_26_22_19_f35f/checkpoint.pth'