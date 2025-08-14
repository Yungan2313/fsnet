#!/bin/bash

# 參數設定
i=1
n=2
bsz=1
pred_len=10
method="fsnet"

# 自訂資料集
data_name="3045"
data_file="3045.csv"
root="./data/"

#要修改input 維度記得修改 => enc_in
#--tickers '1216,2303,2308,2317,2330,2357,2412,2603,2881,2882,2884,2886,2891,3711'
#--tickers '2330,2357,2376,2379,3017,3034,4919,6531,8299' 2454_like
#--pretrained './checkpoints/fsnet/checkpoint.pth'
# --loss_mode diff
#--diff_weight 2.0      =>目前這個效果不佳
#--diff_threshold 0.10 
#--date_from 2019-01-01 => 記得考慮有些資料的頭跟尾不是同一天
#--date_to 2018-12-31
#--pct_limit 0.10 => 記得把inverse關掉
#--limit_col -1(已經是default不需要特別去開)
# --aux_tickers '2330,2357,2376,2379,3017,3034,4919,6531,8299'
# --aux_lr_scale 0.2
# --aux_inner 1
# --aux_update_order 'before'  

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
  --train_epochs 200 \
  --patience 20 \
  --learning_rate 1e-4 \
  --online_learning full \
  --itr $i \
  --des 'Exp'\
  \
  --non_overlap_test \
  --pct_limit 0.10 \
  --test_stride 8 \
  # --aux_tickers '2330,2357,2376,2379,3017,3034,4919,6531,8299' \
  # --aux_lr_scale 0.2 \
  # --aux_inner 1 \
  # --aux_update_order 'before' \
  # --pretrained './checkpoints/fsnet_2454_pl5_olfull_optadam_tb1_2025_08_10_22_02_7205/checkpoint.pth' \
  # --date_from 2019-01-01 \
  