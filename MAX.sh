#!/bin/bash

# 假设你的细胞系列表如下
cell_lines=('A549' 'GM12878' 'H1-hESC' 'HCT116' 'HeLa-S3' 'HepG2' 'K562')

# 获取细胞系的数量
len=${#cell_lines[@]}

# 使用嵌套循环来进行两两组合
for (( i=0; i<$len; i++ )); do
  for (( j=i+1; j<$len; j++ )); do
    python3 ./train.py \
    --transcription_factor MAX \
    --train "${cell_lines[$i]}" \
    --valid "${cell_lines[$j]}" \
    --gpu 3 \
    --name "${cell_lines[$i]}_${cell_lines[$j]}" >"../${cell_lines[$i]}_${cell_lines[$j]}.log"
  done
done