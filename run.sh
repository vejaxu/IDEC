#!/bin/bash

dataset_name="non_spherical_3"  # 替换成实际的数据集名称
gamma_values=(0.01 0.02 0.05 0.1 0.2 0.5 1.0)

for gamma in "${gamma_values[@]}"
do
    echo "Running IDEC with dataset=${dataset_name}, gamma=${gamma}"
    python idec.py --dataset "$dataset_name" --gamma "$gamma"
done
