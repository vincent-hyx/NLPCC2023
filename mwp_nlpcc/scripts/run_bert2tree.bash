#!/bin/bash


python bert2tree.py \
                --pretrain_path=PLMs/bert-base-chinese \
                --batch_size=16 \
                --n_epochs=50 \
                --learning_rate=1e-4 \
                --dropout=0.5 \
                --schedule=linear \
                --hidden_size=768 \
                --embedding_size=128 \
                --train_data_file=./data/math23k_train_test/combined_train23k_processed_nodup.json \
                --val_data_file=./data/val1.json \
                --seed=42 \
                --results_dir=results/bert-base-chinese  \
                --output_dir=models  \
                --warmup_steps=3000 \
                --beam_size=3 > logs/bert-base-chinese.log 2>&1






