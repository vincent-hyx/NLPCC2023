#!/bin/bash


python seq2tree.py \
                --pretrain_path=PLMs/bert-base-chinese \
                --batch_size=64 \
                --n_epochs=80 \
                --learning_rate=1e-3 \
                --dropout=0.5 \
                --weight_decay=1e-5 \
                --hidden_size=512 \
                --embedding_size=128 \
                --train_data_file=./data/train.json \
                --val_data_file=./data/test.json \
                --seed=42 \
                --output_dir=models  \
                --beam_size=5 > logs/seq2tree.log 2>&1






