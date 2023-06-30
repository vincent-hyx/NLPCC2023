#!/bin/bash
model_names=(mengzi-bert-base)
#lr=(5e-5 4e-5 3e-5 2e-5 1e-5)
lr=(8e-5)
#lr=(1e-4)


for (( e=0; e<${#model_names[@]}; e++  )) do
  model_name=${model_names[$e]}
  for (( e=0; e<${#lr[@]}; e++  )) do
      lr_item=${lr[$e]}
      python plm2tree.py \
                --pretrain_path=PLMs/${model_name} \
                --batch_size=16 \
                --n_epochs=50 \
                --learning_rate=${lr_item} \
                --dropout=0.5 \
                --schedule=linear \
                --hidden_size=768 \
                --embedding_size=128 \
                --train_data_file=./data/training.json \
                --val_data_file=./data/val1.json \
                --seed=42 \
                --output_dir=models/${model_name}_${lr_item}  \
                --results_dir=results/${model_name}_${lr_item}  \
                --warmup_steps=3000 \
                --beam_size=3 > logs/${model_name}_${lr_item}.log 2>&1
  done
done
