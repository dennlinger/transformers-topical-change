#!/bin/bash

# seed 7
python3 run_glue.py --data_dir AGB_consec/ --model_type roberta --model_name_or_path roberta-base --task_name qqp --output_dir roberta_og_consec_2/ --max_seq_length 512 --do_train --do_eval --per_gpu_train_batch_size 24 --per_gpu_eval_batch_size 24 --gradient_accumulation_steps 1 --logging_steps 50000 --save_steps 50000 --num_train_epochs 2.0 --eval_all_checkpoints --seed 7

# seed 16
python3 run_glue.py --data_dir AGB_consec/ --model_type roberta --model_name_or_path roberta-base --task_name qqp --output_dir roberta_og_consec_3/ --max_seq_length 512 --do_train --do_eval --per_gpu_train_batch_size 24 --per_gpu_eval_batch_size 24 --gradient_accumulation_steps 1 --logging_steps 50000 --save_steps 50000 --num_train_epochs 2.0 --eval_all_checkpoints --seed 16

# seed 1000000
# python3 run_glue.py --data_dir AGB_consec/ --model_type bert --model_name_or_path bert-base-uncased --task_name qqp --output_dir bert_og_consec_4/ --max_seq_length 512 --do_train --do_eval --per_gpu_train_batch_size 24 --per_gpu_eval_batch_size 24 --gradient_accumulation_steps 1 --logging_steps 10000 --save_steps 10000 --num_train_epochs 2.0 --eval_all_checkpoints --seed 1000000

# seed 2020
# python3 run_glue.py --data_dir AGB_consec/ --model_type roberta --model_name_or_path roberta-base --task_name qqp --output_dir roberta_og_consec_5/ --max_seq_length 512 --do_train --do_eval --per_gpu_train_batch_size 24 --per_gpu_eval_batch_size 24 --gradient_accumulation_steps 1 --logging_steps 50000 --save_steps 50000 --num_train_epochs 2.0 --eval_all_checkpoints --seed 2020
