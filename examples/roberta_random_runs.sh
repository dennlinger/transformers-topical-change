#!/bin/bash

# seed 42
python3 run_glue.py --data_dir AGB_og_random/ --model_type roberta --model_name_or_path roberta-base --task_name qqp --output_dir roberta_og_random_1/ --max_seq_length 512 --do_train --do_eval --per_gpu_train_batch_size 24 --per_gpu_eval_batch_size 24 --gradient_accumulation_steps 1 --logging_steps 50000 --save_steps 50000 --max_steps 75000 --eval_all_checkpoints

# seed 7
python3 run_glue.py --data_dir AGB_og_random/ --model_type roberta --model_name_or_path roberta-base --task_name qqp --output_dir roberta_og_random_2/ --max_seq_length 512 --do_train --do_eval --per_gpu_train_batch_size 24 --per_gpu_eval_batch_size 24 --gradient_accumulation_steps 1 --logging_steps 50000 --save_steps 50000 --max_steps 75000 --eval_all_checkpoints --seed 7

# seed 16
python3 run_glue.py --data_dir AGB_og_random/ --model_type roberta --model_name_or_path roberta-base --task_name qqp --output_dir roberta_og_random_3/ --max_seq_length 512 --do_train --do_eval --per_gpu_train_batch_size 24 --per_gpu_eval_batch_size 24 --gradient_accumulation_steps 1 --logging_steps 50000 --save_steps 50000 --max_steps 75000 --eval_all_checkpoints --seed 16

# seed 1000000
python3 run_glue.py --data_dir AGB_og_random/ --model_type roberta --model_name_or_path roberta-base --task_name qqp --output_dir roberta_og_random_4/ --max_seq_length 512 --do_train --do_eval --per_gpu_train_batch_size 24 --per_gpu_eval_batch_size 24 --gradient_accumulation_steps 1 --logging_steps 50000 --save_steps 50000 --max_steps 75000 --eval_all_checkpoints --seed 1000000

# seed 2020
python3 run_glue.py --data_dir AGB_og_random/ --model_type roberta --model_name_or_path roberta-base --task_name qqp --output_dir roberta_og_random_5/ --max_seq_length 512 --do_train --do_eval --per_gpu_train_batch_size 24 --per_gpu_eval_batch_size 24 --gradient_accumulation_steps 1 --logging_steps 50000 --save_steps 50000 --max_steps 75000 --eval_all_checkpoints --seed 2020

