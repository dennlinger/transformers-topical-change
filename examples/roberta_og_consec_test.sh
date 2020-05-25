#!/bin/bash

# Need 0 to offset indices with naming convention of folders.
# naming starts at 1, seed array indexing starts at 0
seed=(0 42 12 321 1000000 2020)

model='roberta'

# roberta runs
for i in {1..5}; do
    model_path='/data/daumiller/transformers/examples/roberta_og_consec_'$i

    python3 run_glue.py --data_dir AGB_consec_test/ --model_type $model \
     --model_name_or_path $model_path --task_name qqp \
     --output_dir $model_path --max_seq_length 512 --do_eval \
     --per_gpu_eval_batch_size 24 --seed ${seed[$i]}
done

