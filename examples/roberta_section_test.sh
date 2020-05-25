#!/bin/bash

model='roberta'

seed=(0 42 7 16 1000000 2020)

# roberta runs
for i in {1..5}; do
    model_path='/data/daumiller/transformers/examples/roberta_section_'$i

    python3 run_glue.py --data_dir AGB_test/ --model_type $model \
     --model_name_or_path $model_path --task_name qqp \
     --output_dir $model_path --max_seq_length 512 --do_eval \
     --per_gpu_eval_batch_size 24 --seed ${seed[$i]}
done

