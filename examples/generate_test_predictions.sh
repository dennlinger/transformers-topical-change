#!/bin/bash

# Roberta Models
python3 eval_match.py roberta ./roberta_og_consec

# Bert Models
python3 eval_match.py bert ./bert_og_consec_1
python3 eval_match.py bert ./bert_og_consec_2
python3 eval_match.py bert ./bert_og_consec_5
