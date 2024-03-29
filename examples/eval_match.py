"""
Very rudimentary evaluation method for inputs that are sequential in nature.
"""

from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import RobertaTokenizer, RobertaForSequenceClassification, \
    BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers.data.processors.utils import InputExample
from tqdm import tqdm
import numpy as np
import pickle
import torch
import json
import sys
import os

TEST_FOLDER = "./og-test"

print(sys.argv[1:])
if len(sys.argv) < 3:
    print("Missing argument. Please provide model and model path")

MODEL_NAME = sys.argv[1]  # roberta or bert
MODEL_PATH = sys.argv[2]  # ./roberta_og_consec

if len(sys.argv) > 3:
    DEVICE = sys.argv[3]
else:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

if MODEL_NAME == "bert":
    MODEL = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    TOKENIZER = BertTokenizer.from_pretrained(MODEL_PATH)
elif MODEL_NAME == "roberta":
    MODEL = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    TOKENIZER = RobertaTokenizer.from_pretrained(MODEL_PATH)
MODEL.to(DEVICE)

SAME_SECTION_FLAG = "1"
DIFF_SECTION_FLAG = "0"

MAX_LENGTH = 512
MAX_BATCH_SIZE = 24


def generate_samples_per_file(file):
    with open(file) as f:
        data = json.load(f)

    # only use first-level headings
    data = data["level1_headings"]
    
    examples = []
    prev_text = data[0]["text"]
    prev_label = data[0]["section"]
    label_list = ["0", "1"]
    output_mode = "classification"
    if len(data) < 2:
        return None

    for i, paragraph in enumerate(data[1:]):
        guid = "%s-%s" % (file, i)
        if paragraph["section"] == prev_label:
            temp_label = SAME_SECTION_FLAG
        else:
            temp_label = DIFF_SECTION_FLAG

        examples.append(InputExample(guid=guid, text_a=prev_text, text_b=paragraph["text"], label=temp_label))

        prev_text = paragraph["text"]
        prev_label = paragraph["section"]

    features = convert_examples_to_features(
        examples,
        TOKENIZER,
        label_list=label_list,
        max_length=MAX_LENGTH,
        output_mode=output_mode,
        pad_on_left=False,
        pad_token=TOKENIZER.convert_tokens_to_ids([TOKENIZER.pad_token])[0],
        pad_token_segment_id=0,
    )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


if __name__ == "__main__":
    all_preds = []
    all_labels = []
    tp = 0
    fp = 0
    for file in tqdm(sorted(os.listdir(TEST_FOLDER))):
        file_dataset = generate_samples_per_file(os.path.join(TEST_FOLDER, file))

        if not file_dataset:
            continue
        eval_sampler = SequentialSampler(file_dataset)
        # Maximize GPU usage. Datasets per file vary in length though
        batch_size = min(MAX_BATCH_SIZE, len(file_dataset))
        eval_dataloader = DataLoader(file_dataset, sampler=eval_sampler, batch_size=batch_size)

        # Taken from run_glue's evaluate()
        preds = None
        out_label_ids = None
        for batch in eval_dataloader:
            MODEL.eval()
            batch = tuple(t.to(DEVICE) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                inputs["token_type_ids"] = (
                    batch[2] if MODEL_NAME in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = MODEL(**inputs)
                tmp_eval_loss, logits = outputs[:2]
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)
        exact_match = np.array_equal(preds, out_label_ids)
        if exact_match:
            tp += 1
        else:
            fp += 1
        all_preds.append(preds)
        all_labels.append(out_label_ids)

    print(f"Accuracy was {tp/(tp+fp)*100:.6f}%")
    with open(f"preds_{MODEL_PATH.strip('/.')}.pkl", "wb") as f:
        pickle.dump(all_preds, f)
    with open(f"labels_{MODEL_PATH.strip('/.')}.pkl", "wb") as f:
        pickle.dump(all_labels, f)


