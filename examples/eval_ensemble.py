"""
Evals ensemble methods, based on pickled results from individual runs.

"""

import matplotlib.pyplot as plt
import pickle
import numpy as np


def count_mistakes(label, preds):
    """

    :param labels: Ground truth label for each section
    :param preds: Predicted labels from different models
    :return: Number of differing predictions ("mistakes") across document
    """

    # Majority vote. NumPy by default rounds 0.5 to 0.
    # Offset by small amount to avoid. Note this should never happen,
    # unless we have an even number of models in the ensemble.
    majority_pred = np.round(np.average(preds, axis=0)+0.001)
    # Count the differing labels
    num_mistakes = sum(np.abs(label - majority_pred))
    return num_mistakes


if __name__ == "__main__":
    with open("labels_bert_og_consec_1.pkl", "rb") as f:
        labels = pickle.load(f)
    with open("preds_bert_og_consec_1.pkl", "rb") as f:
        preds1 = pickle.load(f)
    with open("preds_bert_og_consec_2.pkl", "rb") as f:
        preds2 = pickle.load(f)
    with open("preds_bert_og_consec_5.pkl", "rb") as f:
        preds3 = pickle.load(f)
    with open("preds_roberta_og_consec_1.pkl", "rb") as f:
        preds4 = pickle.load(f)
    avg_lens = 0
    mistakes = []
    for label, pred1, pred2, pred3, pred4 in zip(labels, preds1, preds2, preds3, preds4):
        # Simulate ensemble for now
        pred = np.stack([pred1, pred2, pred3, pred4])
        num_mistakes = count_mistakes(label, pred)
        print(num_mistakes, len(label))
        mistakes.append(num_mistakes)
        avg_lens += len(label)
    # convert so binary functions work
    mistakes = np.array(mistakes)
    x = list(range(10))
    y = []
    for i in x:
        y.append(sum(mistakes <= i)/len(mistakes))

    plt.plot(x, y)
    plt.show()

    print(y)

    print(f"Average length: {avg_lens / len(labels):.2f} paragraphs")
