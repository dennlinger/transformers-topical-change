"""
Evals ensemble methods, based on pickled results from individual runs.

"""

import matplotlib.pyplot as plt
import numpy as np
import segeval
import pickle


def convert_to_masses(label):
    curr_len = 1
    masses = []
    for el in label:
        # If next section starts, increase id
        if el == 0:
            masses.append(curr_len)
            curr_len = 1
        else:
            curr_len += 1
    masses.append(curr_len)
    return tuple(masses)


def count_mistakes(label, preds):
    """

    :param labels: Ground truth label for each section
    :param preds: Predicted labels.
    :return: Number of differing predictions ("mistakes") across document
    """
    # Count the differing labels
    num_mistakes = sum(np.abs(label - majority_pred))
    return num_mistakes


def calculate_pk(preds, labels, name=""):
    res = []
    for pred, label in zip(preds, labels):
        res.append(segeval.pk(pred, label))

    res = np.array(res)
    print(f"P_k error rate of {name} is : {np.mean(res)*100:.2f}%")



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
    majority_preds = []
    for label, pred1, pred2, pred3, pred4 in zip(labels, preds1, preds2, preds3, preds4):
        # Simulate ensemble for now
        pred = np.stack([pred1, pred2, pred3, pred4])

        # Majority vote. NumPy by default rounds 0.5 to 0.
        # Offset by small amount to avoid. Note this should never happen,
        # unless we have an even number of models in the ensemble.
        majority_pred = np.round(np.average(pred, axis=0) + 0.001)
        majority_preds.append(majority_pred)

        num_mistakes = count_mistakes(label, majority_pred)
        print(num_mistakes, len(label))
        mistakes.append(num_mistakes)
        avg_lens += len(label)
    # convert so binary functions work
    mistakes = np.array(mistakes)
    x = list(range(10))
    y = []
    for i in x:
        y.append(sum(mistakes <= i)/len(mistakes))

    plt.plot(x, y, marker="o")
    plt.xticks(x)
    plt.yticks(np.arange(0, 1.01, 0.1))
    plt.xlim([0, 9])
    plt.ylim([0, 1])
    plt.xlabel("Number of allowed mistakes")
    plt.ylabel("Fraction of samples with up to k mistakes")
    plt.show()

    print(y)

    print(f"Average length: {avg_lens / len(labels):.2f} paragraphs")

    label_masses = [convert_to_masses(label) for label in labels]
    ensemble_masses = [convert_to_masses(pred) for pred in majority_preds]
    roberta_masses = [convert_to_masses(pred) for pred in preds4]
    bert_masses = [convert_to_masses(pred) for pred in preds1]

    calculate_pk(ensemble_masses, label_masses, "ensemble")
    calculate_pk(roberta_masses, label_masses, "roberta")
    calculate_pk(bert_masses, label_masses, "bert")
