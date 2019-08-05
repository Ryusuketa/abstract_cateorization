import pandas as pd
import numpy as np
from typing import Dict, List, Callable
import csv

import torch


def get_glove_vectors(path):
    df = pd.read_csv(path, sep=' ', header=None, engine='c',
                     quoting=csv.QUOTE_ALL, error_bad_lines=False)
    token2id = {t: i for i, t in enumerate(df.iloc[:, 0])}

    return df.iloc[:, 1:].values, token2id


def formatting_data(df: pd.DataFrame, tokenizer: Callable[[str], None], section2label: Dict[str, int]):
    # ToDo: completion of out-of-vocabulary
    gr = df.groupby('paper_id')
    documents = gr['sentense'].apply(lambda sentenses: [tokenizer.encode(x) for x in sentenses])
    labels = gr['section'].apply(lambda section_names: [section2label[name] for name in section_names]) 

    return documents, labels


def calculate_transition_matrix(df: pd.DataFrame, section2label: Dict[str, int]):
    labels_all_paper = df.groupby('paper_id')['section'].apply(lambda x: [section2label[s] for s in x]).tolist()
    n_labels = pd.Series(list(section2label.values())).nunique()
    str2labels = {str([l1, l2]): [l1, l2] for l1 in range(n_labels) for l2 in range(n_labels)}
    label_pairs = [str([labels[i-1], labels[i]]) for labels in labels_all_paper for i in range(1, len(labels))]

    label_pairs = pd.Series(label_pairs).value_counts().to_dict()
    transition = np.zeros([n_labels, n_labels])

    for label, value in label_pairs.items():
        l1, l2 = str2labels[label]
        transition[l1, l2] = value

    transition = np.log(transition / np.sum(transition))

    return transition.T


def data_to_LongTensor(documents: List[List[List[int]]], labels: List[List[int]]):
    documents = [[torch.LongTensor(token_list).cuda() for token_list in sentenses] for sentenses in documents]
    labels = [torch.LongTensor(label).cuda() for label in labels]

    return documents, labels
