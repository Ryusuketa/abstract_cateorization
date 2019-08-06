from typing import List, Callable
import luigi
import gokart

import pandas as pd
from torch import nn
import torch
import numpy as np
import itertools

from model.classifier.train import SentenceClassifierModelTrain
from model.data.make_data import MakeSentenceData, PrepareWordEmbedding
from model.data.utils import formatting_data, data_to_LongTensor
from model.classifier.model_conf import SentenceCategoryLabels, TokenLayerConfig
from model.classifier.train import get_tokenizer


def validation(model: nn.Module,
               documents: List[List[int]],
               dataset: List[List[List[str]]],
               evaluation: List[Callable[..., None]]) -> None:
    pass


class ModelValidation(gokart.TaskOnKart):
    def requires(self):
        return dict(model=SentenceClassifierModelTrain(),
                    validation_data=MakeSentenceData(pubmed_rct_path='localfile/pubmed-rct-master/PubMed_200k_RCT_numbers_replaced_with_at_sign/test.txt'),
                    embedding=PrepareWordEmbedding())

    def output(self):
        return self.make_target('validation_result.csv')

    def run(self):
        model = self.load('model')
        model.cuda()
        model.transition_matrix = model.transition_matrix.cuda()
        validation_data = self.load('validation_data')
        section2label = SentenceCategoryLabels().section2label
        token_layer_config = TokenLayerConfig()

        tokenizer = get_tokenizer(token_layer_config)
        df = self._validation(model, validation_data, section2label, tokenizer)
        print(np.mean([p == t for t, p in zip(df['ground_truth'],df['predicted'])]))
        self.dump(df)

    def _validation(self, model, validation_data, section2label, tokenizer):
        df = validation_data.copy()
        documents, labels = formatting_data(validation_data, tokenizer, section2label)
        documents, labels = data_to_LongTensor(documents, labels)

        predicted_labels = []

        for document in documents:
            p = model.predict(document).data.cpu().numpy()
            p = np.argmax(p, axis=1)
            predicted_labels.append(list(p))

        predicted_labels = itertools.chain.from_iterable(predicted_labels)

        labels = itertools.chain.from_iterable([list(l.cpu().numpy()) for l in labels])
        df['ground_truth'] = labels
        df['predicted'] = predicted_labels

        return df
