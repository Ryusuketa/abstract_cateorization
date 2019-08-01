from typing import List, Callable
import luigi
import gokart

import pandas as pd
from torch import nn
import torch

from model.train import SentenceClassifierModelTrain
from model.data.make_data import MakeSentenceData, PrepareWordEmbedding
from model.data.utils import formatting_data, data_to_LongTensor
from model.classifier.model_conf import SentenceCategoryLabels


def validation(model: nn.Module,
               documents: List[List[int]],
               dataset: List[List[List[str]]],
               evaluation: List[Callable[..., None]]) -> None:
    pass


class ModelValidation(gokart.TaskOnKart):
    def requires(self):
        return dict(model=SentenceClassifierModelTrain(),
                    validation_data=MakeSentenceData('localfile/pubmed-rct-master/PubMed_200k_RCT_numbers_replaced_with_at_sign/test.txt'),
                    embedding=PrepareWordEmbedding())

    def output(self):
        return self.make_target('validation_result.csv')

    def run(self):
        model = self.load('model')
        validation_data = self.load('validation_data')
        token2id = self.load('embedding')['token2id']
        section2label = SentenceCategoryLabels().section2label

        result = self._validation(model, validation_data, section2label, token2id)
        self.dump(result)

    def _validation(self, model, validation_data, section2label, token2id):
        paper_id = validation_data['paper_id'].unique()
        documents, labels = formatting_data(validation_data, token2id, section2label)
        documents, labels = data_to_LongTensor(documents, labels)

        predicted_labels = []

        for document in documents:
            p = list(model(document).data.cpu().numpy())
            predicted_labels.append(p)

        return pd.DataFrame(dict(paper_id=paper_id, ground_truth=labels, predicted=predicted_labels))
