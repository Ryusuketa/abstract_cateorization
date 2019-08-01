from typing import List, Callable
import torch
from torch import nn
from torch import optim
from tqdm import tqdm

import numpy as np
import pandas as pd
import gokart
from functools import partial

from model.classifier.classifier import SentenceClassifier
from model.data.utils import formatting_data, data_to_LongTensor
from model.data.make_data import MakeSentenceData, MakeTransitionMatrix, PrepareWordEmbedding
from model.classifier.model_conf import SentenceCategoryLabels, SentenceClassifierModelParameters, SentenceClassifierTrainParameters


def get_optimizer(optimizer_name: str):
    if optimizer_name == 'Adam':
        optimizer = optim.Adam
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop

    return optimizer


def train(model: nn.Module,
          optimizer: optim.Optimizer,
          documents: List[List[torch.LongTensor]],
          labels: List[torch.LongTensor],
          epoch: int,
          loss_function: Callable[[torch.FloatTensor], None],
          minibatch_size: int) -> None:

    for _ in range(epoch):
        randperm = np.random.permutation(len(documents))
        softmax = nn.CrossEntropyLoss()
        for i in tqdm(range(len(documents) // minibatch_size)):
            input_minibatch = [documents[j] for j in randperm[i * minibatch_size:(i + 1) * minibatch_size]]
            label_minibatch = [labels[j] for j in randperm[i * minibatch_size:(i + 1) * minibatch_size]]

            model.zero_grad()
            loss = 0
            ce = 0
            for input, label in zip(input_minibatch, label_minibatch):
                output = model(input)
                loss += loss_function(output, label)
                ce += softmax(output, label)

            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
            print('cost_func:', loss)
            print('cross entropy:', ce / minibatch_size)


class SentenceClassifierModelTrain(gokart.TaskOnKart):
    def requires(self):
        return dict(data=MakeSentenceData(),
                    transition=MakeTransitionMatrix(),
                    embedding=PrepareWordEmbedding())

    def output(self):
        return self.make_model_target('model/sentence_classifier.zip',
                                      save_function=torch.save,
                                      load_function=partial(torch.load, map_location='cpu'),
                                      use_unique_id=False)

    def run(self):
        torch.cuda.set_device('cuda:3')
        data = self.load_data_frame('data')
        transition_matrix = self.load('transition')
        dic_embedding = self.load('embedding')

        token2id = dic_embedding['token2id']
        embedding = torch.FloatTensor(dic_embedding['embedding'])
        section2label = SentenceCategoryLabels().section2label
        n_labels = pd.Series(list(section2label.values())).nunique()

        documents, labels = formatting_data(data, token2id, section2label)
        documents, labels = data_to_LongTensor(documents, labels)

        n_tokens = embedding.shape[0]
        embed_features = embedding.shape[1]

        transition_matrix = torch.FloatTensor(transition_matrix).cuda()
        model_params = SentenceClassifierModelParameters().param_kwargs
        model = SentenceClassifier(n_tokens=n_tokens,
                                   embed_features=embed_features,
                                   transition_matrix=transition_matrix,
                                   pretrain_embedding=embedding,
                                   n_labels=n_labels,
                                   **model_params)
        model.cuda()

        train_params = SentenceClassifierTrainParameters().param_kwargs
        optimizer = get_optimizer(train_params['optimizer'])
        optimizer = optimizer(model.parameters(), train_params['lr'])
        train(model=model,
              optimizer=optimizer,
              documents=documents,
              labels=labels,
              epoch=train_params['epoch'],
              loss_function=model.calc_log_loss,
              minibatch_size=train_params['minibatch_size'])

        self.dump(model)
