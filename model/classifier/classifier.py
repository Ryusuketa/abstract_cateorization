import csv
import pandas as pd
import numpy as np

from typing import List

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

import torch.nn.functional as F
from torch.nn.parameter import Parameter


class MultiHeadAttention(nn.Linear):
    def __init__(self, in_features, out_features, attention_hop, bias=True):
        super(MultiHeadAttention, self).__init__(in_features, out_features)
        self.context_matrix = Parameter(torch.Tensor(attention_hop, out_features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        hidden = torch.matmul(input, self.weight.transpose(0, 1)) + self.bias
        hidden = torch.matmul(hidden, self.context_matrix.transpose(0, 1))
        attention = self.softmax(hidden)
        sentence = torch.matmul(input.transpose(1, 2), attention)
        sentence = sentence.view(1, -1, np.prod(sentence.shape[1:]))
        print(sentence.shape)

        return sentence


class SentenceClassifier(nn.Module):
    def __init__(self,
                 n_tokens: int,
                 embed_features: int,
                 encoded_features: int,
                 attention_features: int,
                 attention_hop: int,
                 n_labels: int,
                 dropout_rate: float,
                 transition_matrix: torch.FloatTensor,
                 pretrain_embedding: torch.FloatTensor):
        super(SentenceClassifier, self).__init__()
        self.embedding = nn.Embedding(n_tokens, embed_features)
        self.token_lstm = nn.LSTM(embed_features, encoded_features, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.sentence_lstm = nn.LSTM(encoded_features, attention_hop, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.attention = MultiHeadAttention(encoded_features, attention_features, attention_hop)
        self.linear = nn.Linear(encoded_features * attention_hop, n_labels)
        self.loss = nn.CrossEntropyLoss()

        self.transition_matrix = transition_matrix
        self.n_labels = n_labels

    def initialize_embedding(self, pretrain_embeddings):
        self.embedding.weight.data.copy_(pretrain_embeddings)

    def _lstm_forward(self, lstm_layer, embedded):
        packed = pad_sequence(embedded, batch_first=True)
        output, _ = lstm_layer(packed)
        output = output[:, :, :output.shape[-1] // 2] + output[:, :, output.shape[-1] // 2:]

        return output

    def _calc_viterbi(self, prob_features):
        prob_sequence = [prob_features[0, :]]

        for n in range(1, prob_features.shape[1]):
            prob = prob_features[n, :] + self.transition_matrix + prob_sequence[n - 1]
            max_prob = torch.max(prob, dim=1, keepdim=True)
            prob_sequence += [max_prob]

        return torch.cat(prob_sequence, dim=1)

    def forward(self, tokens):
        embedded = [self.embedding(t) for t in tokens]
        encoded = self._lstm_forward(self.token_lstm, embedded)
        encoded = self.attention(encoded)
        encoded = self._lstm_forward(self.sentence_lstm, encoded)
        prob_features = self.linear(encoded)
        probabilities = self._calc_viterbi(prob_features)

        return probabilities 

    def predict(self, tokens):
        probabilities = self.forward(tokens)

        return F.softmax(probabilities, dim=1)
