import csv
import pandas as pd
import numpy as np

from typing import List

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

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
        self.initialize_embedding(pretrain_embedding)
        self.token_lstm = nn.LSTM(embed_features, encoded_features, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.sentence_lstm = nn.LSTM(encoded_features * attention_hop, encoded_features, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.attention = MultiHeadAttention(encoded_features, attention_features, attention_hop)
        self.linear = nn.Linear(encoded_features, n_labels)
        self.loss = nn.CrossEntropyLoss()

        self.transition_matrix = transition_matrix
        self.n_labels = n_labels

    def initialize_embedding(self, pretrain_embedding):
        self.embedding.weight.data.copy_(pretrain_embedding)

    def _lstm_forward(self, lstm_layer, embedded):
        padded = pad_sequence(embedded, batch_first=True)
        output, _ = lstm_layer(padded)
        output = output[:, :, :output.shape[-1] // 2] + output[:, :, output.shape[-1] // 2:]

        return output

    @staticmethod
    def log_sum_exp(tensor):
        max_value = torch.max(tensor)
        return max_value + torch.log(torch.sum(torch.exp(tensor - max_value), dim=1, keepdim=True))

    def _forward_denom(self, prob_features):
        forward_sequence = prob_features[0:1, :]

        for n in range(1, prob_features.shape[0]):
            current_prob_expand = prob_features[n:n + 1, :].expand(self.n_labels, self.n_labels).t()
            forward_expand = forward_sequence.expand(self.n_labels, self.n_labels)
            prob = current_prob_expand + self.transition_matrix + forward_expand
            forward_sequence = self.log_sum_exp(prob).t()

        return self.log_sum_exp(forward_sequence).squeeze()

    def _sequence_score(self, prob_features, label_seq):
        score = prob_features[0, label_seq[0]]
        for n in range(len(1, label_seq)):
            score += prob_features[n, label_seq[n]] + self.transition_matrix[label_seq[n - 1], label_seq[n]]

        return score

    def _viterbi_decode(self, prob_features):
        prob_sequence = [prob_features[0:1, :]]
        for n in range(1, prob_features.shape[0]):
            prob = prob_features[n:n + 1, :] + self.transition_matrix + prob_sequence[n - 1]
            max_prob, _ = torch.max(prob, dim=1, keepdim=True)
            prob_sequence += [max_prob.t()]

        return torch.cat(prob_sequence, dim=0)

    def calc_log_loss(self, prob_features, label_seq):
        denom = self._forward_denom(prob_features)
        score = self._sequence_score(prob_features, label_seq)

        return -(score - denom)

    def forward(self, tokens):
        embedded = [self.embedding(t) for t in tokens]
        encoded = self._lstm_forward(self.token_lstm, embedded)
        encoded = self.attention(encoded)
        encoded = self._lstm_forward(self.sentence_lstm, encoded)
        prob_features = self.linear(torch.squeeze(encoded))

        return prob_features

    def predict(self, tokens):
        prob_features = self.forward(tokens)

        return self._viterbi_decode(prob_features)
