
import os
import torch
from torch import nn

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.n_words = 0

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

activations = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}

class AttentionLayer(nn.Module):
    def __init__(self, input_size, output_size, activation):
        super().__init__()
        linear = nn.Linear(input_size, output_size)
        self.stack = nn.Sequential(
            linear,
            activations[activation](),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        logits = self.stack(x)
        return logits

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, = self.lstm(embedded)
        return output

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hs = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = AttentionLayer(hidden_size, hidden_size, activation="relu")
        self.lstm = nn.LSTMCell(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_outputs):
        attention_output = self.attention(encoder_outputs)
        hx = torch.zeros(1, self.hs)
        cx = torch.zeros(1, self.hs)
        output = []
        for i in range(attention_output.size()[0]):
            hx, cx = self.lstm(attention_output[i], (hx, cx))
            output.append(hx)
        output = torch.stack(output, dim=0)
        print(input, "\n-----\n", output)