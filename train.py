import pickle
import torch
from torch import nn
from attention import *

def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def load_vocab(human_vocab, inv_machine_vocab, machine_vocab):
    pass

def load_dataset(filename):
    data = load_data(filename)
    input_lang = Lang("input")
    output_lang = Lang("output")
    for d in data:
        input_lang.addWord(d[0])
        output_lang.addWord(d[0])
    return input_lang, output_lang, data

#rnn = nn.LSTMCell(10, 20)  # (input_size, hidden_size)
#input = torch.randn(2, 3, 10)  # (time_steps, batch, input_size)
#hx = torch.randn(3, 20)  # (batch, hidden_size)
#cx = torch.randn(3, 20)
#output = []
#for i in range(input.size()[0]):
#    hx, cx = rnn(input[i], (hx, cx))
#    output.append(hx)
#output = torch.stack(output, dim=0)
#print(input, "\n-----\n", output)

rnn = nn.LSTM(10, 10, bidirectional=True)
input = torch.randn(3, 10)
output, (hn, cn) = rnn(input)
softmax = nn.Softmax(dim=1)
output = softmax(output)
print(output)