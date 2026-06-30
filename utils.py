import pickle
import torch
from torch import nn
from string_to_int import string_to_int
from torch.utils.data import TensorDataset

def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

# Transforma dados de entrada numa representação one-hot.
# Transforma dados de saída numa representação ordenada.

def load_dataset(filename, input_vocab, input_length, output_vocab, output_length=10):
    dataset = load_data(filename)

    input_lang = torch.LongTensor(
        [string_to_int(data[0], input_length, input_vocab) for data in dataset]
        )
    input_lang = nn.functional.one_hot(input_lang, len(input_vocab))

    output_lang = torch.LongTensor(
        [string_to_int(data[1], output_length, output_vocab) for data in dataset]
        )
    
    dataset = TensorDataset(input_lang, output_lang)
    return input_lang, output_lang, dataset

def get_padding_mask(X):
    pad_vector = nn.functional.one_hot(torch.zeros((1, ), dtype=torch.long), num_classes=X.size(2))
    padding_mask = (X == pad_vector).all(dim=-1)
    return padding_mask