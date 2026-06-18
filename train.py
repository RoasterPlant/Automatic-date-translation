import pickle
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