import torch
from torch import nn
from attention import *
from utils import *

if __name__ == "__main__":

    model = torch.load('attention_model.pth', weights_only=False)
    test = "may five 2021"
    input_vocab = load_data("human_vocab.pkl")
    output_vocab = load_data("inv_machine_vocab.pkl")
    input_lang = torch.LongTensor([string_to_int(test, 20, input_vocab)])
    input_lang = nn.functional.one_hot(input_lang, len(input_vocab))
    output = model(input_lang).argmax(dim=2)
    print(output)

