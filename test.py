import torch
from torch import nn
from attention import *
from utils import *

if __name__ == "__main__":

    model = torch.load('attention_model.pth', weights_only=False)
    test = "december 25th 2018"
    input_vocab = load_data("human_vocab.pkl")
    output_vocab = load_data("inv_machine_vocab.pkl")
    input_lang = torch.LongTensor([string_to_int(test, 20, input_vocab)])
    input_lang = nn.functional.one_hot(input_lang, len(input_vocab))
    padding_mask = get_padding_mask(input_lang)
    output, attention = model(input_lang, padding_mask)
    print(output.argmax(dim=2))

