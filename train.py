import pickle
import torch
from torch import nn
from attention import *
from string_to_int import string_to_int

def load_data(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

# Transforma dados numa representação one-hot.
def prepare_dataset(dataset, lenght, vocab):
    n = len(vocab)
    translation = [string_to_int(data, lenght, vocab) for data in dataset]
    representation = nn.functional.one_hot(torch.LongTensor(translation), n)
    return representation

def load_dataset(filename):
    data = load_data(filename)
    input_lang = Lang("input")
    output_lang = Lang("output")
    for d in data:
        input_lang.addWord(d[0])
        output_lang.addWord(d[0])
    return input_lang, output_lang, data

if __name__ == "__main__":

    # Hyperparameters

    epochs = 1
    learning_rate = 1e-4
    decay = 1e-3
    gamma = 0.9
    lenght = 20

    # Model setup

    vocab = load_data("human_vocab.pkl")
    file_name = "data.txt"
    generate_random_numbers_to_file(file_name, 10, 200, interval[0], interval[1])
    dataset = open("data.txt")
    
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    model = RNN().to(device)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=gamma)

    # Training

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(dataset, model, loss_fn, optimizer)
        scheduler.step()

    print("Done!")