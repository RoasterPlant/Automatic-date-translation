import torch
from torch import nn
from attention import *
from utils import *
from torch.utils.data import DataLoader

if __name__ == "__main__":

    # Hyperparameters

    epochs = 80
    learning_rate = 1e-3
    batch_size = 32
    gamma = 1.0
    lenght = 20
    hidden_size = 32

    # Model setup

    input_vocab = load_data("human_vocab.pkl")
    output_vocab = load_data("machine_vocab.pkl")
    filename = "dataset.pkl"
    input_lang, output_lang, dataset = load_dataset(filename, input_vocab, lenght, output_vocab)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = RNN(len(input_vocab), hidden_size, len(output_vocab)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=gamma)

    # Training

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(dataloader, model, loss_fn, optimizer)
        scheduler.step()

    print("Done!")
    torch.save(model, "attention_model.pth")
    print("Saved PyTorch Model State to attention_model.pth")