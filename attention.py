
import os
import torch
from torch import nn

activations = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}

# Camada intermediária entre o encoder e o decoder.
class AttentionLayer(nn.Module):
    def __init__(self, input_size, output_size, activation):
        super().__init__()
        linear1 = nn.Linear(input_size, input_size)
        linear2 = nn.Linear(input_size, output_size)
        self.stack = nn.Sequential(
            linear1,
            activations[activation](),
            linear2,
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        logits = self.stack(x)
        return logits

class EncoderRNN(nn.Module):
    # input_size = Número de componentes de x num estado de tempo. Nesse caso, é o tamanho da representação dos caracteres de entrada (37).
    # hidden_size = Número de componentes da saída num estado tempo.

    def __init__(self, input_size, hidden_size, dropout_p = 0.01):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        droped = self.dropout(input.to(torch.float32))
        output, hx = self.lstm(droped)
        return output

class DecoderRNN(nn.Module):
    # hidden_size = O dobro do número de componentes da saída do encoder num estado tempo.
    # output_size = Número de componentes da saída do decoder num estado tempo. Nesse caso, é o tamanho da representação dos caracteres de entrada (11).

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hs = hidden_size
        self.os = output_size
        self.T_y = 10 # Tamanho da data padrão.
        self.attention = AttentionLayer(hidden_size + output_size, 1, activation="relu")
        self.lstm = nn.LSTMCell(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, encoder_outputs):
        batch_size, T_x = encoder_outputs.size()[:2]
        hx = torch.zeros(batch_size, self.os)
        cx = torch.zeros(batch_size, self.os)
        output = []
        for _ in range(self.T_y):
            attention_input = torch.cat([hx.repeat(T_x, 1, 1).transpose(0, 1), encoder_outputs], dim=2)
            attention_output = self.attention(attention_input)
            context = torch.sum(torch.mul(encoder_outputs, attention_output), dim=1)
            hx, cx = self.lstm(context, (hx, cx))
            output.append(hx)
        output = self.softmax(torch.stack(output, dim=1))
        return output
    
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.encoder = EncoderRNN(input_size, hidden_size)
        self.decoder = DecoderRNN(2 * hidden_size, output_size)

    def forward(self, x):
        x = self.encoder(x)
        logits = self.decoder(x)
        return logits
    
def train(dataloader, model, loss_fn, optimizer):
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred.view(-1, pred.size(-1)), y.view(-1))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        #for name, param in model.encoder.lstm.named_parameters():
        #    print(f"Layer Parameter: {name}")
        #    print(f"Parameter values:  {param}")

    print(f"loss: {total_loss:>7f}")

def test(dataset, model, loss_fn):
    # Unnecessary in this situation but added for best practices
    model.eval()
    test_loss, correct = 0, 0
    size = len(dataset)

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for data in dataset:
            x, y = data
            
            # Compute prediction error
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
