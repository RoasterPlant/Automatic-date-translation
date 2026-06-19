
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

# Camada intermediária entre o encoder e o decoder.
class AttentionLayer(nn.Module):
    def __init__(self, input_size, output_size, activation):
        super().__init__()
        linear = nn.Linear(input_size, output_size)
        self.stack = nn.Sequential(
            linear,
            activations[activation](),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        logits = self.stack(x)
        return logits

class EncoderRNN(nn.Module):
    # input_size = Número de componentes de x num estado de tempo. Nesse caso, é o tamanho da representação dos caracteres de entrada (38).
    # hidden_size = Número de componentes da saída num estado tempo. Ness 

    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        drop = self.dropout(input)
        output, = self.lstm(drop)
        return output

class DecoderRNN(nn.Module):
    # hidden_size = O dobro do número de componentes da saída do encoder num estado tempo.
    # hidden_size = Número de componentes da saída do decoder num estado tempo. Nesse caso, é o tamanho da representação dos caracteres de entrada (11).

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hs = hidden_size
        self.os = output_size
        self.T_y = 10 # Tamanho da data padrão.
        self.attention = AttentionLayer(hidden_size + output_size, 1, activation="relu")
        self.lstm = nn.LSTMCell(hidden_size, output_size, batch_first=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_outputs):
        T_x = encoder_outputs.size()[0]
        hx = torch.zeros(self.os)
        cx = torch.zeros(self.os)
        output = []
        for _ in range(self.T_y):
            attention_input = torch.cat([hx.repeat(T_x, 1), encoder_outputs], dim=1)
            attention_output = self.attention(attention_input)
            context = torch.sum(torch.mul(encoder_outputs, attention_output), dim=0)
            hx, cx = self.lstm(context, (hx, cx))
            output.append(hx)
        output = self.softmax(torch.stack(output, dim=0))
        return output
    
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer_stack = nn.Sequential(
            EncoderRNN(input_size, hidden_size),
            DecoderRNN(2 * hidden_size, output_size)
        )

    def forward(self, x):
        logits = self.layer_stack(x)
        return logits
    
def train(dataset, model, loss_fn, optimizer):
    model.train()
    for data in dataset:
        x, y = data
        
        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y) 

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    loss = loss.item()
    print(f"loss: {loss:>7f}")

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
