# Efficient implementation equivalent to the following with bidirectional=False
import os
import torch
from torch import nn

activations = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
}

class AttentionLayer(nn.Module):
    def __init__(self, input_size, output_size, activation):
        super().__init__()
        linear = nn.Linear(input_size, output_size)
        if activation != "cubic":
            nn.init.kaiming_normal_(linear.weight, mode="fan_in", nonlinearity=activation)
        else:
            nn.init.xavier_uniform_(linear.weight, gain=0.5)
        self.stack = nn.Sequential(
            linear,
            activations[activation](),
            nn.Softmax()
        )

    def forward(self, x):
        logits = self.stack(x)
        return logits

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        self.pre_lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, device=device)
        self.params = dict(self.rnn.named_parameters())
        
