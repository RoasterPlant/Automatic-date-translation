import torch
from torch import nn
from utils import get_padding_mask

# Camada intermediária entre o encoder e o decoder.
class AttentionLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        linear1 = nn.Linear(input_size, input_size)
        linear2 = nn.Linear(input_size, output_size)
        self.stack = nn.Sequential(
            linear1,
            nn.Tanh(),
            linear2,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x,  padding_mask):
        logits = self.stack(x).squeeze(-1)
        logits = logits.masked_fill(padding_mask, float('-inf'))
        scores = self.softmax(logits)
        return scores

class EncoderRNN(nn.Module):
    # input_size = Número de componentes de x num estado de tempo. Nesse caso, é o tamanho da representação dos caracteres de entrada (37).
    # hidden_size = Número de componentes da saída num estado tempo.

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)

    def forward(self, input):
        output, hx = self.lstm(input.to(torch.float32))
        return output

class DecoderRNN(nn.Module):
    # enc_hidden_size = O dobro do número de componentes da saída do encoder num estado tempo.
    # dec_hidden_size = Tamanho da saída do pós-LSTM
    # output_size = Número de componentes da saída do decoder num estado tempo. Nesse caso, é o tamanho da representação dos caracteres de saída (11).

    def __init__(self, enc_hidden_size, dec_hidden_size, output_size):

        # IMPORTANTE: O decoder NÃO deve possuir aplicar softmax explícitamente na camada de saída, 
        # pois ele é implicitamente aplicado pelo nn.CrossEntropyLoss do Pytorch.

        super().__init__()
        self.es = enc_hidden_size
        self.ds = dec_hidden_size
        self.os = output_size
        self.T_y = 10 # Tamanho da data padrão.
        self.attention = AttentionLayer(enc_hidden_size + dec_hidden_size, 1)
        self.lstm = nn.LSTMCell(enc_hidden_size, dec_hidden_size)
        self.linear = nn.Linear(dec_hidden_size, output_size)

    def forward(self, encoder_outputs, padding_mask):
        batch_size, T_x = encoder_outputs.size()[:2]
        device = encoder_outputs.device

        hx = torch.zeros(batch_size, self.ds, device=device)
        cx = torch.zeros(batch_size, self.ds, device=device)
        prev = torch.zeros(batch_size, self.T_y, self.ds, device=device)

        attention_maps = []

        # Loop manual da LSTM pós-atenção.
        for i in range(self.T_y):
            attention_input = torch.cat(
                [hx.repeat(T_x, 1, 1).transpose(0, 1), encoder_outputs],
                dim=2
            )

            attention_output = self.attention(attention_input, padding_mask).unsqueeze(-1)

            attention_maps.append(attention_output.squeeze(-1))

            context = torch.sum(
                torch.mul(encoder_outputs, attention_output),
                dim=1
            )

            hx, cx = self.lstm(context, (hx, cx))
            prev[:, i, :] = hx

        output = self.linear(prev)

        attention_maps = torch.stack(attention_maps, dim=1)

        return output, attention_maps # Também retorna os pesos da atenção para visualização.
    
class RNN(nn.Module):
    def __init__(self, input_size, enc_hidden_size, dec_hidden_size, output_size):
        super().__init__()
        self.encoder = EncoderRNN(input_size, enc_hidden_size)
        self.decoder = DecoderRNN(2 * enc_hidden_size, dec_hidden_size, output_size)

    def forward(self, x, padding_mask):
        x = self.encoder(x)
        logits = self.decoder(x, padding_mask)
        return logits
    
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(dataloader):

        # Obtém a máscara de padding
        padding_mask = get_padding_mask(X)

        # Computa a previsão e a função de perda
        pred, attention = model(X, padding_mask)
        loss = loss_fn(pred.view(-1, pred.size(-1)), y.view(-1))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    print(f"loss: {total_loss:>7f}")
