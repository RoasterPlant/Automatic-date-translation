import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from utils import load_data, get_padding_mask
from string_to_int import string_to_int


def int_to_string(indices, inv_vocab):
    return "".join(inv_vocab[int(i)] for i in indices)


def plot_attention_map(model, input_vocabulary, inv_output_vocabulary, text):

    Ty = 10
    Tx = 30

    model.eval()

    # Prepara o exemplo
    encoded = torch.LongTensor([
        string_to_int(text, Tx, input_vocabulary)
    ])

    encoded = nn.functional.one_hot(
        encoded,
        num_classes=len(input_vocabulary)
    ).float()

    padding_mask = get_padding_mask(encoded)

    device = next(model.parameters()).device
    encoded = encoded.to(device)

    # Obtém os pesos de atenção
    with torch.no_grad():
        prediction, attention_map = model(encoded, padding_mask)

    predicted_ids = prediction.argmax(dim=2)[0].cpu()
    predicted_text = int_to_string(predicted_ids, inv_output_vocabulary)

    attention_map = attention_map[0].cpu().numpy()

    row_max = attention_map.max(axis=1)
    attention_map = attention_map / row_max[:, None]

    text_ = list(text)

    input_length = len(text)
    output_length = Ty

    # Plota o mapa de atenção.
    plt.clf()
    f = plt.figure(figsize=(8, 8.5))
    ax = f.add_subplot(1, 1, 1)

    i = ax.imshow(attention_map, interpolation="nearest", cmap="Blues")

    cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
    cbar = f.colorbar(i, cax=cbaxes, orientation="horizontal")
    cbar.ax.set_xlabel('Alpha value (Probability output of the "softmax")', labelpad=2)

    ax.set_yticks(range(output_length))
    ax.set_yticklabels(predicted_text[:output_length])

    ax.set_xticks(range(input_length))
    ax.set_xticklabels(text_[:input_length], rotation=45)

    ax.set_xlabel("Input Sequence")
    ax.set_ylabel("Output Sequence")

    ax.grid()

    return attention_map

if __name__ == "__main__":
    model = torch.load("attention_model.pth", weights_only=False)

    human_vocab = load_data("human_vocab.pkl")
    inv_machine_vocab = load_data("inv_machine_vocab.pkl")

    plot_attention_map(
        model,
        human_vocab,
        inv_machine_vocab,
        "tuesday december 8 1998" # Exemplo de teste
    )

    plt.show()