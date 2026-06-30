import torch
from utils import * # Funções para a preparação do dataset

def testar(modelo, arquivo_teste, tamanho=30):
    human_vocab = load_data("human_vocab.pkl")
    machine_vocab = load_data("machine_vocab.pkl")
    inv_machine_vocab = load_data("inv_machine_vocab.pkl")

    dados = load_data(arquivo_teste)
    X, Y, _ = load_dataset(arquivo_teste, human_vocab, tamanho, machine_vocab)
    padding_mask = get_padding_mask(X)

    modelo.eval()

    with torch.no_grad():
        pred = modelo(X, padding_mask)[0].argmax(dim=2)

    # Calcula número de predições totalmente corretas.
    acertos = 0
    for i, ((entrada, esperado), y_pred) in enumerate(zip(dados, pred)):
        previsto = "".join(inv_machine_vocab[int(c)] for c in y_pred)
        ok = previsto == esperado
        acertos += ok
        print(f"{i+1:2d}) {entrada:30s} -> {previsto} {'✓' if ok else '✗'}")

    print(f"\nAcurácia: {acertos}/{len(dados)} = {100*acertos/len(dados):.2f}%")

if __name__ == "__main__":
    model = torch.load("attention_model.pth", weights_only=False)
    testar(model, "test_dataset.pkl")