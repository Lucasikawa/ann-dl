"""Perceptron de uma camada, escrito do zero em NumPy (Exercícios 1 e 2).

Rótulos em {0, 1}. Predição: y_hat = step(w·x + b), com step(z) = 1 se z >= 0.
Regra de atualização (amostra a amostra):
    w <- w + eta (y - y_hat) x,     b <- b + eta (y - y_hat)
O erro (y - y_hat) vale 0 num acerto (não atualiza) e +1/-1 nos dois tipos de erro.

A mesma classe serve aos dois exercícios, sem nenhuma alteração. Ela também guarda o
"bolso" (pocket): a melhor (w, b) já vista na base inteira. No Exercício 1 o bolso só
acompanha a convergência; no Exercício 2 é ele que importa.
"""
import numpy as np


def step(z):
    """Função degrau: 1 se z >= 0, 0 caso contrário."""
    return (np.asarray(z) >= 0).astype(int)


class Perceptron:
    def __init__(self, w_init, eta, b_init=0.0):
        """w_init vem de fora (rng.normal(0, 0.01, size=2)) para que duas execuções possam
        partir exatamente do mesmo ponto e diferir só em eta (Exercício 1D)."""
        self.w = np.array(w_init, dtype=float)  # cópia: não altera o vetor de quem chamou
        self.b = float(b_init)
        self.eta = float(eta)

    def net_input(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return step(self.net_input(X))

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    def fit(self, X, y, max_epochs=100):
        """Treina na ordem das linhas de X até uma época sem nenhuma atualização ou max_epochs.

        Devolve um histórico com, por época: acurácia na base inteira ao fim da época,
        nº de atualizações e acurácia do bolso; e, por atualização: posição (em épocas),
        acurácia logo após ela e os pesos (w, b) logo após ela.
        """
        n = len(y)
        best_acc = self.accuracy(X, y)          # o bolso começa com os pesos iniciais
        pocket = {"w": self.w.copy(), "b": self.b, "accuracy": best_acc, "epoch": 0}
        self.initial_accuracy = best_acc
        hist = {"accuracy": [], "updates": [], "pocket_accuracy": [],
                "update_position": [], "update_accuracy": [], "update_w": [], "update_b": []}
        converged = False

        for epoch in range(1, max_epochs + 1):
            n_updates = 0
            for i in range(n):
                y_hat = step(self.w @ X[i] + self.b)
                error = y[i] - y_hat                  # 0, +1 ou -1
                if error != 0:
                    self.w += self.eta * error * X[i]
                    self.b += self.eta * error
                    n_updates += 1
                    # bolso: a única coisa a mais no laço. Só uma atualização muda a acurácia.
                    acc = self.accuracy(X, y)
                    hist["update_position"].append(epoch - 1 + (i + 1) / n)
                    hist["update_accuracy"].append(acc)
                    hist["update_w"].append(self.w.copy())
                    hist["update_b"].append(self.b)
                    if acc > best_acc:
                        best_acc = acc
                        pocket = {"w": self.w.copy(), "b": self.b, "accuracy": acc, "epoch": epoch}
            hist["accuracy"].append(self.accuracy(X, y))
            hist["updates"].append(n_updates)
            hist["pocket_accuracy"].append(best_acc)
            if n_updates == 0:                        # uma passada inteira sem erro: parou
                converged = True
                break

        self.pocket = pocket
        self.epochs = epoch
        self.converged = converged
        return {k: np.array(v) for k, v in hist.items()}
