"""Perceptron de camada única, escrito do zero (Exercício 1, item B).

A MESMA classe é reutilizada, sem alteração, no Exercício 2. A única coisa que
o Exercício 2 acrescenta ao laço de treino é a cópia dos melhores pesos vistos
até o momento (algoritmo "pocket"), ligada pelo parâmetro `pocket=True` de
`fit`. Com `pocket=False` (Exercício 1) o laço é exatamente o mesmo, só sem a
cópia.

Convenção do enunciado: rótulos y e previsões y_hat em {0, 1}.

    y_hat = step(w . x + b),  step(z) = 1 se z >= 0, senão 0
    w <- w + eta * (y - y_hat) * x
    b <- b + eta * (y - y_hat)

O erro (y - y_hat) vale 0 num acerto (nenhuma atualização), +1 num falso
negativo (y = 1, y_hat = 0) e -1 num falso positivo (y = 0, y_hat = 1).

Só NumPy é usado: nenhum modelo de biblioteca externa.
"""
import numpy as np


class Perceptron:
    """Perceptron com ativação degrau e regra de atualização dirigida pelo erro.

    Parâmetros
    ----------
    eta : float
        Taxa de aprendizado (enunciado: 0.01; o item D do Exercício 1 repete
        com 1.0).
    max_epochs : int
        Limite de épocas (enunciado: 100).

    Atributos preenchidos por `fit`
    -------------------------------
    w, b                   pesos e viés finais (os que o laço tinha na última época)
    w0, b0                 pesos e viés iniciais
    n_epochs               épocas executadas
    converged              True se uma passada completa não produziu atualização
    epoch_accuracy         acurácia no dataset inteiro ao fim de cada época
    epoch_updates          número de atualizações (erros) em cada época
    epoch_weights          (w, b) ao fim de cada época
    pocket_w, pocket_b     melhores pesos vistos (só com pocket=True)
    pocket_acc             acurácia dos pesos do pocket
    pocket_epoch           época em que o pocket foi copiado pela última vez
    pocket_update          número da atualização, dentro dessa época, em que isso ocorreu
    epoch_pocket_accuracy  melhor acurácia vista até o fim de cada época
    update_log             (época, índice da amostra, w1, w2, b, acurácia) após CADA
                           atualização (só com pocket=True; serve para as figuras)
    """

    def __init__(self, eta=0.01, max_epochs=100):
        self.eta = float(eta)
        self.max_epochs = int(max_epochs)
        self.w = None
        self.b = 0.0

    # ------------------------------------------------------------ modelo
    @staticmethod
    def step(z):
        """Função degrau: 1 se z >= 0, senão 0. Funciona para escalar ou array."""
        return (np.asarray(z) >= 0).astype(int)

    def net_input(self, X):
        """z = w . x + b para cada linha de X."""
        return X @ self.w + self.b

    def predict(self, X):
        """y_hat = step(w . x + b), em {0, 1}."""
        return self.step(self.net_input(X))

    def accuracy(self, X, y):
        """Fração de acertos no conjunto (X, y)."""
        return float(np.mean(self.predict(X) == y))

    # ------------------------------------------------------------ treino
    def fit(self, X, y, rng=None, w0=None, b0=0.0, pocket=False):
        """Treina por, no máximo, `max_epochs` épocas, ou até uma passada sem erro.

        Inicialização (item B): w0 = rng.normal(0, 0.01, size=d) e b0 = 0. Passar
        `w0` explicitamente permite repetir o treino com OUTRA taxa de aprendizado
        sem mudar mais nada (item D.2) ou testar o início em zero (item D.3).

        As amostras são visitadas na ordem em que estão em X, sem embaralhar,
        e a acurácia no dataset inteiro é registrada ao fim de cada época.

        Com `pocket=True`, depois de cada atualização a acurácia no dataset
        inteiro é medida e, se for maior que qualquer uma vista antes, (w, b) é
        copiado para o pocket. Essa cópia é a ÚNICA diferença em relação ao laço
        do Exercício 1; ela não altera a trajetória de w e b.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n, d = X.shape

        if w0 is None:
            w0 = rng.normal(0.0, 0.01, size=d)  # inicialização NÃO nula (item B)
        self.w0 = np.array(w0, dtype=float).copy()
        self.b0 = float(b0)
        self.w = self.w0.copy()
        self.b = self.b0

        self.epoch_accuracy = []
        self.epoch_updates = []
        self.epoch_weights = []
        self.epoch_pocket_accuracy = []
        self.update_log = []
        self.converged = False

        # o pocket começa com os pesos iniciais, que são "os melhores vistos até aqui"
        self.pocket_w, self.pocket_b = self.w.copy(), self.b
        self.pocket_acc = self.accuracy(X, y) if pocket else None
        self.pocket_epoch, self.pocket_update = 0, 0

        for epoch in range(1, self.max_epochs + 1):
            n_updates = 0
            for i in range(n):
                y_hat = 1 if self.w @ X[i] + self.b >= 0 else 0  # step(w . x + b)
                error = y[i] - y_hat                              # 0, +1 ou -1
                if error == 0:
                    continue                                      # acerto: nada muda
                self.w = self.w + self.eta * error * X[i]         # w <- w + eta (y - y_hat) x
                self.b = self.b + self.eta * error                # b <- b + eta (y - y_hat)
                n_updates += 1
                if pocket:  # algoritmo pocket: guarda a melhor (w, b) vista até agora
                    acc = self.accuracy(X, y)
                    self.update_log.append((epoch, i, *self.w, self.b, acc))
                    if acc > self.pocket_acc:
                        self.pocket_w, self.pocket_b, self.pocket_acc = self.w.copy(), self.b, acc
                        self.pocket_epoch, self.pocket_update = epoch, n_updates

            self.epoch_accuracy.append(self.accuracy(X, y))
            self.epoch_updates.append(n_updates)
            self.epoch_weights.append((self.w.copy(), self.b))
            if pocket:
                self.epoch_pocket_accuracy.append(self.pocket_acc)
            if n_updates == 0:          # passada completa sem erro: convergiu
                self.converged = True
                break

        self.n_epochs = len(self.epoch_accuracy)
        return self

    # ------------------------------------------------------------ utilidades
    def unit_direction(self):
        """w / ||w||: a direção do vetor de pesos (normal à fronteira)."""
        return self.w / np.linalg.norm(self.w)


def boundary_offset(w, b):
    """Distância com sinal da origem até a reta w . x + b = 0, medida ao longo de w.

    Um ponto x está do lado positivo (y_hat = 1) quando w . x + b >= 0; a reta
    cruza o eixo definido por w em t = -b / ||w||.
    """
    return float(-b / np.linalg.norm(w))
