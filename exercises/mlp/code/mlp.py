"""MLP escrito do zero em NumPy (Exercícios 2, 3 e 4; também confere o Exercício 1).

Convenções (as mesmas do enunciado):
  * camada l tem pesos W[l] de shape (n_saída, n_entrada) e viés b[l] de shape (n_saída,);
  * com um lote X de shape (N, n_entrada), a pré-ativação é U = X @ W.T + b, ou seja,
    cada linha é u = W x + b para uma amostra;
  * camadas ocultas usam tanh; a ativação da camada de saída vem junto com a perda
    (sigmoide + BCE, softmax + CE, tanh + MSE), porque o gradiente dL/du na saída
    depende do PAR ativação+perda.

A classe MLP NÃO sabe qual problema está resolvendo: tamanho das camadas e perda são
argumentos. forward, backward e step são o mesmo código para binário e multiclasse.
"""
import numpy as np

EPS = 1e-12  # evita log(0) nas entropias cruzadas


# ---------------------------------------------------------------------------
# Ativações
# ---------------------------------------------------------------------------
def tanh(u):
    return np.tanh(u)


def tanh_grad_from_output(h):
    """d tanh(u)/du escrita em função da saída h = tanh(u): 1 - h^2."""
    return 1.0 - h ** 2


def sigmoid(u):
    # forma estável: evita exp de números grandes positivos
    out = np.empty_like(u, dtype=float)
    pos = u >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-u[pos]))
    e = np.exp(u[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def softmax(U):
    # subtrair o máximo da linha não muda o resultado e evita overflow
    E = np.exp(U - U.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Perdas: cada uma traz a ativação de saída e dL/du (u = pré-ativação da saída)
# Todas são médias sobre as N amostras do lote, por isso o 1/N em grad_u.
# ---------------------------------------------------------------------------
class BinaryCrossEntropy:
    """Saída sigmoide (1 unidade) + entropia cruzada binária."""

    name = "BCE"

    def encode(self, y):
        return np.asarray(y, dtype=float).reshape(-1, 1)

    def output(self, U):
        return sigmoid(U)

    def value(self, P, Y):
        P = np.clip(P, EPS, 1 - EPS)
        return float(-np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P)))

    def grad_u(self, P, Y):
        # dL/dp = (p - y) / (p (1 - p)) e dp/du = p (1 - p)  ->  dL/du = p - y
        return (P - Y) / Y.shape[0]

    def predict(self, P):
        return (P[:, 0] >= 0.5).astype(int)


class CategoricalCrossEntropy:
    """Saída softmax (K unidades) + entropia cruzada categórica."""

    name = "CE"

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def encode(self, y):
        return np.eye(self.n_classes)[np.asarray(y, dtype=int)]  # one-hot

    def output(self, U):
        return softmax(U)

    def value(self, P, Y):
        return float(-np.mean(np.sum(Y * np.log(np.clip(P, EPS, 1.0)), axis=1)))

    def grad_u(self, P, Y):
        # derivação no relatório (Exercício 3B): dL/du_k = p_k - y_k
        return (P - Y) / Y.shape[0]

    def predict(self, P):
        return P.argmax(axis=1)


class MeanSquaredErrorTanh:
    """Saída tanh + erro quadrático médio. Usada só para conferir o Exercício 1."""

    name = "MSE"

    def encode(self, y):
        return np.asarray(y, dtype=float).reshape(-1, 1)

    def output(self, U):
        return np.tanh(U)

    def value(self, Yhat, Y):
        return float(np.mean(np.sum((Y - Yhat) ** 2, axis=1)))

    def grad_u(self, Yhat, Y):
        # dL/dŷ = -2 (y - ŷ) / N   e   dŷ/du = 1 - ŷ^2
        return -2.0 * (Y - Yhat) * (1.0 - Yhat ** 2) / Y.shape[0]

    def predict(self, Yhat):
        return Yhat[:, 0]


# ---------------------------------------------------------------------------
# A rede
# ---------------------------------------------------------------------------
class MLP:
    def __init__(self, layer_sizes, loss, rng):
        """layer_sizes = [n_entrada, n_oculta_1, ..., n_saída]; loss = objeto de perda acima."""
        self.sizes = list(layer_sizes)
        self.loss = loss
        self.W, self.b = [], []
        for fan_in, fan_out in zip(self.sizes[:-1], self.sizes[1:]):
            # Xavier/Glorot normal: mantém a variância das ativações tanh entre camadas
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self.W.append(rng.normal(0.0, std, size=(fan_out, fan_in)))
            self.b.append(np.zeros(fan_out))

    @classmethod
    def from_params(cls, W, b, loss):
        """Rede com pesos dados, sem sorteio (usada para conferir o Exercício 1)."""
        net = cls.__new__(cls)
        net.W = [np.array(w, dtype=float) for w in W]
        net.b = [np.array(v, dtype=float) for v in b]
        net.sizes = [net.W[0].shape[1]] + [w.shape[0] for w in net.W]
        net.loss = loss
        return net

    @property
    def n_layers(self):
        return len(self.W)

    def n_params(self):
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))

    # ---- forward ----------------------------------------------------------
    def forward(self, X):
        """Propaga X (N, n_entrada) e guarda as ativações de cada camada para o backward."""
        H = X
        self.cache = [H]  # cache[l] = entrada da camada l (cache[0] = X)
        for l in range(self.n_layers):
            U = H @ self.W[l].T + self.b[l]
            last = l == self.n_layers - 1
            H = self.loss.output(U) if last else tanh(U)
            self.cache.append(H)
        return H

    # ---- backward ---------------------------------------------------------
    def backward(self, Y):
        """Gradientes de L em relação a todos os W e b, usando o cache do último forward."""
        gW = [None] * self.n_layers
        gb = [None] * self.n_layers
        delta = self.loss.grad_u(self.cache[-1], Y)  # dL/du da camada de saída, (N, n_saída)
        for l in reversed(range(self.n_layers)):
            H_in = self.cache[l]                     # entrada da camada l, (N, n_entrada)
            gW[l] = delta.T @ H_in                   # dL/dW = soma_n delta_n h_n^T
            gb[l] = delta.sum(axis=0)                # dL/db = soma_n delta_n
            if l > 0:
                dH = delta @ self.W[l]               # dL/dh da camada anterior
                delta = dH * tanh_grad_from_output(H_in)  # dL/du = dL/dh * (1 - h^2)
        return gW, gb

    # ---- update -----------------------------------------------------------
    def step(self, gW, gb, lr):
        """Descida do gradiente: theta <- theta - lr * dL/dtheta."""
        for l in range(self.n_layers):
            self.W[l] -= lr * gW[l]
            self.b[l] -= lr * gb[l]

    # ---- laço de treino ---------------------------------------------------
    def fit(self, X, y, epochs, lr, batch_size, rng):
        """SGD em mini-lotes. Devolve a perda no treino inteiro ao fim de cada época."""
        Y = self.loss.encode(y)
        n = X.shape[0]
        history = []
        for _ in range(epochs):
            order = rng.permutation(n)  # embaralha a cada época
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                self.forward(X[idx])
                gW, gb = self.backward(Y[idx])
                self.step(gW, gb, lr)
            history.append(self.loss.value(self.forward(X), Y))
        return np.array(history)

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        return self.loss.predict(self.forward(X))


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def standardize(X_train, *others):
    """z-score com média e desvio do TREINO (sem vazamento do teste)."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    return [(X_train - mu) / sd] + [(X - mu) / sd for X in others], (mu, sd)


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
