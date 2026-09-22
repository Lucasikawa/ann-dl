"""Geração dos dados dos dois exercícios (itens A).

As duas classes vêm de normais multivariadas 2D com os parâmetros do enunciado.
Os pontos ficam na ordem em que são gerados: primeiro os 1000 da classe 0,
depois os 1000 da classe 1. O treino visita as amostras nessa ordem, sem
embaralhar (o enunciado não pede embaralhamento; assim o `rng` só é usado para
gerar dados e sortear os pesos iniciais, e as verificações extras com ordem
embaralhada são feitas à parte).
"""
import numpy as np

N_PER_CLASS = 1000

# Exercício 1: médias distantes, variância 0.5 -> separável
EX1_MEANS = (np.array([1.5, 1.5]), np.array([5.0, 5.0]))
EX1_COV = np.array([[0.5, 0.0], [0.0, 0.5]])

# Exercício 2: médias próximas, variância 1.5 -> sobreposição forte
EX2_MEANS = (np.array([3.0, 3.0]), np.array([4.0, 4.0]))
EX2_COV = np.array([[1.5, 0.0], [0.0, 1.5]])


def generate_data(rng, means, cov, n=N_PER_CLASS):
    """n pontos por classe de N(means[k], cov). Retorna X (2n, 2) e y (2n,) em {0, 1}."""
    X = np.vstack([rng.multivariate_normal(m, cov, size=n) for m in means])
    y = np.repeat(np.arange(len(means)), n)
    return X, y


def describe(X, y):
    """Média e covariância amostrais por classe, para conferir com o enunciado."""
    out = {}
    for k in np.unique(y):
        Xk = X[y == k]
        out[str(k)] = {
            "n": int(len(Xk)),
            "mean": Xk.mean(axis=0).tolist(),
            "cov": np.cov(Xk, rowvar=False).tolist(),
        }
    return out
