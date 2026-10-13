"""Exercício 1 — conferência numérica da conta feita à mão.

A conta do Exercício 1 está no relatório, passo a passo. Este script NÃO a
substitui: ele só roda a mesma rede (2-2-1, tanh nas duas camadas, MSE) na classe
MLP de mlp.py e confere que forward, backward e update batem com os valores
escritos à mão. Também compara os gradientes com diferenças finitas, o que
valida o backward que os Exercícios 2-4 usam.
"""
import numpy as np

from mlp import MLP, MeanSquaredErrorTanh

# valores do enunciado
X = np.array([[0.5, -0.2]])
Y = np.array([[1.0]])
W1 = np.array([[0.3, -0.1], [0.2, 0.4]])
B1 = np.array([0.1, -0.2])
W2 = np.array([[0.5, -0.3]])
B2 = np.array([0.2])
ETA = 0.3

# valores obtidos À MÃO no relatório (6 casas)
HAND = {
    "z1": [0.270000, -0.180000],
    "h1": [0.263625, -0.178081],
    "u2": 0.385237,
    "y_hat": 0.367247,
    "loss": 0.400377,
    "dL_dW2": [-0.288624, 0.194968],
    "dL_db2": -1.094828,
    "dL_dW1": [[-0.254685, 0.101874], [0.159016, -0.063606]],
    "dL_db1": [-0.509370, 0.318032],
    "W2_new": [0.586587, -0.358490],
    "b2_new": 0.528448,
    "W1_new": [[0.376405, -0.130562], [0.152295, 0.419082]],
    "b1_new": [0.252811, -0.295410],
    "loss_after": 0.083417,
}


def finite_difference_max_error(net, h=1e-6):
    """Maior |grad analítico - grad numérico| sobre todos os parâmetros."""
    loss = net.loss
    net.forward(X)
    gW, gb = net.backward(Y)
    worst = 0.0
    for params, grads in ((net.W, gW), (net.b, gb)):
        for P, G in zip(params, grads):
            for idx in np.ndindex(P.shape):
                old = P[idx]
                P[idx] = old + h
                lp = loss.value(net.forward(X), Y)
                P[idx] = old - h
                lm = loss.value(net.forward(X), Y)
                P[idx] = old
                worst = max(worst, abs((lp - lm) / (2 * h) - G[idx]))
    return worst


def run():
    net = MLP.from_params([W1, W2], [B1, B2], MeanSquaredErrorTanh())

    # forward
    y_hat = net.forward(X)
    h1 = net.cache[1][0]
    z1 = W1 @ X[0] + B1
    u2 = float(np.arctanh(y_hat[0, 0]))
    loss = net.loss.value(y_hat, Y)

    # backward
    fd_error = finite_difference_max_error(net)
    net.forward(X)
    gW, gb = net.backward(Y)

    # update
    net.step(gW, gb, ETA)
    loss_after = net.loss.value(net.forward(X), Y)

    code = {
        "z1": z1.tolist(),
        "h1": h1.tolist(),
        "u2": u2,
        "y_hat": float(y_hat[0, 0]),
        "loss": loss,
        "dL_dW2": gW[1][0].tolist(),
        "dL_db2": float(gb[1][0]),
        "dL_dW1": gW[0].tolist(),
        "dL_db1": gb[0].tolist(),
        "W2_new": net.W[1][0].tolist(),
        "b2_new": float(net.b[1][0]),
        "W1_new": net.W[0].tolist(),
        "b1_new": net.b[0].tolist(),
        "loss_after": loss_after,
    }
    max_diff = max(
        float(np.max(np.abs(np.asarray(code[k], dtype=float) - np.asarray(HAND[k], dtype=float))))
        for k in HAND
    )
    ok = max_diff < 1e-6
    print(f"[Ex1] conta à mão x código: maior diferença = {max_diff:.2e} -> {'OK' if ok else 'DIVERGE'}")
    print(f"[Ex1] gradiente analítico x diferenças finitas: maior erro = {fd_error:.2e}")
    return {
        "code_values": code,
        "max_abs_diff_hand_vs_code": max_diff,
        "hand_matches_code": ok,
        "finite_difference_max_error": fd_error,
    }
