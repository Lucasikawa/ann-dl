"""Exercício 2 — Classificação binária que uma reta não resolve.

Classe 1 = dois clusters em lados opostos da Classe 0. Baseline linear
(LogisticRegression, permitida só aqui) contra o MLP de mlp.py.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlp import MLP, BinaryCrossEntropy, accuracy, standardize

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# hiperparâmetros (os mesmos do Exercício 3; só mudam a saída e a perda)
LAYERS = [2, 8, 1]
LR = 0.05
BATCH = 32
EPOCHS = 300

COLORS = ("tab:blue", "tab:red")
CLASS_NAMES = ("Classe 0", "Classe 1")
CENTERS = np.array([[0.0, 0.0], [3.0, 3.0], [-3.0, -3.0]])
STD = 1.2


def make_data():
    """Parâmetros do enunciado, sem alteração."""
    X, c = make_blobs(n_samples=[500, 250, 250], centers=CENTERS,
                      cluster_std=STD, random_state=42)
    y = (c > 0).astype(int)  # classe 0: 1 cluster | classe 1: 2 clusters
    return X, y


def best_line_accuracy(X, y, n_angles=720):
    """Melhor acurácia possível com UMA reta: varre direções w e limiares b (w·x >= b -> classe 1)."""
    best = 0.0
    for theta in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
        s = X @ np.array([np.cos(theta), np.sin(theta)])
        order = np.argsort(s)
        ys = y[order]
        # limiar entre s[k-1] e s[k]: classe 1 para os n-k maiores
        ones_right = np.concatenate([[ys.sum()], ys.sum() - np.cumsum(ys)])
        zeros_left = np.concatenate([[0], np.cumsum(1 - ys)])
        best = max(best, float(((ones_right + zeros_left) / len(y)).max()))
    return best


def bayes_accuracy(X, y):
    """Regra de Bayes com os parâmetros verdadeiros: o teto de acurácia para estes dados."""
    def density(c):
        return np.exp(-np.sum((X - c) ** 2, axis=1) / (2 * STD ** 2))
    p0 = 500 * density(CENTERS[0])
    p1 = 250 * density(CENTERS[1]) + 250 * density(CENTERS[2])
    return accuracy(y, (p1 > p0).astype(int))


def fig1_scatter(X, y):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for k in (0, 1):
        ax.scatter(X[y == k, 0], X[y == k, 1], s=12, alpha=0.6, c=COLORS[k],
                   label=f"{CLASS_NAMES[k]} (n = {np.sum(y == k)})")
    ax.set_title("Figura 1 — Dados do Exercício 2 (1000 pontos)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.legend(title="Classe", loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_data.png", dpi=150)
    plt.close(fig)


def fig2_loss(history):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(np.arange(1, len(history) + 1), history, c="tab:purple",
            label=f"MLP {'-'.join(map(str, LAYERS))} (BCE no treino)")
    ax.set_title("Figura 2 — Perda de treino por época (Exercício 2)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Entropia cruzada binária (treino)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_loss.png", dpi=150)
    plt.close(fig)


def fig3_boundary(net, scaler, X_test, y_test, baseline):
    mu, sd = scaler
    x1 = np.linspace(-7.5, 7.5, 400)
    x2 = np.linspace(-7.5, 7.5, 400)
    G1, G2 = np.meshgrid(x1, x2)
    grid = np.column_stack([G1.ravel(), G2.ravel()])
    P = net.predict_proba((grid - mu) / sd)[:, 0].reshape(G1.shape)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.contourf(G1, G2, P >= 0.5, levels=[-0.5, 0.5, 1.5], colors=["#cfe0f3", "#f6cfcf"])
    ax.contour(G1, G2, P, levels=[0.5], colors="k", linewidths=1.5)
    pred = net.predict((X_test - mu) / sd)
    for k in (0, 1):
        m = y_test == k
        ax.scatter(X_test[m, 0], X_test[m, 1], s=18, c=COLORS[k], edgecolor="k",
                   linewidth=0.3, label=f"{CLASS_NAMES[k]} (teste)")
    wrong = pred != y_test
    ax.scatter(X_test[wrong, 0], X_test[wrong, 1], s=70, facecolor="none", edgecolor="k",
               linewidth=1.2, label=f"erro do MLP ({wrong.sum()})")
    # reta da regressão logística: w·x + b = 0
    w, b = baseline.coef_[0], baseline.intercept_[0]
    ax.plot((-b - w[1] * x2) / w[0], x2, ls="--", c="dimgray", lw=1.5,
            label="fronteira da regressão logística")
    # legendas das regiões sombreadas e da fronteira
    handles, _ = ax.get_legend_handles_labels()
    handles += [Patch(color="#cfe0f3", label="região prevista: Classe 0"),
                Patch(color="#f6cfcf", label="região prevista: Classe 1"),
                Line2D([], [], c="k", lw=1.5, label="fronteira do MLP ($p = 0.5$)")]
    ax.set_xlim(x1[0], x1[-1])
    ax.set_ylim(x2[0], x2[-1])
    ax.set_title("Figura 3 — Fronteira de decisão do MLP sobre o teste (Exercício 2)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_boundary.png", dpi=150)
    plt.close(fig)


def run(rng):
    FIG_DIR.mkdir(exist_ok=True)

    # A — dados
    X, y = make_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    fig1_scatter(X, y)

    # B — baseline linear (nos dados crus; a regressão logística não precisa de escala)
    baseline = LogisticRegression().fit(X_tr, y_tr)
    base_acc = float(baseline.score(X_te, y_te))
    base_proba = baseline.predict_proba(X_te)[:, 1]

    # C — MLP (entradas padronizadas com média/desvio do treino)
    (X_tr_s, X_te_s), scaler = standardize(X_tr, X_te)
    net = MLP(LAYERS, BinaryCrossEntropy(), rng)
    history = net.fit(X_tr_s, y_tr, epochs=EPOCHS, lr=LR, batch_size=BATCH, rng=rng)
    train_acc = accuracy(y_tr, net.predict(X_tr_s))
    test_acc = accuracy(y_te, net.predict(X_te_s))
    fig2_loss(history)
    fig3_boundary(net, scaler, X_te, y_te, baseline)

    # D — referências geométricas para a análise
    best_line = best_line_accuracy(X_tr, y_tr)
    bayes = bayes_accuracy(X_te, y_te)
    # médias das classes no treino: o gradiente da regressão logística em w = 0 é proporcional
    # à diferença entre elas
    class_means = [X_tr[y_tr == k].mean(axis=0).tolist() for k in (0, 1)]
    # passo de Newton a partir de w = 0 (classes balanceadas): w ~ Sigma^-1 (m1 - m0)
    w_newton = np.linalg.solve(np.cov(X_tr.T), np.subtract(class_means[1], class_means[0]))
    cov_eig = np.linalg.eigvalsh(np.cov(X_tr.T))
    # onde a fronteira do MLP (p = 0.5) corta a diagonal (t, t) e a antidiagonal (t, -t)
    mu, sd = scaler
    t = np.linspace(-7, 7, 2801)

    def crossings(points):
        s = np.sign(net.predict_proba((points - mu) / sd)[:, 0] - 0.5)
        return t[np.where(s[:-1] != s[1:])[0]].round(2).tolist()
    diag_cross = crossings(np.column_stack([t, t]))
    anti_cross = crossings(np.column_stack([t, -t]))
    # lado da reta da regressão logística em que cai cada grupo do teste
    side = baseline.predict(X_te)
    upper_tr, upper_te = (y_tr == 1) & (X_tr.sum(axis=1) > 0), (y_te == 1) & (X_te.sum(axis=1) > 0)
    lr_sides = {
        "class1_upper_lower_train": [int(upper_tr.sum()), int(y_tr.sum() - upper_tr.sum())],
        "class1_upper_lower_test": [int(upper_te.sum()), int(y_te.sum() - upper_te.sum())],
        "class0_predicted_1": float(np.mean(side[y_te == 0] == 1)),
        "upper_cluster_predicted_1": float(np.mean(side[(y_te == 1) & (X_te.sum(axis=1) > 0)] == 1)),
        "lower_cluster_predicted_1": float(np.mean(side[(y_te == 1) & (X_te.sum(axis=1) < 0)] == 1)),
    }

    print(f"[Ex2] baseline linear: {base_acc:.4f} | MLP {LAYERS}: teste {test_acc:.4f} "
          f"(treino {train_acc:.4f}) | melhor reta (treino) {best_line:.4f} | Bayes (teste) {bayes:.4f}")
    return {
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "class_counts": np.bincount(y).tolist(),
        "baseline": {
            "test_accuracy": base_acc,
            "coef": baseline.coef_[0].tolist(),
            "intercept": float(baseline.intercept_[0]),
            "test_proba_min": float(base_proba.min()),
            "test_proba_max": float(base_proba.max()),
            "test_sides": lr_sides,
        },
        "class_means_train": class_means,
        "newton_w_from_zero": w_newton.tolist(),
        "train_cov_eigenvalues": cov_eig.tolist(),
        "mlp_boundary_crossings_diagonal": diag_cross,
        "mlp_boundary_crossings_antidiagonal": anti_cross,
        "mlp": {
            "layers": LAYERS, "n_params": net.n_params(), "hidden_activation": "tanh",
            "output": "sigmoid", "loss": "BCE", "lr": LR, "batch_size": BATCH, "epochs": EPOCHS,
            "loss_first_epoch": float(history[0]), "loss_last_epoch": float(history[-1]),
            "train_accuracy": train_acc, "test_accuracy": test_acc,
            "test_errors": int(np.sum(net.predict(X_te_s) != y_te)),
        },
        "best_single_line_train_accuracy": best_line,
        "bayes_test_accuracy": bayes,
    }
