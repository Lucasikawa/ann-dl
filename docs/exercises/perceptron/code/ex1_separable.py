"""Exercício 1 — Dados separáveis: o caso para o qual o perceptron foi feito."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

N_PER_CLASS = 1000
MEAN_0, MEAN_1 = np.array([1.5, 1.5]), np.array([5.0, 5.0])
COV = np.array([[0.5, 0.0], [0.0, 0.5]])
ETA = 0.01
ETA_ALT = 1.0
MAX_EPOCHS = 100

COLORS = ("tab:blue", "tab:red")
CLASS_NAMES = ("Classe 0", "Classe 1")


def make_data(rng, mean_0, mean_1, cov, n=N_PER_CLASS):
    """Duas gaussianas 2D, n pontos cada, embaralhadas UMA vez.

    Sem embaralhar, cada época veria 1000 pontos da classe 0 e depois 1000 da classe 1.
    A ordem embaralhada é sorteada aqui, uma vez só, e fica fixa em todas as épocas
    e em todas as execuções, para que só eta mude entre elas (item D).
    """
    X0 = rng.multivariate_normal(mean_0, cov, size=n)
    X1 = rng.multivariate_normal(mean_1, cov, size=n)
    X = np.vstack([X0, X1])
    y = np.repeat([0, 1], n)
    order = rng.permutation(len(y))
    return X[order], y[order]


def boundary_line(w, b, x1):
    """x2 da reta w1 x1 + w2 x2 + b = 0."""
    return -(w[0] * x1 + b) / w[1]


def unit(w):
    return w / np.linalg.norm(w)


def angle_deg(u, v):
    return float(np.degrees(np.arccos(np.clip(unit(u) @ unit(v), -1, 1))))


def novikoff_bound(X, y, n_angles=7200):
    """Limite de erros do teorema de convergência (Novikoff), com o viés embutido em (x, 1).

    Acha, por varredura de direções u, a reta de MAIOR margem geométrica gamma (metade do vão
    entre as projeções das duas classes) e o viés b* dessa reta. Com ||x|| <= R, o teorema
    garante no máximo (R^2 + 1)(1 + b*^2) / gamma^2 erros a partir de w = 0, b = 0.
    """
    best = (-np.inf, None, None)
    for theta in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
        u = np.array([np.cos(theta), np.sin(theta)])
        s = X @ u
        gap = s[y == 1].min() - s[y == 0].max()
        if gap > best[0]:
            best = (gap, theta, -(s[y == 1].min() + s[y == 0].max()) / 2)
    gap, theta, b_star = best
    gamma = gap / 2
    R = float(np.max(np.linalg.norm(X, axis=1)))
    return {"gamma": float(gamma), "b_star": float(b_star), "R": R,
            "direction": [float(np.cos(theta)), float(np.sin(theta))],
            "mistake_bound": float((R ** 2 + 1) * (1 + b_star ** 2) / gamma ** 2)}


def fig1_scatter(X, y, title, path, means):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for k in (0, 1):
        ax.scatter(X[y == k, 0], X[y == k, 1], s=8, alpha=0.5, c=COLORS[k],
                   label=f"{CLASS_NAMES[k]} (n = {np.sum(y == k)}, média {tuple(means[k])})")
    ax.set_title(title)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig2_boundary(X, y, model):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    pred = model.predict(X)
    for k in (0, 1):
        m = (y == k) & (pred == y)
        ax.scatter(X[m, 0], X[m, 1], s=8, alpha=0.5, c=COLORS[k], label=f"{CLASS_NAMES[k]} (acerto)")
    wrong = pred != y
    ax.scatter(X[wrong, 0], X[wrong, 1], s=60, marker="x", c="k", label=f"classificado errado ({wrong.sum()})")
    x1 = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 2)
    ax.plot(x1, boundary_line(model.w, model.b, x1), c="k", lw=2,
            label=r"fronteira $\mathbf{w}\cdot\mathbf{x} + b = 0$ ($\eta = 0.01$)")
    ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    ax.set_title("Figura 2 — Fronteira de decisão aprendida (Exercício 1)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_boundary.png", dpi=150)
    plt.close(fig)


def fig3_accuracy(runs):
    """(a) acurácia x época; (b) acurácia logo após cada atualização, dentro da época 1."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4))
    styles = {ETA: ("tab:purple", "o", "-"), ETA_ALT: ("tab:orange", "s", "--")}
    for eta, (model, hist) in runs.items():
        c, mk, ls = styles[eta]
        acc = np.concatenate([[model.initial_accuracy], hist["accuracy"]]) * 100
        a.plot(np.arange(len(acc)), acc, c=c, marker=mk, ls=ls, label=rf"$\eta = {eta}$")
        pos = np.concatenate([[0], hist["update_position"]])
        ua = np.concatenate([[model.initial_accuracy], hist["update_accuracy"]]) * 100
        b.step(pos, ua, where="post", c=c, ls=ls, label=rf"$\eta = {eta}$ ({len(hist['update_accuracy'])} atualizações)")
    a.set_xticks(range(0, 3))
    a.set_xlabel("Época (0 = pesos iniciais)")
    a.set_ylabel("Acurácia na base inteira (%)")
    a.set_title("(a) Acurácia ao fim de cada época")
    b.set_xlabel("Posição na época 1 (fração das 2000 amostras)")
    b.set_ylabel("Acurácia na base inteira (%)")
    b.set_title("(b) Acurácia logo após cada atualização")
    b.set_xlim(0, 1)
    for ax in (a, b):
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
    fig.suptitle("Figura 3 — Acurácia × época (Exercício 1)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_accuracy.png", dpi=150)
    plt.close(fig)


def fig_c1_eta(X, y, models):
    """Complementar: as fronteiras de eta = 0.01, eta = 1.0 e da partida do zero, na região do vão."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for k in (0, 1):
        ax.scatter(X[y == k, 0], X[y == k, 1], s=10, alpha=0.5, c=COLORS[k], label=CLASS_NAMES[k])
    x1 = np.linspace(0, 6.5, 2)
    for (label, model), (c, ls) in zip(models.items(), (("k", "-"), ("tab:orange", "--"), ("tab:green", ":"))):
        ax.plot(x1, boundary_line(model.w, model.b, x1), c=c, ls=ls, lw=2, label=label)
    ax.set_xlim(1.0, 5.5)
    ax.set_ylim(1.0, 5.5)
    ax.set_title("Figura C1 (complementar) — Fronteiras por $\\eta$ e partida")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_c1_eta.png", dpi=150)
    plt.close(fig)


def summarize(model, hist):
    return {
        "w": model.w.tolist(), "b": model.b,
        "direction": unit(model.w).tolist(), "norm_w": float(np.linalg.norm(model.w)),
        "distance_origin_to_boundary": float(abs(model.b) / np.linalg.norm(model.w)),
        "epochs": model.epochs, "converged": model.converged,
        "updates_per_epoch": hist["updates"].tolist(),
        "total_updates": int(hist["updates"].sum()),
        "initial_accuracy": model.initial_accuracy,
        "accuracy_per_epoch": hist["accuracy"].tolist(),
        "final_accuracy": float(hist["accuracy"][-1]),
        "last_update_position_epoch1": float(hist["update_position"][-1]) if len(hist["update_position"]) else None,
    }


def run(rng):
    FIG_DIR.mkdir(exist_ok=True)

    # A — dados
    X, y = make_data(rng, MEAN_0, MEAN_1, COV)
    fig1_scatter(X, y, "Figura 1 — Dados separáveis (Exercício 1)", FIG_DIR / "fig1_data.png",
                 (MEAN_0.tolist(), MEAN_1.tolist()))

    # B/C — perceptron com eta = 0.01 a partir de w ~ N(0, 0.01^2), b = 0
    w0 = rng.normal(0, 0.01, size=2)
    model = Perceptron(w0, ETA)
    hist = model.fit(X, y, max_epochs=MAX_EPOCHS)
    fig2_boundary(X, y, model)

    # D — mesma partida (w0, b = 0) e mesma ordem; só eta muda
    model_alt = Perceptron(w0, ETA_ALT)
    hist_alt = model_alt.fit(X, y, max_epochs=MAX_EPOCHS)
    fig3_accuracy({ETA: (model, hist), ETA_ALT: (model_alt, hist_alt)})

    # D — conferência numérica da partida do zero (a prova algébrica está no relatório)
    zero = {}
    for eta in (ETA, ETA_ALT):
        m = Perceptron(np.zeros(2), eta)
        h = m.fit(X, y, max_epochs=MAX_EPOCHS)
        zero[eta] = (m, h)
    (mz1, hz1), (mz2, hz2) = zero[ETA], zero[ETA_ALT]
    ratio_w = (mz2.w / mz1.w).tolist()
    ratio_b = mz2.b / mz1.b
    fig_c1_eta(X, y, {r"$\eta = 0.01$, partida $\mathbf{w}_0$": model,
                      r"$\eta = 1.0$, partida $\mathbf{w}_0$": model_alt,
                      r"partida $\mathbf{w} = \mathbf{0}$ (qualquer $\eta$)": mz1})

    s, s_alt = summarize(model, hist), summarize(model_alt, hist_alt)
    print(f"[Ex1] eta=0.01: {model.epochs} épocas, {s['total_updates']} atualizações, acc {s['final_accuracy']:.4f} | "
          f"eta=1.0: {model_alt.epochs} épocas, {s_alt['total_updates']} atualizações | "
          f"ângulo entre w: {angle_deg(model.w, model_alt.w):.2f}° | zero: razão w {ratio_w}, b {ratio_b:.4f}")
    return {
        "class_means_sample": [X[y == k].mean(axis=0).tolist() for k in (0, 1)],
        "w0": w0.tolist(), "w0_over_eta": (w0 / ETA).tolist(),
        "eta_0.01": s,
        "eta_1.0": s_alt,
        "angle_between_directions_deg": angle_deg(model.w, model_alt.w),
        "zero_start": {
            "eta_0.01": summarize(mz1, hz1), "eta_1.0": summarize(mz2, hz2),
            "w_ratio": ratio_w, "b_ratio": ratio_b,
            "same_predictions": bool(np.array_equal(mz1.predict(X), mz2.predict(X))),
            "angle_to_eta1_run_deg": angle_deg(mz1.w, model_alt.w),
        },
        "novikoff": novikoff_bound(X, y),
        "mean_norm_x": float(np.linalg.norm(X, axis=1).mean()),
    }
