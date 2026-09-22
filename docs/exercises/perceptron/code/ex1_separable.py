"""Exercício 1 - Dados separáveis: o caso para o qual o perceptron foi projetado.

A: gera as duas nuvens e a Figura 1.
B/C: treina o Perceptron (eta = 0.01, até 100 épocas), Figuras 2 e 3.
D: margem máxima e cota de Novikoff (D.1), repete com eta = 1.0 e o mesmo w0
   (D.2) e confere numericamente o argumento do início em zero (D.3). Figura
   complementar com as três retas. No fim, uma verificação extra com a ordem
   das amostras embaralhada.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import EX1_COV, EX1_MEANS, describe, generate_data
from perceptron import Perceptron, boundary_offset
from plots import (draw_boundary, limits, line_label, mark_misclassified, save,
                   scatter_classes, shade_regions)

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
ETA = 0.01
MAX_EPOCHS = 100


def summarize(p, X, y):
    """Números que o relatório cita sobre um treino: pesos, épocas, acurácia, geometria."""
    w, b = p.w, p.b
    norm_w = float(np.linalg.norm(w))
    dist = np.abs(X @ w + b) / norm_w        # distância de cada ponto à reta
    return {
        "eta": p.eta,
        "w0": p.w0.tolist(),
        "b0": p.b0,
        "w": w.tolist(),
        "b": float(b),
        "norm_w": norm_w,
        "direction": p.unit_direction().tolist(),           # w / ||w||
        "angle_deg": float(np.degrees(np.arctan2(w[1], w[0]))),
        "offset": boundary_offset(w, b),                     # -b / ||w||
        "slope": float(-w[0] / w[1]),                        # x2 = slope * x1 + intercept
        "intercept": float(-b / w[1]),
        "n_epochs": p.n_epochs,
        "converged": p.converged,
        "epoch_updates": p.epoch_updates,
        "total_updates": int(sum(p.epoch_updates)),
        "epoch_accuracy": p.epoch_accuracy,
        "final_accuracy": p.epoch_accuracy[-1],
        "n_misclassified": int(np.sum(p.predict(X) != y)),
        "margin": float(dist.min()),                         # ponto mais próximo da reta
    }


def angle_between(u, v):
    """Ângulo em graus entre dois vetores."""
    c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def max_margin(X, y, n_angles=3600):
    """Maior margem possível entre as duas classes, por varredura de direções.

    Para a normal unitária w(theta), a melhor reta é a que passa no meio do vão
    entre max(w . x | classe 0) e min(w . x | classe 1); sua margem é metade do
    vão. O máximo sobre theta é a margem da reta de margem máxima (o que um SVM
    linear acharia), calculado só com NumPy. Serve para a cota de Novikoff.
    """
    theta = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    W = np.stack([np.cos(theta), np.sin(theta)])          # (2, n_angles)
    p0, p1 = X[y == 0] @ W, X[y == 1] @ W
    gap = np.maximum(p1.min(axis=0) - p0.max(axis=0), p0.min(axis=0) - p1.max(axis=0))
    k = int(np.argmax(gap))
    return {"gamma": float(gap[k] / 2), "direction": W[:, k].tolist(),
            "angle_deg": float(np.degrees(theta[k])), "separable": bool(gap[k] > 0)}


# ----------------------------------------------------------------- figuras

def plot_figure1(X, y, path):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    scatter_classes(ax, X, y)
    ax.set_title("Figura 1 — Exercício 1: duas classes separáveis (1000 pontos por classe)")
    ax.set_aspect("equal")
    ax.legend()
    save(fig, path)


def plot_figure2(X, y, p, path):
    xlim, ylim = limits(X)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    shade_regions(ax, p.w, p.b, xlim, ylim)
    scatter_classes(ax, X, y)
    draw_boundary(ax, p.w, p.b, xlim, ylim, label=line_label("Fronteira", p.w, p.b))
    mark_misclassified(ax, X, y, p.predict(X))
    ax.set_title(f"Figura 2 — Fronteira do perceptron (η = {p.eta}, {p.n_epochs} épocas, "
                 f"acurácia = {100 * p.epoch_accuracy[-1]:.1f}%)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    save(fig, path)


def plot_figure3(p, path):
    epochs = np.arange(1, p.n_epochs + 1)
    acc = 100 * np.array(p.epoch_accuracy)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(epochs, acc, "o-", lw=2, color="tab:blue",
            label="Acurácia no dataset inteiro (ao fim da época)")
    ax.annotate(f"{acc[-1]:.1f}%", (epochs[-1], acc[-1]), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("época")
    ax.set_ylabel("acurácia (%)")
    ax.set_xticks(epochs)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.bar(epochs, p.epoch_updates, width=0.35, alpha=0.3, color="tab:red",
            label="Atualizações (erros) na época")
    ax2.set_ylabel("atualizações na época")
    ax2.set_ylim(0, max(p.epoch_updates) * 2.2)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right")
    ax.set_title(f"Figura 3 — Acurácia por época (η = {p.eta}); parou em {p.n_epochs} épocas")
    save(fig, path)


def plot_figure_eta(X, y, runs, path):
    """Figura complementar: as fronteiras de eta = 0.01, eta = 1.0 e do início em zero."""
    xlim, ylim = limits(X)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    scatter_classes(ax, X, y, alpha=0.35)
    styles = (("black", "-"), ("tab:green", "--"), ("tab:purple", ":"))
    for (name, p), (color, ls) in zip(runs, styles):
        draw_boundary(ax, p.w, p.b, xlim, ylim, color=color, ls=ls,
                      label=f"{name}: {p.n_epochs} épocas, -b/||w|| = {boundary_offset(p.w, p.b):.2f}")
    ax.set_title("Figura complementar (Ex. 1, item D) — Fronteiras finais para diferentes η e w0")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, path)


# ------------------------------------------------------------------ roteiro

def run(rng):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # A - dados
    X, y = generate_data(rng, EX1_MEANS, EX1_COV)
    plot_figure1(X, y, FIG_DIR / "fig1_data.png")
    norms = np.linalg.norm(X, axis=1)
    res_a = {"n_total": int(len(X)), "classes": describe(X, y),
             "norm_x_mean": float(norms.mean()), "norm_x_max": float(norms.max()),
             "norm_x_mean_class0": float(norms[y == 0].mean()),
             "norm_x_mean_class1": float(norms[y == 1].mean())}

    # B/C - treino com eta = 0.01, w0 ~ N(0, 0.01), b0 = 0
    p = Perceptron(eta=ETA, max_epochs=MAX_EPOCHS).fit(X, y, rng=rng)
    plot_figure2(X, y, p, FIG_DIR / "fig2_boundary.png")
    plot_figure3(p, FIG_DIR / "fig3_accuracy.png")
    res_c = summarize(p, X, y)
    # contagem de cada tipo de erro: b = b0 + eta (n_FN - n_FP) e total = n_FN + n_FP
    net = round((p.b - p.b0) / p.eta)
    res_c["n_false_negative_updates"] = int((res_c["total_updates"] + net) // 2)
    res_c["n_false_positive_updates"] = int((res_c["total_updates"] - net) // 2)

    # D.1 - margem máxima e cota de Novikoff, (R / gamma)^2, com x aumentado de 1 (viés)
    mm = max_margin(X, y)
    R = float(np.sqrt(norms.max() ** 2 + 1.0))
    res_d1 = {"max_margin": mm, "R_augmented": R,
              "novikoff_bound": float((R / mm["gamma"]) ** 2),
              "total_updates": res_c["total_updates"]}

    # D.2 - eta = 1.0, mudando SÓ a taxa (mesmo w0, mesmo b0, mesmos dados)
    p_big = Perceptron(eta=1.0, max_epochs=MAX_EPOCHS).fit(X, y, w0=p.w0, b0=p.b0)
    res_d2 = summarize(p_big, X, y)
    res_d2["angle_between_directions_deg"] = angle_between(p.w, p_big.w)
    res_d2["norm_w0"] = float(np.linalg.norm(p.w0))
    res_d2["first_update_norm_eta_small"] = float(ETA * norms.mean())   # tamanho típico de UMA atualização
    res_d2["first_update_norm_eta_big"] = float(1.0 * norms.mean())
    # a parte "vinda dos dados": w - w0 = eta * soma(erro_i x_i)
    res_d2["data_part_small"] = (p.w - p.w0).tolist()
    res_d2["data_part_big"] = (p_big.w - p_big.w0).tolist()
    res_d2["angle_w0_vs_data_part_small_deg"] = angle_between(p.w0, p.w - p.w0)
    res_d2["ratio_norm_w0_to_norm_w_small"] = float(np.linalg.norm(p.w0) / np.linalg.norm(p.w))
    res_d2["ratio_norm_w0_to_norm_w_big"] = float(np.linalg.norm(p.w0) / np.linalg.norm(p_big.w))

    # D.3 - início em zero: rodar duas vezes, com eta1 e eta2, e comparar
    eta1, eta2 = 0.01, 1.0
    z1 = Perceptron(eta=eta1, max_epochs=MAX_EPOCHS).fit(X, y, w0=np.zeros(2), b0=0.0)
    z2 = Perceptron(eta=eta2, max_epochs=MAX_EPOCHS).fit(X, y, w0=np.zeros(2), b0=0.0)
    ratio = eta2 / eta1
    res_d3 = {
        "eta1": eta1, "eta2": eta2, "ratio": ratio,
        "run_eta1": summarize(z1, X, y),
        "run_eta2": summarize(z2, X, y),
        "max_abs_diff_w2_minus_ratio_w1": float(np.max(np.abs(z2.w - ratio * z1.w))),
        "abs_diff_b2_minus_ratio_b1": float(abs(z2.b - ratio * z1.b)),
        "same_epoch_updates": bool(z1.epoch_updates == z2.epoch_updates),
        "same_epoch_accuracy": bool(np.allclose(z1.epoch_accuracy, z2.epoch_accuracy)),
        "same_predictions": bool(np.array_equal(z1.predict(X), z2.predict(X))),
        # o run com eta = 1.0 e w0 ~ N(0, 0.01) é, na prática, o run de início em zero
        "eta_big_same_epoch_updates_as_zero_start": bool(p_big.epoch_updates == z2.epoch_updates),
        "angle_between_zero_start_and_eta_big_deg": angle_between(z2.w, p_big.w),
        "angle_between_zero_start_and_eta_small_deg": angle_between(z2.w, p.w),
    }
    plot_figure_eta(X, y, [(f"η = {ETA}, w0 ~ N(0, 0.01)", p),
                           ("η = 1.0, mesmo w0", p_big),
                           ("w0 = 0 (qualquer η)", z2)],
                    FIG_DIR / "figA_ex1_eta.png")

    # verificação extra: a mesma coisa com a ordem das amostras embaralhada (uma vez)
    perm = rng.permutation(len(y))
    s = Perceptron(eta=ETA, max_epochs=MAX_EPOCHS).fit(X[perm], y[perm], w0=p.w0, b0=p.b0)
    res_shuffle = summarize(s, X, y)
    res_shuffle["angle_vs_ordered_run_deg"] = angle_between(s.w, p.w)

    return {"A": res_a, "C": res_c, "D1": res_d1, "D2": res_d2, "D3": res_d3,
            "shuffled_order": res_shuffle}
