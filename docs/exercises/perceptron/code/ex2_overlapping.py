"""Exercício 2 - Dados sobrepostos: o caso que o perceptron não resolve.

A: gera as duas nuvens e a Figura 4.
B: reutiliza o Perceptron do Exercício 1 (eta = 0.01, 100 épocas) com
   pocket=True, que só acrescenta a cópia dos melhores pesos ao laço.
C: Figuras 5 e 6.
D: números de apoio para a análise: referência da melhor reta, posição da
   fronteira final, trajetória atualização a atualização, testes com outros
   eta e mais épocas. Figura complementar com a trajetória inteira.
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import EX2_COV, EX2_MEANS, N_PER_CLASS, describe, generate_data
from perceptron import Perceptron, boundary_offset
from plots import (COLORS, draw_boundary, limits, line_label, mark_misclassified, save,
                   scatter_classes, shade_regions)

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
ETA = 0.01
MAX_EPOCHS = 100
CENTER = (EX2_MEANS[0] + EX2_MEANS[1]) / 2   # centro da nuvem toda, (3.5, 3.5)


def phi(z):
    """Função de distribuição acumulada da normal padrão (sem scipy)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def weights_summary(w, b, X, y):
    """Números de um par (w, b): acurácia, geometria da reta e lado em que a nuvem cai."""
    w = np.asarray(w, dtype=float)
    z = X @ w + b
    y_hat = (z >= 0).astype(int)
    norm_w = float(np.linalg.norm(w))
    return {
        "w": w.tolist(), "b": float(b), "norm_w": norm_w,
        "direction": (w / norm_w).tolist(),
        "angle_deg": float(np.degrees(np.arctan2(w[1], w[0]))),
        "offset": boundary_offset(w, b),                             # -b / ||w||
        "center_signed_distance": float((w @ CENTER + b) / norm_w),  # centro (3.5, 3.5) até a reta
        "slope": float(-w[0] / w[1]), "intercept": float(-b / w[1]),
        "accuracy": float(np.mean(y_hat == y)),
        "share_predicted_1": float(y_hat.mean()),
        "false_positives": int(np.sum((y_hat == 1) & (y == 0))),
        "false_negatives": int(np.sum((y_hat == 0) & (y == 1))),
    }


def best_line_accuracy(X, y, n_angles=1800):
    """Melhor reta possível NESTA amostra, por busca exaustiva: para cada direção w
    (unitária) as projeções são ordenadas e todo limiar entre projeções vizinhas
    é testado. Não é um modelo treinado, é uma referência para o item D.1."""
    n = len(y)
    best = {"accuracy": 0.0}
    n_ones = int(y.sum())
    for theta in np.linspace(0.0, np.pi, n_angles, endpoint=False):
        w = np.array([np.cos(theta), np.sin(theta)])
        proj = X @ w
        order = np.argsort(proj)
        ys = y[order]
        ps = proj[order]
        # prever 1 para os índices >= k (limiar entre ps[k-1] e ps[k]), k = 0..n
        ones_right = np.concatenate([[n_ones], n_ones - np.cumsum(ys)])
        zeros_left = np.concatenate([[0], np.cumsum(1 - ys)])
        acc = (ones_right + zeros_left) / n
        acc_flipped = 1.0 - acc          # mesma reta, lados trocados (w -> -w)
        for a, sign in ((acc, 1.0), (acc_flipped, -1.0)):
            k = int(np.argmax(a))
            if a[k] > best["accuracy"]:
                if k == 0:
                    t = ps[0] - 1.0
                elif k == n:
                    t = ps[-1] + 1.0
                else:
                    t = 0.5 * (ps[k - 1] + ps[k])
                best = {"accuracy": float(a[k]), "w": (sign * w).tolist(), "b": float(-sign * t)}
    return best


def update_trajectory(p):
    """Do update_log: época, índice da amostra, acurácia, ||w||, b e distância com
    sinal do centro (3.5, 3.5) até a reta, após CADA atualização."""
    log = np.array(p.update_log)
    W, B = log[:, 2:4], log[:, 4]
    norm_w = np.linalg.norm(W, axis=1)
    return {
        "epoch": log[:, 0].astype(int), "idx": log[:, 1].astype(int),
        "acc": log[:, 5], "norm_w": norm_w, "b": B,
        "signed": (W @ CENTER + B) / norm_w,
        "is_fp": log[:, 1] < N_PER_CLASS,   # amostra da classe 0 -> erro foi um falso positivo
    }


# ----------------------------------------------------------------- figuras

def plot_figure4(X, y, path):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    scatter_classes(ax, X, y)
    ax.set_title("Figura 4 — Exercício 2: duas classes sobrepostas (1000 pontos por classe)")
    ax.set_aspect("equal")
    ax.legend()
    save(fig, path)


def plot_figure5(X, y, p, path):
    xlim, ylim = limits(X)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    sets = (("Pesos finais", p.w, p.b, "black", "-"),
            ("Pesos do pocket", p.pocket_w, p.pocket_b, "tab:green", "--"))
    for ax, (name, w, b, color, ls) in zip(axes, sets):
        shade_regions(ax, w, b, xlim, ylim)
        scatter_classes(ax, X, y, alpha=0.45)
        # as duas retas em cada painel; a do painel em destaque
        for other_name, ow, ob, ocolor, ols in sets:
            highlight = other_name == name
            draw_boundary(ax, ow, ob, xlim, ylim, color=ocolor, ls=ols,
                          lw=2.4 if highlight else 1.2,
                          label=line_label(other_name, ow, ob) if highlight else f"{other_name} (referência)")
        y_hat = Perceptron.step(X @ w + b)
        n_err = mark_misclassified(ax, X, y, y_hat)
        acc = 100 * np.mean(y_hat == y)
        ax.set_title(f"{name}: acurácia = {acc:.1f}% ({n_err} erros)")
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Figura 5 — Fronteiras final e do pocket sobre os dados do Exercício 2 "
                 "(círculos: pontos mal classificados por cada uma)")
    fig.tight_layout()
    save(fig, path)


def plot_figure6(p, reference_acc, path):
    epochs = np.arange(1, p.n_epochs + 1)
    log = np.array(p.update_log)
    # posição fracionária de cada atualização no eixo das épocas
    t = log[:, 0] - 1 + log[:, 1] / (2 * N_PER_CLASS)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, 100 * log[:, 5], lw=0.6, alpha=0.55, color="tab:blue",
            label="Acurácia dos pesos correntes após cada atualização")
    ax.plot(epochs, 100 * np.array(p.epoch_accuracy), "o", ms=4, color="tab:blue",
            label="Acurácia dos pesos correntes ao fim da época")
    ax.step(epochs, 100 * np.array(p.epoch_pocket_accuracy), where="post", lw=2.2, color="tab:green",
            label="Melhor acurácia até a época (pocket)")
    ax.axhline(100 * reference_acc, color="gray", ls="--", lw=1,
               label=f"Melhor reta possível nesta amostra ({100 * reference_acc:.1f}%)")
    ax.axvline(p.pocket_epoch, color="tab:green", ls=":", lw=1)
    ax.annotate(f"pocket copiado pela última vez: época {p.pocket_epoch}",
                (p.pocket_epoch, 100 * p.pocket_acc), textcoords="offset points",
                xytext=(-8, 10), ha="right", fontsize=8, color="tab:green")
    ax.set_xlabel("época")
    ax.set_ylabel("acurácia (%)")
    ax.set_xlim(0, p.n_epochs)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"Figura 6 — Acurácia corrente e melhor acurácia (pocket) por época (η = {p.eta})")
    save(fig, path)


def plot_figure_trajectory(p, path):
    """Figura complementar: após cada uma das atualizações do treino, a acurácia e a
    posição da reta em relação ao centro da nuvem, com o tipo de erro que a causou."""
    tr = update_trajectory(p)
    k = np.arange(1, len(tr["acc"]) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    axes[0].plot(k, 100 * tr["acc"], lw=0.8, color="tab:blue", label="Acurácia após a atualização")
    axes[0].axhline(100 * p.pocket_acc, color="tab:green", ls="--", lw=1.2,
                    label=f"Acurácia do pocket ({100 * p.pocket_acc:.1f}%)")
    axes[0].set_ylabel("acurácia (%)")
    axes[0].set_ylim(40, 80)
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].plot(k, tr["signed"], lw=0.8, color="gray")
    for mask, color, label in ((tr["is_fp"], COLORS[0], "erro numa amostra da classe 0 (falso positivo: w ← w − ηx)"),
                               (~tr["is_fp"], COLORS[1], "erro numa amostra da classe 1 (falso negativo: w ← w + ηx)")):
        axes[1].scatter(k[mask], tr["signed"][mask], s=12, color=color, label=label, zorder=3)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].axhspan(-2.5, 2.5, color="tab:green", alpha=0.08, label="faixa em que a reta atravessa a nuvem")
    axes[1].set_ylabel("distância do centro (3.5, 3.5) à reta")
    axes[1].set_ylim(-20, 8)
    axes[1].set_xlabel(f"número da atualização (1 a {len(k)}; em média {len(k) / p.n_epochs:.1f} por época)")
    axes[1].legend(loc="lower right", fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[0].set_title("Figura complementar (Ex. 2, item D) — Trajetória do perceptron, atualização a atualização: "
                      "a reta salta de um lado da nuvem para o outro")
    save(fig, path)


# ------------------------------------------------------------------ roteiro

def run(rng):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # A - dados
    X, y = generate_data(rng, EX2_MEANS, EX2_COV)
    plot_figure4(X, y, FIG_DIR / "fig4_data.png")
    norms = np.linalg.norm(X, axis=1)
    res_a = {"n_total": int(len(X)), "classes": describe(X, y),
             "norm_x_mean": float(norms.mean()), "norm_x_std": float(norms.std()),
             "norm_x_max": float(norms.max())}

    # B - mesmo Perceptron, mesmo eta e mesmo limite de épocas; só acrescenta o pocket
    p = Perceptron(eta=ETA, max_epochs=MAX_EPOCHS).fit(X, y, rng=rng, pocket=True)
    tr = update_trajectory(p)
    res_b = {
        "w0": p.w0.tolist(), "b0": p.b0,
        "n_epochs": p.n_epochs, "converged": p.converged,
        "total_updates": int(sum(p.epoch_updates)),
        "n_false_positive_updates": int(tr["is_fp"].sum()),
        "n_false_negative_updates": int((~tr["is_fp"]).sum()),
        "updates_per_epoch_min": int(min(p.epoch_updates)),
        "updates_per_epoch_max": int(max(p.epoch_updates)),
        "updates_per_epoch_mean": float(np.mean(p.epoch_updates)),
        "final": weights_summary(p.w, p.b, X, y),
        "pocket": weights_summary(p.pocket_w, p.pocket_b, X, y),
        "pocket_epoch": p.pocket_epoch,
        "pocket_update_in_epoch": p.pocket_update,
        "pocket_sample_index": int(next(row[1] for row in p.update_log
                                        if row[0] == p.pocket_epoch and row[5] == p.pocket_acc)),
        "pocket_improvements": [(int(e), float(a)) for e, a in
                                zip(range(1, p.n_epochs + 1), p.epoch_pocket_accuracy)
                                if e == 1 or a > p.epoch_pocket_accuracy[e - 2]],
        "epoch_accuracy": p.epoch_accuracy,
        "epoch_pocket_accuracy": p.epoch_pocket_accuracy,
        "epoch_accuracy_min": float(min(p.epoch_accuracy)),
        "epoch_accuracy_max": float(max(p.epoch_accuracy)),
        "update_accuracy_min": float(tr["acc"].min()),
        "update_accuracy_max": float(tr["acc"].max()),
    }

    # referências para o item D.1: regra de Bayes (mediatriz x1 + x2 = 7) e melhor reta na amostra
    d = float(np.linalg.norm(EX2_MEANS[1] - EX2_MEANS[0]))
    sigma = float(np.sqrt(EX2_COV[0, 0]))
    bayes_w = EX2_MEANS[1] - EX2_MEANS[0]
    bayes_b = -float(bayes_w @ CENTER)
    best = best_line_accuracy(X, y)
    res_ref = {
        "bayes_rule": weights_summary(bayes_w, bayes_b, X, y),
        "bayes_theoretical_accuracy": phi(d / (2 * sigma)),
        "best_line_search": best,
        "best_line_search_summary": weights_summary(best["w"], best["b"], X, y),
    }

    # C - figuras
    plot_figure5(X, y, p, FIG_DIR / "fig5_boundaries.png")
    plot_figure6(p, best["accuracy"], FIG_DIR / "fig6_accuracy.png")
    plot_figure_trajectory(p, FIG_DIR / "figB_ex2_trajectory.png")

    # D.1 - o que acontece atualização a atualização
    end_states = np.array([(w @ np.ones(2), b) for w, b in p.epoch_weights])
    share_all_one = np.mean([weights_summary(w, b, X, y)["share_predicted_1"] >= 0.99
                             for w, b in p.epoch_weights])
    inside = np.abs(tr["signed"]) < 2.5
    ep = tr["epoch"]
    per_epoch_fp = [int(np.sum(tr["is_fp"] & (ep == e))) for e in range(1, p.n_epochs + 1)]
    per_epoch_fn = [int(np.sum(~tr["is_fp"] & (ep == e))) for e in range(1, p.n_epochs + 1)]
    res_d1 = {
        "per_mistake_step_w": float(ETA * norms.mean()),   # eta * ||x|| médio
        "per_mistake_step_b": ETA,
        "norm_w_after_updates_min": float(tr["norm_w"].min()),
        "norm_w_after_updates_max": float(tr["norm_w"].max()),
        "norm_w_after_updates_mean": float(tr["norm_w"].mean()),
        "b_min": float(tr["b"].min()), "b_max": float(tr["b"].max()),
        "offset_needed_for_center": float(np.linalg.norm(CENTER)),   # |b|/||w|| para a reta passar no centro
        "share_of_epochs_ending_with_all_predicted_1": float(share_all_one),
        "updates_class0_block_per_epoch_mean": float(np.mean(per_epoch_fp)),
        "updates_class1_block_per_epoch_mean": float(np.mean(per_epoch_fn)),
        "share_updates_landing_inside_cloud": float(inside.mean()),
        "n_updates_landing_inside_cloud": int(inside.sum()),
        "accuracy_when_inside_mean": float(tr["acc"][inside].mean()) if inside.any() else None,
        "accuracy_when_outside_mean": float(tr["acc"][~inside].mean()),
        "signed_distance_min": float(tr["signed"].min()),
        "signed_distance_max": float(tr["signed"].max()),
        "last_epoch": {
            "n_updates": int(p.epoch_updates[-1]),
            "sample_indices": [int(i) for i in tr["idx"][ep == p.n_epochs]],
            "accuracies": [float(a) for a in tr["acc"][ep == p.n_epochs]],
            "signed_distances": [float(s) for s in tr["signed"][ep == p.n_epochs]],
            "norm_w": [float(s) for s in tr["norm_w"][ep == p.n_epochs]],
            "start": weights_summary(*p.epoch_weights[-2], X, y),
        },
    }

    # D.3 - mais épocas e outros eta, sem mudar mais nada (mesmo w0)
    trials = []
    for eta, max_epochs in ((0.1, 100), (0.001, 100), (0.0001, 100), (ETA, 500)):
        q = Perceptron(eta=eta, max_epochs=max_epochs).fit(X, y, w0=p.w0, b0=p.b0, pocket=True)
        trials.append({
            "eta": eta, "max_epochs": max_epochs, "n_epochs": q.n_epochs, "converged": q.converged,
            "total_updates": int(sum(q.epoch_updates)),
            "final_accuracy": q.epoch_accuracy[-1], "pocket_accuracy": q.pocket_acc,
            "pocket_epoch": q.pocket_epoch, "norm_w_final": float(np.linalg.norm(q.w)),
            "epoch_accuracy_min": float(min(q.epoch_accuracy)), "epoch_accuracy_max": float(max(q.epoch_accuracy)),
            "max_abs_diff_epoch_accuracy_vs_main": float(np.max(np.abs(
                np.array(q.epoch_accuracy[:MAX_EPOCHS]) - np.array(p.epoch_accuracy)))),
        })
    # a partir de w0 = 0, eta1 e eta2 dão a MESMA sequência de acurácias (só reescala w)
    z1 = Perceptron(eta=0.01, max_epochs=MAX_EPOCHS).fit(X, y, w0=np.zeros(2), pocket=True)
    z2 = Perceptron(eta=0.001, max_epochs=MAX_EPOCHS).fit(X, y, w0=np.zeros(2), pocket=True)
    res_d3 = {
        "trials": trials,
        "zero_start_same_epoch_accuracy": bool(np.allclose(z1.epoch_accuracy, z2.epoch_accuracy)),
        "zero_start_same_pocket_accuracy": bool(z1.pocket_acc == z2.pocket_acc),
        "zero_start_final_accuracy": z1.epoch_accuracy[-1],
        "zero_start_pocket_accuracy": z1.pocket_acc,
        "zero_start_max_abs_diff_w2_minus_ratio_w1": float(np.max(np.abs(z2.w - 0.1 * z1.w))),
    }

    # verificação extra: e se a ordem das amostras fosse embaralhada (uma vez, com o mesmo rng)?
    perm = rng.permutation(len(y))
    s = Perceptron(eta=ETA, max_epochs=MAX_EPOCHS).fit(X[perm], y[perm], w0=p.w0, b0=p.b0, pocket=True)
    res_shuffle = {
        "n_epochs": s.n_epochs, "converged": s.converged, "total_updates": int(sum(s.epoch_updates)),
        "final_accuracy": s.epoch_accuracy[-1], "pocket_accuracy": s.pocket_acc,
        "pocket_epoch": s.pocket_epoch,
        "epoch_accuracy_min": float(min(s.epoch_accuracy)), "epoch_accuracy_max": float(max(s.epoch_accuracy)),
        "epoch_accuracy_mean": float(np.mean(s.epoch_accuracy)),
        "final": weights_summary(s.w, s.b, X, y),
    }

    return {"A": res_a, "B": res_b, "reference": res_ref, "D1": res_d1, "D3": res_d3,
            "shuffled_order": res_shuffle}
