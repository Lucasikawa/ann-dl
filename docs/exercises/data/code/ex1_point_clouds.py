"""Exercício 1 — Nuvens de pontos: geometria e espalhamento em 2D.

Tudo aqui é geometria com NumPy: nenhum modelo é treinado. As fronteiras
"curvas" do esboço vêm da regra de Bayes calculada com os parâmetros
verdadeiros das gaussianas (conhecidos, porque nós mesmos geramos os dados).
"""
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# Parâmetros do enunciado (item A): média e desvio padrão por eixo de cada classe
MEANS = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 1.0], [15.0, 4.0]])
STDS = np.array([[0.8, 2.5], [1.2, 1.9], [0.9, 0.9], [0.5, 2.0]])
N_PER_CLASS = 100
SCALES = (0.5, 1.0, 2.0, 4.0)
COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red")
N_CLASSES = len(MEANS)


def generate_clouds(rng, scale=1.0, n=N_PER_CLASS):
    """Gera n pontos por classe. Os eixos são independentes (covariância diagonal)
    e todos os desvios são multiplicados por `scale`; as médias nunca mudam."""
    X = np.vstack([rng.normal(MEANS[k], STDS[k] * scale, size=(n, 2))
                   for k in range(N_CLASSES)])
    y = np.repeat(np.arange(N_CLASSES), n)
    return X, y


def separation_ratios(scale=1.0):
    """r_ij = ||mu_i - mu_j|| / (sigma_bar_i + sigma_bar_j), sigma_bar_k = (sigma_kx + sigma_ky) / 2."""
    sigma_bar = STDS.mean(axis=1) * scale
    rows = []
    for i, j in combinations(range(N_CLASSES), 2):
        dist = float(np.linalg.norm(MEANS[i] - MEANS[j]))
        rows.append({"pair": f"{i}-{j}", "distance": dist,
                     "sigma_bar_i": float(sigma_bar[i]), "sigma_bar_j": float(sigma_bar[j]),
                     "r": dist / float(sigma_bar[i] + sigma_bar[j])})
    return rows


def nearest_center(P):
    """Índice do centro (média da classe) mais próximo de cada ponto de P."""
    dist = np.linalg.norm(P[:, None, :] - MEANS[None, :, :], axis=2)  # (N, 4)
    return dist.argmin(axis=1)


def mixing_rate(X, y):
    """Fração de pontos cujo centro mais próximo não é o da própria classe."""
    return float(np.mean(nearest_center(X) != y))


def mixing_pairs(X, y):
    """Quantos pontos misturados há em cada direção (classe verdadeira -> centro mais próximo)."""
    pred = nearest_center(X)
    wrong = pred != y
    pairs = {}
    for true, near in zip(y[wrong], pred[wrong]):
        key = f"{true}->{near}"
        pairs[key] = pairs.get(key, 0) + 1
    return dict(sorted(pairs.items()))


def bayes_label(P, scale=1.0):
    """Classe de maior verossimilhança sob as 4 gaussianas verdadeiras (priors iguais).
    Como cada classe tem variâncias diferentes, as fronteiras resultantes são curvas."""
    var = (STDS * scale) ** 2
    diff2 = (P[:, None, :] - MEANS[None, :, :]) ** 2
    log_lik = -0.5 * (diff2 / var[None]).sum(axis=2) - 0.5 * np.log(var).sum(axis=1)[None]
    return log_lik.argmax(axis=1)


def bayes_error(rng, scale, n=20_000):
    """Erro irredutível estimado por Monte Carlo: nem a melhor fronteira possível acerta esses pontos."""
    X, y = generate_clouds(rng, scale, n)
    return float(np.mean(bayes_label(X, scale) != y))


def linearly_separable(A, B, n_angles=3600):
    """Dois conjuntos 2D são separáveis por uma reta se existe uma direção w com
    max(w·a) < min(w·b) (ou o inverso). Varre n_angles direções em [0, pi)."""
    theta = np.linspace(0.0, np.pi, n_angles, endpoint=False)
    W = np.stack([np.cos(theta), np.sin(theta)])  # (2, n_angles)
    pa, pb = A @ W, B @ W
    gap = np.maximum(pb.min(axis=0) - pa.max(axis=0), pa.min(axis=0) - pb.max(axis=0))
    return bool(gap.max() > 0)


def pairwise_separability(X, y):
    """Para cada par de classes: existe uma reta que separa os dois conjuntos de pontos?"""
    return {f"{i}-{j}": linearly_separable(X[y == i], X[y == j])
            for i, j in combinations(range(N_CLASSES), 2)}


# ----------------------------------------------------------------- figuras

def _scatter_classes(ax, X, y, size=14):
    for k in range(N_CLASSES):
        m = y == k
        ax.scatter(X[m, 0], X[m, 1], s=size, alpha=0.65, color=COLORS[k], label=f"Classe {k}")
    ax.scatter(MEANS[:, 0], MEANS[:, 1], marker="X", s=170, c=COLORS,
               edgecolors="black", linewidths=1.6, zorder=5)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.grid(alpha=0.3)


def _center_handle():
    return Line2D([], [], marker="X", linestyle="", markersize=11, markerfacecolor="white",
                  markeredgecolor="black", label="Centro (média)")


def plot_figure1(X, y, path):
    fig, ax = plt.subplots(figsize=(9, 6))
    _scatter_classes(ax, X, y)
    ax.set_title("Figura 1 — Quatro nuvens gaussianas (100 pontos por classe)")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [_center_handle()], loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_figure1_sketch(X, y, path):
    """Figura 1 com o esboço das fronteiras: curvas (Bayes) e lineares (centro mais próximo)."""
    xlim = (X[:, 0].min() - 1.5, X[:, 0].max() + 1.5)
    ylim = (X[:, 1].min() - 1.5, X[:, 1].max() + 1.5)
    gx, gy = np.meshgrid(np.linspace(*xlim, 700), np.linspace(*ylim, 500))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    curved = bayes_label(grid).reshape(gx.shape)
    linear = nearest_center(grid).reshape(gx.shape)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.contourf(gx, gy, curved, levels=np.arange(-0.5, N_CLASSES), colors=COLORS, alpha=0.12)
    for k in range(N_CLASSES):
        ax.contour(gx, gy, (curved == k).astype(float), levels=[0.5], colors="black", linewidths=1.8)
        ax.contour(gx, gy, (linear == k).astype(float), levels=[0.5], colors="dimgray",
                   linestyles="--", linewidths=1.1)
    _scatter_classes(ax, X, y)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title("Figura 1 (esboço) — Fronteiras que uma rede treinada tenderia a aprender")
    handles, _ = ax.get_legend_handles_labels()
    extra = [_center_handle(),
             Line2D([], [], color="black", lw=1.8, label="Fronteira curva (esboço da rede)"),
             Line2D([], [], color="dimgray", lw=1.1, ls="--", label="Fronteira linear (centro mais próximo)")]
    ax.legend(handles=handles + extra, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_figure2(datasets, rates, path):
    all_points = np.vstack([X for X, _ in datasets.values()])
    xlim = (all_points[:, 0].min() - 1, all_points[:, 0].max() + 1)
    ylim = (all_points[:, 1].min() - 1, all_points[:, 1].max() + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, s in zip(axes.ravel(), SCALES):
        X, y = datasets[s]
        _scatter_classes(ax, X, y, size=10)
        ax.set_title(f"s = {s}  —  taxa de mistura = {100 * rates[s]:.2f}%")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    handles, _ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles=handles + [_center_handle()], loc="lower center", ncol=5)
    fig.suptitle("Figura 2 — As mesmas 4 classes com os desvios multiplicados por s (mesmos eixos)")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_figure3(rates, bayes, path):
    s = np.array(SCALES)
    mix = 100 * np.array([rates[v] for v in SCALES])
    irreducible = 100 * np.array([bayes[v] for v in SCALES])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s, mix, "o-", lw=2, label="Taxa de mistura (centro mais próximo = fronteiras retas)")
    ax.plot(s, irreducible, "s--", lw=1.5, label="Erro de Bayes (melhor fronteira possível, curva)")
    for si, mi in zip(s, mix):
        ax.annotate(f"{mi:.2f}%", (si, mi), textcoords="offset points", xytext=(0, 9), ha="center")
    ax.set_ylim(-2, 1.15 * mix.max())
    ax.set_xscale("log", base=2)
    ax.set_xticks(s)
    ax.set_xticklabels([str(v) for v in SCALES])
    ax.set_xlabel("fator de escala s (todos os desvios × s)")
    ax.set_ylabel("pontos misturados (%)")
    ax.set_title("Figura 3 — Taxa de mistura × fator de escala s")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ roteiro

def run(rng):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # A — as 4 nuvens (100 pontos por classe, s = 1)
    X, y = generate_clouds(rng)
    plot_figure1(X, y, FIG_DIR / "fig1_clouds.png")
    res_a = {
        "samples_per_class": np.bincount(y).tolist(),
        "empirical_means": [X[y == k].mean(axis=0).tolist() for k in range(N_CLASSES)],
        "empirical_stds": [X[y == k].std(axis=0, ddof=1).tolist() for k in range(N_CLASSES)],
    }

    # B — as mesmas 4 classes, geradas de novo para cada s (4 datasets de 4 classes)
    datasets = {s: generate_clouds(rng, s) for s in SCALES}
    rates = {s: mixing_rate(*datasets[s]) for s in SCALES}
    ratios = separation_ratios(1.0)
    smallest = min(ratios, key=lambda row: row["r"])
    bayes = {s: bayes_error(rng, s) for s in SCALES}
    plot_figure2(datasets, rates, FIG_DIR / "fig2_scales.png")
    plot_figure3(rates, bayes, FIG_DIR / "fig3_mixing_rate.png")
    res_b = {
        "separation_ratios_s1": ratios,
        "smallest_pair": smallest["pair"],
        "smallest_r_s1": smallest["r"],
        "smallest_r_by_scale": {str(s): smallest["r"] / s for s in SCALES},  # r_ij escala com 1/s
        "mixing_rate": {str(s): rates[s] for s in SCALES},
        "mixing_pairs": {str(s): mixing_pairs(*datasets[s]) for s in SCALES},
        "pairwise_linearly_separable": {str(s): pairwise_separability(*datasets[s]) for s in SCALES},
        "bayes_error": {str(s): bayes[s] for s in SCALES},
    }

    # C — esboço das fronteiras sobre os dados da Figura 1
    plot_figure1_sketch(X, y, FIG_DIR / "fig1_sketch.png")
    res_c = {
        "figure1_mixing_rate": mixing_rate(X, y),
        "figure1_mixing_pairs": mixing_pairs(X, y),
        "figure1_bayes_rule_errors": int(np.sum(bayes_label(X) != y)),
        "figure1_pairwise_linearly_separable": pairwise_separability(X, y),
        # faixa vazia em x1 entre a classe 3 e as outras (onde cabe uma reta vertical)
        "class3_min_x1": float(X[y == 3, 0].min()),
        "others_max_x1": float(X[y != 3, 0].max()),
    }
    return {"A": res_a, "B": res_b, "C": res_c}
