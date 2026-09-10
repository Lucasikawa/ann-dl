"""Exercício 2 — Não-linearidade em 5 dimensões.

Dois datasets 5D: gaussianas deslocadas (Dataset I) e cascas concêntricas
(Dataset II). Nenhum modelo é treinado: as regras de decisão avaliadas aqui
usam apenas parâmetros conhecidos ou fixados antes de olhar os dados.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

N_PER_CLASS = 500

# Dataset I — parâmetros do enunciado
MU_A = np.zeros(5)
SIGMA_A = np.array([
    [1.0, 0.8, 0.1, 0.0, 0.0],
    [0.8, 1.0, 0.3, 0.0, 0.0],
    [0.1, 0.3, 1.0, 0.5, 0.0],
    [0.0, 0.0, 0.5, 1.0, 0.2],
    [0.0, 0.0, 0.0, 0.2, 1.0],
])
MU_B = np.full(5, 1.5)
SIGMA_B = np.array([
    [1.5, -0.7, 0.2, 0.0, 0.0],
    [-0.7, 1.5, 0.4, 0.0, 0.0],
    [0.2, 0.4, 1.5, 0.6, 0.0],
    [0.0, 0.0, 0.6, 1.5, 0.3],
    [0.0, 0.0, 0.0, 0.3, 1.5],
])

# Dataset II — raio ~ N(média, desvio padrão)
R_CORE, R_SHELL, R_STD = 2.0, 5.0, 0.4
R_THRESHOLD = (R_CORE + R_SHELL) / 2  # 3.5: ponto médio dos raios, fixado antes de gerar os dados

LABELS = {"I": ("Classe A", "Classe B"), "II": ("Classe C (núcleo)", "Classe D (casca)")}
COLORS = ("tab:blue", "tab:red")


def dataset_one(rng):
    XA = rng.multivariate_normal(MU_A, SIGMA_A, size=N_PER_CLASS)
    XB = rng.multivariate_normal(MU_B, SIGMA_B, size=N_PER_CLASS)
    return np.vstack([XA, XB]), np.repeat([0, 1], N_PER_CLASS)


def unit_directions(rng, n, dim=5):
    """Direções uniformes na esfera unitária de R^dim: v ~ N(0, I) e u = v / ||v||."""
    v = rng.standard_normal((n, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def shell(rng, mean_radius, n=N_PER_CLASS):
    u = unit_directions(rng, n)
    rho = rng.normal(mean_radius, R_STD, size=n)
    return rho[:, None] * u, u


def dataset_two(rng):
    XC, uC = shell(rng, R_CORE)
    XD, uD = shell(rng, R_SHELL)
    unit_norm_error = float(np.abs(np.linalg.norm(np.vstack([uC, uD]), axis=1) - 1).max())
    return np.vstack([XC, XD]), np.repeat([0, 1], N_PER_CLASS), unit_norm_error


def center_distance(X, y):
    """||mu_1 - mu_2|| com os centros empíricos, calculado em 5D."""
    return float(np.linalg.norm(X[y == 0].mean(axis=0) - X[y == 1].mean(axis=0)))


def midpoint_hyperplane(X, mu0, mu1):
    """Hiperplano perpendicular ao segmento entre os centros, passando pelo ponto médio."""
    w = mu1 - mu0
    b = w @ (mu0 + mu1) / 2
    return (X @ w - b > 0).astype(int)


def radial_rule(X):
    """f(x) = ||x||^2 - 3.5^2: f > 0 -> casca (D), f < 0 -> núcleo (C)."""
    return (np.sum(X ** 2, axis=1) - R_THRESHOLD ** 2 > 0).astype(int)


def pca_2d(X):
    """PCA não usa o rótulo: é uma projeção linear escolhida pela variância."""
    pca = PCA(n_components=2, svd_solver="full").fit(X)
    return pca, pca.transform(X)


# ----------------------------------------------------------------- figuras

def plot_figure4(Z1, y1, evr1, Z2, y2, evr2, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, Z, y, evr, key, name in ((axes[0], Z1, y1, evr1, "I", "Dataset I — gaussianas deslocadas"),
                                     (axes[1], Z2, y2, evr2, "II", "Dataset II — cascas concêntricas")):
        for k in (0, 1):
            m = y == k
            ax.scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.55, color=COLORS[k], label=LABELS[key][k])
        ax.set_title(f"{name}\nvariância explicada: PC1 {100 * evr[0]:.1f}% + PC2 {100 * evr[1]:.1f}%"
                     f" = {100 * evr.sum():.1f}%")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Figura 4 — Projeção PCA (5D → 2D) dos dois datasets")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_figure5(r1, y1, r2, y2, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, r, y, key, name in ((axes[0], r1, y1, "I", "Dataset I"), (axes[1], r2, y2, "II", "Dataset II")):
        bins = np.linspace(r.min(), r.max(), 45)
        for k in (0, 1):
            ax.hist(r[y == k], bins=bins, alpha=0.55, color=COLORS[k], label=LABELS[key][k])
        if key == "II":
            ax.axvline(R_THRESHOLD, color="black", ls="--", lw=1.5,
                       label=f"fronteira ||x|| = {R_THRESHOLD}")
        ax.set_title(f"{name} — raio ||x|| calculado em 5D")
        ax.set_xlabel("||x||")
        ax.set_ylabel("número de pontos")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Figura 5 — Histogramas do raio por classe (sobrepostos)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ roteiro

def run(rng):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # A — Dataset I
    X1, y1 = dataset_one(rng)
    res_a = {
        "min_eigenvalue_sigma_A": float(np.linalg.eigvalsh(SIGMA_A).min()),  # > 0: matriz válida
        "min_eigenvalue_sigma_B": float(np.linalg.eigvalsh(SIGMA_B).min()),
        "empirical_mean_A": X1[y1 == 0].mean(axis=0).tolist(),
        "empirical_mean_B": X1[y1 == 1].mean(axis=0).tolist(),
        "max_abs_cov_error_A": float(np.abs(np.cov(X1[y1 == 0].T) - SIGMA_A).max()),
        "max_abs_cov_error_B": float(np.abs(np.cov(X1[y1 == 1].T) - SIGMA_B).max()),
        "empirical_corr_x1x2_A": float(np.corrcoef(X1[y1 == 0].T)[0, 1]),
        "empirical_corr_x1x2_B": float(np.corrcoef(X1[y1 == 1].T)[0, 1]),
    }

    # B — Dataset II
    X2, y2, unit_err = dataset_two(rng)
    r2 = np.linalg.norm(X2, axis=1)
    res_b = {
        "max_unit_norm_error": unit_err,
        "radius_mean_C": float(r2[y2 == 0].mean()), "radius_std_C": float(r2[y2 == 0].std(ddof=1)),
        "radius_mean_D": float(r2[y2 == 1].mean()), "radius_std_D": float(r2[y2 == 1].std(ddof=1)),
    }

    # C — PCA + medidas geométricas em 5D
    pca1, Z1 = pca_2d(X1)
    pca2, Z2 = pca_2d(X2)
    r1 = np.linalg.norm(X1, axis=1)
    plot_figure4(Z1, y1, pca1.explained_variance_ratio_, Z2, y2, pca2.explained_variance_ratio_,
                 FIG_DIR / "fig4_pca.png")
    plot_figure5(r1, y1, r2, y2, FIG_DIR / "fig5_radius.png")

    # Regra linear sem treino no Dataset I: hiperplano mediador entre os centros verdadeiros,
    # em 5D e na projeção 2D (centros projetados pela mesma PCA)
    lin_5d = midpoint_hyperplane(X1, MU_A, MU_B)
    mu_proj = pca1.transform(np.vstack([MU_A, MU_B]))
    lin_2d = midpoint_hyperplane(Z1, mu_proj[0], mu_proj[1])
    # No Dataset II os centros verdadeiros coincidem (w = 0): usamos os centros empíricos
    c0, c1 = X2[y2 == 0].mean(axis=0), X2[y2 == 1].mean(axis=0)
    lin_ii = midpoint_hyperplane(X2, c0, c1)
    rad = radial_rule(X2)
    # Na projeção 2D do Dataset II: quantos pontos da casca caem no disco ocupado pelo núcleo?
    zr = np.linalg.norm(Z2, axis=1)
    shell_inside_core_disk = float(np.mean(zr[y2 == 1] <= zr[y2 == 0].max()))

    res_c = {
        "explained_variance_I": pca1.explained_variance_ratio_.tolist(),
        "explained_variance_I_sum": float(pca1.explained_variance_ratio_.sum()),
        "explained_variance_II": pca2.explained_variance_ratio_.tolist(),
        "explained_variance_II_sum": float(pca2.explained_variance_ratio_.sum()),
        "center_distance_I": center_distance(X1, y1),
        "center_distance_I_theoretical": float(np.linalg.norm(MU_B - MU_A)),
        "center_distance_II": center_distance(X2, y2),
        "radius_I_mean_A": float(r1[y1 == 0].mean()), "radius_I_mean_B": float(r1[y1 == 1].mean()),
        "radius_II_max_C": float(r2[y2 == 0].max()), "radius_II_min_D": float(r2[y2 == 1].min()),
    }
    res_d = {
        "I_midpoint_hyperplane_accuracy_5d": float(np.mean(lin_5d == y1)),
        "I_midpoint_hyperplane_accuracy_pca2d": float(np.mean(lin_2d == y1)),
        "II_midpoint_hyperplane_accuracy_5d": float(np.mean(lin_ii == y2)),
        "II_radial_rule_accuracy_5d": float(np.mean(rad == y2)),
        "II_pca2d_shell_points_inside_core_disk": shell_inside_core_disk,
    }
    return {"A": res_a, "B": res_b, "C": res_c, "D": res_d}
