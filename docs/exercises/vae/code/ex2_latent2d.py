"""Exercício 2 — Um espaço latente que dá para ver (d = 2)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.stats import norm

from ex1_mnist import first_of_each_digit, image_rows
from vae import VAE, evaluate, train

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
LATENT_DIM = 2
GRID_N = 15
Q = np.linspace(0.05, 0.95, GRID_N)        # quantis igualmente espaçados
GRID_Z = norm.ppf(Q)                       # z = Phi^-1(q): mesma massa da priori entre pontos vizinhos
INTERP_A, INTERP_B = 1, 4                  # imagens de teste interpoladas: #1 é um "2", #4 é um "4"
N_STEPS = 10


def fig4_scatter(mu, y):
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab10")
    order = torch.randperm(len(y), generator=torch.Generator().manual_seed(0))  # mistura a ordem de desenho
    m, yy = mu[order].numpy(), y[order].numpy()
    for k in range(10):
        ax.scatter([], [], s=25, color=cmap(k), label=f"{k}")
    ax.scatter(m[:, 0], m[:, 1], s=3, c=[cmap(k) for k in yy], alpha=0.6, linewidths=0)
    lim = GRID_Z[-1]
    ax.plot([-lim, lim, lim, -lim, -lim], [-lim, -lim, lim, lim, -lim], "k--", lw=1,
            label="região da Figura 5")
    ax.set_xlabel("$\\mu_1$")
    ax.set_ylabel("$\\mu_2$")
    ax.set_title("Figura 4 — μ das 10 000 imagens de teste, por dígito (d = 2)")
    ax.legend(title="Dígito", markerscale=1, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_latent_scatter.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def fig5_plane(model):
    # linha 0 da imagem = maior z2, para a orientação bater com a Figura 4 (z2 cresce para cima)
    z1, z2 = np.meshgrid(GRID_Z, GRID_Z[::-1])
    Z = torch.tensor(np.column_stack([z1.ravel(), z2.ravel()]), dtype=torch.float32)
    imgs = model.decode(Z).reshape(GRID_N, GRID_N, 28, 28)
    tiled = imgs.permute(0, 2, 1, 3).reshape(GRID_N * 28, GRID_N * 28)
    fig, ax = plt.subplots(figsize=(9, 9.4))
    ax.imshow(tiled, cmap="gray_r", vmin=0, vmax=1)
    ticks = np.arange(14, GRID_N * 28, 28)
    labels = [f"{0.0 if abs(v) < 1e-9 else v:.2f}" for v in GRID_Z]   # evita "-0.00" no centro
    ax.set_xticks(ticks, labels, rotation=90, fontsize=7)
    ax.set_yticks(ticks, labels[::-1], fontsize=7)
    ax.set_xlabel("$z_1 = \\Phi^{-1}(q)$,  q = 0.05 … 0.95")
    ax.set_ylabel("$z_2 = \\Phi^{-1}(q)$")
    ax.set_title("Figura 5 — Plano latente decodificado numa grade 15 × 15 de quantis (d = 2)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_latent_plane.png", dpi=150)
    plt.close(fig)


def to_grid_cell(z):
    """Posição (coluna, linha a partir do topo) de um ponto latente na grade da Figura 5, em unidades de célula."""
    step = (Q[-1] - Q[0]) / (GRID_N - 1)
    col = (norm.cdf(z[0]) - Q[0]) / step
    row_from_top = (GRID_N - 1) - (norm.cdf(z[1]) - Q[0]) / step
    return float(col) + 1, float(row_from_top) + 1   # 1-indexado, como se conta numa figura


def overlap_pairs(mu, y, k=15):
    """Pares de dígitos que se misturam em mu: para cada ponto, o dígito mais comum entre seus k vizinhos."""
    tree = cKDTree(mu.numpy())
    _, nn = tree.query(mu.numpy(), k=k + 1)
    neigh = y.numpy()[nn[:, 1:]]                       # exclui o próprio ponto
    majority = np.array([np.bincount(row, minlength=10).argmax() for row in neigh])
    yy = y.numpy()
    conf = np.zeros((10, 10), dtype=int)
    np.add.at(conf, (yy, majority), 1)
    pairs = sorted(((int(conf[i, j] + conf[j, i]), i, j) for i in range(10) for j in range(i + 1, 10)),
                   reverse=True)
    return conf, pairs, float(np.mean(majority == yy))


def run(data, model16):
    FIG_DIR.mkdir(exist_ok=True)
    X_train, X_val, X_test, y_test = data["X_train"], data["X_val"], data["X_test"], data["y_test"]

    # A — o mesmo modelo, só d = 2; mesmos dados, seed, beta e orçamento
    torch.manual_seed(42)
    model = VAE(LATENT_DIM)
    hist = train(model, X_train, X_val)
    test_rec, test_kl = evaluate(model, X_test)
    model.eval()
    model16.eval()

    with torch.no_grad():
        mu, logvar = model.encode(X_test)
        rec2 = model.decode(mu)
        mu16, _ = model16.encode(X_test)
        rec16 = model16.decode(mu16)

    # B — o conjunto de teste codificado
    fig4_scatter(mu, y_test)
    conf, pairs, knn_agree = overlap_pairs(mu, y_test)
    medians = {k: mu[y_test == k].median(dim=0).values.numpy() for k in range(10)}

    # C — o plano decodificado na grade de quantis
    fig5_plane(model)

    # D — interpolação: pixels x latente (modelo d = 16 do Exercício 1)
    t = torch.linspace(0, 1, N_STEPS)
    xa, xb = X_test[INTERP_A], X_test[INTERP_B]
    pixel_row = (1 - t)[:, None] * xa + t[:, None] * xb
    with torch.no_grad():
        latent_row = model16.decode((1 - t)[:, None] * mu16[INTERP_A] + t[:, None] * mu16[INTERP_B])
    image_rows([pixel_row, latent_row],
               ["pixels\n(1−t)x_a + t x_b", "latente, d = 16\ndecodifica\n(1−t)μ_a + t μ_b"],
               [f"t = {v:.2f}" for v in t.tolist()],
               f"Figura 6 — Interpolação entre o teste #{INTERP_A} (dígito {int(y_test[INTERP_A])}) "
               f"e o teste #{INTERP_B} (dígito {int(y_test[INTERP_B])})",
               FIG_DIR / "fig6_interpolation.png")

    # complementar — reconstruções d = 16 x d = 2 das mesmas imagens da Figura 2
    idx = first_of_each_digit(y_test)
    image_rows([X_test[idx], rec16[idx], rec2[idx]],
               ["original", "d = 16\n(decodifica μ)", "d = 2\n(decodifica μ)"],
               [f"dígito {k}" for k in range(10)],
               "Figura C1 (complementar) — As mesmas imagens reconstruídas com d = 16 e com d = 2",
               FIG_DIR / "fig_c1_d16_vs_d2.png")

    gray = lambda im: ((im > 0.1) & (im < 0.9)).float().mean(dim=1)  # noqa: E731
    return {
        "latent_dim": LATENT_DIM,
        "history": hist,
        "test_reconstruction": test_rec, "test_kl": test_kl, "test_neg_elbo": test_rec + test_kl,
        "val_kl_first_epoch": hist["val_kl"][0], "val_kl_last_epoch": hist["val_kl"][-1],
        "knn_majority_agrees_with_label": knn_agree,
        "top_overlapping_pairs": [{"digits": [i, j], "count": c} for c, i, j in pairs[:5]],
        "knn_confusion": conf.tolist(),
        "digit_medians_mu": {str(k): v.tolist() for k, v in medians.items()},
        "digit_medians_grid_cell_col_row": {str(k): to_grid_cell(v) for k, v in medians.items()},
        "grid_z": GRID_Z.tolist(),
        "interpolation": {
            "index_a": INTERP_A, "digit_a": int(y_test[INTERP_A]),
            "index_b": INTERP_B, "digit_b": int(y_test[INTERP_B]),
            "t": t.tolist(),
            "gray_fraction_pixel_row": gray(pixel_row).tolist(),
            "gray_fraction_latent_row": gray(latent_row).tolist(),
        },
    }
