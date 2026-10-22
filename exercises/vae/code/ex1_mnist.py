"""Exercício 1 — Um VAE no MNIST (d = 16)."""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from vae import VAE, evaluate, kl_per_dimension, reconstruction_term, sample_prior, train

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
LATENT_DIM = 16


def first_of_each_digit(y):
    """Índice da primeira imagem de cada dígito 0-9."""
    return [int((y == k).nonzero()[0]) for k in range(10)]


@torch.no_grad()
def aggregate_log_density(Z, mu, logvar, chunk=1000):
    """log q(z) da posteriori agregada q(z) = (1/N) sum_n N(z; mu_n, diag sigma_n^2), avaliada em cada linha de Z.

    Expande (z - mu)^2 / sigma^2 em produtos de matrizes para não montar um tensor N x M x d.
    """
    mu, var = mu.double(), logvar.double().exp()
    inv = 1.0 / var
    const = -0.5 * (torch.sum(mu.pow(2) * inv, dim=1) + torch.sum(var.log(), dim=1)
                    + mu.shape[1] * math.log(2 * math.pi))
    out = []
    for start in range(0, Z.shape[0], chunk):
        z = Z[start:start + chunk].double()
        quad = -0.5 * (z.pow(2) @ inv.T - 2 * z @ (mu * inv).T)
        out.append(torch.logsumexp(quad + const, dim=1) - math.log(mu.shape[0]))
    return torch.cat(out)


def bernoulli_entropy(p):
    """Entropia (nats) de cada pixel Bernoulli(p), somada na imagem: incerteza que o decodificador admite."""
    p = p.clamp(1e-7, 1 - 1e-7)
    return -(p * p.log() + (1 - p) * (1 - p).log()).sum(dim=1)


def gray_fraction(images):
    """Fração de pixels "indecisos", nem apagados nem acesos: valor em (0.1, 0.9)."""
    return float(((images > 0.1) & (images < 0.9)).float().mean())


def fig1_curves(hist):
    epochs = np.arange(1, len(hist["train_rec"]) + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, key, name in ((a, "rec", "Reconstrução (BCE)"), (b, "kl", "KL")):
        ax.plot(epochs, hist[f"train_{key}"], "o-", ms=3, c="tab:blue", label="treino (média da época)")
        ax.plot(epochs, hist[f"val_{key}"], "s-", ms=3, c="tab:orange", label="validação (fim da época)")
        ax.set_xlabel("Época")
        ax.set_ylabel(f"{name} (nats por imagem)")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(alpha=0.3)
        ax.legend()
    a.set_title("(a) Reconstrução × época")
    a.set_ylim(75, 180)
    b.set_title(f"(b) KL × época (validação antes do treino: {hist['val_kl_epoch0']:.2f})")
    b.set_ylim(0, 22)
    fig.suptitle("Figura 1 — Os dois termos da perda, treino e validação (Exercício 1, d = 16)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_loss_terms.png", dpi=150)
    plt.close(fig)


def image_rows(rows, row_labels, col_labels, title, path, col_title=None):
    """Grade de imagens 28x28: uma linha por lista em rows."""
    n_rows, n_cols = len(rows), len(rows[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.25 * n_cols, 1.45 * n_rows + 0.6))
    axes = np.atleast_2d(axes)
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(rows[r][c].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_labels[c], fontsize=9)
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=9)
            if r == n_rows - 1 and col_title and c == n_cols // 2:
                ax.set_xlabel(col_title, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig3_samples(images, n=8):
    grid = images.reshape(n, n, 28, 28).permute(0, 2, 1, 3).reshape(n * 28, n * 28)
    fig, ax = plt.subplots(figsize=(6.4, 6.8))
    ax.imshow(grid, cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks(np.arange(14, n * 28, 28), range(1, n + 1))
    ax.set_yticks(np.arange(14, n * 28, 28), range(1, n + 1))
    ax.set_xlabel("Coluna da grade")
    ax.set_ylabel("Linha da grade")
    ax.set_title("Figura 3 — 64 amostras de z ~ N(0, I) decodificadas\n(torch.manual_seed(0); d = 16)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_prior_samples.png", dpi=150)
    plt.close(fig)


def run(data):
    FIG_DIR.mkdir(exist_ok=True)
    X_train, X_val, X_test, y_test = data["X_train"], data["X_val"], data["X_test"], data["y_test"]

    # B/C — modelo e treino (a seed vem antes de construir o modelo)
    torch.manual_seed(42)
    model = VAE(LATENT_DIM)
    hist = train(model, X_train, X_val)
    fig1_curves(hist)

    # C — números finais no teste (só agora o teste é usado)
    test_rec, test_kl = evaluate(model, X_test)
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(X_test)
        rec_from_mu = model.decode(mu)
        test_rec_mu = float(reconstruction_term(rec_from_mu, X_test))

    # D — reconstruções (decodificando mu) e amostras da priori
    idx = first_of_each_digit(y_test)
    image_rows([X_test[idx], rec_from_mu[idx]], ["original", "reconstrução\n(decodifica μ)"],
               [f"dígito {k}\n(teste #{i})" for k, i in enumerate(idx)],
               "Figura 2 — Primeira imagem de teste de cada dígito e sua reconstrução (d = 16)",
               FIG_DIR / "fig2_reconstructions.png")
    torch.manual_seed(0)
    samples = sample_prior(model, 64)
    fig3_samples(samples)

    # E — medidas de apoio à análise
    with torch.no_grad():
        kl_dims = kl_per_dimension(mu, logvar)
        var_mu = mu.var(dim=0)
        # (a) onde caem as amostras da priori: densidade da posteriori agregada (componentes = teste)
        torch.manual_seed(1)
        z_prior = torch.randn(10_000, LATENT_DIM)
        mu_v, logvar_v = model.encode(X_val)
        z_post = mu_v + torch.exp(0.5 * logvar_v) * torch.randn(mu_v.shape)  # 1 z por imagem de validação
        logq_prior = aggregate_log_density(z_prior, mu, logvar)
        logq_post = aggregate_log_density(z_post, mu, logvar)
        p1 = torch.quantile(logq_post, 0.01)
        # (b) o que o decodificador devolve: probabilidades por pixel, não imagens binárias
        many_samples = model.decode(z_prior)
    return model, {
        "latent_dim": LATENT_DIM,
        "history": hist,
        "test_reconstruction": test_rec, "test_kl": test_kl, "test_neg_elbo": test_rec + test_kl,
        "test_reconstruction_decoding_mu": test_rec_mu,
        "val_kl_first_epoch": hist["val_kl"][0], "val_kl_last_epoch": hist["val_kl"][-1],
        "figure2_test_indices": idx,
        "kl_per_dimension_test": kl_dims.tolist(),
        "active_units": int((var_mu > 0.01).sum()),
        "var_mu_per_dimension_test": var_mu.tolist(),
        "prior_holes": {
            "median_logq_prior_samples": float(logq_prior.median()),
            "median_logq_posterior_samples_val": float(logq_post.median()),
            "fraction_prior_below_p1_of_posterior": float((logq_prior < p1).float().mean()),
            "fraction_prior_below_median_of_posterior": float((logq_prior < logq_post.median()).float().mean()),
        },
        "blur": {
            "gray_fraction_real_test": gray_fraction(X_test),
            "gray_fraction_reconstructions_mu": gray_fraction(rec_from_mu),
            "gray_fraction_prior_samples": gray_fraction(many_samples),
            "bernoulli_entropy_reconstructions_mu": float(bernoulli_entropy(rec_from_mu).mean()),
            "bernoulli_entropy_prior_samples": float(bernoulli_entropy(many_samples).mean()),
        },
    }
