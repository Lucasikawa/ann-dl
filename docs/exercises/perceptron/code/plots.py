"""Funções de desenho compartilhadas pelos dois exercícios.

Toda figura tem título, rótulos de eixo e legenda de classe (regra do enunciado).
"""
import matplotlib.pyplot as plt
import numpy as np

COLORS = ("tab:blue", "tab:orange")


def limits(X, pad=0.6):
    return ((X[:, 0].min() - pad, X[:, 0].max() + pad),
            (X[:, 1].min() - pad, X[:, 1].max() + pad))


def scatter_classes(ax, X, y, size=10, alpha=0.6):
    """Nuvem de pontos, uma cor por classe."""
    for k, color in enumerate(COLORS):
        m = y == k
        ax.scatter(X[m, 0], X[m, 1], s=size, alpha=alpha, color=color, label=f"Classe {k}")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.grid(alpha=0.3)


def shade_regions(ax, w, b, xlim, ylim, alpha=0.08):
    """Pinta de leve os dois semiplanos: onde o perceptron prevê 0 e onde prevê 1."""
    gx, gy = np.meshgrid(np.linspace(*xlim, 300), np.linspace(*ylim, 300))
    z = (gx * w[0] + gy * w[1] + b >= 0).astype(float)
    ax.contourf(gx, gy, z, levels=[-0.5, 0.5, 1.5], colors=COLORS, alpha=alpha)


def draw_boundary(ax, w, b, xlim, ylim, color="black", ls="-", lw=2.0, label=None):
    """Reta w . x + b = 0, parametrizada pelo ponto mais próximo da origem e pela
    direção perpendicular a w (funciona também quando a reta é quase vertical)."""
    norm = np.linalg.norm(w)
    p0 = -b * w / norm**2                     # ponto da reta mais próximo da origem
    d = np.array([-w[1], w[0]]) / norm        # direção ao longo da reta
    P = np.vstack([p0 - 100 * d, p0 + 100 * d])
    ax.plot(P[:, 0], P[:, 1], color=color, ls=ls, lw=lw, label=label)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def mark_misclassified(ax, X, y, y_hat, label="Ponto mal classificado", color="black"):
    """Círculo vazio em volta de cada ponto cuja previsão difere do rótulo."""
    m = y_hat != y
    ax.scatter(X[m, 0], X[m, 1], s=34, facecolors="none", edgecolors=color,
               linewidths=0.9, label=f"{label} ({int(m.sum())})")
    return int(m.sum())


def line_label(name, w, b):
    return f"{name}: ${w[0]:.3f}\\,x_1 {w[1]:+.3f}\\,x_2 {b:+.3f} = 0$"


def save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
