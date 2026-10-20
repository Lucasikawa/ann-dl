"""Exercício 2 — Dados sobrepostos: o caso que o perceptron não resolve.

O perceptron é o mesmo de perceptron.py, sem nenhuma alteração; o bolso (pocket)
já é guardado por ele.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ex1_separable import CLASS_NAMES, COLORS, boundary_line, fig1_scatter, make_data
from perceptron import Perceptron

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

MEAN_0, MEAN_1 = np.array([3.0, 3.0]), np.array([4.0, 4.0])
COV = np.array([[1.5, 0.0], [0.0, 1.5]])
ETA = 0.01
MAX_EPOCHS = 100
CENTER = (MEAN_0 + MEAN_1) / 2  # (3.5, 3.5): meio do caminho entre as duas médias


def best_line_accuracy(X, y, n_angles=1440):
    """Melhor acurácia com UMA reta: varre direções e todos os limiares (w·x >= t -> classe 1)."""
    best = 0.0
    for theta in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
        s = X @ np.array([np.cos(theta), np.sin(theta)])
        ys = y[np.argsort(s)]
        ones_right = np.concatenate([[ys.sum()], ys.sum() - np.cumsum(ys)])
        zeros_left = np.concatenate([[0], np.cumsum(1 - ys)])
        best = max(best, float(((ones_right + zeros_left) / len(y)).max()))
    return best


def signed_distance(w, b, point):
    """Distância com sinal do ponto à reta w·x + b = 0 (positivo = lado da classe 1)."""
    return float((w @ point + b) / np.linalg.norm(w))


def describe(w, b, X, y):
    w = np.asarray(w)
    pred = (X @ w + b >= 0).astype(int)
    return {
        "w": w.tolist(), "b": float(b), "accuracy": float(np.mean(pred == y)),
        "norm_w": float(np.linalg.norm(w)),
        "distance_origin_to_boundary": float(abs(b) / np.linalg.norm(w)),
        "signed_distance_center_to_boundary": signed_distance(w, b, CENTER),
        "fraction_predicted_class1": float(pred.mean()),
    }


def fig5_boundaries(X, y, final, pocket):
    fig, axes = plt.subplots(1, 2, figsize=(11, 7), sharex=True, sharey=True)
    x1 = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 2)
    lines = {"pocket": (pocket, "k", "-", "bolso"), "final": (final, "tab:green", "--", "final")}
    for ax, key in zip(axes, ("pocket", "final")):
        (w, b), _, _, _ = lines[key]
        pred = (X @ np.asarray(w) + b >= 0).astype(int)
        for k in (0, 1):
            m = (y == k) & (pred == y)
            ax.scatter(X[m, 0], X[m, 1], s=7, alpha=0.45, c=COLORS[k], label=f"{CLASS_NAMES[k]} (acerto)")
        wrong = pred != y
        for k, mk in ((0, "x"), (1, "+")):
            m = wrong & (y == k)
            ax.scatter(X[m, 0], X[m, 1], s=22, marker=mk, c=COLORS[k], linewidths=1.1,
                       label=f"{CLASS_NAMES[k]} classificada errado ({m.sum()})")
        for (wl, bl), c, ls, nm in lines.values():
            acc = np.mean((X @ np.asarray(wl) + bl >= 0).astype(int) == y)
            ax.plot(x1, boundary_line(np.asarray(wl), bl, x1), c=c, ls=ls, lw=2.2,
                    label=f"fronteira {nm} ({acc:.2%})")
        ax.set_title("(a) Erros marcados: pesos do bolso" if key == "pocket"
                     else "(b) Erros marcados: pesos finais")
        ax.set_xlabel("$x_1$")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    axes[0].set_ylabel("$x_2$")
    axes[0].set_xlim(x1[0], x1[1])
    axes[0].set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    fig.suptitle("Figura 5 — Fronteiras final e do bolso (Exercício 2)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG_DIR / "fig5_boundaries.png", dpi=150)
    plt.close(fig)


def fig6_accuracy(hist, initial_accuracy, best_line):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(hist["update_position"], 100 * hist["update_accuracy"], c="tab:gray", lw=0.3, alpha=0.5,
            label="pesos atuais, logo após cada atualização")
    ep = np.arange(0, len(hist["accuracy"]) + 1)
    ax.plot(ep, 100 * np.concatenate([[initial_accuracy], hist["accuracy"]]), c="tab:green", marker="o", ms=3,
            lw=1.5, label="pesos atuais, ao fim de cada época")
    ax.plot(ep, 100 * np.concatenate([[initial_accuracy], hist["pocket_accuracy"]]), c="k", lw=2.2,
            label="bolso (melhor até agora)")
    ax.axhline(100 * best_line, c="tab:red", ls=":", lw=1.5, label=f"melhor reta possível ({best_line:.2%})")
    ax.axhline(50, c="tab:blue", ls=":", lw=1, label="chute (50%)")
    ax.set_xlim(0, len(hist["accuracy"]))
    ax.set_ylim(40, 75)
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia na base inteira (%)")
    ax.set_title("Figura 6 — Acurácia dos pesos atuais e do bolso × época (Exercício 2)")
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_accuracy.png", dpi=150)
    plt.close(fig)


def fig_c2_dynamics(hist, step_size):
    """Complementar: o que acontece a cada atualização durante a última época."""
    last = hist["update_position"] > len(hist["accuracy"]) - 1  # atualizações da última época
    pos = hist["update_position"][last]
    W = np.array(hist["update_w"])[last]
    B = np.array(hist["update_b"])[last]
    norm_w = np.linalg.norm(W, axis=1)
    dist = (W @ CENTER + B) / norm_w

    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    a.plot(pos, norm_w, c="tab:purple", lw=1, label=r"$\|\mathbf{w}\|$")
    a.plot(pos, np.abs(B), c="tab:orange", lw=1.5, label="$|b|$")
    a.axhline(step_size, c="k", ls="--", lw=1,
              label=rf"tamanho de um passo em $\mathbf{{w}}$: $\eta\,\overline{{\|\mathbf{{x}}\|}} = {step_size:.3f}$")
    a.set_ylabel("Magnitude")
    a.set_title("(a) Tamanho dos pesos depois de cada atualização")
    a.legend(loc="center right", fontsize=9)
    b.plot(pos, np.clip(dist, -6, 6), c="tab:green", lw=0.8, label="distância com sinal do centro (3.5, 3.5) à fronteira")
    b.axhspan(-np.sqrt(1.5), np.sqrt(1.5), color="tab:gray", alpha=0.2, label=r"$\pm 1$ desvio-padrão da nuvem")
    b.axhline(0, c="k", lw=0.8)
    b.set_ylim(-6.3, 6.3)
    b.set_ylabel("Distância (cortada em ±6)")
    b.set_xlabel("Época")
    b.set_title("(b) Onde a fronteira está em relação ao centro da nuvem")
    b.legend(loc="lower right", fontsize=9)
    for ax in (a, b):
        ax.grid(alpha=0.3)
    fig.suptitle("Figura C2 (complementar) — Última época do Exercício 2, atualização por atualização")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_c2_dynamics.png", dpi=150)
    plt.close(fig)
    return {
        "updates_in_last_epoch": int(last.sum()),
        "norm_w_range": [float(norm_w.min()), float(norm_w.max())],
        "b_range": [float(B.min()), float(B.max())],
        "center_distance_percentiles_5_50_95": np.percentile(dist, [5, 50, 95]).tolist(),
        "fraction_updates_center_beyond_1sd": float(np.mean(np.abs(dist) > np.sqrt(1.5))),
    }


def variant(X, y, w0, eta, epochs, order=None):
    """Mesmos dados e mesma partida w0; muda só o que for pedido (eta, épocas ou ordem)."""
    if order is not None:
        X, y = X[order], y[order]
    m = Perceptron(w0, eta)
    h = m.fit(X, y, max_epochs=epochs)
    tail = h["accuracy"][-50:]
    return {
        "eta": eta, "epochs": epochs,
        "final_accuracy": float(h["accuracy"][-1]),
        "final_fraction_predicted_class1": float(m.predict(X).mean()),
        "last50_epoch_accuracy_range": [float(tail.min()), float(tail.max())],
        "mean_updates_per_epoch": float(h["updates"].mean()),
        "pocket_accuracy": m.pocket["accuracy"],
    }


def run(rng):
    FIG_DIR.mkdir(exist_ok=True)

    # A — dados (mesma função do Exercício 1, com os parâmetros deste exercício)
    X, y = make_data(rng, MEAN_0, MEAN_1, COV)
    fig1_scatter(X, y, "Figura 4 — Dados sobrepostos (Exercício 2)", FIG_DIR / "fig4_data.png",
                 (MEAN_0.tolist(), MEAN_1.tolist()))

    # B — o mesmo perceptron, eta = 0.01, 100 épocas; o bolso é guardado pelo próprio fit
    w0 = rng.normal(0, 0.01, size=2)
    model = Perceptron(w0, ETA)
    hist = model.fit(X, y, max_epochs=MAX_EPOCHS)
    final = describe(model.w, model.b, X, y)
    pocket = describe(model.pocket["w"], model.pocket["b"], X, y)
    pocket["epoch"] = model.pocket["epoch"]

    # C — figuras
    best_line = best_line_accuracy(X, y)
    fig5_boundaries(X, y, (model.w, model.b), (model.pocket["w"], model.pocket["b"]))
    fig6_accuracy(hist, model.initial_accuracy, best_line)
    step_size = ETA * float(np.linalg.norm(X, axis=1).mean())
    dynamics = fig_c2_dynamics(hist, step_size)

    # D — referências e verificações (as respostas vêm da regra; isto só confirma)
    ua = hist["update_accuracy"]
    checks = {
        "class_sorted_order": variant(X, y, w0, ETA, MAX_EPOCHS, order=np.argsort(y, kind="stable")),
        "more_epochs": variant(X, y, w0, ETA, 1000),
        "smaller_eta": variant(X, y, w0, ETA / 10, MAX_EPOCHS),
        "smaller_eta_more_epochs": variant(X, y, w0, ETA / 10, 1000),
    }
    print(f"[Ex2] final: acc {final['accuracy']:.4f} | bolso: acc {pocket['accuracy']:.4f} (época {pocket['epoch']}) | "
          f"melhor reta {best_line:.4f} | por atualização: {ua.min():.3f}–{ua.max():.3f}")
    return {
        "class_means_sample": [X[y == k].mean(axis=0).tolist() for k in (0, 1)],
        "w0": w0.tolist(),
        "epochs_run": model.epochs, "converged": model.converged,
        "updates_per_epoch": hist["updates"].tolist(),
        "total_updates": int(hist["updates"].sum()),
        "accuracy_per_epoch": hist["accuracy"].tolist(),
        "final": final,
        "pocket": pocket,
        "best_single_line_accuracy": best_line,
        "bayes_rule_accuracy": float(np.mean((X.sum(axis=1) >= CENTER.sum()).astype(int) == y)),
        "mean_norm_x": float(np.linalg.norm(X, axis=1).mean()),
        "step_size_w": step_size, "step_size_b": ETA,
        "per_update_accuracy": {
            "n": int(len(ua)), "min": float(ua.min()), "p5": float(np.percentile(ua, 5)),
            "median": float(np.median(ua)), "p95": float(np.percentile(ua, 95)), "max": float(ua.max()),
            "fraction_at_or_below_50pct": float(np.mean(ua <= 0.5)),
        },
        "end_of_epoch_accuracy_range": [float(hist["accuracy"].min()), float(hist["accuracy"].max())],
        "last_epoch_dynamics": dynamics,
        "checks": checks,
    }
