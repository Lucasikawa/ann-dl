"""Exercício 4 — Uma segunda camada oculta ajuda?

Comparação controlada: mesmos dados, divisão, padronização, taxa de aprendizado,
tamanho de lote, nº de épocas e inicialização (Xavier) do Exercício 3. A única
mudança é a profundidade: [4, 16, 3] -> [4, 16, 16, 3] (mesma largura, 16).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

import ex3_multiclass as ex3
from mlp import MLP, CategoricalCrossEntropy, accuracy

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

SHALLOW = ex3.LAYERS            # [4, 16, 3]     — a rede do Exercício 3
DEEP = [4, 16, 16, 3]           # mesma largura, uma camada oculta a mais
LR, BATCH, EPOCHS = ex3.LR, ex3.BATCH, ex3.EPOCHS  # orçamento idêntico ao do Exercício 3

SEEDS = (42, 43, 44, 45, 46)    # >= 3 seeds; cada seed inicializa as DUAS redes
# controle extra: mesma quantidade de parâmetros, profundidades diferentes
MATCHED = {"~135 parâmetros": ([4, 16, 3], [4, 8, 8, 3]),
           "~395 parâmetros": ([4, 48, 3], [4, 16, 16, 3])}
# diagnóstico (não usado para escolher nada): o que acontece se o treino continuar
LONG_EPOCHS, EVAL_EVERY = 1000, 10

COLORS = {"shallow": "tab:green", "deep": "tab:orange"}


def train(layers, seed, data, epochs=EPOCHS, eval_every=None):
    """Treina uma rede com np.random.default_rng(seed) (inicialização + embaralhamento).

    Com eval_every, treina em blocos e mede a acurácia de teste a cada bloco. Isso não
    muda o treino: fit consome o rng só com uma permutação por época, então 10 blocos de
    30 épocas = 1 bloco de 300 épocas, bit a bit. As métricas "no orçamento" são tiradas
    exatamente na época EPOCHS.
    """
    X_tr, X_te, y_tr, y_te = data
    rng = np.random.default_rng(seed)
    net = MLP(layers, CategoricalCrossEntropy(ex3.N_CLASSES), rng)
    step = eval_every or epochs
    history, checkpoints, at_budget = [], [], None
    done = 0
    while done < epochs:
        history.extend(net.fit(X_tr, y_tr, epochs=step, lr=LR, batch_size=BATCH, rng=rng))
        done += step
        checkpoints.append((done, accuracy(y_te, net.predict(X_te))))
        if done == EPOCHS:
            pred = net.predict(X_te)
            at_budget = {
                "final_train_loss": float(history[-1]),
                "train_accuracy": accuracy(y_tr, net.predict(X_tr)),
                "test_accuracy": accuracy(y_te, pred),
                "confusion_matrix": confusion_matrix(y_te, pred),
            }
    return {"n_params": net.n_params(), "history": np.array(history),
            "checkpoints": checkpoints, "at_budget": at_budget}


def summarize(runs):
    keys = ("final_train_loss", "train_accuracy", "test_accuracy")
    return {k: {"mean": float(np.mean([r["at_budget"][k] for r in runs])),
                "std": float(np.std([r["at_budget"][k] for r in runs])),
                "per_seed": [r["at_budget"][k] for r in runs]} for k in keys}


def fig6_loss(shallow_runs, deep_runs):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ep = np.arange(1, EPOCHS + 1)
    for key, runs, layers in (("shallow", shallow_runs, SHALLOW), ("deep", deep_runs, DEEP)):
        H = np.array([r["history"][:EPOCHS] for r in runs])
        name = "Exercício 3" if key == "shallow" else "Exercício 4"
        ax.plot(ep, H.mean(axis=0), c=COLORS[key], lw=2,
                label=f"{name}: {'-'.join(map(str, layers))} (média de {len(runs)} seeds)")
        ax.fill_between(ep, H.min(axis=0), H.max(axis=0), color=COLORS[key], alpha=0.2,
                        label=f"{name}: faixa mín–máx entre seeds")
    ax.set_title("Figura 6 — Perda de treino: 1 vs. 2 camadas ocultas (mesmo orçamento)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Entropia cruzada categórica (treino)")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_depth_loss.png", dpi=150)
    plt.close(fig)


def fig_c1_long(shallow_runs, deep_runs):
    """Figura complementar: treino estendido até LONG_EPOCHS, perda de treino x acurácia de teste."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4))
    for key, runs, layers in (("shallow", shallow_runs, SHALLOW), ("deep", deep_runs, DEEP)):
        name = f"{'Ex. 3' if key == 'shallow' else 'Ex. 4'}: {'-'.join(map(str, layers))}"
        H = np.array([r["history"] for r in runs])
        a.plot(np.arange(1, H.shape[1] + 1), H.mean(axis=0), c=COLORS[key], lw=2, label=name)
        cp = np.array([[acc for _, acc in r["checkpoints"]] for r in runs])
        epochs = [e for e, _ in runs[0]["checkpoints"]]
        b.plot(epochs, 100 * cp.mean(axis=0), c=COLORS[key], lw=2, label=name)
        b.fill_between(epochs, 100 * cp.min(axis=0), 100 * cp.max(axis=0), color=COLORS[key], alpha=0.15)
    for ax in (a, b):
        ax.axvline(EPOCHS, c="gray", ls="--", lw=1, label=f"orçamento usado ({EPOCHS} épocas)")
        ax.set_xlabel("Época")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)
    a.set_yscale("log")
    a.set_ylabel("Entropia cruzada (treino)")
    a.set_title("(a) Perda de treino — média de 5 seeds")
    b.set_ylabel("Acurácia de teste (%)")
    b.set_title("(b) Acurácia de teste — média e faixa mín–máx")
    fig.suptitle("Figura C1 (complementar) — Treino estendido até 1000 épocas, só para diagnóstico")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_c1_long_training.png", dpi=150)
    plt.close(fig)


def run(rng):
    FIG_DIR.mkdir(exist_ok=True)
    _, _, _, _, X_tr_s, X_te_s, y_tr, y_te = ex3.make_data()
    data = (X_tr_s, X_te_s, y_tr, y_te)

    # A — rede mais funda, treinada com o rng compartilhado do relatório (como a do Ex. 3)
    net = MLP(DEEP, CategoricalCrossEntropy(ex3.N_CLASSES), rng)
    history = net.fit(X_tr_s, y_tr, epochs=EPOCHS, lr=LR, batch_size=BATCH, rng=rng)
    main_run = {
        "layers": DEEP, "n_params": net.n_params(), "lr": LR, "batch_size": BATCH, "epochs": EPOCHS,
        "final_train_loss": float(history[-1]),
        "train_accuracy": accuracy(y_tr, net.predict(X_tr_s)),
        "test_accuracy": accuracy(y_te, net.predict(X_te_s)),
    }

    # B — comparação controlada em várias seeds (treino até LONG_EPOCHS; tabela em EPOCHS)
    shallow = [train(SHALLOW, s, data, LONG_EPOCHS, EVAL_EVERY) for s in SEEDS]
    deep = [train(DEEP, s, data, LONG_EPOCHS, EVAL_EVERY) for s in SEEDS]
    fig6_loss(shallow, deep)
    fig_c1_long(shallow, deep)
    s_sum, d_sum = summarize(shallow), summarize(deep)
    diffs = np.array(d_sum["test_accuracy"]["per_seed"]) - np.array(s_sum["test_accuracy"]["per_seed"])
    p = s_sum["test_accuracy"]["mean"]

    # robustez do "par mais confundido" do Exercício 3 entre seeds (rede rasa)
    def pair_counts(cm):
        return {f"{i}-{j}": int(cm[i, j] + cm[j, i]) for i in range(3) for j in range(i + 1, 3)}
    shallow_pairs = [pair_counts(r["at_budget"]["confusion_matrix"]) for r in shallow]

    # C — controle: mesma contagem de parâmetros, profundidade diferente
    matched = {}
    for label, (wide, narrow_deep) in MATCHED.items():
        matched[label] = {}
        for layers in (wide, narrow_deep):
            runs = [train(layers, s, data) for s in SEEDS]
            matched[label]["-".join(map(str, layers))] = {"n_params": runs[0]["n_params"], **summarize(runs)}

    # diagnóstico do treino estendido: acurácia média de teste em algumas épocas
    def acc_at(runs, epoch):
        return float(np.mean([dict(r["checkpoints"])[epoch] for r in runs]))
    long_view = {str(e): {"shallow_test_acc": acc_at(shallow, e), "deep_test_acc": acc_at(deep, e),
                          "shallow_train_loss": float(np.mean([r["history"][e - 1] for r in shallow])),
                          "deep_train_loss": float(np.mean([r["history"][e - 1] for r in deep]))}
                 for e in (50, 100, 200, 300, 500, 1000)}

    print(f"[Ex4] principal {DEEP}: teste {main_run['test_accuracy']:.4f} | "
          f"seeds {SEEDS}: raso {s_sum['test_accuracy']['mean']:.4f} x fundo {d_sum['test_accuracy']['mean']:.4f} "
          f"(diferença média {diffs.mean():+.4f})")
    return {
        "main_run": main_run,
        "seeds": list(SEEDS),
        "shallow": {"layers": SHALLOW, "n_params": shallow[0]["n_params"], **s_sum},
        "deep": {"layers": DEEP, "n_params": deep[0]["n_params"], **d_sum},
        "test_acc_diff_deep_minus_shallow": {
            "per_seed": diffs.tolist(), "mean": float(diffs.mean()), "std": float(diffs.std(ddof=1)),
            "seeds_where_deep_wins": int(np.sum(diffs > 0)),
        },
        "test_accuracy_binomial_se": float(np.sqrt(p * (1 - p) / len(y_te))),
        "shallow_pair_confusions_per_seed": shallow_pairs,
        "parameter_matched": matched,
        "long_training": long_view,
    }
