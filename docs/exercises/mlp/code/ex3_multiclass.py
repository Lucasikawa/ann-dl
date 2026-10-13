"""Exercício 3 — Multiclasse com a MESMA rede.

A classe MLP é a do Exercício 2, sem nenhuma alteração: só mudam os argumentos
(camadas [4, 16, 3] e a perda CategoricalCrossEntropy = softmax + CE).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from mlp import MLP, CategoricalCrossEntropy, accuracy, standardize

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

# mesmos hiperparâmetros de treino do Exercício 2
LAYERS = [4, 16, 3]
LR = 0.05
BATCH = 32
EPOCHS = 300

N_CLASSES = 3
CLASS_NAMES = [f"Classe {k}" for k in range(N_CLASSES)]
DATA_KW = dict(n_samples=1500, n_features=4, n_informative=4, n_redundant=0, n_repeated=0,
               n_classes=3, n_clusters_per_class=2, class_sep=1.0, random_state=42)


def make_data():
    """Parâmetros do enunciado + divisão 80/20 estratificada; entradas padronizadas pelo treino."""
    X, y = make_classification(**DATA_KW)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    (X_tr_s, X_te_s), _ = standardize(X_tr, X_te)
    return X, y, X_tr, X_te, X_tr_s, X_te_s, y_tr, y_te


def recover_clusters(X):
    """Devolve, para cada amostra de X, o índice (0-5) do cluster gaussiano que a gerou.

    make_classification sorteia tudo (centros, pontos, transformações, rótulos trocados)
    ANTES de embaralhar; com shuffle=False ela devolve os mesmos pontos, em ordem de
    cluster (250 por cluster; cluster k tem rótulo k % 3) e sem permutar as colunas.
    Basta achar a permutação das colunas e casar as linhas.
    """
    Xn, _ = make_classification(shuffle=False, **DATA_KW)
    perm = [int(np.argmin([np.abs(np.sort(X[:, j]) - np.sort(Xn[:, k])).max() for k in range(4)]))
            for j in range(4)]
    Xn = Xn[:, perm]
    row_of = {tuple(np.round(r, 9)): i for i, r in enumerate(Xn)}
    idx = np.array([row_of[tuple(np.round(r, 9))] for r in X])
    assert len(set(idx.tolist())) == len(X), "casamento de linhas falhou"
    return idx // (DATA_KW["n_samples"] // 6)  # 250 amostras por cluster


def gaussian_loglik(X, mu, S):
    """log N(x; mu, S) a menos de uma constante (igual para todos os clusters em R^4)."""
    D = X - mu
    _, logdet = np.linalg.slogdet(S)
    return -0.5 * np.einsum("ij,jk,ik->i", D, np.linalg.inv(S), D) - 0.5 * logdet


def cluster_structure(X, y, X_te, y_te, pred_te):
    """Estrutura dos 6 clusters e de onde vêm os erros do MLP.

    Usa o índice de cluster, que o MLP NÃO vê. As gaussianas de cada cluster (média e
    covariância) são estimadas só no treino. Com elas:
      * sobreposição de um par = fração dos pontos dos dois clusters que a própria
        gaussiana do outro cluster explica melhor;
      * "Bayes plug-in" = classificador ideal que conhece as 6 gaussianas
        (prevê o cluster mais verossímil e devolve a classe dele). É o teto de referência.
    """
    cl = recover_clusters(X)
    # mesma chamada da divisão original -> mesmos índices; o assert confere
    X_tr_check, X_te_check, cl_tr, cl_te = train_test_split(X, cl, test_size=0.2, random_state=42, stratify=y)
    assert np.array_equal(X_te_check, X_te)
    cls_of = np.arange(6) % N_CLASSES
    mus = [X_tr_check[cl_tr == k].mean(axis=0) for k in range(6)]
    covs = [np.cov(X_tr_check[cl_tr == k].T) for k in range(6)]

    pairs = []
    for a in range(6):
        for b in range(a + 1, 6):
            if cls_of[a] == cls_of[b]:
                continue
            m = (cl == a) | (cl == b)
            closer_to_a = gaussian_loglik(X[m], mus[a], covs[a]) > gaussian_loglik(X[m], mus[b], covs[b])
            pairs.append({
                "clusters": [a, b], "classes": [int(cls_of[a]), int(cls_of[b])],
                "distance": float(np.linalg.norm(mus[b] - mus[a])),
                # centros ~ vértices (±1, ±1, ±1, ±1) do hipercubo: em quantas coordenadas diferem
                "n_coords_differ": int(np.sum(np.sign(mus[a]) != np.sign(mus[b]))),
                "overlap": float(np.mean(np.where(closer_to_a, a, b) != cl[m])),
            })
    pairs.sort(key=lambda p: -p["overlap"])

    per_cluster = []
    for k in range(6):
        m = cl_te == k
        per_cluster.append({
            "cluster": k, "class": int(cls_of[k]), "n_test": int(m.sum()),
            "predicted_counts": np.bincount(pred_te[m], minlength=N_CLASSES).tolist(),
            "errors": int(np.sum(pred_te[m] != y_te[m])),
            # desvios-padrão ao longo dos eixos principais (a transformação aleatória deforma cada cluster)
            "principal_stds": np.round(np.sqrt(np.linalg.eigvalsh(covs[k])), 2).tolist(),
        })

    LL = np.column_stack([gaussian_loglik(X_te, mus[k], covs[k]) for k in range(6)])
    bayes_pred = cls_of[LL.argmax(axis=1)]
    bayes_cm = confusion_matrix(y_te, bayes_pred)
    flipped = int(sum(np.sum(y[cl == k] != cls_of[k]) for k in range(6)))
    return {
        "centroids": np.round(mus, 3).tolist(),
        "cross_class_pairs": pairs,
        "test_errors_by_cluster": per_cluster,
        "bayes_plugin_test_accuracy": accuracy(y_te, bayes_pred),
        "bayes_plugin_confusion_matrix": bayes_cm.tolist(),
        "labels_flipped_by_flip_y": flipped,
    }


def fig4_loss(history):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(np.arange(1, len(history) + 1), history, c="tab:green",
            label=f"MLP {'-'.join(map(str, LAYERS))} (CE no treino)")
    ax.set_title("Figura 4 — Perda de treino por época (Exercício 3)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Entropia cruzada categórica (treino)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_loss.png", dpi=150)
    plt.close(fig)


def fig5_confusion(cm):
    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    row_tot = cm.sum(axis=1, keepdims=True)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / row_tot[i, 0]:.0%})", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks(range(N_CLASSES), CLASS_NAMES)
    ax.set_yticks(range(N_CLASSES), CLASS_NAMES)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe verdadeira")
    ax.set_title("Figura 5 — Matriz de confusão (teste, Ex. 3)")
    fig.colorbar(im, ax=ax, label="nº de amostras de teste (% da linha)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_confusion.png", dpi=150)
    plt.close(fig)


def run(rng):
    FIG_DIR.mkdir(exist_ok=True)

    # A — dados
    X, y, X_tr, X_te, X_tr_s, X_te_s, y_tr, y_te = make_data()

    # baseline linear (para a análise D)
    baseline = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    base_acc = float(baseline.score(X_te, y_te))

    # B/C — o MESMO MLP, com saída de 3 unidades e softmax + CE
    net = MLP(LAYERS, CategoricalCrossEntropy(N_CLASSES), rng)
    history = net.fit(X_tr_s, y_tr, epochs=EPOCHS, lr=LR, batch_size=BATCH, rng=rng)
    pred_te = net.predict(X_te_s)
    train_acc = accuracy(y_tr, net.predict(X_tr_s))
    test_acc = accuracy(y_te, pred_te)
    cm = confusion_matrix(y_te, pred_te)
    fig4_loss(history)
    fig5_confusion(cm)

    # D — par mais confundido (soma dos dois sentidos) e estrutura dos clusters
    pair_conf = {f"{i}-{j}": int(cm[i, j] + cm[j, i])
                 for i in range(N_CLASSES) for j in range(i + 1, N_CLASSES)}
    most = max(pair_conf, key=pair_conf.get)
    structure = cluster_structure(X, y, X_te, y_te, pred_te)

    print(f"[Ex3] baseline linear: {base_acc:.4f} | MLP {LAYERS}: teste {test_acc:.4f} "
          f"(treino {train_acc:.4f}) | par mais confundido: {most} ({pair_conf[most]} erros)")
    return {
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "baseline_test_accuracy": base_acc,
        "mlp": {
            "layers": LAYERS, "n_params": net.n_params(), "hidden_activation": "tanh",
            "output": "softmax", "loss": "CE", "lr": LR, "batch_size": BATCH, "epochs": EPOCHS,
            "loss_first_epoch": float(history[0]), "loss_last_epoch": float(history[-1]),
            "train_accuracy": train_acc, "test_accuracy": test_acc,
        },
        "confusion_matrix": cm.tolist(),
        "pair_confusions": pair_conf,
        "most_confused_pair": most,
        "cluster_structure": structure,
    }
