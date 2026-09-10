"""Exercício 3 — Preparando o Spaceship Titanic para uma rede com tanh.

Regra de ouro: a divisão treino/teste acontece ANTES de qualquer estatística
usada numa transformação. Mediana, moda, categorias, média e desvio padrão
são ajustados (fit) só no treino e depois aplicados (transform) ao teste.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "train.csv"
FIG_DIR = HERE.parent / "figures"

TARGET = "Transported"
SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERIC = ["Age"] + SPEND
CATEGORICAL = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
DROP = ["PassengerId", "Cabin", "Name"]
LOG_COLS = SPEND + ["TotalSpend"]
CLASS_LABELS = ("Transported = False", "Transported = True")
CLASS_COLORS = ("tab:blue", "tab:orange")


# ------------------------------------------------------------ A — descrição

def describe(df):
    """Estatísticas puramente descritivas do arquivo inteiro: nenhuma delas alimenta uma transformação."""
    counts = df[TARGET].value_counts()
    missing = pd.DataFrame({"ausentes": df.isna().sum(), "percentual": 100 * df.isna().mean()})
    spend = df[SPEND].agg(["mean", "median", "max"]).T
    spend["zeros_pct"] = 100 * (df[SPEND] == 0).sum() / df[SPEND].notna().sum()
    spend["skew"] = df[SPEND].skew()
    return {
        "shape": list(df.shape),
        "target_counts": {str(k): int(v) for k, v in counts.items()},
        "positive_share": float(df[TARGET].mean()),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "n_unique": {c: int(df[c].nunique()) for c in df.columns},
        "missing": {c: {"count": int(r.ausentes), "pct": float(r.percentual)} for c, r in missing.iterrows()},
        "rows_with_any_missing_pct": float(100 * df.isna().any(axis=1).mean()),
        "spending_stats": {c: {k: float(v) for k, v in r.items()} for c, r in spend.iterrows()},
    }


# -------------------------------------------------------------- B — divisão

def stratified_split(y, rng, test_size=0.2):
    """Divisão estratificada com o rng do relatório: sorteia 20% de cada classe para o teste."""
    test_parts = []
    for label in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == label))
        test_parts.append(idx[: int(round(test_size * len(idx)))])
    test_idx = np.sort(np.concatenate(test_parts))
    train_idx = np.setdiff1d(np.arange(len(y)), test_idx)
    return train_idx, test_idx


# -------------------------------------------------------- C — pré-processo

def categorical_array(df):
    """Categorias como texto ("True"/"False" para booleanos); ausentes continuam np.nan."""
    cols = [[np.nan if pd.isna(v) else str(v) for v in df[c]] for c in CATEGORICAL]
    return np.array(cols, dtype=object).T


class Preprocessor:
    """Imputação -> TotalSpend -> log(1 + x) -> padronização (numéricas) + one-hot (categóricas).
    Toda estatística é aprendida em fit() com o treino e reaproveitada em transform()."""

    def fit(self, df):
        # Numéricas: mediana do treino (robusta às caudas longas dos gastos)
        self.num_imputer = SimpleImputer(strategy="median").fit(df[NUMERIC])
        # Categóricas: moda do treino
        cat = categorical_array(df)
        self.cat_imputer = SimpleImputer(strategy="most_frequent").fit(cat)
        # One-hot com as categorias vistas no treino; categoria nova no teste -> vetor de zeros
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder.fit(self.cat_imputer.transform(cat))
        # Padronização: média e desvio de cada coluna numérica do treino (depois do log)
        self.scaler = StandardScaler().fit(self.numeric_block(df))
        return self

    def numeric_block(self, df):
        """Numéricas imputadas + TotalSpend + log(1 + x) nas colunas de gasto (antes do escalonamento)."""
        num = pd.DataFrame(self.num_imputer.transform(df[NUMERIC]), columns=NUMERIC, index=df.index)
        num["TotalSpend"] = num[SPEND].sum(axis=1)
        num[LOG_COLS] = np.log1p(num[LOG_COLS])
        return num

    def transform(self, df):
        num = self.scaler.transform(self.numeric_block(df))
        cat = self.encoder.transform(self.cat_imputer.transform(categorical_array(df)))
        return np.hstack([num, cat])

    def feature_names(self):
        return NUMERIC + ["TotalSpend"] + list(self.encoder.get_feature_names_out(CATEGORICAL))


def unseen_category_demo(prep, df):
    """Troca HomePlanet de uma linha por uma categoria que não existe no treino."""
    row = df.iloc[[0]].copy()
    row["HomePlanet"] = "Pluto"
    x = prep.transform(row)[0]
    return {name: float(v) for name, v in zip(prep.feature_names(), x) if name.startswith("HomePlanet_")}


# --------------------------------------------------------------- figuras

def _hist_by_class(ax, values, y, bins):
    for k in (0, 1):
        ax.hist(values[y == k], bins=bins, alpha=0.55, color=CLASS_COLORS[k], label=CLASS_LABELS[k])


def plot_log_spending(train_df, y, path):
    """Complemento do item C: as 5 colunas de gasto antes e depois de log(1 + x)."""
    fig, axes = plt.subplots(2, len(SPEND), figsize=(20, 7.5))
    for j, col in enumerate(SPEND):
        ok = train_df[col].notna().to_numpy()
        raw = train_df[col].to_numpy()[ok]
        for row, (values, xlabel) in enumerate(((raw, f"{col} (créditos)"),
                                                (np.log1p(raw), f"log(1 + {col})"))):
            ax = axes[row, j]
            _hist_by_class(ax, values, y[ok], bins=40)
            ax.set_title(f"{col} — {'antes' if row == 0 else 'depois'} de log(1 + x)")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("passageiros (treino)")
            ax.grid(alpha=0.3)
    axes[0, 0].legend()
    fig.suptitle("Figura complementar (item C) — Gastos no treino antes e depois de log(1 + x)")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_figure6(raw, logged, final, y, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = ((raw, "FoodCourt bruto (créditos)", "(a) antes: valor bruto"),
              (logged, "log(1 + FoodCourt)", "(b) depois de log(1 + x)"),
              (final, "FoodCourt final (padronizado, z)", "(c) final: log(1 + x) + padronização"))
    axes[2].axvspan(-1, 1, color="gray", alpha=0.12, label="faixa [-1, 1]")
    for ax, (values, xlabel, title) in zip(axes, panels):
        _hist_by_class(ax, values, y, bins=50)
        s = pd.Series(values)
        ax.text(0.97, 0.62, f"média = {s.mean():.2f}\nmediana = {s.median():.2f}\n"
                            f"máximo = {s.max():.2f}\nassimetria = {s.skew():.2f}",
                transform=ax.transAxes, ha="right", va="top",
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9})
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("passageiros (treino)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Figura 6 — FoodCourt no treino antes e depois do pré-processamento")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ roteiro

def run(rng):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    # A — conhecer os dados
    res_a = describe(df)

    # B — divide ANTES de qualquer estatística de transformação
    y = df[TARGET].astype(int).to_numpy()
    train_idx, test_idx = stratified_split(y, rng, test_size=0.2)
    features = df.drop(columns=DROP + [TARGET])  # Cabin, Name e PassengerId saem aqui
    train_df, test_df = features.iloc[train_idx], features.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    res_b = {
        "n_train": len(train_idx), "n_test": len(test_idx),
        "positive_share_train": float(y_train.mean()), "positive_share_test": float(y_test.mean()),
        "overlap": int(np.intersect1d(train_idx, test_idx).size),
        "foodcourt_train_mean": float(train_df["FoodCourt"].mean()),
        "foodcourt_train_median": float(train_df["FoodCourt"].median()),
    }

    # C — ajusta no treino, aplica em treino e teste
    prep = Preprocessor().fit(train_df)
    X_train, X_test = prep.transform(train_df), prep.transform(test_df)
    names = prep.feature_names()
    n_num = len(NUMERIC) + 1
    logged_train = prep.numeric_block(train_df)
    imputed_train = pd.DataFrame(prep.num_imputer.transform(train_df[NUMERIC]), columns=NUMERIC)
    imputed_train["TotalSpend"] = imputed_train[SPEND].sum(axis=1)

    # Alternativa descartada: Min-Max para [-1, 1] nas colunas log (só para justificar a escolha)
    lg = logged_train[LOG_COLS]
    minmax = 2 * (lg - lg.min()) / (lg.max() - lg.min()) - 1
    # O que aconteceria sem o log: FoodCourt padronizado direto do valor bruto
    fc = imputed_train["FoodCourt"]
    z_raw = (fc - fc.mean()) / fc.std(ddof=0)
    z_log = X_train[:, names.index("FoodCourt")]

    res_c = {
        "imputed_counts_train": {c: int(train_df[c].isna().sum()) for c in NUMERIC + CATEGORICAL},
        "imputed_counts_test": {c: int(test_df[c].isna().sum()) for c in NUMERIC + CATEGORICAL},
        "medians_train": dict(zip(NUMERIC, map(float, prep.num_imputer.statistics_))),
        "modes_train": dict(zip(CATEGORICAL, map(str, prep.cat_imputer.statistics_))),
        "categories": {c: list(map(str, cats)) for c, cats in zip(CATEGORICAL, prep.encoder.categories_)},
        "unseen_category_demo": unseen_category_demo(prep, test_df),
        "total_spend_train": {"mean": float(imputed_train["TotalSpend"].mean()),
                              "median": float(imputed_train["TotalSpend"].median()),
                              "max": float(imputed_train["TotalSpend"].max())},
        "skew_before_log": {c: float(imputed_train[c].skew()) for c in LOG_COLS},
        "skew_after_log": {c: float(logged_train[c].skew()) for c in LOG_COLS},
        "scaler_mean": dict(zip(prep.scaler.feature_names_in_, map(float, prep.scaler.mean_))),
        "scaler_std": dict(zip(prep.scaler.feature_names_in_, map(float, prep.scaler.scale_))),
        "minmax_alternative_mean": {c: float(minmax[c].mean()) for c in LOG_COLS},
        "minmax_alternative_share_at_minus1": {c: float((minmax[c] == -1).mean()) for c in LOG_COLS},
        "foodcourt_z_without_log_max": float(z_raw.max()),
        "foodcourt_z_with_log_max": float(z_log.max()),
        "foodcourt_std_without_log": float(fc.std(ddof=0)),
        "foodcourt_z_of_zero_without_log": float(-fc.mean() / fc.std(ddof=0)),
        "numeric_min_max_train": {n: [float(X_train[:, i].min()), float(X_train[:, i].max())]
                                  for i, n in enumerate(names[:n_num])},
        "numeric_min_max_test": {n: [float(X_test[:, i].min()), float(X_test[:, i].max())]
                                 for i, n in enumerate(names[:n_num])},
        "numeric_mean_std_train": {n: [float(X_train[:, i].mean()), float(X_train[:, i].std())]
                                   for i, n in enumerate(names[:n_num])},
    }

    plot_log_spending(train_df, y_train, FIG_DIR / "fig_c_log_spending.png")
    plot_figure6(train_df["FoodCourt"].dropna().to_numpy(), np.log1p(train_df["FoodCourt"].dropna().to_numpy()),
                 z_log[train_df["FoodCourt"].notna().to_numpy()], y_train[train_df["FoodCourt"].notna().to_numpy()],
                 FIG_DIR / "fig6_foodcourt.png")

    # D — verificações finais
    res_d = {
        "feature_names": names,
        "shape_train": list(X_train.shape), "shape_test": list(X_test.shape),
        "nan_train": int(np.isnan(X_train).sum()), "nan_test": int(np.isnan(X_test).sum()),
        "min_max_train": [float(X_train.min()), float(X_train.max())],
        "min_max_test": [float(X_test.min()), float(X_test.max())],
        "numeric_min_max_train_all": [float(X_train[:, :n_num].min()), float(X_train[:, :n_num].max())],
        "numeric_min_max_test_all": [float(X_test[:, :n_num].min()), float(X_test[:, :n_num].max())],
        "share_numeric_within_3sd_train": float((np.abs(X_train[:, :n_num]) <= 3).mean()),
        "share_numeric_within_3sd_test": float((np.abs(X_test[:, :n_num]) <= 3).mean()),
    }
    return {"A": res_a, "B": res_b, "C": res_c, "D": res_d}
