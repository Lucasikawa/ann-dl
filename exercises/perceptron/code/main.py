"""Ponto de entrada da atividade 2 (Perceptron).

Roda os dois exercícios com UM único gerador aleatório (seed 42), gera as
figuras em ../figures/ e grava todos os números em results.json.

Uso, a partir da raiz do repositório:
    python docs/exercises/perceptron/code/main.py
"""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só salva arquivos, não abre janelas

import numpy as np  # noqa: E402

import ex1_separable  # noqa: E402
import ex2_overlap  # noqa: E402


def to_builtin(obj):
    """Converte tipos do NumPy em tipos nativos para o JSON."""
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_builtin(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main():
    t0 = time.time()
    # o mesmo rng é usado no relatório inteiro, na ordem 1 -> 2: dados do Ex. 1, w inicial do Ex. 1,
    # dados do Ex. 2, w inicial do Ex. 2. As execuções extras reaproveitam esses pesos iniciais.
    rng = np.random.default_rng(42)
    results = {
        "exercise_1": ex1_separable.run(rng),
        "exercise_2": ex2_overlap.run(rng),
    }
    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(to_builtin(results), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resultados gravados em {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
