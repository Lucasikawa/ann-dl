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
import ex2_overlapping  # noqa: E402


def to_builtin(obj):
    """Converte tipos do NumPy em tipos nativos para o JSON."""
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main():
    t0 = time.time()
    rng = np.random.default_rng(42)  # o mesmo rng no relatório inteiro, na ordem Ex. 1 -> Ex. 2
    results = {
        "exercise_1": ex1_separable.run(rng),
        "exercise_2": ex2_overlapping.run(rng),
    }
    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(to_builtin(results), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resultados gravados em {out} ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
