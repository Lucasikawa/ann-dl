"""Ponto de entrada da atividade 4 (VAE).

Carrega o MNIST com a divisão do enunciado, treina o VAE com d = 16 (Exercício 1) e
com d = 2 (Exercício 2), gera as figuras em ../figures/ e grava os números em results.json.
Cada modelo é construído logo depois de torch.manual_seed(42). Roda na CPU, ~2 min.

Uso, a partir da raiz do repositório:
    python docs/exercises/vae/code/main.py
"""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só salva arquivos, não abre janelas

import torch  # noqa: E402

import ex1_mnist  # noqa: E402
import ex2_latent2d  # noqa: E402
from data import load_mnist  # noqa: E402


def to_builtin(obj):
    """Converte tensores e tipos do NumPy em tipos nativos para o JSON."""
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return to_builtin(obj.tolist())
    if hasattr(obj, "item"):          # escalares do NumPy
        return obj.item()
    return obj


def main():
    t0 = time.time()
    torch.set_num_threads(4)          # a ordem das somas em ponto flutuante depende do nº de threads
    data = load_mnist()
    model16, ex1 = ex1_mnist.run(data)
    print(f"[Ex1] d=16: teste rec {ex1['test_reconstruction']:.2f} + KL {ex1['test_kl']:.2f} "
          f"= {ex1['test_neg_elbo']:.2f} nats ({time.time() - t0:.0f} s)")
    ex2 = ex2_latent2d.run(data, model16)
    print(f"[Ex2] d=2: teste rec {ex2['test_reconstruction']:.2f} + KL {ex2['test_kl']:.2f} "
          f"= {ex2['test_neg_elbo']:.2f} nats ({time.time() - t0:.0f} s)")
    out = Path(__file__).resolve().parent / "results.json"
    out.write_text(json.dumps(to_builtin({"exercise_1": ex1, "exercise_2": ex2}), indent=2,
                              ensure_ascii=False), encoding="utf-8")
    print(f"Resultados gravados em {out} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
