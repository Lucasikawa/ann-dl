# Redes Neurais e Deep Learning — 2026.2

**Lucas Ikawa** — Ciência da Computação, Insper.

Este site é o portfólio das entregas da disciplina
[Artificial Neural Networks and Deep Learning](https://insper.github.io/ann-dl/){:target="_blank"}.
Cada entrega fica em `docs/exercises/<slug>/`, com o relatório (`index.md`),
o código que foi de fato executado (`code/`) e as figuras (`figures/`).

## Entregas

| Entrega | Status | Relatório |
|---------|--------|-----------|
| 1. Data | entregue | [Preparação e análise de dados](exercises/data/index.md) |
| 2. Perceptron | a fazer | — |
| 3. MLP | a fazer | — |
| 4. VAE | a fazer | — |
| Projetos | a fazer | — |

## Como reproduzir

```bash
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
python3 docs/exercises/data/code/main.py   # regenera figuras e números da entrega 1
mkdocs serve -o                            # visualiza o site localmente
```
