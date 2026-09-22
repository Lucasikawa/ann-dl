# Redes Neurais e Deep Learning — Lucas Ikawa (Insper, 2026.2)

Portfólio das entregas da disciplina, publicado com MkDocs + Material no GitHub Pages:
<https://lucasikawa.github.io/ann-dl/>

## Estrutura

```
docs/
  index.md                    # página inicial
  exercises/
    data/
      index.md                # relatório da entrega 1
      code/                   # scripts executados (main.py roda tudo)
      figures/                # figuras exibidas no relatório
    perceptron/
      index.md                # relatório da entrega 2
      code/                   # perceptron.py (do zero), dados, exercícios, main.py
      figures/                # figuras 1 a 6 e figuras complementares
mkdocs.yml
requirements.txt
```

## Setup

```shell
python3 -m venv env
source ./env/bin/activate
python3 -m pip install -r requirements.txt --upgrade
```

## Reproduzir a entrega 1 (Data)

```shell
python3 docs/exercises/data/code/main.py
```

O script usa um único `np.random.default_rng(42)` para os três exercícios,
regrava as figuras em `docs/exercises/data/figures/` e os números em
`docs/exercises/data/code/results.json`.

## Reproduzir a entrega 2 (Perceptron)

```shell
python3 docs/exercises/perceptron/code/main.py
```

O script usa um único `np.random.default_rng(42)` para os dois exercícios,
regrava as figuras em `docs/exercises/perceptron/figures/` e os números em
`docs/exercises/perceptron/code/results.json` (cerca de 10 s). O perceptron
está em `docs/exercises/perceptron/code/perceptron.py`, escrito só com NumPy.

## Site

```shell
mkdocs serve -o      # local
```

O workflow `.github/workflows/main.yaml` publica o site a cada push na `main`
(`mkdocs gh-deploy` para a branch `gh-pages`).
