# sparse-experiment

[MovieLens 100K Dataset](https://grouplens.org/datasets/movielens/)を使ったスパース性が推薦に与える影響を調べる実験

## Requirements

- Python3.8~
- [Poetry](https://github.com/python-poetry/poetry)

## Get started

```sh
$ wget -P dataset https://files.grouplens.org/datasets/movielens/ml-100k.zip
$ unzip -d dataset dataset/ml-100k.zip
$ poetry install
$ poetry run python main.py
```
