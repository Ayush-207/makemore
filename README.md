# makemore — character-level language models from scratch

Building a character-level language model step by step, from a counting-based bigram model up to a WaveNet-style hierarchical network, in PyTorch. Trained on ~32K human names (`names.txt`), the model learns to generate new, name-like strings.

Inspired by Andrej Karpathy's *Neural Networks: Zero to Hero* series.

## Notebooks (in order)

| # | Notebook | What it covers | Result |
|---|----------|----------------|--------|
| 1 | [`makemore.ipynb`](makemore.ipynb) | Bigram model, first by counting then as a single-layer neural net trained with gradient descent; negative log-likelihood loss | loss ≈ 2.48 |
| 2 | [`build_makemore_mlp.ipynb`](build_makemore_mlp.ipynb) | MLP with character embeddings (3-char context); fixing initialization (softmax overconfidence, saturated tanh units), Kaiming init, batch normalization | val 2.106 |
| 3 | [`build_makemore_mlp2.ipynb`](build_makemore_mlp2.ipynb) | Rewriting the network as PyTorch-like modules (`Linear`, `BatchNorm1D`, `Tanh`); a deeper 5-layer MLP; checking activation, gradient and update-to-data ratio statistics | — |
| 4 | [`makemore_backprop_part4.ipynb`](makemore_backprop_part4.ipynb) | Manual backprop through the whole network, without autograd: cross-entropy, batchnorm, tanh, linear and embedding layers, each gradient checked against PyTorch | gradients match |
| 5 | [`makemore_part5.ipynb`](makemore_part5.ipynb) | WaveNet-style hierarchical model: an 8-char context fused in pairs using `FlattenConsecutive`, `Embedding` and `Sequential` | **val 2.020** |

## Generated samples

From the final model:

```
kamorah  khani  stentar  jaslaun  fynner  celvina  raleah
jaylan  chantley  annalyze  gwindro  pennessa  blaylak
```

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch matplotlib jupyter
jupyter notebook
```
