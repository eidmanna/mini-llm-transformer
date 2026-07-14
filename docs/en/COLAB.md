# Mini-LLM Transformer – Google Colab Guide

This guide shows how to run the project directly in Google Colab – without any local installation.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Setting Up the Notebook](#2-setting-up-the-notebook)
3. [Cloning the Repository](#3-cloning-the-repository)
4. [Starting Training](#4-starting-training)
5. [Using the Model](#5-using-the-model)
6. [Saving the Checkpoint](#6-saving-the-checkpoint)
7. [Tips for Colab](#7-tips-for-colab)

---

## 1. Prerequisites

- Google account (free)
- Browser (Chrome, Firefox, Safari)
- No local Python or PyTorch needed – everything runs in the cloud

> **GPU recommended:** Under *Runtime → Change runtime type → T4 GPU*. Training is approximately 10× faster than on CPU.

---

## 2. Setting Up the Notebook

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Click **New notebook**
3. Optionally select a GPU runtime:
   - Menu → **Runtime** → **Change runtime type** → **T4 GPU** → Save

---

## 3. Cloning the Repository

Add these cells to the notebook one after the other and run them:

### Cell 1 – Navigate to the content folder and clone

```python
%cd /content/
!git clone https://github.com/eidmanna/mini-llm-transformer.git
```

### Cell 2 – Switch to the cloned directory

```python
%cd /content/mini-llm-transformer
```

> **Note:** `%cd` is an IPython magic command that permanently changes the working directory for all following cells. `!` runs shell commands once.

---

## 4. Starting Training

### Cell 3 – Install dependencies

Colab already has PyTorch pre-installed. `tqdm` is also available. A separate installation step is therefore usually not necessary. If an import error occurs:

```python
!pip install tqdm -q
```

### Cell 4 – Run training

```python
!python train.py
```

The script runs through and prints an intermediate report with the current loss and a generated text sample every `eval_interval` iterations:

```
═════════════════════════════════════════════════════════════════
  Mini-Transformer – Learning Experiment
═════════════════════════════════════════════════════════════════
  Device         : cuda      ← GPU active
  Total chars    : 5,306
  Vocabulary size: 68 unique characters
  ...

─────────────────────────────────────────────────────────────────
  Iter   250/5000  ( 5.0%)  Time: 4s
  Train-Loss: 2.8134  |  Val-Loss: 2.9021  |  LR: 9.50e-04

  ▶ Generated text:
  'Der Wald ist ein wich...'
```

### Typical runtimes in Colab

| Runtime | `max_iters = 5000` |
|---|---|
| CPU (free) | ~8–12 minutes |
| T4 GPU (free) | ~1–2 minutes |

---

## 5. Using the Model

After training, `model_checkpoint.pt` is located in the project folder. Text generation:

### Cell 5 – Generate text

```python
!python generate.py
```

With custom parameters:

```python
!python generate.py --start "Die Wissenschaft" --tokens 400 --temperature 0.7
```

| Option | Default | Description |
|---|---|---|
| `--start` | `"Der"` | Seed text for generation |
| `--tokens` | `200` | Number of tokens to generate |
| `--temperature` | `0.8` | `< 1.0` focused · `> 1.0` creative |
| `--top_k` | `40` | Only the k most probable candidates |

---

## 6. Saving the Checkpoint

Colab deletes all files when the session ends. Save the checkpoint beforehand:

### Option A – Download via Colab UI

```python
from google.colab import files
files.download("model_checkpoint.pt")
```

### Option B – Save to Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")

!cp model_checkpoint.pt /content/drive/MyDrive/mini-llm-checkpoint.pt
print("Checkpoint saved.")
```

To reload in a new session:

```python
from google.colab import drive
drive.mount("/content/drive")

%cd /content/
!git clone https://github.com/eidmanna/mini-llm-transformer.git
%cd /content/mini-llm-transformer

!cp /content/drive/MyDrive/mini-llm-checkpoint.pt model_checkpoint.pt
!python generate.py
```

---

## 7. Tips for Colab

| Topic | Recommendation |
|---|---|
| **Use GPU** | Set runtime type to *T4 GPU* – significantly faster than CPU |
| **Session timeout** | Colab disconnects after ~90 min of inactivity – save checkpoint to Drive beforehand |
| **Hyperparameters** | Edit `train.py` directly in Colab: double-click the file in the file browser on the left |
| **More data** | `fetch_wikipedia.py` also works in Colab – simply call it with `!python fetch_wikipedia.py` |
| **Scroll output** | For long training runs: right-click the cell → *Clear output* to reduce browser load |
