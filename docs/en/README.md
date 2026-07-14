# Mini-LLM: Decoder-Only Transformer

A complete, trainable **Decoder-Only Transformer built from scratch** – as an interactive learning environment where you can modify hyperparameters and observe their effects live.

---

## Quick Start

```bash
# 1. Install dependencies (once)
uv sync --python 3.12

# 2. Start training
uv run python train.py
```

Training runs on the CPU and prints the current loss plus a generated text snippet every `eval_interval` iterations.

---

## Project Structure

```
.
├── pyproject.toml          # Project configuration & dependencies (uv)
├── model.py                # Transformer architecture (commented)
├── train.py                # Training loop + EXPERIMENTATION CENTRE
├── ARCHITECTURE.md         # Detailed architecture documentation (with diagrams)
├── USAGE.md                # User guide & experiments
└── data/
    └── training_text.txt   # German training text (~5,300 characters)
```

---

## Documentation

| Document | Contents |
|---|---|
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Detailed explanation of the entire Transformer architecture with Mermaid diagrams – ideal for beginners |
| **[USAGE.md](USAGE.md)** | User guide, hyperparameter tips, and experiments |
