# Mini-LLM: Decoder-Only Transformer

Ein vollständiger, lernbarer **Decoder-Only Transformer von Grund auf** – als interaktive Lernumgebung, mit der du Hyperparameter verändern und deren Wirkung live beobachten kannst.

---

## Schnellstart

```bash
# 1. Abhängigkeiten installieren (einmalig)
uv sync --python 3.12

# 2. Training starten
uv run python train.py
```

Das Training läuft auf der CPU und gibt alle `eval_interval` Iterationen den aktuellen Loss sowie einen generierten Textschnipsel aus.

---

## Projektstruktur

```
.
├── pyproject.toml          # Projekt-Konfiguration & Abhängigkeiten (uv)
├── model.py                # Transformer-Architektur (kommentiert)
├── train.py                # Trainings-Schleife + EXPERIMENTIER-ZENTRALE
├── ARCHITECTURE.md         # Ausführliche Architektur-Dokumentation (mit Diagrammen)
├── USAGE.md                # Bedienungsanleitung & Experimente
├── docs/
│   ├── en/
│   │   ├── COLAB.md        # Google Colab — GPU-Training in der Cloud
│   │   ├── ARCHITECTURE.md # Architektur-Dokumentation (English)
│   │   └── USAGE.md        # Usage guide (English)
│   └── explainers/
│       ├── math-basics.md  # Vektor · Matrix · Tensor — Crashkurs
│       └── attention-head.md  # Scaled Dot-Product Attention erklärt
└── data/
    └── training_text.txt   # Deutschsprachiger Trainingstext (~5.300 Zeichen)
```

---

## Dokumentation

| Dokument | Inhalt |
|---|---|
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Ausführliche Erklärung der gesamten Transformer-Architektur mit Mermaid-Diagrammen – ideal für Einsteiger |
| **[USAGE.md](USAGE.md)** | Bedienungsanleitung, Hyperparameter-Tipps und Experimente |
| **[COLAB.md](docs/en/COLAB.md)** | Google Colab – Modell auf GPU trainieren, ohne lokale Installation |

### Explainer-Reihe

| Dokument | Inhalt |
|---|---|
| **[how-it-works.md](docs/explainers/how-it-works.md)** | Wie der gesamte Transformer funktioniert — Bibliotheks-Analogie + Glossar aller Fachbegriffe |
| **[math-basics.md](docs/explainers/math-basics.md)** | Vektor · Matrix · Tensor — visueller Crashkurs (keine Vorkenntnisse nötig) |
| **[attention-head.md](docs/explainers/attention-head.md)** | Scaled Dot-Product Attention — `class Head` Schritt für Schritt erklärt |
