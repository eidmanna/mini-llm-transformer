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

Das Training läuft auf der CPU. Du siehst alle `eval_interval` Iterationen den aktuellen Loss und einen generierten Textschnipsel, der mit `"Der"` beginnt.

---

## Projektstruktur

```
.
├── pyproject.toml          # Projekt-Konfiguration & Abhängigkeiten (uv)
├── model.py                # Transformer-Architektur (kommentiert)
├── train.py                # Trainings-Schleife + EXPERIMENTIER-ZENTRALE
└── data/
    └── training_text.txt   # Deutschsprachiger Trainingstext (~5.300 Zeichen)
```

---

## Die Experimentier-Zentrale

Alle Hyperparameter befinden sich im `CONFIG`-Dictionary am Anfang von [`train.py`](train.py). Ändere die Werte und starte das Training neu.

### Kontextlänge: `block_size`
Wie viele Zeichen das Modell gleichzeitig als Kontext sieht.

| Wert | Effekt |
|------|--------|
| `32` | Sehr schnell, kurzer Kontext – lernt kurze Muster |
| `64` | **Standard** – guter Kompromiss |
| `128` | Längerer Kontext, aber langsamer |

### Batch-Größe: `batch_size`
Wie viele Textausschnitte gleichzeitig verarbeitet werden.

| Wert | Effekt |
|------|--------|
| `16` | Wenig RAM, rauschigere Gradienten |
| `32` | **Standard** |
| `64` | Stabilere Updates, mehr RAM |

### Trainings-Dauer: `max_iters`
Gesamtzahl der Trainingsschritte. Auf Intel-Mac-CPU:

| Wert | Dauer |
|------|-------|
| `3000` | ~5 Minuten |
| `5000` | ~8–10 Minuten |

### Modell-Größe

```python
"n_embd":   64,   # Embedding-Dimension (Breite)
"n_heads":  4,    # Attention-Heads (n_embd muss teilbar sein!)
"n_layers": 4,    # Anzahl Transformer-Blöcke (Tiefe)
```

> **Wichtig:** `n_embd` muss immer durch `n_heads` teilbar sein.

### Lernrate & Scheduler

```python
"learning_rate":    1e-3,   # Startwert
"use_lr_scheduler": True,   # Lineare Abnahme bis 10% des Startwertes
```

### Zwischen-Generierung

```python
"eval_interval":   250,    # Alle X Iterationen: Loss + Textbeispiel
"gen_start_text":  "Der",  # Startwort für die Generierung
"gen_temperature": 0.8,    # < 1.0 schärfer | > 1.0 kreativer
"gen_top_k":       40,     # Nur die k besten Kandidaten
```

---

## Beobachtbare Lernphasen

| Loss-Bereich | Was du im generierten Text siehst |
|---|---|
| ~4.2 | Reiner Buchstabensalat, keine Muster |
| ~3.5 | Häufige Zeichen (Leerzeichen, `e`, `n`) tauchen auf |
| ~2.5 | Wortähnliche Strukturen, gelegentlich echte Wörter |
| ~2.0 | Kurze deutsche Wörter, einfache Wortfolgen |
| ~1.5 | Echte Wörter überwiegen, etwas Grammatik |

---

## Architektur-Überblick

```
Text → Character-Tokenizer → Token-IDs
                                  ↓
                    Token-Embedding  (n_embd Dimensionen)
                  + Position-Embedding (position 0…block_size-1)
                                  ↓
                    ┌─── N × Transformer-Block ───┐
                    │  LayerNorm                  │
                    │  → Masked Multi-Head Attention  (kausale Maske)
                    │  LayerNorm                  │
                    │  → Feed-Forward-Netz (4×n_embd, ReLU)  │
                    └─────────────────────────────┘
                                  ↓
                    LayerNorm → Linear → Logits (vocab_size)
                                  ↓
                    Cross-Entropy Loss / Softmax-Sampling
```

**Kausal** bedeutet: Position `i` kann nur auf Positionen `0…i` schauen – nie in die Zukunft (Decoder-Only / GPT-Stil).

---

## Tipps für Experimente

1. **Starte klein:** `n_embd=32`, `n_layers=2` – verstehe die Ausgabe, dann skaliere hoch.
2. **Beobachte den Val-Loss:** Wenn `val_loss >> train_loss`, überanpasst das Modell → `dropout` erhöhen.
3. **Lernrate anpassen:** Bei plateau → `learning_rate` halbieren.
4. **Mehr Daten:** Ersetze `data/training_text.txt` durch einen längeren deutschen Text für bessere Ergebnisse.
5. **Temperature spielen:** Nach dem Training ändere `gen_temperature` von `0.2` bis `1.5` und sieh den Effekt.
