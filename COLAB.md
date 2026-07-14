# Mini-LLM Transformer – Google Colab Anleitung

Diese Anleitung zeigt, wie du das Projekt direkt in Google Colab ausführst – ohne lokale Installation.

---

## Inhaltsverzeichnis

1. [Voraussetzungen](#1-voraussetzungen)
2. [Notebook einrichten](#2-notebook-einrichten)
3. [Repository klonen](#3-repository-klonen)
4. [Training starten](#4-training-starten)
5. [Modell verwenden](#5-modell-verwenden)
6. [Checkpoint speichern](#6-checkpoint-speichern)
7. [Tipps für Colab](#7-tipps-für-colab)

---

## 1. Voraussetzungen

- Google-Konto (kostenlos)
- Browser (Chrome, Firefox, Safari)
- Kein Python oder PyTorch lokal nötig – alles läuft in der Cloud

> **GPU empfohlen:** Unter *Laufzeit → Laufzeittyp ändern → T4 GPU* auswählen. Das Training ist damit ca. 10× schneller als auf der CPU.

---

## 2. Notebook einrichten

1. Öffne [colab.research.google.com](https://colab.research.google.com)
2. Klicke auf **Neue Notebook**
3. Wähle optional eine GPU-Laufzeit:
   - Menü → **Laufzeit** → **Laufzeittyp ändern** → **T4 GPU** → Speichern

---

## 3. Repository klonen

Füge diese Zellen nacheinander ins Notebook ein und führe sie aus:

### Zelle 1 – In den Content-Ordner wechseln und klonen

```python
%cd /content/
!git clone https://github.com/eidmanna/mini-llm-transformer.git
```

### Zelle 2 – In das geklonte Verzeichnis wechseln

```python
%cd /content/mini-llm-transformer
```

> **Hinweis:** `%cd` ist ein IPython-Magic-Befehl, der das Arbeitsverzeichnis für alle folgenden Zellen dauerhaft ändert. `!` führt Shell-Befehle einmalig aus.

---

## 4. Training starten

### Zelle 3 – Abhängigkeiten installieren

Colab hat PyTorch bereits vorinstalliert. `tqdm` ist ebenfalls vorhanden. Ein separater Installationsschritt ist daher in der Regel nicht nötig. Falls doch ein Import-Fehler auftritt:

```python
!pip install tqdm -q
```

### Zelle 4 – Training ausführen

```python
!python train.py
```

Das Skript läuft durch und gibt alle `eval_interval` Iterationen einen Zwischen-Report mit aktuellem Loss und einem generierten Textbeispiel aus:

```
═════════════════════════════════════════════════════════════════
  Mini-Transformer – Lernexperiment
═════════════════════════════════════════════════════════════════
  Gerät          : cuda      ← GPU aktiv
  Zeichen gesamt : 5.306
  Vokabular-Größe: 68 eindeutige Zeichen
  ...

─────────────────────────────────────────────────────────────────
  Iter   250/5000  ( 5.0%)  Zeit: 4s
  Train-Loss: 2.8134  |  Val-Loss: 2.9021  |  LR: 9.50e-04

  ▶ Generierter Text:
  'Der Wald ist ein wich...'
```

### Typische Laufzeiten in Colab

| Laufzeit | `max_iters = 5000` |
|---|---|
| CPU (kostenlos) | ~8–12 Minuten |
| T4 GPU (kostenlos) | ~1–2 Minuten |

---

## 5. Modell verwenden

Nach dem Training liegt `model_checkpoint.pt` im Projektordner. Textgenerierung:

### Zelle 5 – Text generieren

```python
!python generate.py
```

Mit eigenen Parametern:

```python
!python generate.py --start "Die Wissenschaft" --tokens 400 --temperature 0.7
```

| Option | Standard | Beschreibung |
|---|---|---|
| `--start` | `"Der"` | Starttext für die Generierung |
| `--tokens` | `200` | Anzahl zu generierender Tokens |
| `--temperature` | `0.8` | `< 1.0` fokussiert · `> 1.0` kreativ |
| `--top_k` | `40` | Nur die k wahrscheinlichsten Kandidaten |

---

## 6. Checkpoint speichern

Colab löscht alle Dateien, wenn die Sitzung endet. Den Checkpoint vorher sichern:

### Option A – Download über Colab-UI

```python
from google.colab import files
files.download("model_checkpoint.pt")
```

### Option B – In Google Drive speichern

```python
from google.colab import drive
drive.mount("/content/drive")

!cp model_checkpoint.pt /content/drive/MyDrive/mini-llm-checkpoint.pt
print("Checkpoint gespeichert.")
```

Zum Wiederladen in einer neuen Sitzung:

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

## 7. Tipps für Colab

| Thema | Empfehlung |
|---|---|
| **GPU nutzen** | Laufzeittyp auf *T4 GPU* setzen – deutlich schnellerer als CPU |
| **Sitzungsabbruch** | Colab trennt nach ~90 min Inaktivität – Checkpoint vorher in Drive sichern |
| **Hyperparameter** | `train.py` direkt in Colab editieren: Doppelklick auf die Datei im Datei-Browser links |
| **Mehr Daten** | `fetch_wikipedia.py` funktioniert auch in Colab – einfach mit `!python fetch_wikipedia.py` aufrufen |
| **Ausgabe scrollen** | Bei langen Trainingsläufen: Rechtsklick auf die Zelle → *Ausgabe löschen* entlastet den Browser |
