# Wie funktioniert der llm-mini-transformer?

> **Ziel dieses Dokuments:** Das gesamte Modell in einem durchgängigen Bild erklären — von rohem Text bis zum generierten Satz — mit einer Analogie, die sich leicht einprägen lässt, und einem Glossar mit Lernhilfen für jedes Fachwort.

---

## Die durchgängige Analogie: die Bibliothek mit Lese-AGs

Stell dir vor, du bist **Bibliothekar** und möchtest automatisch den nächsten Satz eines Buches weiterschreiben. Dazu rufst du eine **Lese-AG** (das Modell).

Die AG arbeitet in fünf Stationen — **genau die fünf Stufen des Transformers**:

```
Rohtext → [Station 1] Übersetzer → [Station 2] Positionsschilder
        → [Station 3] Lese-Tische (Attention) → [Station 4] Notizen aufarbeiten (FFN)
        → [Station 5] Schätzung: Was kommt als Nächstes?
```

Diese fünf Stationen werden als **Schleife** mehrfach durchlaufen (= `n_layers` Transformer-Blöcke), bevor am Ende eine Antwort entsteht.

---

## Station 1 — Der Übersetzer (Token-Embedding)

**Was passiert?**  
Jedes Zeichen (oder Wort-Stück) wird in eine **Zahlen-Wolke** (Vektor) verwandelt.

**Bibliotheks-Analogie:**  
Jedes Wort bekommt eine **Karteikarte** mit 64–128 Zahlen drauf — wie ein Steckbrief, der beschreibt, welche Bedeutung das Wort hat, wie es sich anfühlt, ob es eher Nomen oder Verb ist usw. Anfangs sind die Zahlen zufällig; das Modell lernt im Training die sinnvollen Steckbriefe.

**Im Code:**
```python
tok_emb = self.token_embedding(idx)   # (B, T, n_embd)
```

| Fachwort | Lernhilfe |
|---|---|
| **Token** | Ein Stück Text — hier: einzelnes Zeichen oder Silbe. Denk an Legostein. |
| **Embedding** | Die Karteikarte eines Tokens: ein Vektor mit z.B. 96 Zahlen. |
| **Vokabular (`vocab_size`)** | Die Gesamtzahl aller möglichen Legostein-Typen. |
| **`n_embd`** | Wie viele Zahlen pro Karteikarte. Größer = mehr Ausdrucksvermögen. |

---

## Station 2 — Die Positionsschilder (Positions-Embedding)

**Was passiert?**  
Zu jeder Karteikarte wird noch eine zweite Karteikarte **addiert**, die nur die **Position** (Stelle 0, 1, 2, …) beschreibt.

**Bibliotheks-Analogie:**  
Alle Bücher in der Bibliothek haben denselben Grundsteckbrief für das Wort „Katze" — aber ein Buch, das auf **Seite 3** ein Wort hat, weiß durch das Positionsschild, dass es auf Seite 3 steht. Ohne dieses Schild würde das Modell keinen Unterschied machen, ob „Katze" am Anfang oder am Ende des Satzes steht.

```python
pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, n_embd)
x = tok_emb + pos_emb   # addiert: gleiche Form bleibt (B, T, n_embd)
```

| Fachwort | Lernhilfe |
|---|---|
| **Positions-Embedding** | Das Positionsschild. Sagt: „Ich bin das 5. Token in diesem Satz." |
| **`block_size`** | Maximale Satzlänge, die das Modell kennt. Wie viele Seiten ein Buch maximal hat. |

---

## Station 3 — Die Lese-Tische (Multi-Head Self-Attention)

Das ist **der Kern** des Transformers. Hier passiert die eigentliche Intelligenz.

### Die Idee in einem Satz

Jedes Token fragt alle **früheren** Tokens: *„Bist du gerade für mich relevant?"* — und mischt dann deren Inhalt proportional zur Relevanz zusammen.

### Bibliotheks-Analogie: Frage–Angebot–Inhalt

Jede Karteikarte bekommt **drei Rollen**:

| Rolle | Frage | Bibliotheks-Bild |
|---|---|---|
| **Query (Q)** | *Was suche ich gerade?* | Das Recherche-Zettelchen, das du in die Kartothek steckst. |
| **Key (K)** | *Was biete ich an?* | Das Beschriftungsschild eines Regals. |
| **Value (V)** | *Was steht wirklich drin?* | Der eigentliche Buch-Inhalt im Regal. |

**Ablauf:**
1. Dein Query-Zettel wird mit allen Key-Schildern verglichen → **Ähnlichkeits-Score**
2. Ähnlichste Regale bekommen hohe Gewichte (Softmax → Summe = 1)
3. Dein Output = gewichteter Durchschnitt aller Buch-Inhalte (Values)

```
Token i fragt:   q_i · k_0   k_1   k_2   k_3
                 ──────────────────────────────
Scores:           0.1    0.6   0.2   0.1   → nach Softmax: Gewichte
Output:           0.1·v_0 + 0.6·v_1 + 0.2·v_2 + 0.1·v_3
```

### Die Kausale Maske — kein Blick in die Zukunft

Token 2 darf **nicht** sehen, was Token 3 sagt (das würde schummeln beim Schreiben). Zukünftige Positionen werden auf $-\infty$ gesetzt, nach Softmax also 0:

```
Wer darf wen sehen? (✓ = ja, ✗ = geblockt)
         T0  T1  T2  T3
Token 0 [ ✓   ✗   ✗   ✗ ]
Token 1 [ ✓   ✓   ✗   ✗ ]
Token 2 [ ✓   ✓   ✓   ✗ ]
Token 3 [ ✓   ✓   ✓   ✓ ]
```

### Warum mehrere Heads?

**Multi-Head Attention** betreibt mehrere Lese-Tische *gleichzeitig* — jeder sucht nach etwas anderem:

- Head 1: grammatikalische Abhängigkeiten
- Head 2: semantische Ähnlichkeit
- Head 3: Abstand im Satz
- …

Alle Ergebnisse werden zusammengefügt und auf die ursprüngliche Größe projiziert.

**Im Code:**
```python
# In MultiHeadAttention.forward:
out = torch.cat([h(x) for h in self.heads], dim=-1)  # Alle Heads zusammenklappen
return self.dropout(self.proj(out))                    # Auf n_embd projizieren
```

| Fachwort | Lernhilfe |
|---|---|
| **Self-Attention** | „Self" = das Modell achtet auf sich selbst (seinen eigenen Satz), kein externes Wörterbuch. |
| **Scaled Dot-Product** | Skalarprodukt ÷ √head_size. Das Teilen verhindert, dass Zahlen zu groß werden und Softmax einfriert. |
| **Softmax** | Verwandelt beliebige Zahlen in Wahrscheinlichkeiten (alle ≥ 0, Summe = 1). Wie Abstimmen: hohe Zahlen gewinnen. |
| **Kausale Maske** | Der Sichtschutz: zukünftige Tokens werden versteckt. Autoregressive Eigenschaft. |
| **`n_heads`** | Anzahl der Lese-Tische. z.B. 6 parallele Heads. |
| **`head_size`** | Breite eines einzelnen Heads = `n_embd / n_heads`. |

---

## Station 4 — Notizen aufarbeiten (Feed-Forward-Netz)

**Was passiert?**  
Nach der Attention bekommt jedes Token **für sich allein** nochmal einen Denk-Schritt: ein kleines neuronales Netz, das die gemischten Informationen verarbeitet und komplexere Muster extrahiert.

**Bibliotheks-Analogie:**  
Nachdem du alle relevanten Bücher durchgesehen hast, setzt du dich hin und schreibst **eigene Notizen** daraus — du destillierst die Information. Das geschieht für jedes Wort einzeln, ohne nochmal andere Worte anzusehen.

```python
# FeedForward: 2-schichtiges MLP, inner 4× größer als n_embd
self.net = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),  # Aufweiten
    nn.ReLU(),                        # Nicht-Linearität
    nn.Linear(4 * n_embd, n_embd),  # Zusammenziehen
    nn.Dropout(dropout),
)
```

| Fachwort | Lernhilfe |
|---|---|
| **MLP / Feed-Forward** | Multi-Layer Perceptron. Klassisches neuronales Netz: Eingabe → Versteckt → Ausgabe. |
| **ReLU** | „Rectified Linear Unit" = `max(0, x)`. Negative Zahlen → 0, positive bleiben. Wie ein Filter: nur gute Signale durch. |
| **`4 × n_embd`** | Die innere Schicht ist bewusst 4× breiter — mehr Platz zum „Denken". |

---

## Station 3+4 zusammen: der Transformer-Block

Stationen 3 und 4 zusammen bilden **einen Transformer-Block**. Das Modell stapelt `n_layers` solcher Blöcke übereinander. Jeder Block verfeinert das Verständnis ein bisschen mehr.

Zwei wichtige Tricks machen das Stapeln stabil:

### Residual-Verbindung — „Vergiss nicht, was du weißt"

```python
x = x + self.sa(self.ln1(x))   # Attention-Ergebnis wird zum Eingang addiert
x = x + self.ff(self.ln2(x))   # FFN-Ergebnis wird zum Eingang addiert
```

**Analogie:** Du hast einen langen Brief geschrieben. Statt ihn komplett zu ersetzen, fügst du nur *Korrekturen am Rand* ein. Das Original bleibt erhalten; die Schicht ändert nur das, was nötig ist.

### Layer Normalization — „Gleiche Maßstäbe"

Vor jedem Block wird der Vektor normiert (Mittelwert → 0, Varianz → 1). Verhindert, dass einzelne Werte explodieren.

**Analogie:** Alle Schüler einer Klasse bekommen Noten auf der gleichen Skala (1–6), egal wie schwer der Test war.

| Fachwort | Lernhilfe |
|---|---|
| **Residual-Verbindung** | Abkürzungs-Kabel: Ergebnis = Eingang + kleiner Korrektur. Gradient fließt ungehindert zurück. |
| **LayerNorm** | Normierung pro Token-Vektor. Pre-Norm (vor der Schicht) ist stabiler als Post-Norm. |
| **`n_layers`** | Anzahl gestapelter Blöcke. Tiefer = mehr Abstraktionsebenen, aber langsamer. |

---

## Station 5 — Was kommt als Nächstes? (Language Model Head)

**Was passiert?**  
Der letzte Vektor jedes Tokens wird auf die Größe des Vokabulars projiziert. Daraus entstehen **Logits** — eine Zahl pro möglichem nächstem Token. Die größte Zahl gewinnt (mit etwas Zufall).

```python
x = self.ln_final(x)       # Letzte Normierung
logits = self.lm_head(x)   # (B, T, vocab_size) — ein Score pro möglichem Zeichen
```

**Bibliotheks-Analogie:**  
Am Ende schreibt der Bibliothekar für jedes Wort im Vokabular auf einen Zettel: *„Wie wahrscheinlich ist es, dass dieses Wort als Nächstes kommt?"* Das wahrscheinlichste Wort (oder ein zufällig gewähltes aus den Top-k) wird ausgegeben.

| Fachwort | Lernhilfe |
|---|---|
| **Logits** | Rohe, unnormierte Scores. Noch keine Wahrscheinlichkeiten — erst nach Softmax. |
| **Cross-Entropy-Loss** | Strafe beim Training: Wie weit liegt der vorhergesagte Token vom richtigen entfernt? |
| **Decoder-Only** | Das Modell schreibt nur vorwärts — kein separater Encoder für die Eingabe (wie bei GPT). |

---

## Das Gesamtbild auf einen Blick

```
Eingabe-Text:   "Der Mond"
                     │
    ┌────────────────▼────────────────┐
    │  Token-Embedding  (Station 1)   │  "Der" → [0.2, -0.1, 0.8, ...]
    │  + Positions-Embedding (Stat.2) │  + [0.0, 0.3, -0.1, ...]
    └────────────────┬────────────────┘
                     │  x: (B, T, n_embd)
         ┌───────────▼───────────┐
         │   Transformer-Block   │  ← n_layers mal wiederholen
         │   ┌─────────────────┐ │
         │   │  LayerNorm      │ │
         │   │  Multi-Head     │ │  ← Wer achtet auf wen?
         │   │  Attention      │ │
         │   └────────┬────────┘ │
         │   + Residual          │
         │   ┌─────────────────┐ │
         │   │  LayerNorm      │ │
         │   │  Feed-Forward   │ │  ← Pro Token denken
         │   └────────┬────────┘ │
         │   + Residual          │
         └───────────┬───────────┘
                     │
    ┌────────────────▼────────────────┐
    │  LayerNorm + LM Head (Stat. 5)  │  (B, T, vocab_size) = Logits
    └────────────────┬────────────────┘
                     │
    Nächstes Token: "scheint" ← Sampling mit Temperature + Top-k
```

---

## Training vs. Generierung

### Training — Fehler messen und korrigieren

Das Modell bekommt echten Text und muss vorhersagen, was das nächste Zeichen ist. Der Abstand zwischen Vorhersage und Wahrheit = **Loss**. Dieser Fehler wird durch das Netz zurückgespielt (**Backpropagation**) und die Gewichte werden ein kleines Stück korrigiert.

```
Eingabe: "Der Mo"   → Ziel: "n"
Modell sagt: "n" mit 5% Wahrscheinlichkeit   → hoher Loss
Modell sagt: "n" mit 85% Wahrscheinlichkeit  → niedriger Loss
```

**6.000 Iterationen × 32 Batches** = das Modell sieht ~192.000 Beispiele.

| Fachwort | Lernhilfe |
|---|---|
| **Backpropagation** | Rückwärtsrechnung: Fehler wird schichtweise zurückgeleitet wie ein Echo. |
| **Gradient** | Richtungsweiser für jede Gewichtszahl: In welche Richtung muss sie sich ändern? |
| **AdamW-Optimizer** | Schlaues Update-Verfahren: passt die Lernrate pro Gewicht individuell an. |
| **Gradient Clipping** | Sicherheits-Bremse: Wenn der Gradient zu groß wird, wird er gekappt. |
| **Learning Rate** | Schrittgröße beim Lernen. Zu groß → überschießen. Zu klein → ewig dauern. |
| **Dropout** | Beim Training werden zufällig Verbindungen gekappt. Verhindert, dass das Modell auf Auswendig-Lernen setzt. |
| **Überanpassung (Overfitting)** | Modell lernt den Trainingstext auswendig, aber kann nicht verallgemeinern. |

### Generierung — Token für Token

```python
for _ in range(max_new_tokens):
    logits, _ = self(idx_cond)          # Vorhersage
    logits = logits[:, -1, :] / temperature
    # Top-k: nur die k besten Kandidaten
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, 1)  # zufällig sampeln
    idx = torch.cat([idx, idx_next], dim=1) # ans Ende hängen
```

| Fachwort | Lernhilfe |
|---|---|
| **Autoregressive Generierung** | Das Modell schreibt Zeichen für Zeichen — jedes neue Token wird zum neuen Kontext. |
| **Temperature** | < 1.0 = konservativ/sicher, > 1.0 = kreativ/zufällig. Wie die Risikobereitschaft des Autors. |
| **Top-k Sampling** | Nur die k wahrscheinlichsten Kandidaten kommen in die Lostrommel. Vermeidet seltsame Außenseiter-Token. |

---

## Hyperparameter — die Stellschrauben

| Parameter | Was er steuert | Bibliotheks-Bild |
|---|---|---|
| `n_embd` | Breite des Modells (Karteikarten-Größe) | Wie viele Spalten eine Karteikarte hat |
| `n_heads` | Anzahl paralleler Lese-Tische | Wie viele AGs gleichzeitig arbeiten |
| `n_layers` | Tiefe des Modells (Wiederholungen) | Wie viele Überarbeitungsrunden |
| `block_size` | Maximale Kontextlänge | Wie viele vorherige Seiten das Modell kennt |
| `batch_size` | Parallele Trainings-Beispiele | Wie viele Bücher gleichzeitig auf dem Tisch liegen |
| `dropout` | Regularisierungs-Stärke | Wie oft der Bibliothekar absichtlich etwas ignoriert |

---

## Alles in einem Merksatz

> **Der Transformer liest den bisherigen Text, lässt jedes Wort entscheiden, auf welche anderen Wörter es achten soll (Attention), destilliert das Ergebnis durch ein kleines Netz (FFN), wiederholt das mehrfach (n_layers), und schätzt dann, welches Wort als nächstes am wahrscheinlichsten ist.**

---

## Weiterführende Explainer in dieser Reihe

| Dokument | Inhalt |
|---|---|
| [`math-basics.md`](math-basics.md) | Vektor · Matrix · Tensor — visueller Crashkurs |
| [`attention-head.md`](attention-head.md) | `class Head` — Scaled Dot-Product Attention Schritt für Schritt |
