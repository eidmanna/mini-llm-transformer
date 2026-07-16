# Neuronale Netze — was sie sind und was sie tun

> **Ziel dieses Dokuments:** Verstehen, was ein neuronales Netz grundsätzlich ist — und dann direkt nachvollziehen, wo und warum genau diese Konzepte in [`model.py`](../../model.py) des MiniTransformers stecken.

---

## Was ist ein neuronales Netz überhaupt?

Ein neuronales Netz ist im Kern eine **Funktion, die Zahlen in Zahlen umwandelt** — und die Gewichte dieser Funktion werden durch Training automatisch angepasst.

Die Grundidee in drei Schritten:

1. **Eingabe** → Zahlenvektor (z. B. ein Token-Vektor mit 96 Zahlen)
2. **Transformation** → multipliziere mit einer lernbaren Gewichtsmatrix, addiere optional einen Bias-Vektor, wende optional eine Aktivierungsfunktion an
3. **Ausgabe** → neuer Zahlenvektor (z. B. ein Score pro möglichem nächstem Token)

Das klingt simpel. Die Stärke kommt davon, dass man viele solcher Transformationsschritte hintereinander schaltet — und die Gewichte durch Fehlerrückrechnung (Backpropagation) so justiert, dass das Netz immer bessere Vorhersagen macht.

---

## Die drei Grundbausteine

### 1. Lineare Schicht — `nn.Linear`

```
Eingabe x  →  Ausgabe y = x · W + b
```

- `W` ist eine **Gewichtsmatrix** (lernbar)
- `b` ist ein **Bias-Vektor** (optional, lernbar)
- Das ist eine reine Matrixmultiplikation — keine Kurven, keine Sprünge, nur eine gerade Transformation

**Warum braucht man das?** Eine lineare Schicht kann jede beliebige lineare Beziehung zwischen Eingang und Ausgang lernen. Sie dreht, streckt und verschiebt den Eingangsvektor in einen anderen Raum.

**Limitation:** Mehrere lineare Schichten hintereinander sind immer noch linear — man könnte sie zu einer einzigen zusammenfassen. Um komplexere, nichtlineare Muster zu lernen, braucht es Aktivierungsfunktionen.

---

### 2. Aktivierungsfunktion — `nn.ReLU`

```
Eingabe x  →  Ausgabe y = max(0, x)
```

ReLU (Rectified Linear Unit) ist die einfachste und am häufigsten verwendete Aktivierungsfunktion:

- Negative Werte werden auf **0 gesetzt**
- Positive Werte bleiben **unverändert**

```
Eingang:  [-3,  0,  2, -1,  5]
Ausgang:  [ 0,  0,  2,  0,  5]
```

**Warum braucht man das?** Erst durch ReLU kann ein Netz nichtlineare Zusammenhänge modellieren. Ohne Aktivierungsfunktion wäre ein 10-schichtiges Netz mathematisch identisch zu einer einzigen linearen Schicht — egal wie viele Ebenen man stapelt.

---

### 3. Gewichts-Tabelle — `nn.Embedding`

```
Eingabe: Ganzzahl (Token-ID)  →  Ausgabe: Zahlenvektor (lernbar)
```

Ein Embedding ist eine spezielle Form des Lernens: statt eine Funktion zu berechnen, schlägt das Netz einfach in einer **Tabelle** nach. Zeile 42 der Tabelle ist der Vektor für Token 42.

- Im Training wird nur die Zeile der aktuell verwendeten Token-ID aktualisiert
- Alle anderen Zeilen bleiben unangetastet

---

## Wie lernt ein neuronales Netz?

Das Training läuft in einer Schleife:

```
1. Forward Pass  — Eingabe durch alle Schichten schicken → Vorhersage
2. Loss          — Wie falsch ist die Vorhersage? (Cross-Entropy)
3. Backward Pass — Fehler rückwärts durch alle Schichten rechnen (Backpropagation)
4. Update        — Alle Gewichte ein kleines Stück in die richtige Richtung verschieben (AdamW)
```

Der Schlüssel ist Schritt 3: PyTorch berechnet für jedes Gewicht automatisch, ob es erhöht oder verringert werden soll, um den Fehler zu reduzieren. Das funktioniert, weil alle Operationen differenzierbar sind — man kann berechnen, wie stark sich der Loss ändert, wenn man ein bestimmtes Gewicht leicht verändert.

---

## Neuronale Netze im MiniTransformer

Der MiniTransformer in [`model.py`](../../model.py) besteht vollständig aus PyTorch-NN-Bausteinen. Hier ist jeder Einsatzort — und warum dort ein neuronales Netz die richtige Wahl ist:

---

### 1. Token-Embedding — [`model.py:157`](../../model.py:157)

```python
self.token_embedding = nn.Embedding(vocab_size, n_embd)
```

**Was passiert:** Jede Token-ID (z. B. `42` für den Buchstaben `"H"`) wird in einen Vektor mit `n_embd` Zahlen übersetzt.

**Warum ein neuronales Netz?**  
Das Modell soll selbst lernen, welche Bedeutung jeder Token hat. Am Anfang sind alle Vektoren zufällig. Nach dem Training stehen ähnliche Tokens (z. B. `"a"` und `"e"`) in ähnlichen Regionen des Vektorraums. Diese Bedeutungsstruktur kann nicht vorgegeben werden — das Netz muss sie aus dem Trainingstext ableiten.

> 🔗 Bibliotheks-Analogie: „Karteikarte eines Wortes" → [how-it-works.md Station 1](how-it-works.md#station-1--der-übersetzer-token-embedding)

---

### 2. Positions-Embedding — [`model.py:158`](../../model.py:158)

```python
self.position_embedding = nn.Embedding(block_size, n_embd)
```

**Was passiert:** Für jede Position im Satz (0, 1, 2, … bis `block_size − 1`) gibt es einen eigenen lernbaren Vektor. Dieser wird zum Token-Vektor **addiert**.

**Warum ein neuronales Netz?**  
Der Attention-Mechanismus ist von sich aus positionsblind — er sieht nur, welche Token es gibt, nicht wo sie stehen. Das Positions-Embedding injiziert die Reihenfolge als lernbares Signal. Das Netz lernt dann selbst, was „an Position 3 stehen" bedeutet — z. B. dass das dritte Wort eines Satzes oft ein Verb ist.

> 🔗 Bibliotheks-Analogie: „Positionsschild auf der Karteikarte" → [how-it-works.md Station 2](how-it-works.md#station-2--die-positionsschilder-positions-embedding)

---

### 3. Query, Key, Value — [`model.py:37-39`](../../model.py:37)

```python
self.key   = nn.Linear(n_embd, head_size, bias=False)
self.query = nn.Linear(n_embd, head_size, bias=False)
self.value = nn.Linear(n_embd, head_size, bias=False)
```

**Was passiert:** Jeder Token-Vektor wird dreimal durch je eine lineare Schicht geschickt und landet so in drei verschiedenen „Rollen": Was suche ich? (Query) — Was biete ich an? (Key) — Was gebe ich weiter? (Value).

**Warum neuronale Netze?**  
Das Netz muss lernen, welche Aspekte eines Tokens für das Suchen, das Anbieten und das Weitergeben relevant sind — das sind drei völlig verschiedene Perspektiven auf denselben Vektor. Lineare Schichten ohne Bias sind hier ausreichend, weil das Positions- und Token-Embedding bereits genug Information kodiert; der Bias würde nur unnötige Parameter hinzufügen.

> 🔗 Bibliotheks-Analogie: „Recherche-Zettelchen, Regal-Schild, Buch-Inhalt" → [how-it-works.md Station 3](how-it-works.md#station-3--die-lese-tische-multi-head-self-attention)

---

### 4. Attention-Ausgabeprojektion — [`model.py:74`](../../model.py:74)

```python
self.proj = nn.Linear(n_heads * head_size, n_embd)
```

**Was passiert:** Die Ausgaben aller Attention-Heads (jeder hat `head_size` Dimensionen) werden zusammengeklebt und durch eine lineare Schicht wieder auf `n_embd` komprimiert.

**Warum ein neuronales Netz?**  
Die verschiedenen Heads haben unabhängig voneinander verschiedene Aspekte des Kontexts analysiert. Diese Erkenntnisse müssen sinnvoll **gemischt** werden. Eine lineare Schicht lernt genau diese Mischgewichtung — welche Information welches Heads wie stark in den nächsten Schritt einfließt.

---

### 5. Feed-Forward-Netz (MLP) — [`model.py:93-98`](../../model.py:93)

```python
self.net = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),  # Aufweiten:      96 → 384 Neuronen (advanced)
    nn.ReLU(),                        # Nichtlinearität: negative Aktivierungen → 0
    nn.Linear(4 * n_embd, n_embd),  # Komprimieren:   384 → 96 zurück
    nn.Dropout(dropout),
)
```

**Was passiert:** Jeder Token-Vektor durchläuft nach der Attention ein klassisches 2-schichtiges MLP — vollständig unabhängig von allen anderen Token-Positionen.

**Warum ein neuronales Netz?**  
Die Attention hat entschieden, *welche* anderen Token relevant sind, und die Informationen gemischt. Das FFN verarbeitet jetzt dieses Ergebnis für sich und extrahiert **komplexere, nichtlineare Muster**. Ohne `ReLU` wäre das FFN nur eine weitere lineare Umformung — es könnte nichts lernen, was nicht schon durch die linearen Attention-Projektionen darstellbar ist.

Das FFN ist das **einzige klassische MLP** im gesamten Transformer. Alles andere sind lineare Projektionen (ohne Aktivierungsfunktion) oder Lookup-Tabellen.

> 🔗 Bibliotheks-Analogie: „Eigene Notizen aus den Büchern destillieren" → [how-it-works.md Station 4](how-it-works.md#station-4--notizen-aufarbeiten-feed-forward-netz)

---

### 6. Layer Normalization — [`model.py:119-120`](../../model.py:119), [`model.py:162`](../../model.py:162)

```python
self.ln1 = nn.LayerNorm(n_embd)   # vor Attention
self.ln2 = nn.LayerNorm(n_embd)   # vor FFN
# ...
self.ln_final = nn.LayerNorm(n_embd)  # nach allen Blöcken
```

**Was passiert:** Vor jeder Attention- und FFN-Schicht wird der Token-Vektor normiert: Mittelwert → 0, Standardabweichung → 1. Danach skaliert und verschiebt LayerNorm mit lernbaren Parametern `γ` und `β`.

**Warum ein neuronales Netz?**  
LayerNorm ist ein Grenzfall: Es hat lernbare Gewichte (`γ`, `β`), ist also technisch ein NN-Baustein — aber die Gewichte kodieren **keine semantische Information**. Sie stellen nur sicher, dass die Zahlenbereiche stabil bleiben. Ohne Normierung würden Werte nach mehreren Schichten explodieren oder verschwinden, was das Training unmöglich macht.

> 🔗 Bibliotheks-Analogie: „Alle Schüler bekommen Noten auf derselben Skala" → [how-it-works.md Station 3+4](how-it-works.md#layer-normalization--gleiche-maßstäbe)

---

### 7. Language Model Head — [`model.py:163`](../../model.py:163)

```python
self.lm_head = nn.Linear(n_embd, vocab_size)
```

**Was passiert:** Der finale Token-Vektor (nach allen Blöcken und der letzten LayerNorm) wird durch eine lineare Schicht auf `vocab_size` Ausgabe-Zahlen projiziert — einen Score (Logit) pro möglichem nächstem Token.

**Warum ein neuronales Netz?**  
Das ist die Entscheidungsschicht. Sie lernt, welche Kombination von Werten im Token-Vektor für welches nächste Zeichen spricht. Eine lineare Schicht reicht hier aus, weil die nichtlineare Komplexität bereits in den vorherigen Schichten aufgebaut wurde.

> 🔗 Bibliotheks-Analogie: „Zettel mit Wahrscheinlichkeiten für alle Wörter" → [how-it-works.md Station 5](how-it-works.md#station-5--was-kommt-als-nächstes-language-model-head)

---

## Gesamtbild: lernbare vs. feste Operationen

| Operation | Lernbar? | PyTorch-Klasse | Grund |
|---|---|---|---|
| Token-Embedding | ✅ ja | `nn.Embedding` | Bedeutung muss aus Daten gelernt werden |
| Positions-Embedding | ✅ ja | `nn.Embedding` | Reihenfolge-Semantik muss gelernt werden |
| Q/K/V-Projektion | ✅ ja | `nn.Linear` | Relevanzkriterien müssen gelernt werden |
| Attention-Ausgabe | ✅ ja | `nn.Linear` | Mischgewichtung der Heads muss gelernt werden |
| FFN (beide Schichten) | ✅ ja | `nn.Linear` + `nn.ReLU` | Nichtlineare Muster müssen gelernt werden |
| LayerNorm (γ, β) | ✅ ja | `nn.LayerNorm` | Skalierung muss angepasst werden |
| LM Head | ✅ ja | `nn.Linear` | Entscheidung muss gelernt werden |
| Softmax | ❌ nein | `F.softmax` | Feste Normierungsformel |
| Scaled Dot-Product | ❌ nein | `@` + `* head_size**-0.5` | Feste Ähnlichkeitsberechnung |
| Kausale Maske | ❌ nein | `torch.tril` | Feste Regel (keine Zukunft) |
| Residual-Addition | ❌ nein | `+` | Feste Kurzschluss-Verbindung |

---

## Weiterführende Explainer in dieser Reihe

| Dokument | Inhalt |
|---|---|
| [how-it-works.md](how-it-works.md) | Der gesamte Transformer als Bibliotheks-Analogie |
| [attention-head.md](attention-head.md) | Der Attention-Mechanismus im Detail |
| [math-basics.md](math-basics.md) | Vektoren, Matrizen, Softmax — die mathematischen Grundlagen |
| [ARCHITECTURE.md](../../ARCHITECTURE.md) | Vollständige technische Architekturbeschreibung |
