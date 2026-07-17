# Neuronale Netze — was sie sind und was sie tun

> **Ziel dieses Dokuments:** Verstehen, was ein neuronales Netz grundsätzlich ist — und dann direkt nachvollziehen, wo und warum genau diese Konzepte in [`model.py`](../../model.py) des MiniTransformers stecken.

---

## Neuronale Netze als gelernte Formel — die Physik-Analogie

### Die Black-Box-Sichtweise

Eine physikalische Formel wie `F = m · a` ist eine Black Box: `m` und `a` kommen herein, `F` kommt heraus. Man muss nicht wissen, wie das Universum „intern" funktioniert — man steckt Werte rein und bekommt eine Antwort.

Ein trainiertes neuronales Netz ist **dasselbe Prinzip** — nur dass die Formel nicht vom menschlichen Verstand entdeckt, sondern automatisch aus Daten optimiert wurde:

| | Physikalische Formel | Neuronales Netz |
|---|---|---|
| **Eingabe** | `m`, `a` | Eingabevektor `x` |
| **Funktion** | `F = m · a` (bekannt, analytisch) | `y = W₂ · ReLU(W₁ · x + b₁) + b₂` (gelernt) |
| **Ausgabe** | `F` | Vorhersage `y` |
| **Woher kommt die Funktion?** | Menschlicher Verstand + Experiment | Automatisch aus Trainingsdaten optimiert |
| **Interpretierbar?** | Ja — jeder Term hat eine Bedeutung | Nein — die Gewichte `W` sind nur Zahlen |

Der einzige strukturelle Unterschied: Bei `F = m·a` kennt man die Formel. Beim NN kennt man nur die Gewichte `W`, die die Formel **implizit** kodieren.

---

### Kann ein NN Physik aus Messdaten nachbauen?

**Ja — und das passiert tatsächlich in der Forschung.** Das nennt sich **Scientific Machine Learning** oder **Physics Discovery with NNs**.

Konkrete Methoden, die real existieren:

- **SINDy** (Sparse Identification of Nonlinear Dynamics): Findet aus Messdaten symbolische Formeln — entdeckt z. B. selbst, dass `F = m·a` die beste Erklärung ist
- **PINNs** (Physics-Informed Neural Networks): Ein NN löst Differentialgleichungen nur aus Randbedingungen, ohne dass die Gleichung explizit vorgegeben wird
- **AI Feynman** (Tegmark et al., MIT): Rekonstruiert aus Messdaten symbolische physikalische Gleichungen — und hat Formeln wie `E = ½mv²` tatsächlich wiederentdeckt

---

### Gemeinsamkeiten und Unterschiede

#### Was beide gemeinsam haben

- Beide sind **Eingabe → Funktion → Ausgabe**: Black Boxes, die Zahlen in Zahlen abbilden
- Beide können **interpolieren**: für Eingaben zwischen bekannten Werten funktionieren beide zuverlässig
- Beide sind **deterministisch**: gleiche Eingabe → gleiche Ausgabe (kein Zufall zur Laufzeit)
- Beide **approximieren die Realität**: auch `F = m·a` ist eine Näherung (gilt streng nur für konstante Masse, ohne Relativität usw.)

#### Wo sie sich unterscheiden

| Eigenschaft | Physikalisches Gesetz | Neuronales Netz |
|---|---|---|
| **Herkunft** | Menschliche Einsicht + Experiment | Gradientenabstieg auf Trainingsdaten |
| **Parameterzahl** | Oft 1–5 Parameter | Tausende bis Milliarden Gewichte |
| **Extrapolation** | Gilt universell (innerhalb des Gültigkeitsbereichs) | Bricht außerhalb der Trainingsdaten oft zusammen |
| **Interpretierbarkeit** | Jeder Term hat physikalische Bedeutung | Gewichte sind bedeutungslose Zahlen |
| **Kompaktheit** | `F = m·a` passt auf eine Zeile | Nicht in einer Formel darstellbar |
| **Rauschen** | Filtert Messfehler durch Ableitung | Kann Messrauschen einlernen (Overfitting) |
| **Generalisierung** | Gilt für alle `m` und `a` | Nur zuverlässig im Trainingsbereich |

---

### Die entscheidende Einschränkung: Extrapolation

`F = m·a` gilt für alle `m` und `a` — auch für `m = 10.000 kg`, das nie gemessen wurde. Ein NN, das nur mit `m` zwischen 1 und 10 kg trainiert wurde, wird für `m = 1000 kg` vermutlich falsch liegen — weil es die lineare Beziehung nicht als **universelles Gesetz** gelernt hat, sondern als **lokale Kurvenanpassung** der Trainingsdaten.

```
Physikalisches Gesetz:   ──────────────────────────────────────►  (gerade Linie, immer gültig)
                         0        10       100      1000    m →

Neuronales Netz:         ──────────╌╌╌╌╌╌╌╌?????????             (sicher nur im Trainingsbereich)
                         0        10       100      1000    m →
                              ↑
                         Trainingsbereich
```

---

### Fazit

Das Bild ist präzise: Ein trainiertes NN **ist** eine gelernte Black-Box-Formel. Man kann damit Naturgesetze aus Messdaten rekonstruieren — das Prinzip funktioniert. Die Grenzen liegen nicht im Prinzip, sondern in **Generalisierung** (kein universelles Gesetz), **Interpretierbarkeit** (keine Bedeutung hinter den Gewichten) und **Datenbedarf** (braucht viel mehr Beispiele als ein Mensch, um dasselbe zu lernen).

---

## Warum heißt es „neuronales Netz"?

Der Name kommt aus der Biologie: Das Gehirn besteht aus Milliarden von **Neuronen** (Nervenzellen), die über **Synapsen** miteinander verbunden sind. Jedes Neuron empfängt Signale von anderen Neuronen, summiert sie, und feuert — oder nicht — ein eigenes Signal weiter.

```
Biologisches Neuron:
  Eingangssignale (x₁, x₂, x₃)
       ↓   ↓   ↓
  Gewichtete Summe:  x₁·w₁ + x₂·w₂ + x₃·w₃  +  Schwellwert
       ↓
  Aktivierungsfunktion:  "Feuere, wenn der Wert groß genug ist"
       ↓
  Ausgangssignal
```

Das **künstliche Neuron** ist eine stark vereinfachte mathematische Nachbildung genau dieses Prinzips:

```
Künstliches Neuron:
  Eingaben:           x₁, x₂, x₃
  Gewichte:           w₁, w₂, w₃  (lernbar)
  Bias:               b            (lernbar, entspricht dem Schwellwert)
  Gewichtete Summe:   z = x₁·w₁ + x₂·w₂ + x₃·w₃ + b
  Aktivierung:        a = f(z)     (z. B. ReLU: max(0, z))
```

Ein **neuronales Netz** ist eine Ansammlung vieler solcher künstlichen Neuronen — in Schichten organisiert, miteinander verbunden. Der Name ist also eine Analogie: genauso wie Neuronen im Gehirn durch Aktivierungs- und Hemmungsmuster zusammen Entscheidungen treffen, tun es auch die mathematischen Neuronen in einem NN.

> **Wichtig:** Die Analogie zum Gehirn ist oberflächlich. Echte Neuronen sind viel komplexer. Künstliche neuronale Netze sind primär gut funktionierende **Funktionsapproximatoren** — sie können (mit genug Parametern) nahezu jede beliebige Funktion lernen.

---

## Eine Schicht oder mehrere? — Das mehrschichtige Netz

### Ein einzelnes Neuron

Ein einzelnes Neuron kann nur **lineare Entscheidungsgrenzen** lernen — es kann z. B. unterscheiden, ob eine Zahl größer oder kleiner als 5 ist, aber nicht, ob sie *zwischen* 3 und 7 liegt.

### Eine Schicht (= viele Neuronen parallel)

Eine **Schicht** besteht aus mehreren Neuronen, die alle **dieselbe Eingabe** bekommen, aber **unterschiedliche Gewichte** haben. Jedes Neuron lernt dabei ein anderes Merkmal.

```
Eingabe x (3 Werte)     Schicht (4 Neuronen)     Ausgabe (4 Werte)

  x₁ ─┬──────────────── Neuron 1 (Gewichte w₁₁, w₁₂, w₁₃)  →  a₁
  x₂ ─┼──────────────── Neuron 2 (Gewichte w₂₁, w₂₂, w₂₃)  →  a₂
  x₃ ─┴──────────────── Neuron 3 (Gewichte w₃₁, w₃₂, w₃₃)  →  a₃
                   └─── Neuron 4 (Gewichte w₄₁, w₄₂, w₄₃)  →  a₄
```

Das ist genau das, was `nn.Linear(3, 4)` macht: eine Matrix mit 4 × 3 = 12 lernbaren Gewichten.

### Mehrere Schichten hintereinander — warum?

**Muss** man mehrere Schichten verwenden? Nein — aber man sollte es fast immer tun, wenn das Problem komplex genug ist.

| Tiefe | Was lernbar ist | Beispiel |
|---|---|---|
| 1 Schicht | Lineare Muster | „Ist Wort A größer als Wert B?" |
| 2 Schichten + Aktivierung | Einfache nichtlineare Muster | „Liegt A zwischen B und C?" |
| Viele Schichten | Hierarchische, abstrakte Muster | „Ist das ein grammatisch korrekter Satz?" |

Der **Universelle Approximationssatz** besagt: Ein Netz mit *einer* ausreichend breiten versteckten Schicht und einer Aktivierungsfunktion kann theoretisch jede stetige Funktion approximieren. In der Praxis sind aber mehrere, schmälere Schichten effizienter — sie lernen hierarchische Merkmale (erst einfache Muster, dann komplexere Kombinationen davon).

> **Im MiniTransformer:** Die Q/K/V-Projektionen und der LM Head sind jeweils **1-schichtig** (eine `nn.Linear`). Das FeedForward-Netz ist **2-schichtig** (`Linear → ReLU → Linear`). Der gesamte Transformer mit `n_layers=4` Blöcken ist effektiv ein sehr tiefes Netz — aber die Tiefe kommt durch das Stapeln der Blöcke, nicht durch ein einzelnes tiefes MLP.

---

## Die mathematische Funktion eines neuronalen Netzes

### Schritt 1 — Ein einzelnes Neuron

Gegeben einen Eingabevektor **x** mit `n` Werten:

```
x = [x₁, x₂, ..., xₙ]
```

Ein Neuron berechnet:

```
z = w₁·x₁ + w₂·x₂ + ... + wₙ·xₙ + b
  = Σᵢ (wᵢ · xᵢ) + b
  = w · x  +  b        (als Vektorprodukt geschrieben)
```

Dann kommt die Aktivierungsfunktion:

```
a = f(z)   →   z. B. ReLU: a = max(0, z)
```

### Schritt 2 — Eine ganze Schicht als Matrixoperation

Statt jeden Neuron einzeln zu berechnen, fasst man alle Neuronen einer Schicht in eine **Matrix** zusammen. Das ist der Kern von `nn.Linear`:

```
Eingabe:    x  ∈ ℝⁿ         (Vektor mit n Werten)
Gewichte:   W  ∈ ℝᵐˣⁿ      (Matrix: m Neuronen, je n Gewichte)
Bias:       b  ∈ ℝᵐ         (Vektor mit m Werten)

Ausgabe:    z = W · x + b   ∈ ℝᵐ
```

Konkret für `nn.Linear(3, 4)` — 3 Eingaben, 4 Ausgabe-Neuronen:

```
     W (4×3)          x (3)     b (4)     z (4)

  [w₁₁ w₁₂ w₁₃]   [x₁]   [b₁]   [w₁₁x₁ + w₁₂x₂ + w₁₃x₃ + b₁]
  [w₂₁ w₂₂ w₂₃] · [x₂] + [b₂] = [w₂₁x₁ + w₂₂x₂ + w₂₃x₃ + b₂]
  [w₃₁ w₃₂ w₃₃]   [x₃]   [b₃]   [w₃₁x₁ + w₃₂x₂ + w₃₃x₃ + b₃]
  [w₄₁ w₄₂ w₄₃]          [b₄]   [w₄₁x₁ + w₄₂x₂ + w₄₃x₃ + b₄]
```

### Schritt 3 — Mehrere Schichten hintereinander

Die Ausgabe einer Schicht wird zur Eingabe der nächsten. Für ein 2-schichtiges Netz (wie das FFN in [`model.py:93-98`](../../model.py:93)):

```
Eingabe:     x

Schicht 1:   z₁ = W₁ · x  + b₁        (Linear: n_embd → 4·n_embd)
Aktivierung: a₁ = ReLU(z₁)             (Nichtlinearität)
Schicht 2:   z₂ = W₂ · a₁ + b₂        (Linear: 4·n_embd → n_embd)

Ausgabe:     z₂
```

Eingesetzt:

```
Ausgabe = W₂ · ReLU(W₁ · x + b₁) + b₂
```

Das `ReLU` in der Mitte ist entscheidend: Ohne es wäre `W₂ · (W₁ · x + b₁) + b₂` nur eine einzige lineare Transformation `(W₂·W₁)·x + (W₂·b₁+b₂)` — mathematisch identisch zu einer einzelnen Schicht.

### Schritt 4 — Wie lernen die Gewichte?

Das Ziel ist, die Gewichte `W` und `b` so zu finden, dass der Vorhersagefehler (Loss) möglichst klein wird. Das geschieht über den **Gradienten**:

```
Für jedes Gewicht w:
  ∂Loss/∂w  →  "Wenn ich w um ε erhöhe, wie ändert sich der Loss?"
```

- Ist `∂Loss/∂w > 0` → w erhöhen macht den Loss größer → w verringern
- Ist `∂Loss/∂w < 0` → w erhöhen macht den Loss kleiner → w erhöhen

Der **Gradient-Descent**-Schritt:

```
w_neu = w_alt  −  lernrate · (∂Loss/∂w)
```

PyTorch berechnet alle diese Ableitungen automatisch via `loss.backward()` ([`train.py:396`](../../train.py:396)). AdamW ([`train.py:339`](../../train.py:339)) ist eine verbesserte Variante: es merkt sich für jedes Gewicht, wie stark und in welche Richtung es in der Vergangenheit aktualisiert wurde, und passt die effektive Lernrate individuell an.

---

## Die drei Grundbausteine in PyTorch

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
