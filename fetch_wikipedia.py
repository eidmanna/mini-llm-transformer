"""
fetch_wikipedia.py – Wikipedia-Artikel laden, bereinigen und an Trainingsdaten anhängen

Verwendung:
  uv run python fetch_wikipedia.py
  uv run python fetch_wikipedia.py --output data/training_text.txt
  uv run python fetch_wikipedia.py --max-chars 50000
  uv run python fetch_wikipedia.py --dry-run        (nur anzeigen, nicht schreiben)
  uv run python fetch_wikipedia.py --chunk-size 5   (Artikel pro Block, Standard: 8)
  uv run python fetch_wikipedia.py --pause 30       (Pause zwischen Blöcken in Sek.)
  uv run python fetch_wikipedia.py --delay 2        (Pause zwischen einzelnen Artikeln)

Artikel-Liste unten in ARTIKEL anpassen.
Bereits geladene Artikel werden automatisch übersprungen (Resume-Funktion).
"""

import argparse
import json
import re
import subprocess
import sys
import time


# ══════════════════════════════════════════════════════════════════════════════
#  Artikel-Liste – hier beliebig erweitern oder kürzen
# ══════════════════════════════════════════════════════════════════════════════
ARTIKEL = [
    # Geografie & Natur
    "Elbe",
    "Schwarzwald",
    "Alpen",
    "Rhein",
    "Ozean",
    "Klimawandel",
    "Evolution",
    "Astronomie",
    # Wissenschaft & Technik
    "Sonnensystem",
    "Photosynthese",
    "Quantenmechanik",
    "Relativit%C3%A4tstheorie",   # Relativitätstheorie (URL-kodiert)
    "Mathematik",
    "Physik",
    "Biologie",
    "Computer",
    "Internet",
    # Geschichte & Gesellschaft
    "Zweiter_Weltkrieg",
    "Erster_Weltkrieg",
    "Demokratie",
    "Geschichte",
    "Wirtschaft",
    "Europa",
    "Deutschland",
    # Sprache & Kultur
    "Deutsche_Sprache",
    "Philosophie",
    "Literatur",
    "Musik",
    # Personen
    "Johannes_Gutenberg",
    "Albert_Einstein",
    "Goethe",
    "Beethoven",
    "Marie_Curie",
    "Isaac_Newton",
]
# ══════════════════════════════════════════════════════════════════════════════


def lade_artikel(titel: str) -> str:
    """Reinen Plaintext eines deutschen Wikipedia-Artikels per API laden."""
    url = (
        "https://de.wikipedia.org/w/api.php?action=query"
        f"&titles={titel}"
        "&prop=extracts&explaintext=true&exsectionformat=plain&format=json"
    )
    result = subprocess.run(
        ["curl", "-s", "--retry", "3", "--retry-delay", "5", url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl fehlgeschlagen: {result.stderr.strip()}")
    if not result.stdout.strip():
        raise RuntimeError("Leere Antwort vom Server (Rate-Limit?)")
    data = json.loads(result.stdout)
    page = next(iter(data["query"]["pages"].values()))
    return page.get("extract", "")


def bereinige(text: str) -> str:
    """Sonderzeichen normalisieren, Wiki-Artefakte entfernen."""
    ersetzungen = [
        # Typografische Zeichen → ASCII
        ("\u2013", "-"),    # En-Dash
        ("\u2014", "-"),    # Em-Dash
        ("\u201e", '"'),    # „
        ("\u201c", '"'),    # "
        ("\u201d", '"'),    # "
        ("\u2018", "'"),    # '
        ("\u2019", "'"),    # '
        ("\u201a", "'"),    # ‚
        ("\xa0",   " "),    # Non-breaking space
        # Hochgestellte Ziffern
        ("\u00b2", "2"),
        ("\u00b3", "3"),
        ("\u00b0", " Grad"),
        # Fremdsprachige Buchstaben
        ("\u00c6", "Ae"),   # Æ
        ("\u00dc", "Ue"),   # Ü  (Großbuchstabe, selten)
        ("\u00c4", "Ae"),   # Ä  (Großbuchstabe, selten)
        ("\u014d", "o"),    # ō
        ("\u02b0", "h"),    # ʰ
        ("\u00f0", "d"),    # ð
        ("\u00fd", "y"),    # ý
        ("\u00fe", "th"),   # þ
    ]
    for alt, neu in ersetzungen:
        text = text.replace(alt, neu)

    # Griechische Buchstaben, IPA-Lautschrift und sonstige Unicode > U+02FF entfernen
    text = re.sub(r"[^\x00-\u02FF]", "", text)

    # Wiki-Markup: Überschriften (== ... ==), Fußnoten-Zahlen ([1])
    text = re.sub(r"^=+[^=]+=+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[\d+\]", "", text)

    # Mehrfache Leerzeilen auf eine reduzieren
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def bereits_geladen(output: str) -> set[str]:
    """Liest die Zieldatei und gibt alle Artikel-Titel zurück, die bereits enthalten sind."""
    try:
        with open(output, encoding="utf-8") as f:
            inhalt = f.read()
    except FileNotFoundError:
        return set()
    # Sucht nach Markierungen "### Titel ###" die beim Schreiben eingefügt werden
    return set(re.findall(r"### (.+?) ###", inhalt))


def main() -> None:
    parser = argparse.ArgumentParser(description="Wikipedia-Artikel an Trainingsdaten anhängen")
    parser.add_argument(
        "--output", default="data/training_text.txt",
        help="Ziel-Datei (Standard: data/training_text.txt)",
    )
    parser.add_argument(
        "--max-chars", type=int, default=50000,
        help="Maximale Zeichenanzahl pro Artikel (Standard: 50000, 0 = unbegrenzt)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=8,
        help="Anzahl Artikel pro Block vor der langen Pause (Standard: 8)",
    )
    parser.add_argument(
        "--pause", type=int, default=30,
        help="Pause in Sekunden zwischen zwei Blöcken (Standard: 30)",
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Pause in Sekunden zwischen einzelnen Artikeln (Standard: 2.0)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Nur anzeigen, nicht in Datei schreiben",
    )
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print("  Wikipedia → Trainingsdaten")
    print(f"{'═'*60}")
    print(f"  Zieldatei        : {args.output}")
    print(f"  Max. Zeichen     : {args.max_chars if args.max_chars > 0 else 'unbegrenzt'}")
    print(f"  Block-Größe      : {args.chunk_size} Artikel")
    print(f"  Pause/Block      : {args.pause} s")
    print(f"  Delay/Artikel    : {args.delay} s")
    print(f"  Dry-run          : {args.dry_run}")

    # ── Resume: bereits vorhandene Artikel überspringen ─────────────────────
    vorhanden = bereits_geladen(args.output)
    todo = [t for t in ARTIKEL if t.replace("_", " ").replace("%C3%A4", "ä") not in vorhanden]

    print(f"{'─'*60}")
    print(f"  Artikel gesamt   : {len(ARTIKEL)}")
    print(f"  Bereits geladen  : {len(vorhanden)}  (werden übersprungen)")
    print(f"  Noch zu laden    : {len(todo)}")
    print(f"{'─'*60}")

    if not todo:
        print("  Alle Artikel bereits vorhanden. Nichts zu tun.")
        print(f"{'═'*60}\n")
        sys.exit(0)

    # ── Artikel in Blöcke aufteilen ─────────────────────────────────────────
    chunks = [todo[i : i + args.chunk_size] for i in range(0, len(todo), args.chunk_size)]
    gesamt_geladen = 0
    gesamt_zeichen = 0

    for chunk_nr, chunk in enumerate(chunks, start=1):
        print(f"\n  ── Block {chunk_nr}/{len(chunks)} ({len(chunk)} Artikel) ──")
        block_texte = []

        for titel in chunk:
            anzeige = titel.replace("_", " ").replace("%C3%A4", "ä")
            print(f"  Lade: {anzeige} ...", end=" ", flush=True)
            try:
                text = lade_artikel(titel)
                text = bereinige(text)
                if args.max_chars > 0:
                    text = text[: args.max_chars]
                if not text:
                    print("LEER (Artikel nicht gefunden?)")
                else:
                    block_texte.append(f"\n\n### {anzeige} ###\n\n{text}")
                    gesamt_geladen += 1
                    gesamt_zeichen += len(text)
                    print(f"OK ({len(text):,} Zeichen)")
            except Exception as exc:
                print(f"FEHLER: {exc}")
            time.sleep(args.delay)

        # Block in Datei schreiben (sofort, nicht erst am Ende)
        if block_texte and not args.dry_run:
            with open(args.output, "a", encoding="utf-8") as f:
                f.write("\n" + "\n".join(block_texte))
            print(f"  ✓ Block {chunk_nr} gespeichert ({len(block_texte)} Artikel)")
        elif block_texte and args.dry_run:
            print(f"  [Dry-run] Block {chunk_nr} nicht gespeichert.")

        # Pause zwischen Blöcken (außer nach dem letzten)
        if chunk_nr < len(chunks):
            print(f"\n  ⏸  Pause {args.pause} s vor Block {chunk_nr + 1} ...", end=" ", flush=True)
            time.sleep(args.pause)
            print("weiter.")

    print(f"\n{'═'*60}")
    print(f"  Fertig!")
    print(f"  Neu geladen      : {gesamt_geladen}/{len(todo)} Artikel")
    print(f"  Neue Zeichen     : {gesamt_zeichen:,}")
    if not args.dry_run:
        print(f"  Angehängt an     : {args.output}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
