"""
fetch_wikipedia.py – Wikipedia-Artikel laden, bereinigen und an Trainingsdaten anhängen

Verwendung:
  uv run python fetch_wikipedia.py
  uv run python fetch_wikipedia.py --output data/training_text.txt
  uv run python fetch_wikipedia.py --max-chars 6000
  uv run python fetch_wikipedia.py --dry-run   (nur anzeigen, nicht schreiben)

Artikel-Liste unten in ARTIKEL anpassen.
"""

import argparse
import json
import re
import subprocess
import sys


# ══════════════════════════════════════════════════════════════════════════════
#  Artikel-Liste – hier beliebig erweitern oder kürzen
# ══════════════════════════════════════════════════════════════════════════════
ARTIKEL = [
    "Elbe",
    "Schwarzwald",
    "Sonnensystem",
    "Photosynthese",
    "Johannes_Gutenberg",
    "Relativit%C3%A4tstheorie",   # Relativitätstheorie (URL-kodiert)
    "Zweiter_Weltkrieg",
    "Deutsche_Sprache",
]
# ══════════════════════════════════════════════════════════════════════════════


def lade_artikel(titel: str) -> str:
    """Reinen Plaintext eines deutschen Wikipedia-Artikels per API laden."""
    url = (
        "https://de.wikipedia.org/w/api.php?action=query"
        f"&titles={titel}"
        "&prop=extracts&explaintext=true&exsectionformat=plain&format=json"
    )
    result = subprocess.run(["curl", "-s", url], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"curl fehlgeschlagen: {result.stderr.strip()}")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Wikipedia-Artikel an Trainingsdaten anhängen")
    parser.add_argument(
        "--output", default="data/training_text.txt",
        help="Ziel-Datei (Standard: data/training_text.txt)",
    )
    parser.add_argument(
        "--max-chars", type=int, default=4000,
        help="Maximale Zeichenanzahl pro Artikel (Standard: 4000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Nur anzeigen, nicht in Datei schreiben",
    )
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print("  Wikipedia → Trainingsdaten")
    print(f"{'═'*60}")
    print(f"  Zieldatei  : {args.output}")
    print(f"  Max. Zeichen/Artikel: {args.max_chars}")
    print(f"  Dry-run    : {args.dry_run}")
    print(f"{'─'*60}")

    texte = []
    for titel in ARTIKEL:
        anzeige = titel.replace("_", " ").replace("%C3%A4", "ä")
        print(f"  Lade: {anzeige} ...", end=" ", flush=True)
        try:
            text = lade_artikel(titel)
            text = bereinige(text)
            text = text[: args.max_chars]
            if not text:
                print("LEER (Artikel nicht gefunden?)")
                continue
            texte.append(f"\n\n### {anzeige} ###\n\n{text}")
            print(f"OK ({len(text):,} Zeichen)")
        except Exception as exc:
            print(f"FEHLER: {exc}")

    gesamt = "\n".join(texte)
    print(f"{'─'*60}")
    print(f"  Geladene Artikel : {len(texte)}/{len(ARTIKEL)}")
    print(f"  Neue Zeichen     : {len(gesamt):,}")

    if args.dry_run:
        print("\n  [Dry-run] Nichts geschrieben.")
        print(gesamt[:500] + "...")
        sys.exit(0)

    with open(args.output, "a", encoding="utf-8") as f:
        f.write("\n" + gesamt)

    print(f"  Angehängt an     : {args.output}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
