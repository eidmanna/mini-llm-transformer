"""
tokenizer.py – Tokenizer-Implementierungen für den Mini-Transformer

Zwei Modi:
  "bpe"  – Byte Pair Encoding (Standard): lernt Subword-Tokens aus dem Text
  "char" – Zeichen-Level (wie bisher): jedes Zeichen = ein Token

Verwendung:
  from tokenizer import build_tokenizer

  # BPE (Standard)
  tok = build_tokenizer(text, mode="bpe", vocab_size=2000)

  # Zeichen-Level
  tok = build_tokenizer(text, mode="char")

  ids   = tok.encode("Hallo Welt")
  text  = tok.decode([23, 47, 12])
  vocab = tok.vocab_size
"""

from __future__ import annotations
from collections import defaultdict


# ──────────────────────────────────────────────────────────────────────────────
# Gemeinsames Interface
# ──────────────────────────────────────────────────────────────────────────────

class Tokenizer:
    """Basis-Interface – wird von CharTokenizer und BPETokenizer implementiert."""

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    def state(self) -> dict:
        """Serialisierbarer Zustand für torch.save / torch.load."""
        raise NotImplementedError

    @classmethod
    def from_state(cls, state: dict) -> "Tokenizer":
        """Tokenizer aus gespeichertem Zustand wiederherstellen."""
        if state["mode"] == "char":
            return CharTokenizer.from_state(state)
        elif state["mode"] == "bpe":
            return BPETokenizer.from_state(state)
        raise ValueError(f"Unbekannter Tokenizer-Modus: {state['mode']}")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Zeichen-Level-Tokenizer (wie bisher)
# ──────────────────────────────────────────────────────────────────────────────

class CharTokenizer(Tokenizer):
    """Ein Token = ein Zeichen. Vokabular = alle eindeutigen Zeichen im Text."""

    def __init__(self, stoi: dict[str, int], itos: dict[int, str]):
        self._stoi = stoi
        self._itos = itos

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi  = {ch: i for i, ch in enumerate(chars)}
        itos  = {i: ch for i, ch in enumerate(chars)}
        return cls(stoi, itos)

    @property
    def vocab_size(self) -> int:
        return len(self._stoi)

    def encode(self, text: str) -> list[int]:
        return [self._stoi[c] for c in text if c in self._stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._itos[i] for i in ids if i in self._itos)

    def state(self) -> dict:
        return {"mode": "char", "stoi": self._stoi, "itos": self._itos}

    @classmethod
    def from_state(cls, state: dict) -> "CharTokenizer":
        itos = {int(k): v for k, v in state["itos"].items()}
        return cls(state["stoi"], itos)


# ──────────────────────────────────────────────────────────────────────────────
# 2. BPE-Tokenizer
# ──────────────────────────────────────────────────────────────────────────────

class BPETokenizer(Tokenizer):
    """
    Byte Pair Encoding (BPE) – Subword-Tokenizer.

    Algorithmus:
      1. Starte mit einem Zeichen-Level-Vokabular.
      2. Finde das häufigste benachbarte Token-Paar.
      3. Füge es zu einem neuen Token zusammen und ersetze alle Vorkommen.
      4. Wiederhole bis vocab_size erreicht ist.

    Ergebnis: häufige Wörter bekommen eigene Tokens,
              seltene Wörter werden in bekannte Subwords zerlegt.
    """

    def __init__(
        self,
        stoi:   dict[str, int],
        itos:   dict[int, str],
        merges: list[tuple[str, str]],
    ):
        self._stoi   = stoi
        self._itos   = itos
        self._merges = merges
        # Schnelle Merge-Lookup-Tabelle: (a, b) → zusammengeführtes Token
        self._merge_map: dict[tuple[str, str], str] = {
            (a, b): a + b for a, b in merges
        }

    @property
    def vocab_size(self) -> int:
        return len(self._stoi)

    @classmethod
    def train(cls, text: str, vocab_size: int = 2000, verbose: bool = True) -> "BPETokenizer":
        """
        BPE-Vokabular aus Text lernen (wort-basierter Algorithmus für Effizienz).

        Statt den gesamten Text als eine flache Sequenz zu verwalten, werden
        zunächst Wort-Häufigkeiten gezählt. Jedes Wort wird als Zeichen-Tupel
        gespeichert. Merges werden auf diesen kompakten Wort-Repräsentationen
        durchgeführt – das ist O(vocab_size × unique_words) statt
        O(vocab_size × total_chars) und damit deutlich schneller.
        """
        import re

        # ── Schritt 1: Zeichen-Level-Startvokabular ────────────────────────
        base_chars = sorted(set(text))
        stoi: dict[str, int] = {ch: i for i, ch in enumerate(base_chars)}
        itos: dict[int, str] = {i: ch for i, ch in enumerate(base_chars)}
        merges: list[tuple[str, str]] = []

        if verbose:
            print(f"  BPE-Training: Startvokabular {len(stoi)} Zeichen → Ziel {vocab_size} Tokens")

        # ── Schritt 2: Wörter tokenisieren (Zeichen + Wortende-Marker) ─────
        # Wort = Folge von Nicht-Leerzeichen; Wortende als separates Zeichen
        word_pattern = re.compile(r"\S+")
        word_freq: dict[tuple[str, ...], int] = defaultdict(int)
        for m in word_pattern.finditer(text):
            word = tuple(m.group())   # ("H","a","l","l","o")
            word_freq[word] += 1

        def get_pair_counts(
            wf: dict[tuple[str, ...], int]
        ) -> dict[tuple[str, str], int]:
            counts: dict[tuple[str, str], int] = defaultdict(int)
            for word, freq in wf.items():
                for a, b in zip(word, word[1:]):
                    counts[(a, b)] += freq
            return counts

        def merge_word(
            word: tuple[str, ...], a: str, b: str, ab: str
        ) -> tuple[str, ...]:
            new: list[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    new.append(ab)
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            return tuple(new)

        # ── Schritt 3: Merge-Iterationen ───────────────────────────────────
        while len(stoi) < vocab_size:
            pair_counts = get_pair_counts(word_freq)
            if not pair_counts:
                break

            best_pair  = max(pair_counts, key=lambda p: pair_counts[p])
            best_count = pair_counts[best_pair]
            if best_count < 2:
                break

            a, b      = best_pair
            new_token = a + b
            new_id    = len(stoi)
            stoi[new_token] = new_id
            itos[new_id]    = new_token
            merges.append((a, b))

            # Wort-Frequenz-Tabelle aktualisieren
            new_word_freq: dict[tuple[str, ...], int] = {}
            for word, freq in word_freq.items():
                new_word = merge_word(word, a, b, new_token)
                new_word_freq[new_word] = new_word_freq.get(new_word, 0) + freq
            word_freq = new_word_freq

            if verbose and len(stoi) % 200 == 0:
                print(f"    Vokabular: {len(stoi):>5} Tokens  (letztes Merge: {new_token!r})")

        # ── Finale Kompression messen ──────────────────────────────────────
        if verbose:
            total_tokens = sum(len(w) * f for w, f in word_freq.items())
            # Leerzeichen/Zeilenumbrüche einzeln zählen (nicht in word_freq)
            ws_count = sum(1 for c in text if c not in set(re.compile(r"\S").pattern))
            compression = len(text) / max(total_tokens + ws_count, 1)
            print(f"  BPE-Training abgeschlossen: {len(stoi)} Tokens, {len(merges)} Merges")
            print(f"  Kompression: ~{compression:.2f}x")

        return cls(stoi, itos, merges)

    def encode(self, text: str) -> list[int]:
        """Text zu Token-IDs kodieren."""
        # Unbekannte Zeichen überspringen
        tokens = [ch for ch in text if ch in self._stoi]
        # Merges in derselben Reihenfolge wie beim Training anwenden
        for a, b in self._merges:
            merged = a + b
            if merged not in self._stoi:
                continue
            new_tokens: list[str] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return [self._stoi[t] for t in tokens if t in self._stoi]

    def decode(self, ids: list[int]) -> str:
        """Token-IDs zurück zu Text dekodieren."""
        return "".join(self._itos[i] for i in ids if i in self._itos)

    def state(self) -> dict:
        return {
            "mode":   "bpe",
            "stoi":   self._stoi,
            "itos":   self._itos,
            "merges": self._merges,
        }

    @classmethod
    def from_state(cls, state: dict) -> "BPETokenizer":
        itos = {int(k): v for k, v in state["itos"].items()}
        return cls(state["stoi"], itos, state["merges"])


# ──────────────────────────────────────────────────────────────────────────────
# Factory-Funktion
# ──────────────────────────────────────────────────────────────────────────────

def build_tokenizer(
    text:       str,
    mode:       str   = "bpe",
    vocab_size: int   = 2000,
    verbose:    bool  = True,
) -> Tokenizer:
    """
    Tokenizer erstellen und auf 'text' trainieren.

    mode       : "bpe" (Standard) oder "char"
    vocab_size : Ziel-Vokabulargröße für BPE (bei "char" ignoriert)
    verbose    : Fortschrittsausgabe während des BPE-Trainings
    """
    if mode == "char":
        return CharTokenizer.train(text)
    elif mode == "bpe":
        return BPETokenizer.train(text, vocab_size=vocab_size, verbose=verbose)
    else:
        raise ValueError(f"Unbekannter Tokenizer-Modus: {mode!r}. Wähle 'bpe' oder 'char'.")
