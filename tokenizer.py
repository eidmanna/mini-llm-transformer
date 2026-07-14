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
import heapq
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

        # ── Schritt 2: Wörter tokenisieren ────────────────────────────────
        word_pattern = re.compile(r"\S+")
        word_freq: dict[tuple[str, ...], int] = defaultdict(int)
        for m in word_pattern.finditer(text):
            word_freq[tuple(m.group())] += 1

        # ── Inkrementeller Pair-Count-Index ───────────────────────────────
        # pair_counts[pair]  = aktueller Gesamtcount (kann veraltet sein im Heap)
        # pair_to_words[pair] = set aller Wörter (als Index in word_list), die
        #                       dieses Paar enthalten
        # Heap: max-heap via negative counts → (-count, pair)
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        pair_to_words: dict[tuple[str, str], set] = defaultdict(set)

        # Stabile Wortliste – Wörter werden nie gelöscht, nur ersetzt
        word_list: list[tuple[str, ...] | None] = list(word_freq.keys())
        # word_index: aktuelles Tupel → Index in word_list
        word_index: dict[tuple[str, ...], int] = {w: i for i, w in enumerate(word_list)}

        for idx, word in enumerate(word_list):
            freq = word_freq[word]
            for a, b in zip(word, word[1:]):
                pair_counts[(a, b)] += freq
                pair_to_words[(a, b)].add(idx)

        # Heap: Einträge können veraltet sein → lazy removal via Gegencheck
        heap: list[tuple[int, tuple[str, str]]] = [
            (-cnt, pair) for pair, cnt in pair_counts.items()
        ]
        heapq.heapify(heap)

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

        # ── Schritt 3: Merge-Iterationen (inkrementell) ───────────────────
        while len(stoi) < vocab_size:
            # Bestes Paar aus Heap holen (veraltete Einträge überspringen)
            best_pair = None
            while heap:
                neg_cnt, pair = heapq.heappop(heap)
                if pair_counts.get(pair, 0) == -neg_cnt:
                    best_pair = pair
                    best_count = -neg_cnt
                    break

            if best_pair is None or best_count < 2:
                break

            a, b      = best_pair
            new_token = a + b
            new_id    = len(stoi)
            stoi[new_token] = new_id
            itos[new_id]    = new_token
            merges.append((a, b))

            # Nur die Wörter aktualisieren, die das Paar (a, b) enthalten
            affected_indices = list(pair_to_words.get(best_pair, set()))
            for idx in affected_indices:
                old_word = word_list[idx]
                if old_word is None:
                    continue
                freq = word_freq.get(old_word, 0)
                if freq == 0:
                    continue

                new_word = merge_word(old_word, a, b, new_token)
                if new_word == old_word:
                    continue

                # Altes Wort aus Datenstrukturen entfernen
                del word_freq[old_word]
                word_list[idx] = None  # Slot als verwaist markieren

                # Pair-Counts für das alte Wort dekrementieren
                for x, y in zip(old_word, old_word[1:]):
                    pair_counts[(x, y)] -= freq
                    pair_to_words[(x, y)].discard(idx)

                # Neues Wort einfügen oder mit vorhandenem zusammenführen
                if new_word in word_index:
                    existing_idx = word_index[new_word]
                    word_freq[new_word] = word_freq.get(new_word, 0) + freq
                    # Pair-Counts für das neue Wort inkrementieren
                    for x, y in zip(new_word, new_word[1:]):
                        pair_counts[(x, y)] += freq
                        pair_to_words[(x, y)].add(existing_idx)
                else:
                    new_idx = len(word_list)
                    word_list.append(new_word)
                    word_index[new_word] = new_idx
                    word_freq[new_word] = freq
                    for x, y in zip(new_word, new_word[1:]):
                        pair_counts[(x, y)] += freq
                        pair_to_words[(x, y)].add(new_idx)
                        heapq.heappush(heap, (-pair_counts[(x, y)], (x, y)))

                # Geänderte Pair-Counts in Heap pushen
                for x, y in zip(new_word, new_word[1:]):
                    heapq.heappush(heap, (-pair_counts[(x, y)], (x, y)))

            # best_pair-Count aktualisieren (kann durch obige Schleife bereits 0 sein)
            if pair_counts.get(best_pair, 0) > 0:
                heapq.heappush(heap, (-pair_counts[best_pair], best_pair))

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
