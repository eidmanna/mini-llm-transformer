"""
generate.py – Textgenerierung mit einem gespeicherten Modell-Checkpoint

Verwendung:
  uv run python generate.py
  uv run python generate.py --start "Die Wissenschaft" --tokens 300
  uv run python generate.py --start "Der Wald" --temperature 0.5 --top_k 20
"""

import argparse
import torch
from model import MiniTransformer
from tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini-LLM Textgenerierung")
    parser.add_argument(
        "--checkpoint", default="model_checkpoint.pt",
        help="Pfad zum gespeicherten Checkpoint (Standard: model_checkpoint.pt)"
    )
    parser.add_argument(
        "--start", default="Der",
        help="Startwort / Starttext für die Generierung (Standard: 'Der')"
    )
    parser.add_argument(
        "--tokens", type=int, default=200,
        help="Anzahl zu generierender Zeichen (Standard: 200)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Temperatur: < 1.0 fokussierter | > 1.0 kreativer (Standard: 0.8)"
    )
    parser.add_argument(
        "--top_k", type=int, default=40,
        help="Top-K Sampling: nur die k wahrscheinlichsten Kandidaten (Standard: 40)"
    )
    args = parser.parse_args()

    # ── Checkpoint laden ───────────────────────────────────────────────────
    print(f"\nLade Checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    cfg        = checkpoint["config"]
    vocab_size = checkpoint["vocab_size"]
    tokenizer  = Tokenizer.from_state(checkpoint["tokenizer"])

    # ── Modell aufbauen & Gewichte laden ───────────────────────────────────
    model = MiniTransformer(
        vocab_size = vocab_size,
        block_size = cfg["block_size"],
        n_embd     = cfg["n_embd"],
        n_heads    = cfg["n_heads"],
        n_layers   = cfg["n_layers"],
        dropout    = 0.0,   # Im Inferenz-Modus immer 0
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    tok_mode = cfg.get("tokenizer", "char")
    print(f"Modell geladen  – {n_params:,} Parameter")
    print(f"Tokenizer       – {tok_mode.upper()}  (Vokabular: {vocab_size} Tokens)")
    print(f"Starttext       – \"{args.start}\"")
    print(f"Tokens          – {args.tokens}")
    print(f"Temperature     – {args.temperature}")
    print(f"Top-K           – {args.top_k}")
    print(f"\n{'─' * 60}\n")

    # ── Starttext kodieren ─────────────────────────────────────────────────
    start_ids = tokenizer.encode(args.start)
    if not start_ids:
        print(f"Warnung: kein Token aus \"{args.start}\" im Vokabular – nutze erstes Token.")
        start_ids = [0]

    context = torch.tensor([start_ids], dtype=torch.long)

    # ── Generieren ─────────────────────────────────────────────────────────
    with torch.no_grad():
        output = model.generate(context, args.tokens, args.temperature, args.top_k)

    result = tokenizer.decode(output[0].tolist())
    print(result)
    print(f"\n{'─' * 60}\n")


if __name__ == "__main__":
    main()
