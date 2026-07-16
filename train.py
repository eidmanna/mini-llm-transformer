"""
train.py – Trainings-Skript für den Mini-Transformer

════════════════════════════════════════════════════════════════════════════════
  ██████   EXPERIMENTIER-ZENTRALE – Alle Hyperparameter auf einen Blick  ██████
════════════════════════════════════════════════════════════════════════════════

Starte das Training:
  uv run python train.py                   # simple-Modus (Standard)
  uv run python train.py --mode simple     # explizit simple-Modus
  uv run python train.py --mode advanced   # advanced-Modus (großer Datensatz)

Einzelne Parameter übersteuern:
  uv run python train.py --max_iters 1000 --learning_rate 5e-4
  uv run python train.py --mode advanced --batch_size 64

Tipp: Ändere die Werte im CONFIG-Block und beobachte, wie sich Training-Kurve
und generierter Text verändern.
"""

import argparse
import os
import time
import torch
from model import MiniTransformer
from tokenizer import build_tokenizer, Tokenizer

# ══════════════════════════════════════════════════════════════════════════════
#  ░░  VOREINSTELLUNGEN – Zwei Modi als Schnellkonfiguration  ░░
# ══════════════════════════════════════════════════════════════════════════════

# ── simple-Modus: kurzer Trainingstext, Zeichen-Tokenizer, kleines Modell ──
CONFIG_SIMPLE = {
    "data_path":       "data/training_text_simple.txt",
    "tokenizer":       "char",
    "bpe_vocab_size":  200,          # wird bei char ignoriert
    "block_size":      64,
    "batch_size":      16,
    "max_iters":       1000,
    "learning_rate":   1e-3,
    "use_lr_scheduler": True,
    "eval_interval":   100,
    "train_split":     0.9,
    "n_embd":          32,
    "n_heads":         4,
    "n_layers":        2,
    "dropout":         0.0,
    "gen_start_text":  "Die",
    "gen_max_tokens":  80,
    "gen_temperature": 0.8,
    "gen_top_k":       20,
    "seed":            42,
}

# ── advanced-Modus: großer Trainingstext, BPE, Standard-Architektur ─────────
CONFIG_ADVANCED = {
    "data_path":       "data/training_text.txt",
    "tokenizer":       "bpe",
    "bpe_vocab_size":  2000,
    "block_size":      128,
    "batch_size":      32,
    "max_iters":       6000,
    "learning_rate":   1e-3,
    "use_lr_scheduler": True,
    "eval_interval":   250,
    "train_split":     0.9,
    "n_embd":          96,
    "n_heads":         6,
    "n_layers":        4,
    "dropout":         0.2,
    "gen_start_text":  "Der",
    "gen_max_tokens":  120,
    "gen_temperature": 0.8,
    "gen_top_k":       40,
    "seed":            42,
}

PRESETS = {
    "simple":   CONFIG_SIMPLE,
    "advanced": CONFIG_ADVANCED,
}

# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# CLI-Argumente
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mini-Transformer Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Modus ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--mode",
        choices=["simple", "advanced"],
        default="simple",
        help="Voreinstellung: 'simple' (kurzer Text, char-Tokenizer, kleines Modell) "
             "oder 'advanced' (großer Text, BPE, Standard-Architektur). "
             "Alle übrigen Flags übersteuern den Modus.",
    )

    # ── Daten ─────────────────────────────────────────────────────────────────
    p.add_argument("--data_path",      type=str,   default=None, help="Pfad zum Trainingstext")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    p.add_argument("--tokenizer",      type=str,   default=None, choices=["char", "bpe"],
                   help="Tokenizer: 'char' oder 'bpe'")
    p.add_argument("--bpe_vocab_size", type=int,   default=None, help="BPE-Vokabular-Größe")

    # ── Modell ────────────────────────────────────────────────────────────────
    p.add_argument("--block_size",     type=int,   default=None, help="Kontext-Länge in Tokens")
    p.add_argument("--n_embd",         type=int,   default=None, help="Embedding-Dimension")
    p.add_argument("--n_heads",        type=int,   default=None, help="Anzahl Attention-Heads")
    p.add_argument("--n_layers",       type=int,   default=None, help="Anzahl Transformer-Blöcke")
    p.add_argument("--dropout",        type=float, default=None, help="Dropout-Rate")

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--batch_size",     type=int,   default=None, help="Batch-Größe")
    p.add_argument("--max_iters",      type=int,   default=None, help="Maximale Trainings-Iterationen")
    p.add_argument("--learning_rate",  type=float, default=None, help="Lernrate")
    p.add_argument("--use_lr_scheduler", type=lambda x: x.lower() in ("true", "1", "yes"),
                   default=None, metavar="BOOL", help="Lernraten-Scheduler (true/false)")
    p.add_argument("--eval_interval",  type=int,   default=None, help="Evaluierungs-Intervall")
    p.add_argument("--train_split",    type=float, default=None, help="Train-Anteil (0–1)")
    p.add_argument("--seed",           type=int,   default=None, help="Zufalls-Seed")

    # ── Generierung ───────────────────────────────────────────────────────────
    p.add_argument("--gen_start_text",  type=str,   default=None, help="Starttext für Zwischen-Generierung")
    p.add_argument("--gen_max_tokens",  type=int,   default=None, help="Max. Tokens pro Zwischen-Generierung")
    p.add_argument("--gen_temperature", type=float, default=None, help="Sampling-Temperatur")
    p.add_argument("--gen_top_k",       type=int,   default=None, help="Top-K Sampling (0 = deaktiviert)")

    return p.parse_args()


def build_config(args: argparse.Namespace) -> dict:
    """Modus-Voreinstellung laden und mit expliziten CLI-Werten übersteuern."""
    cfg = dict(PRESETS[args.mode])   # Kopie des Presets

    overridable = [
        "data_path", "tokenizer", "bpe_vocab_size",
        "block_size", "n_embd", "n_heads", "n_layers", "dropout",
        "batch_size", "max_iters", "learning_rate", "use_lr_scheduler",
        "eval_interval", "train_split", "seed",
        "gen_start_text", "gen_max_tokens", "gen_temperature", "gen_top_k",
    ]
    for key in overridable:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    # gen_top_k=0 → None (= kein Top-K Sampling)
    if cfg.get("gen_top_k") == 0:
        cfg["gen_top_k"] = None

    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ──────────────────────────────────────────────────────────────────────────────

def load_text(path: str) -> str:
    """Trainingstext aus Datei lesen."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zufällige Batch-Ausschnitte aus den Daten laden."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i     : i + block_size    ] for i in ix])
    y  = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    if device is not None:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model: MiniTransformer,
    train_data: torch.Tensor,
    val_data:   torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_batches: int = 20,
    device: torch.device | None = None,
) -> dict[str, float]:
    """
    Durchschnittlichen Loss auf Train- und Val-Split schätzen.
    'eval_batches' Batches werden gemittelt, damit die Zahl stabiler ist.
    """
    model.eval()
    losses = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        batch_losses = []
        for _ in range(eval_batches):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            batch_losses.append(loss.item())
        losses[split] = sum(batch_losses) / len(batch_losses)
    model.train()
    return losses


def generate_sample(
    model: MiniTransformer,
    start_text: str,
    tokenizer: Tokenizer,
    max_tokens: int,
    temperature: float,
    top_k: int | None,
    device: torch.device | None = None,
) -> str:
    """Einen kurzen Text aus dem Modell generieren."""
    model.eval()
    start_ids = tokenizer.encode(start_text)
    if not start_ids:
        start_ids = [0]
    context = torch.tensor([start_ids], dtype=torch.long)
    if device is not None:
        context = context.to(device)
    generated = model.generate(context, max_tokens, temperature, top_k)
    model.train()
    return tokenizer.decode(generated[0].tolist())


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


# ──────────────────────────────────────────────────────────────────────────────
# Haupt-Trainings-Schleife
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg  = build_config(args)

    # ── Reproduzierbarkeit ──────────────────────────────────────────────────
    torch.manual_seed(cfg["seed"])

    # ── Gerät automatisch erkennen: CUDA → MPS (Apple Silicon) → CPU ───────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n{'═'*65}")
    print("  Mini-Transformer – Lernexperiment")
    print(f"{'═'*65}")
    print(f"  Modus          : {args.mode}")
    print(f"  Gerät          : {device}")

    # ── Daten laden & Tokenizer trainieren ─────────────────────────────────
    text = load_text(cfg["data_path"])

    print(f"  Datei          : {cfg['data_path']}")
    print(f"  Zeichen gesamt : {len(text):,}")
    print(f"  Tokenizer      : {cfg['tokenizer'].upper()}", end="")
    if cfg["tokenizer"] == "bpe":
        print(f"  (Ziel-Vokabular: {cfg['bpe_vocab_size']} Tokens)")
    else:
        print()

    tokenizer = build_tokenizer(
        text,
        mode       = cfg["tokenizer"],
        vocab_size = cfg["bpe_vocab_size"],
        verbose    = True,
    )
    vocab_size = tokenizer.vocab_size
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n_train = int(len(data) * cfg["train_split"])
    train_data = data[:n_train]
    val_data   = data[n_train:]

    compression = len(text) / len(data) if cfg["tokenizer"] == "bpe" else 1.0
    print(f"  Vokabular-Größe: {vocab_size} Tokens")
    if cfg["tokenizer"] == "bpe":
        print(f"  Kompression    : {compression:.2f}x  ({len(text):,} Zeichen → {len(data):,} Tokens)")
    print(f"  Train-Tokens   : {len(train_data):,}  |  Val-Tokens: {len(val_data):,}")

    # ── Modell initialisieren ───────────────────────────────────────────────
    model = MiniTransformer(
        vocab_size  = vocab_size,
        block_size  = cfg["block_size"],
        n_embd      = cfg["n_embd"],
        n_heads     = cfg["n_heads"],
        n_layers    = cfg["n_layers"],
        dropout     = cfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'─'*65}")
    print(f"  Modell-Architektur")
    print(f"{'─'*65}")
    print(f"  block_size     : {cfg['block_size']} Zeichen Kontext")
    print(f"  n_embd         : {cfg['n_embd']}  (Embedding-Dimension)")
    print(f"  n_heads        : {cfg['n_heads']}  (Attention-Heads)")
    print(f"  n_layers       : {cfg['n_layers']}  (Transformer-Blöcke)")
    print(f"  dropout        : {cfg['dropout']}")
    print(f"  Parameter      : {n_params:,} gesamt")

    print(f"\n{'─'*65}")
    print(f"  Training-Konfiguration")
    print(f"{'─'*65}")
    print(f"  batch_size     : {cfg['batch_size']}")
    print(f"  max_iters      : {cfg['max_iters']}")
    print(f"  learning_rate  : {cfg['learning_rate']}")
    print(f"  lr_scheduler   : {'aktiv (linear decay)' if cfg['use_lr_scheduler'] else 'aus'}")
    print(f"  eval_interval  : alle {cfg['eval_interval']} Iterationen")
    print(f"  gen_start_text : \"{cfg['gen_start_text']}\"")
    print(f"{'═'*65}\n")

    # ── Optimizer & Scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    if cfg["use_lr_scheduler"]:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor = 1.0,
            end_factor   = 0.1,
            total_iters  = cfg["max_iters"],
        )
    else:
        scheduler = None

    # ── Training ────────────────────────────────────────────────────────────
    start_time = time.time()

    for iteration in range(1, cfg["max_iters"] + 1):

        # Gelegentlich Zwischen-Report ausgeben
        if iteration == 1 or iteration % cfg["eval_interval"] == 0:
            elapsed = time.time() - start_time
            progress_pct = 100 * iteration / cfg["max_iters"]
            current_lr   = optimizer.param_groups[0]["lr"]

            losses = estimate_loss(
                model, train_data, val_data,
                cfg["block_size"], cfg["batch_size"],
                device=device,
            )

            sample_text = generate_sample(
                model,
                cfg["gen_start_text"],
                tokenizer,
                cfg["gen_max_tokens"],
                cfg["gen_temperature"],
                cfg["gen_top_k"],
                device=device,
            )

            print(f"{'─'*65}")
            print(
                f"  Iter {iteration:>5}/{cfg['max_iters']}  "
                f"({progress_pct:5.1f}%)  "
                f"Zeit: {format_duration(elapsed)}"
            )
            print(
                f"  Train-Loss: {losses['train']:.4f}  |  "
                f"Val-Loss: {losses['val']:.4f}  |  "
                f"LR: {current_lr:.2e}"
            )
            print(f"\n  ▶ Generierter Text:\n  {sample_text!r}\n")

        # ── Einen Trainings-Schritt ────────────────────────────────────────
        x, y = get_batch(train_data, cfg["block_size"], cfg["batch_size"], device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient Clipping: verhindert explodierende Gradienten
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    # ── Abschluss ──────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    print(f"{'═'*65}")
    print(f"  Training abgeschlossen in {format_duration(total_time)}")

    final_losses = estimate_loss(
        model, train_data, val_data,
        cfg["block_size"], cfg["batch_size"],
        eval_batches=50,
        device=device,
    )
    print(f"  Finaler Train-Loss : {final_losses['train']:.4f}")
    print(f"  Finaler Val-Loss   : {final_losses['val']:.4f}")

    # ── Abschließende Generierung (länger) ─────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Abschließende Textgenerierung (200 Zeichen):")
    print(f"{'─'*65}")
    long_sample = generate_sample(
        model,
        cfg["gen_start_text"],
        tokenizer,
        max_tokens  = 200,
        temperature = cfg["gen_temperature"],
        top_k       = cfg["gen_top_k"],
        device      = device,
    )
    print(f"  {long_sample!r}")
    print(f"{'═'*65}\n")

    # ── Modell speichern ───────────────────────────────────────────────────
    save_path = "model_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config":           cfg,
            "vocab_size":       vocab_size,
            "tokenizer":        tokenizer.state(),
        },
        save_path,
    )
    print(f"  Modell gespeichert: {save_path}")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
