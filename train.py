"""
train.py – Trainings-Skript für den Mini-Transformer

════════════════════════════════════════════════════════════════════════════════
  ██████   EXPERIMENTIER-ZENTRALE – Alle Hyperparameter auf einen Blick  ██████
════════════════════════════════════════════════════════════════════════════════

Starte das Training:
  uv run python train.py

Tipp: Ändere die Werte im CONFIG-Block und beobachte, wie sich Training-Kurve
und generierter Text verändern.
"""

import os
import time
import torch
from model import MiniTransformer

# ══════════════════════════════════════════════════════════════════════════════
#  ░░  EXPERIMENTIER-ZENTRALE – Hier kannst du alle Werte verändern  ░░
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Daten ──────────────────────────────────────────────────────────────
    "data_path": "data/training_text.txt",

    # ── Kontextlänge ───────────────────────────────────────────────────────
    #   Wie viele Zeichen das Modell gleichzeitig als Kontext bekommt.
    #   Kleiner → schneller, aber weniger Zusammenhang.
    #   Empfehlung: 32 (schnell) | 64 (Standard) | 128 (langsamer, besser)
    "block_size": 128,

    # ── Batch-Größe ────────────────────────────────────────────────────────
    #   Wie viele Textausschnitte gleichzeitig verarbeitet werden.
    #   Kleiner → weniger RAM, rauschigere Updates.
    #   Empfehlung: 16 (sparsam) | 32 (Standard) | 64 (mehr RAM nötig)
    "batch_size": 32,

    # ── Trainings-Dauer ────────────────────────────────────────────────────
    #   Erhöhe max_iters für längeres Training.
    #   Auf Intel-Mac-CPU: 3000 ≈ 5 Min | 6000 ≈ 10 Min
    "max_iters": 5000,

    # ── Lernrate ───────────────────────────────────────────────────────────
    #   Zu hoch → Training explodiert, zu niedrig → zu langsam.
    #   Empfehlung: 1e-3 (Start) → bei Plateau auf 1e-4 reduzieren
    "learning_rate": 1e-3,

    # ── Lernraten-Scheduler ────────────────────────────────────────────────
    #   True → Lernrate nimmt über das Training linear ab (oft besser)
    #   False → konstante Lernrate
    "use_lr_scheduler": True,

    # ── Zwischen-Evaluierung ───────────────────────────────────────────────
    #   Alle X Iterationen: Loss ausgeben + kurzen Text generieren.
    #   Empfehlung: 250 (viel Feedback) | 500 (Standard) | 1000 (wenig)
    "eval_interval": 250,

    # ── Trainings/Validierungs-Split ───────────────────────────────────────
    #   Anteil der Daten für das Training (Rest = Validierung).
    "train_split": 0.9,

    # ── Modell-Architektur ─────────────────────────────────────────────────
    #   n_embd  : Embedding-Dimension (Breite des Modells)
    #             Kleiner → schneller | 32 (minimal) | 64 | 128 | 256
    #   n_heads : Anzahl Attention-Heads (n_embd muss durch n_heads teilbar sein!)
    #             Typisch: n_embd=64 → n_heads=4 | n_embd=128 → n_heads=8
    #   n_layers: Anzahl gestapelter Transformer-Blöcke (Tiefe des Modells)
    #             1 = sehr flach, 4 = Standard für Mini-Modelle, 6 = tiefer
    "n_embd":   64,
    "n_heads":  4,
    "n_layers": 4,

    # ── Regularisierung ────────────────────────────────────────────────────
    #   Dropout verhindert Überanpassung. Bei kleinen Daten eher niedrig halten.
    #   Empfehlung: 0.0 (aus) | 0.1 (leicht) | 0.2 (Standard)
    "dropout": 0.2,

    # ── Text-Generierung ───────────────────────────────────────────────────
    #   Startwort für die Zwischen-Generierung während des Trainings.
    "gen_start_text":  "Der",
    #   Anzahl der Zeichen, die pro Zwischen-Generierung erzeugt werden.
    "gen_max_tokens":  120,
    #   Temperature: < 1.0 → konservativer | 1.0 → neutral | > 1.0 → kreativer
    "gen_temperature": 0.8,
    #   Top-K: Nur die k wahrscheinlichsten Kandidaten (None = alle)
    "gen_top_k":       40,

    # ── Reproduzierbarkeit ─────────────────────────────────────────────────
    "seed": 42,
}
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ──────────────────────────────────────────────────────────────────────────────

def load_text(path: str) -> str:
    """Trainingstext aus Datei lesen."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text: str) -> tuple[dict, dict, int]:
    """
    Character-Level Tokenizer.
    Gibt zurück: (char→int, int→char, vocab_size)
    """
    chars = sorted(set(text))
    stoi  = {ch: i for i, ch in enumerate(chars)}
    itos  = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos, len(chars)


def encode(text: str, stoi: dict) -> list[int]:
    return [stoi[c] for c in text]


def decode(ids: list[int], itos: dict) -> str:
    return "".join(itos[i] for i in ids)


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
    stoi: dict,
    itos:  dict,
    max_tokens: int,
    temperature: float,
    top_k: int | None,
    device: torch.device | None = None,
) -> str:
    """Einen kurzen Text aus dem Modell generieren."""
    model.eval()
    # Startwort codieren (unbekannte Zeichen überspringen)
    start_ids = [stoi[c] for c in start_text if c in stoi]
    if not start_ids:
        start_ids = [0]
    context = torch.tensor([start_ids], dtype=torch.long)
    if device is not None:
        context = context.to(device)
    generated = model.generate(context, max_tokens, temperature, top_k)
    model.train()
    return decode(generated[0].tolist(), itos)


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


# ──────────────────────────────────────────────────────────────────────────────
# Haupt-Trainings-Schleife
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = CONFIG

    # ── Reproduzierbarkeit ──────────────────────────────────────────────────
    torch.manual_seed(cfg["seed"])

    # ── Gerät automatisch erkennen: MPS (Apple Silicon) → CPU ──────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n{'═'*65}")
    print("  Mini-Transformer – Lernexperiment")
    print(f"{'═'*65}")
    print(f"  Gerät          : {device}")

    # ── Daten laden & Vokabular bauen ───────────────────────────────────────
    text = load_text(cfg["data_path"])
    stoi, itos, vocab_size = build_vocab(text)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)

    n_train = int(len(data) * cfg["train_split"])
    train_data = data[:n_train]
    val_data   = data[n_train:]

    print(f"  Datei          : {cfg['data_path']}")
    print(f"  Zeichen gesamt : {len(text):,}")
    print(f"  Vokabular-Größe: {vocab_size} eindeutige Zeichen")
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
                stoi, itos,
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
        stoi, itos,
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
            "stoi":             stoi,
            "itos":             itos,
        },
        save_path,
    )
    print(f"  Modell gespeichert: {save_path}")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
