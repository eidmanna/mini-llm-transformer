"""
model.py – Decoder-Only Transformer (Mini-GPT style)

Architektur-Überblick
─────────────────────
  Token-Embedding  ←  Jedes Zeichen wird in einen Vektor der Größe n_embd umgewandelt
  + Positions-Embedding  ←  Die Position im Kontext wird als weiterer Vektor addiert
  → N × Transformer-Block
       ├─ LayerNorm
       ├─ Masked Multi-Head Self-Attention  (kein Blick in die Zukunft)
       ├─ LayerNorm
       └─ Feed-Forward-Netz (2-schichtig, ReLU)
  → LayerNorm
  → Linear-Projektion auf Vokabular-Größe
  → Logits (werden außerhalb zu Wahrscheinlichkeiten / Loss)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 1. Ein einzelner Attention-Head
# ──────────────────────────────────────────────────────────────────────────────
class Head(nn.Module):
    """
    Ein Attention-Head mit kausaler Maske.

    Jede Position darf nur auf sich selbst und frühere Positionen schauen
    (Decoder-Only / Autoregressive Property).
    """

    def __init__(self, head_size: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 'tril' ist kein lernbarer Parameter, sondern ein fester Buffer
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape                          # Batch, Zeitschritte, Kanäle
        k = self.key(x)                            # (B, T, head_size)
        q = self.query(x)                          # (B, T, head_size)
        head_size = k.shape[-1]

        # Scaled Dot-Product Attention
        wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)   # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)                          # (B, T, head_size)
        return wei @ v                             # (B, T, head_size)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Multi-Head Attention
# ──────────────────────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    """
    Mehrere parallele Attention-Heads, deren Ausgaben konkateniert
    und dann durch eine lineare Projektion gemischt werden.
    """

    def __init__(self, n_heads: int, head_size: int, n_embd: int, block_size: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, n_embd, block_size, dropout) for _ in range(n_heads)
        ])
        self.proj    = nn.Linear(n_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Feed-Forward-Netz (pro Position unabhängig)
# ──────────────────────────────────────────────────────────────────────────────
class FeedForward(nn.Module):
    """
    Einfaches 2-schichtiges MLP mit ReLU.
    Der innere Zustand ist 4× so groß wie n_embd (klassische GPT-Skalierung).
    """

    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Ein vollständiger Transformer-Block
# ──────────────────────────────────────────────────────────────────────────────
class Block(nn.Module):
    """
    Transformer-Block mit Pre-LayerNorm (stabiler als Post-Norm):
      x = x + Attention(LayerNorm(x))
      x = x + FFN(LayerNorm(x))
    """

    def __init__(self, n_embd: int, n_heads: int, block_size: int, dropout: float):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa   = MultiHeadAttention(n_heads, head_size, n_embd, block_size, dropout)
        self.ff   = FeedForward(n_embd, dropout)
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))   # Residual-Verbindung um Attention
        x = x + self.ff(self.ln2(x))   # Residual-Verbindung um FFN
        return x


# ──────────────────────────────────────────────────────────────────────────────
# 5. Das vollständige Sprachmodell
# ──────────────────────────────────────────────────────────────────────────────
class MiniTransformer(nn.Module):
    """
    Decoder-Only Transformer (GPT-ähnlich) auf Zeichen-Ebene.

    Parameter
    ─────────
    vocab_size  : Anzahl der eindeutigen Zeichen im Trainings-Text
    block_size  : Maximale Kontextlänge (wie weit das Modell zurückschauen darf)
    n_embd      : Embedding-Dimension (Breite des Modells)
    n_heads     : Anzahl der Attention-Heads (n_embd muss durch n_heads teilbar sein)
    n_layers    : Anzahl der gestapelten Transformer-Blöcke (Tiefe des Modells)
    dropout     : Dropout-Rate (verhindert Überanpassung)
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.block_size = block_size

        self.token_embedding    = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_heads, block_size, dropout) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head  = nn.Linear(n_embd, vocab_size)

        # Gewichts-Initialisierung (verbessert Trainings-Stabilität)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        # Token- + Positions-Embedding addieren
        tok_emb = self.token_embedding(idx)                           # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb                                         # Broadcasting über Batch

        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)                                      # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int | None = 40,
    ) -> torch.Tensor:
        """
        Autoregressiv neue Tokens generieren.

        temperature  : < 1.0 → konservativer/schärfer, > 1.0 → kreativer/zufälliger
        top_k        : Nur die k wahrscheinlichsten Kandidaten berücksichtigen
        """
        for _ in range(max_new_tokens):
            # Kontext auf block_size begrenzen
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature    # Nur letztes Zeichen

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, idx_next], dim=1)
        return idx
