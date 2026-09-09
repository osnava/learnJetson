"""Shared MiniLM embedder (issue #28).

Both the index builder (`build.py`) and the query path (`search.py`) embed
with the exact same math — mean-pool + L2-normalize — because retrieval is
a dot product between the two. Keeping the model id, token window and
pooling HERE is what guarantees cosine compatibility; changing one side
silently breaks the other.
"""
from __future__ import annotations

MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 23M params, 384-dim
MAX_TOKENS = 256  # MiniLM's window; also the chunker's max_tokens
BATCH = 64

_MODEL = None
_TOK = None
_DEVICE = None


def _load():
    global _MODEL, _TOK, _DEVICE
    if _MODEL is None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        _TOK = AutoTokenizer.from_pretrained(MODEL)
        _MODEL = AutoModel.from_pretrained(MODEL).to(_DEVICE).eval()
    return _MODEL, _TOK, _DEVICE


def embed_texts(texts: list[str]):
    """Embed a batch -> (N, 384) float32, L2-normalized (GPU if present)."""
    import torch

    model, tok, device = _load()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=MAX_TOKENS, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(emb, dim=-1).cpu())
    return torch.cat(out).numpy().astype("float32")


def embed_query(text: str):
    """Single-text convenience over embed_texts."""
    return embed_texts([text])[0]
