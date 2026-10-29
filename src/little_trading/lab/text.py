"""Stage 2: text readers. N0 lexicon, N1 FinBERT sentiment, N3-lite a learned head on frozen FinBERT embeddings.

Point in time: a headline counts from the decision at the close of the day after its date, a conservative rule
for date-only stamps (a decision at a close cannot know headlines published later that day) that is applied to
timestamped headlines too, for one rule everywhere. Features per (session, company) over the last 3 and 14 days: mean sentiment, recency-weighted
sentiment, log count. Missing news is 0 with count 0, so the model can tell "no news" from "neutral news".
"""
from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime, timedelta, timezone

import numpy as np

FINBERT = "ProsusAI/finbert"
TEXT_FEATURES = ("sent_3d", "sent_14d", "sentw_14d", "cnt_3d", "cnt_14d", "maxabs_14d")


def embed_headlines(news_db: str, out_db: str, batch: int = 64, log=print, limit: int | None = None) -> int:
    """FinBERT class probabilities (positive, negative, neutral) and the [CLS] vector for every stored headline."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(12)
    tok = AutoTokenizer.from_pretrained(FINBERT)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT, use_safetensors=True).eval()
    src = sqlite3.connect(f"file:{news_db}?mode=ro", uri=True)
    dst = sqlite3.connect(out_db)
    dst.execute("CREATE TABLE IF NOT EXISTS emb (id TEXT PRIMARY KEY, pos REAL, neg REAL, neu REAL, vec BLOB)")
    done = {r[0] for r in dst.execute("SELECT id FROM emb")}
    rows = [r for r in src.execute("SELECT id, title FROM news ORDER BY published DESC") if r[0] not in done]
    if limit:
        rows = rows[:limit]
    log(f"{len(rows)} headlines to embed")
    n = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        enc = tok([t for _, t in chunk], padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        p = torch.softmax(out.logits, dim=-1).numpy()  # FinBERT order: positive, negative, neutral
        cls = out.hidden_states[-1][:, 0].numpy().astype(np.float16)
        dst.executemany("INSERT OR REPLACE INTO emb VALUES (?,?,?,?,?)",
                        [(chunk[j][0], float(p[j, 0]), float(p[j, 1]), float(p[j, 2]), cls[j].tobytes()) for j in range(len(chunk))])
        n += len(chunk)
        if (i // batch) % 50 == 0:
            dst.commit()
            log(f"  {n}/{len(rows)}")
    dst.commit()
    return n


def load_headlines(news_db: str, emb_db: str | None, symbols: list[str]):
    """(symbol, available_date, lexicon, sent, vec-or-None) rows for the given companies."""
    con = sqlite3.connect(f"file:{news_db}?mode=ro", uri=True)
    emb = {}
    if emb_db and os.path.exists(emb_db):
        e = sqlite3.connect(f"file:{emb_db}?mode=ro", uri=True)
        emb = {r[0]: (r[1] - r[2], r[4]) for r in e.execute("SELECT id, pos, neg, neu, vec FROM emb")}
        e.close()
    out = []
    marks = ",".join("?" * len(symbols))
    for sym, pub, day_only, lex, nid in con.execute(
            f"SELECT symbol, published, day_only, lexicon, id FROM news WHERE symbol IN ({marks})", symbols):
        d = datetime.fromisoformat(pub).astimezone(timezone.utc).date()
        avail = d + timedelta(days=1)  # a decision at close t sees only headlines dated before t (dates and times alike)
        sent, vec = emb.get(nid, (None, None))
        out.append((sym, avail, lex, sent, vec))
    con.close()
    return out


def text_features(panel, headlines, source: str = "finbert") -> tuple[np.ndarray, np.ndarray]:
    """(T, N, len(TEXT_FEATURES)) float32 features and (T, N) count of headlines in the last 14 days.
    source: 'lexicon' (the live word list) or 'finbert'."""
    T, N = panel.close.shape
    idx = {s: i for i, s in enumerate(panel.symbols)}
    cal = np.array([d.toordinal() for d in panel.dates])
    per = np.zeros((T + 40, N, 3), np.float32)  # daily sum of sentiment, count, max abs
    for sym, avail, lex, sent, _ in headlines:
        if sym not in idx:
            continue
        v = lex if source == "lexicon" else sent
        if v is None:
            continue
        t = np.searchsorted(cal, avail.toordinal(), side="left")  # first session on/after the available date
        if t >= T:
            continue
        per[t, idx[sym], 0] += v
        per[t, idx[sym], 1] += 1
        per[t, idx[sym], 2] = max(per[t, idx[sym], 2], abs(v))
    per = per[:T]
    cs = lambda a, w: np.cumsum(a, axis=0) - np.vstack([np.zeros((w, a.shape[1]), a.dtype), np.cumsum(a, axis=0)[:-w]])  # noqa: E731
    s3, c3, s14, c14 = cs(per[..., 0], 3), cs(per[..., 1], 3), cs(per[..., 0], 14), cs(per[..., 1], 14)
    wdec = 0.5 ** (np.arange(14) / 3.0)
    sw = np.zeros((T, N), np.float32)
    for k in range(14):
        sw[k:] += wdec[k] * per[: T - k, :, 0]
    mx = np.zeros((T, N), np.float32)
    for k in range(14):
        mx[k:] = np.maximum(mx[k:], per[: T - k, :, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        feats = np.stack([np.where(c3 > 0, s3 / c3, 0), np.where(c14 > 0, s14 / c14, 0), sw / np.maximum(c14, 1),
                          np.log1p(c3), np.log1p(c14), mx], axis=-1).astype(np.float32)
    return feats, c14
