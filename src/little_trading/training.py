"""Trains every decision model on the stored history, validates it walk-forward and weighs it by skill.

Labels never leak into features: a price sample at session i uses bars up to i's close and is labelled by the
move from session i + 1's open to the close `horizon` sessions after i. A news item counts from the first session
that opens after it was published (a day-only date counts from the next day). Validation is the latest quarter
of the sessions, with `horizon` sessions of embargo before it so no training label overlaps it; the reported
metrics come from that split, and the model kept for trading is then refitted on everything.

Calibration: naive Bayes, nearest neighbours and the word list give badly scaled scores, so each is fitted on all
but the latest fifth of its training sessions and Platt scaling fitted on that fifth maps its scores to
probabilities. Logistic models are calibrated by their own loss and use all their training rows.

Continuity: learning is kept across trainings, restarts and fresh starts. Every training run trains on every
company the server has ever watched (`Server.retrain`), so data is never dropped when the picks change; the
count-based models (naive Bayes, neighbours) are their data, so keeping all of it keeps what they learned. The
logistic models continue from their previous weights (`Logistic.fit` with a prior). Each training run is also
saved to `models/history/` (without the neighbours' shapes, which are the price data). Validation stays honest:
the validation fit starts from the latest saved weights trained only on data before the validation period, or
from scratch while none exists, never from weights that saw the validation sessions.

Weights: a model earns weight only for positive Brier skill on enough validation samples; that includes the
hand-written technical rule, measured as the `technical` model. Nothing unmeasured decides. `conviction` scales
the decision down while the combined skill is below `FULL_SKILL`, to 0 when no model has any.
"""
from __future__ import annotations

import bisect
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from . import features as F
from .data import align
from .jeff import FEATURES as JEFF_FEATURES, features as jeff_features
from .horizons import HorizonSpec
from .learners import Calibrated, Lexicon, Logistic, NewsBayes, PatternKnn, TechnicalRule, from_json, metrics
from .models import Bar
from .news import score_text
from .store import Store

ET = ZoneInfo("America/New_York")
MARKET = "SPY"
MIN_VAL = {"price": 150, "news": 60}
FULL_SKILL = 0.02
PRICE_STRIDE = 3  # price samples: every third session (neighbours share 4 of 5 label days), the newest included
KNN_STRIDE = 3  # neighbouring sessions have near-identical shapes: keep every third
KNN_MAX_VAL = 600
KNN_MAX_SHAPES = 8000  # stored shapes, thinned evenly: prediction cost grows with them
HISTORY_KEEP = 60  # saved training runs in models/history/
MAX_FIT = 12_000  # logistic fits are pure Python: thin evenly beyond this (neighbouring days overlap anyway)
CAL_FRACTION = 0.2  # latest share of a model's training sessions held out to calibrate it
MIN_CAL = 50


def _thin(rows: list[dict], cap: int = MAX_FIT) -> list[dict]:
    return rows[:: -(-len(rows) // cap)] if len(rows) > cap else rows


def _mean_y(rows: list[dict]) -> float:
    return sum(r["y"] for r in rows) / len(rows) if rows else 0.5


@dataclass
class TrainedModel:
    name: str
    model: object
    weight: float
    metrics: dict
    group: str  # "price" or "news"
    details: dict = field(default_factory=dict)


@dataclass
class ModelSet:
    models: dict[str, TrainedModel] = field(default_factory=dict)
    trained_at: str = ""

    @property
    def conviction(self) -> float:
        """How far the decision trusts its signals: combined skill over `FULL_SKILL`, capped at 1; 0 without skill."""
        total = sum(m.metrics.get("skill", 0) for m in self.models.values() if m.weight > 0)
        return round(min(1.0, total / FULL_SKILL), 4) if total > 0 else 0.0

    def to_json(self, skip_kinds: tuple[str, ...] = ()) -> dict:
        return {"trained_at": self.trained_at, "models": {
            n: {"model": m.model.to_json(), "weight": m.weight, "metrics": m.metrics, "group": m.group,
                "details": m.details} for n, m in self.models.items()
            if getattr(getattr(m.model, "inner", m.model), "kind", "") not in skip_kinds}}

    def save(self, directory: str) -> None:
        """models/current.json, plus this run in models/history/ (the newest `HISTORY_KEEP` are kept)."""
        path = os.path.join(directory, "models", "current.json")
        with open(path + ".tmp", "w") as fh:
            json.dump(self.to_json(), fh)
        os.replace(path + ".tmp", path)
        hist = os.path.join(directory, "models", "history")
        os.makedirs(hist, exist_ok=True)
        stem = os.path.join(hist, self.trained_at.replace(":", "").replace("+0000", "Z"))
        name, k = stem + ".json", 1
        while os.path.exists(name):  # two runs in the same second keep both
            name, k = f"{stem}-{k}.json", k + 1
        with open(name + ".tmp", "w") as fh:
            json.dump(self.to_json(skip_kinds=("knn",)), fh)
        os.replace(name + ".tmp", name)
        for old in sorted(os.listdir(hist))[:-HISTORY_KEEP]:
            os.remove(os.path.join(hist, old))

    @classmethod
    def load(cls, directory: str) -> "ModelSet | None":
        path = os.path.join(directory, "models", "current.json")
        if not os.path.exists(path):
            return None
        with open(path) as fh:
            d = json.load(fh)
        return cls({n: TrainedModel(n, from_json(m["model"]), m["weight"], m["metrics"], m["group"], m["details"])
                    for n, m in d["models"].items()}, d["trained_at"])


def history_prior(directory: str, name: str, before: date):
    """The newest saved weights of model `name` trained only on samples before `before` (the inner model), or
    None. Used as the prior of the validation fit, so validation never starts from weights that saw it."""
    hist = os.path.join(directory, "models", "history")
    for fname in sorted(os.listdir(hist), reverse=True) if os.path.isdir(hist) else []:
        try:
            with open(os.path.join(hist, fname)) as fh:
                m = json.load(fh)["models"].get(name)
        except (OSError, ValueError):
            continue
        if m and m["details"].get("data", {}).get("to", "9999") < before.isoformat():
            model = from_json(m["model"])
            return getattr(model, "inner", model)
    return None


def session_open(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET).astimezone(timezone.utc)


def available_from(published: datetime, day_only: bool) -> datetime:
    """When a headline can first be known: a date-only stamp counts from the end of that New York day."""
    if day_only:
        d = published.astimezone(ET).date()
        return datetime(d.year, d.month, d.day, 23, 59, tzinfo=ET).astimezone(timezone.utc)
    return published


def pair_with_market(bars: list[Bar], market: list[Bar]) -> tuple[list[Bar], list[Bar]]:
    if not market:
        return bars, []
    both = align({"s": bars, "m": market})
    return both["s"], both["m"]


def price_samples(store: Store, symbols: list[str], spec: HorizonSpec) -> list[dict]:
    horizon = spec.max_hold
    market = store.bars(MARKET)
    out = []
    for sym in symbols:
        bars, mkt = pair_with_market(store.bars(sym), market) if sym != MARKET else (market, market)
        for i in range(F.WARMUP - 1, len(bars) - horizon):
            if (len(bars) - 1 - i) % PRICE_STRIDE:
                continue
            fwd = F.forward_return(bars, i, horizon)
            if fwd is None:
                continue
            hist = bars[: i + 1]
            closes = [b.close for b in hist]
            out.append({"symbol": sym, "day": bars[i].time.date(), "i": i, "y": 1.0 if fwd > 0 else 0.0, "fwd": fwd,
                        "x": F.price_features(hist, mkt[: i + 1] if mkt else None),
                        "jeff": jeff_features(closes, spec),
                        "shape": F.pattern(hist)})
    return out


def news_samples(store: Store, symbols: list[str], horizon: int) -> list[dict]:
    """Headlines labelled by whether the stock beat the market over `horizon` sessions after they were known."""
    market = {b.time.date(): b for b in store.bars(MARKET)}
    out = []
    for sym in symbols:
        bars = [b for b in store.bars(sym) if b.time.date() in market]
        opens = [session_open(b.time.date()) for b in bars]
        for n in store.news(sym):
            known = available_from(datetime.fromisoformat(n["published"]), bool(n["day_only"]))
            s = bisect.bisect_right(opens, known)
            e = s + horizon - 1
            if e >= len(bars):
                continue
            stock = bars[e].close / bars[s].open - 1.0
            mkt = market[bars[e].time.date()].close / market[bars[s].time.date()].open - 1.0
            out.append({"symbol": sym, "day": bars[s].time.date(), "title": n["title"], "y": 1.0 if stock > mkt else 0.0,
                        "excess": stock - mkt, "lexicon": score_text(n["title"])})
    return out


def split(samples: list[dict], horizon: int, val_fraction: float = 0.25) -> tuple[list[dict], list[dict]]:
    days = sorted({s["day"] for s in samples})
    if len(days) < 4 * horizon:
        return samples, []
    cut = int(len(days) * (1 - val_fraction))
    val_start, train_end = days[cut], days[max(0, cut - horizon)]
    return [s for s in samples if s["day"] < train_end], [s for s in samples if s["day"] >= val_start]


def _weight(m: dict, group: str) -> float:
    return round(m["skill"], 5) if m.get("n", 0) >= MIN_VAL[group] and m.get("skill", 0) > 0 else 0.0


def train_all(store: Store, symbols: list[str], spec: HorizonSpec, software: str, log=None,
              previous: ModelSet | None = None) -> ModelSet:
    """Trains, validates and records every model; returns the set the trader should use. Logistic models continue
    from `previous` (normally the models in use)."""
    log = log or (lambda *a, **k: None)
    h = spec.max_hold
    t0 = time.monotonic()
    price = price_samples(store, symbols, spec)
    news = news_samples(store, symbols, h)
    train, val = split(price, h)
    ntrain, nval = split(news, h)
    log("training", f"datasets: {len(price)} price samples ({len(train)} train, {len(val)} validation), "
                    f"{len(news)} news samples ({len(ntrain)} train, {len(nval)} validation)",
        price=len(price), news=len(news), seconds=round(time.monotonic() - t0, 1))
    first = min((s["day"] for s in price), default=None)
    last = max((s["day"] for s in price), default=None)
    data_range = {"from": str(first), "to": str(last), "horizon_sessions": h}
    out: dict[str, TrainedModel] = {}

    def record(name, model, group, m, details):
        w = _weight(m, group)
        out[name] = TrainedModel(name, model, w, m, group, {**details, "data": data_range})
        store.add_model_run(name, details.get("train_n", 0), m.get("n", 0), m, w, out[name].details, software)
        log("training", f"{name}: skill {m.get('skill', 0):+.4f}, accuracy {m.get('accuracy', 0):.1%} "
                        f"vs majority {m.get('majority', 0):.1%} on {m.get('n', 0)}, weight {w}", model=name,
            metrics=m, weight=w)

    def fit_calibrated(rows, fit, raw, calibrate=True, cal_limit=None):
        """Fit on all but the latest fifth of `rows`, calibrate on that fifth: (model, calibration base rate).

        Without `calibrate` (logistic models, already calibrated by their loss) the model is fitted on all rows.
        """
        if not calibrate:
            return Calibrated(fit(rows)), _mean_y(rows)
        fit_rows, cal_rows = split(rows, h, CAL_FRACTION)
        model = fit(fit_rows or rows)
        if cal_limit and len(cal_rows) > cal_limit:  # slow models: an even thinning calibrates as well
            cal_rows = cal_rows[:: -(-len(cal_rows) // cal_limit)]
        if len(cal_rows) < MIN_CAL:
            return Calibrated(model), _mean_y(fit_rows or rows)
        return Calibrated(model).fit([raw(model, r) for r in cal_rows], [r["y"] for r in cal_rows]), _mean_y(cal_rows)

    def evaluate(name, group, rows, tr, va, fit, raw, details, val_limit=None, calibrate=True, fit_val=None):
        t = time.monotonic()
        model, _ = fit_calibrated(tr, fit_val or fit, raw, calibrate, val_limit)
        va = va[:: max(1, len(va) // val_limit)] if val_limit and va else va
        m = metrics([model._map(raw(model.inner, r)) for r in va], [r["y"] for r in va])
        final, base = fit_calibrated(rows, fit, raw, calibrate, val_limit)
        details = {**details(final.inner), "train_n": len(tr), "train_base_rate": round(base, 4),
                   "calibration": final.platt.to_json() if final.platt else None,
                   "seconds": round(time.monotonic() - t, 1)}
        record(name, final, group, m, details)

    label = f"up over {h} sessions"
    if price:
        val_start = min((r["day"] for r in val), default=None)
        for name, key, names in (("jeff", "jeff", JEFF_FEATURES), ("price-logistic", "x", F.NAMES)):
            prior = previous.models.get(name) if previous else None
            prior = getattr(prior.model, "inner", prior.model) if prior else None
            prior = prior if isinstance(prior, Logistic) and prior.names == list(names) else None
            vprior = history_prior(store.directory, name, val_start) if val_start else None

            def fit(rows, key=key, names=names, prior=prior):
                return Logistic(names).fit([r[key] for r in _thin(rows)], [r["y"] for r in _thin(rows)], prior=prior)

            def fit_val(rows, key=key, names=names, prior=vprior):
                return Logistic(names).fit([r[key] for r in _thin(rows)], [r["y"] for r in _thin(rows)], prior=prior)

            def describe(model, names=names, prior=prior, vprior=vprior):
                out = {"label": label, "weights": dict(zip(names, (round(w, 4) for w in model.weights))),
                       "continued_from": previous.trained_at if prior else None,
                       "validation_prior": "saved weights trained before the validation period" if vprior
                       else "none (from scratch)"}
                if prior:
                    out["weight_change"] = round(math.sqrt(sum((a - b) ** 2 for a, b in zip(model.weights,
                                                                                         prior.weights))), 4)
                return out

            evaluate(name, "price", price, train, val, fit, lambda model, r, key=key: model.predict(r[key]),
                     describe, calibrate=False, fit_val=fit_val)
        evaluate("technical", "price", price, train, val, lambda rows: TechnicalRule(),
                 lambda model, r: model.predict(r["jeff"]),
                 lambda model: {"label": label, "note": "hand-written rule (momentum, trend, RSI), only calibrated"})
        evaluate("pattern-knn", "price", price, train, val,
                 lambda rows: (lambda kept: PatternKnn().fit([r["shape"] for r in kept], [r["y"] for r in kept],
                                                             [f"{r['symbol']} {r['day']}" for r in kept]))(
                     _thin(rows[::KNN_STRIDE], KNN_MAX_SHAPES)),
                 lambda model, r: model.predict(r["shape"]),
                 lambda model: {"label": label, "k": model.k, "shapes": len(model.shapes),
                                "pattern_sessions": F.PATTERN_LEN}, val_limit=KNN_MAX_VAL)
    if news:
        nlabel = f"beat {MARKET} over {h} sessions"
        evaluate("news-bayes", "news", news, ntrain, nval,
                 lambda rows: NewsBayes().fit([r["title"] for r in rows], [r["y"] for r in rows]),
                 lambda model, r: model.predict_text(r["title"]),
                 lambda model: {"label": nlabel, "vocabulary": len(model.counts), "top_tokens": model.top_tokens()})
        evaluate("news-lexicon", "news", news, ntrain, nval, lambda rows: Lexicon(),
                 lambda model, r: model.predict_text(r["title"]),
                 lambda model: {"label": nlabel, "note": "fixed word list, untrained; only calibrated"})
    models = ModelSet(out, datetime.now(timezone.utc).isoformat(timespec="seconds"))
    models.save(store.directory)
    return models


def lessons_from_training(store: Store, models: ModelSet) -> list[str]:
    """Compares this training run with the previous one and records what changed."""
    notes = []
    for name, m in models.models.items():
        prev = store.query("SELECT metrics, weight FROM model_runs WHERE model = ? ORDER BY id DESC LIMIT 1 OFFSET 1",
                           [name])
        skill, n = m.metrics.get("skill", 0.0), m.metrics.get("n", 0)
        if n < MIN_VAL[m.group]:
            notes.append(("models", f"{name}: only {n} validation samples (needs {MIN_VAL[m.group]}), so it gets no "
                                    "weight yet.", {"model": name, "n": n}))
        elif skill <= 0:
            notes.append(("models", f"{name}: no skill on validation (Brier skill {skill:+.4f}, accuracy "
                                    f"{m.metrics['accuracy']:.1%} vs {m.metrics['majority']:.1%} always guessing the "
                                    "majority); it gets no weight.", {"model": name, **m.metrics}))
        if prev:
            old = json.loads(prev[0]["metrics"]).get("skill", 0.0)
            if abs(skill - old) >= 0.005:
                notes.append(("models", f"{name}: skill moved {old:+.4f} -> {skill:+.4f} since the last training.",
                              {"model": name, "old": old, "new": skill}))
    ranked = sorted(models.models.values(), key=lambda m: -m.metrics.get("skill", 0))
    if ranked and ranked[0].metrics.get("skill", 0) > 0:
        best = ranked[0]
        notes.append(("models", f"Best model now: {best.name} (Brier skill {best.metrics['skill']:+.4f}, AUC "
                                f"{best.metrics.get('auc', 0.5):.3f}). Conviction is {models.conviction:.0%}: "
                                "decisions act on that share of the skill-weighted signal.", {"model": best.name}))
    nb, lex = models.models.get("news-bayes"), models.models.get("news-lexicon")
    if nb and lex and nb.metrics.get("n", 0) >= MIN_VAL["news"]:
        better = nb.metrics.get("auc", 0.5) > lex.metrics.get("auc", 0.5)
        notes.append(("news", f"Learned news tokens {'beat' if better else 'did not beat'} the fixed word list "
                              f"(AUC {nb.metrics.get('auc', 0.5):.3f} vs {lex.metrics.get('auc', 0.5):.3f}).", {}))
    added = []
    for cat, text, ev in notes:
        if store.add_lesson(cat, text, ev):
            added.append(text)
    return added
