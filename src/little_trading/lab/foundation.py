"""Pre-trained time-series foundation models (H-F1: Kronos, TimesFM, Chronos) as frozen zero-shot forecasters.

Each adapter turns the price history of one session into one number per company: the expected return over the next
`horizon` sessions (forecast close at t + h over the close at t, minus 1). The models are used frozen, never fine-tuned,
and only forward in time: their pre-training data overlaps every year we could test on, so a historical score would
measure memory, not skill (plan section 5, H-F1). The service `h_f1_foundation` therefore scores only the newest session.

`score_session(panel, row, forecaster, context_len)` is the only way scores are made. It cuts the arrays at `row` (rows
`row - context_len + 1 .. row`, nothing later) before the forecaster sees them, so an adapter cannot read the future even
by accident; `tests/unit/test_lab_f1.py` checks that with the future-perturbation test. It also measures the cost per name
(wall and CPU seconds), which the service logs and stores.

The packages (torch, pandas, timesfm, chronos-forecasting, Kronos from its repository) are imported inside the methods,
never at module import, so the lab and the tests run without them. What each adapter does is taken from the project READMEs
(read 2026-10-30; the Hugging Face model cards were not reachable): see docs/lab/foundation-models.md.
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

CHANNELS = ("open", "high", "low", "close", "volume")


class Forecaster:
    """predict(history) -> (N,) expected return over the next `horizon` sessions, NaN where a name was not forecast.

    `history` holds rows up to the scored session only: "open", "high", "low", "close", "volume" as (L, N) arrays with
    missing bars forward-filled inside the window (NaN before a name's first bar), "rows" (L,) panel row indices,
    "dates" (L,) as datetime.date, and "symbols" (N,)."""
    name = "base"
    horizon = 5

    def predict(self, history: dict) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def version(self) -> str:
        return ""


def _ffill(a: np.ndarray) -> np.ndarray:
    out = np.array(a, dtype=float, copy=True)
    for t in range(1, len(out)):
        m = np.isnan(out[t])
        out[t, m] = out[t - 1, m]
    return out


def session_history(panel, row: int, context_len: int, min_history: int) -> tuple[dict, np.ndarray]:
    """The input of session `row`: the last `context_len` rows up to and including `row`, and a mask of the names that
    are tradable at `row` with at least `min_history` known closes in that window."""
    lo = max(0, row - context_len + 1)
    hist = {k: _ffill(getattr(panel, k)[lo:row + 1]) for k in CHANNELS}
    hist["rows"] = np.arange(lo, row + 1)
    hist["dates"] = list(panel.dates[lo:row + 1])
    hist["symbols"] = list(panel.symbols)
    known = np.isfinite(hist["close"]).sum(axis=0)
    tradable = panel.tradable[row] if panel.tradable is not None else np.isfinite(panel.close[row])
    return hist, tradable & (known >= min_history) & np.isfinite(hist["close"][-1])


def score_session(panel, row: int, forecaster: Forecaster, context_len: int, min_history: int | None = None) -> tuple[np.ndarray, dict]:
    """(scores (N,), cost) for panel session `row`, from rows <= `row` only. Names outside the mask stay NaN. `cost` holds
    the names forecast, wall and CPU seconds, and milliseconds per name."""
    need = context_len if min_history is None else min_history
    hist, ok = session_history(panel, row, context_len, need)
    out = np.full(len(panel.symbols), np.nan)
    t0, c0 = time.perf_counter(), time.process_time()
    if ok.any():
        sub = {k: (v[:, ok] if k in CHANNELS else v) for k, v in hist.items()}
        sub["symbols"] = [s for s, m in zip(hist["symbols"], ok) if m]
        pred = np.asarray(forecaster.predict(sub), dtype=float).reshape(-1)
        if pred.shape != (int(ok.sum()),):
            raise ValueError(f"{forecaster.name} returned {pred.shape[0]} forecasts for {int(ok.sum())} names")
        out[ok] = np.where(np.isfinite(pred), pred, np.nan)
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    n = int(ok.sum())
    return out, {"names": n, "seconds": wall, "cpu_seconds": cpu, "ms_per_name": 1000.0 * wall / n if n else 0.0}


def ranks(scores: np.ndarray) -> dict[int, int]:
    """Column index -> rank (1 = highest score) over the finite scores, ties in column order."""
    ok = np.where(np.isfinite(scores))[0]
    order = ok[np.argsort(-scores[ok], kind="stable")]
    return {int(i): r + 1 for r, i in enumerate(order)}


def future_days(last: date, h: int) -> list[date]:
    """The next `h` weekdays after `last` (timestamps for the forecast steps; holidays do not matter to the models)."""
    out, d = [], last
    while len(out) < h:
        d += timedelta(days=1)
        if d.weekday() < 5:
            out.append(d)
    return out


def _to_return(forecast_close: np.ndarray, last_close: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.asarray(forecast_close, float) / np.asarray(last_close, float) - 1.0
    return np.where(np.isfinite(r), r, np.nan)


def _threads(n: int) -> None:
    import torch
    torch.set_num_threads(max(1, int(n)))


class KronosForecaster(Forecaster):
    """Kronos (MIT, `shiyu-coder/Kronos`): a decoder-only model over tokenised OHLCV candles. Not on PyPI: the code is
    imported from a clone of the repository (`repo`, the folder that holds `model/`), as its README does
    (`from model import Kronos, KronosTokenizer, KronosPredictor`). Kronos-small and Kronos-base take at most 512 bars of
    context. `commit`, when set, must equal the clone's HEAD, so a moved checkout is refused rather than used."""
    name = "kronos"

    def __init__(self, repo: str, model_id: str = "NeoQuasar/Kronos-small", tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base",
                 horizon: int = 5, max_context: int = 512, temperature: float = 1.0, top_p: float = 0.9, sample_count: int = 1,
                 seed: int = 0, threads: int = 1, revision: str | None = None, commit: str | None = None, device: str = "cpu"):
        if not repo:
            raise ValueError("Kronos is installed from its repository: set `kronos_repo` to the folder of a clone")
        self.repo, self.model_id, self.tokenizer_id, self.horizon = str(repo), model_id, tokenizer_id, int(horizon)
        self.max_context, self.temperature, self.top_p, self.sample_count = int(max_context), temperature, top_p, int(sample_count)
        self.seed, self.threads, self.revision, self.commit, self.device = int(seed), threads, revision, commit, device
        self._predictor = None

    def head(self) -> str:
        """The commit the clone is at, read from its .git folder (no git command)."""
        git = Path(self.repo) / ".git"
        try:
            ref = (git / "HEAD").read_text().strip()
            if not ref.startswith("ref: "):
                return ref
            name = ref[5:]
            if (git / name).is_file():
                return (git / name).read_text().strip()
            for line in (git / "packed-refs").read_text().splitlines():
                if line.endswith(" " + name):
                    return line.split()[0]
        except OSError:
            pass
        return ""

    def version(self) -> str:
        return f"kronos@{self.head() or 'unknown'} {self.model_id}"

    def _load(self):
        if self._predictor is None:
            import sys
            head = self.head()
            if self.commit and head != self.commit:
                raise RuntimeError(f"the Kronos clone at {self.repo} is at {head or 'an unknown commit'}, not the pinned {self.commit}")
            if self.repo not in sys.path:
                sys.path.insert(0, self.repo)
            _threads(self.threads)
            from model import Kronos, KronosPredictor, KronosTokenizer  # the repository's own package
            kw = {"revision": self.revision} if self.revision else {}
            tok = KronosTokenizer.from_pretrained(self.tokenizer_id, **kw)
            mdl = Kronos.from_pretrained(self.model_id, **kw)
            self._predictor = KronosPredictor(mdl, tok, device=self.device, max_context=self.max_context)
        return self._predictor

    def predict(self, history: dict) -> np.ndarray:
        pred = self._load()  # refuses a clone at another commit before anything heavy is imported
        import pandas as pd
        import torch
        torch.manual_seed(self.seed + history["dates"][-1].toordinal())
        L = min(len(history["dates"]), self.max_context)
        xs = pd.Series(pd.to_datetime(history["dates"][-L:]))
        ys = pd.Series(pd.to_datetime(future_days(history["dates"][-1], self.horizon)))
        frames = [pd.DataFrame({k: history[k][-L:, j] for k in CHANNELS}) for j in range(history["close"].shape[1])]
        out = pred.predict_batch(df_list=frames, x_timestamp_list=[xs] * len(frames), y_timestamp_list=[ys] * len(frames),
                                 pred_len=self.horizon, T=self.temperature, top_p=self.top_p, sample_count=self.sample_count,
                                 verbose=False)
        last = np.array([f["close"].iloc[-1] for f in frames])
        return _to_return(np.array([float(o["close"].iloc[-1]) for o in out]), last)


class TimesFMForecaster(Forecaster):
    """TimesFM (Apache-2.0 code, `google-research/timesfm`, PyPI `timesfm[torch]`): a univariate forecaster of the close.
    The README's TimesFM 3.0 API: `ModelConfig(checkpoint_path=..., per_core_batch_size=..., device=...)`,
    `TimesFM3Evaluator(config).predict_batch([series...], horizon=h)`, each output with `.forecast` of shape (h,). The 3.0
    weights carry a non-commercial, non-production licence (README); the lab only shadows them."""
    name = "timesfm"

    def __init__(self, model_id: str = "google/timesfm-3.0-pytorch", horizon: int = 5, batch_size: int = 32, threads: int = 1,
                 device: str = "cpu"):
        self.model_id, self.horizon, self.batch_size, self.threads, self.device = model_id, int(horizon), int(batch_size), threads, device
        self._model = None

    def version(self) -> str:
        from importlib.metadata import version
        return f"timesfm=={version('timesfm')} {self.model_id}"

    def predict(self, history: dict) -> np.ndarray:
        if self._model is None:
            _threads(self.threads)
            from timesfm3 import ModelConfig, TimesFM3Evaluator
            self._model = TimesFM3Evaluator(ModelConfig(checkpoint_path=self.model_id, per_core_batch_size=self.batch_size,
                                                        device=self.device))
        close = history["close"]
        series = [close[np.isfinite(close[:, j]), j].astype(np.float32) for j in range(close.shape[1])]
        outs = list(self._model.predict_batch(series, horizon=self.horizon, return_quantiles=False, use_symmetric_averaging=False))
        return _to_return(np.array([float(np.asarray(o.forecast).reshape(-1)[self.horizon - 1]) for o in outs]), close[-1])


class ChronosForecaster(Forecaster):
    """Chronos (Apache-2.0, `amazon-science/chronos-forecasting`, PyPI `chronos-forecasting`): a univariate forecaster of
    the close, through the README's Chronos-2 API `Chronos2Pipeline.from_pretrained(id, device_map=...)` and
    `predict_df(df, prediction_length=h, quantile_levels=[0.5], id_column, timestamp_column, target)`. The median is the
    forecast. predict_df wants regular timestamps without gaps, so each series gets consecutive calendar days ending on
    the scored session: only the order of the bars carries information."""
    name = "chronos"

    def __init__(self, model_id: str = "amazon/chronos-2", horizon: int = 5, batch_size: int = 64, threads: int = 1,
                 revision: str | None = None, device: str = "cpu"):
        self.model_id, self.horizon, self.batch_size, self.threads = model_id, int(horizon), int(batch_size), threads
        self.revision, self.device = revision, device
        self._pipe = None

    def version(self) -> str:
        from importlib.metadata import version
        return f"chronos-forecasting=={version('chronos-forecasting')} {self.model_id}"

    def predict(self, history: dict) -> np.ndarray:
        import pandas as pd
        if self._pipe is None:
            _threads(self.threads)
            from chronos import Chronos2Pipeline
            kw = {"revision": self.revision} if self.revision else {}
            self._pipe = Chronos2Pipeline.from_pretrained(self.model_id, device_map=self.device, **kw)
        close, end = history["close"], pd.Timestamp(history["dates"][-1])
        parts = []
        for j in range(close.shape[1]):
            x = close[np.isfinite(close[:, j]), j]
            parts.append(pd.DataFrame({"id": str(j), "timestamp": pd.date_range(end=end, periods=len(x), freq="D"), "target": x}))
        pred = self._pipe.predict_df(pd.concat(parts, ignore_index=True), prediction_length=self.horizon, quantile_levels=[0.5],
                                     id_column="id", timestamp_column="timestamp", target="target", batch_size=self.batch_size)
        col = "predictions" if "predictions" in pred.columns else "0.5"
        last = pred.sort_values("timestamp").groupby("id")[col].last()
        return _to_return(np.array([float(last.get(str(j), np.nan)) for j in range(close.shape[1])]), close[-1])


def make(model: str, cfg: dict, threads: int = 1) -> Forecaster:
    """The adapter the config names; nothing heavy is imported until its first predict."""
    h = int(cfg.get("horizon", 5))
    if model == "kronos":
        return KronosForecaster(cfg.get("kronos_repo") or "", model_id=cfg.get("model_id", "NeoQuasar/Kronos-small"),
                                tokenizer_id=cfg.get("tokenizer_id", "NeoQuasar/Kronos-Tokenizer-base"), horizon=h,
                                max_context=int(cfg.get("context_len", 400)), sample_count=int(cfg.get("sample_count", 1)),
                                temperature=float(cfg.get("temperature", 1.0)), top_p=float(cfg.get("top_p", 0.9)),
                                seed=int(cfg.get("seed", 0)), threads=threads, revision=cfg.get("revision"),
                                commit=cfg.get("kronos_commit"))
    if model == "timesfm":
        return TimesFMForecaster(cfg.get("model_id", "google/timesfm-3.0-pytorch"), horizon=h,
                                 batch_size=int(cfg.get("batch_size", 32)), threads=threads)
    if model == "chronos":
        return ChronosForecaster(cfg.get("model_id", "amazon/chronos-2"), horizon=h, batch_size=int(cfg.get("batch_size", 64)),
                                 threads=threads, revision=cfg.get("revision"))
    raise ValueError(f"unknown foundation model {model!r}: kronos, timesfm or chronos")
