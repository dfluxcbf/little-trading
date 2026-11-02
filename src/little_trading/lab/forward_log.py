"""What every scorer of the forward log shares (REQ-000000027): which names it may score and how a session's coverage is kept.

The live database holds bars for only some symbols on recent sessions (the newest session of the 2026-10-29 snapshot had bars
for about a quarter of the universe), and `Panel` forward-fills closes for features. Left alone, a rule on a forward-filled
feature scores every name while the trees score only the tradable ones (208 names against 47 that day), so the models of one
session are compared on different names, partly at stale prices. `scorable` is the one mask: a real bar on the scored
session, tradable, and every ranked feature known. It is what `panel.ranked()` (the input of the trees) already requires, so
all scorers can keep to it, and it reads nothing after the session. A name without a bar stays unscored on that session;
nothing is ever back-filled.

`shadow_coverage` holds, per session and model, the names scored, the names with a real bar and the universe, in the
scorer's own database. `shadow-eval` reads it and states it in its report.
"""
from __future__ import annotations

import numpy as np

COVERAGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_coverage (asof TEXT, model TEXT, names INTEGER, bars INTEGER, universe INTEGER,
                                            snapshot TEXT, run_id TEXT, PRIMARY KEY (asof, model));
"""


def real_bars(panel, row: int) -> np.ndarray:
    """(N,) True where the company has its own open and close on panel session `row`, never a forward-filled one."""
    return np.isfinite(panel.open[row]) & np.isfinite(panel.close[row])


def scorable(panel, row: int) -> np.ndarray:
    """(N,) True for the names every forward scorer scores on panel session `row`, from rows up to `row` only."""
    return real_bars(panel, row) & panel.tradable[row] & np.isfinite(panel.ranked()[row]).all(axis=-1)


def record_coverage(db, asof: str, model: str, names: int, panel, row: int, snapshot: str, run_id: str) -> dict:
    """Stores the coverage of one scored session in the scorer's own database and returns it."""
    db.executescript(COVERAGE_SCHEMA)
    out = {"names": int(names), "bars": int(real_bars(panel, row).sum()), "universe": len(panel.symbols)}
    db.execute("INSERT OR REPLACE INTO shadow_coverage VALUES (?,?,?,?,?,?,?)",
               (asof, model, out["names"], out["bars"], out["universe"], snapshot, run_id))
    return out
