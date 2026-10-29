"""python -m little_trading.lab <command>

  snapshot SRC_DB   copy a server database's prices into the lab's own snapshot (prices.db)
  controls          Stage 0: run the controls on the development window and write the verdict
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

from . import experiments


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m little_trading.lab")
    p.add_argument("--dir", default=experiments.default_dir())
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("src")
    nb = sub.add_parser("news-backfill")
    nb.add_argument("--from", dest="first", default="2018-10-01")
    nb.add_argument("--extra", type=int, default=20)
    g1 = sub.add_parser("stage1")
    g1.add_argument("--seeds", type=int, default=3)
    g1.add_argument("--alts", default="P1-trees,P2-mlp,P3-transformer,P3-transformer-scratch,P4-gru")
    g1.add_argument("--windows", default="development,selection")
    sub.add_parser("embed")
    sub.add_parser("stage2")
    sub.add_parser("stage4")
    g3 = sub.add_parser("stage3")
    g3.add_argument("--seeds", type=int, default=3)
    g3.add_argument("--only", default="")
    c = sub.add_parser("controls")
    c.add_argument("--window", default="development")
    c.add_argument("--stage", default="stage0")
    a = p.parse_args(argv)
    os.makedirs(a.dir, exist_ok=True)
    db = os.path.join(a.dir, "prices.db")
    if a.cmd == "snapshot":
        shutil.copy(a.src, db)
        print("snapshot", db)
        return 0
    if a.cmd == "news-backfill":
        from datetime import date as _d
        from . import news_data
        syms = news_data.pick_companies(db, a.extra)
        print(f"{len(syms)} companies: {' '.join(syms)}", flush=True)
        n = news_data.backfill(db, syms, _d.fromisoformat(a.first), _d.today(), log=lambda m: print(m, flush=True))
        print("added", n)
        return 0
    if a.cmd == "embed":
        from . import text
        print("embedded", text.embed_headlines(db, os.path.join(a.dir, "emb.db"), log=lambda m: print(m, flush=True)))
        return 0
    if a.cmd == "stage2":
        from . import ablation
        ablation.stage2(db, os.path.join(a.dir, "emb.db"), experiments.ExperimentLog(os.path.join(a.dir, "lab.db")))
        return 0
    if a.cmd == "stage4":
        from . import decision
        decision.stage4(db, experiments.ExperimentLog(os.path.join(a.dir, "lab.db")), os.path.join(a.dir, "scores_mlp.npz"),
                        out=lambda m: print(m, flush=True))
        return 0
    if a.cmd == "stage3":
        from . import memory
        from .panel import Panel
        from .runner import run
        panel = Panel.from_db(db)
        log = experiments.ExperimentLog(os.path.join(a.dir, "lab.db"))
        params = [r for r in log.runs("stage1-tune") if r["alt"] == "P2-mlp"][-1]["config"]
        params = {k: v for k, v in params.items() if k in ("hidden", "dropout", "lr", "wd", "epochs")}
        print("MLP params", params, flush=True)
        for win in ("development", "selection"):
            for label in memory.variants(params, 0):
                if a.only and a.only not in label:
                    continue
                for seed in range(1 if "retrieval" == label.split()[-1] else a.seeds):
                    make, refit = memory.variants(params, seed)[label]
                    t0 = time.time()
                    r = run(panel, log, "stage3", label, make, seed=seed, win=win, refit_every=refit)
                    print(f"{label:30s} {win:11s} seed {seed} IC {r['ic']:+.4f} (t {r['ic_t']:+.1f}) spread "
                          f"{r['spread']:+.3%} (t {r['spread_t']:+.1f}) monthly {r['account_slow']['total_return']:+.0%} "
                          f"vsEW {r['excess_slow_vs_ew']:+.0%} [{time.time() - t0:.0f}s]", flush=True)
        return 0
    if a.cmd == "stage1":
        from . import alternatives as A
        from .panel import Panel
        from .runner import run
        panel = Panel.from_db(db)
        log = experiments.ExperimentLog(os.path.join(a.dir, "lab.db"))
        for name in a.alts.split(","):
            make, space = A.ALTERNATIVES[name]
            t0 = time.time()
            params, table = A.tune(lambda prm, make=make: make(0, **prm), space, panel)
            log.record("stage1-tune", name, "P", "tuning", 0, params, panel.data_hash(),
                       {"trials": [[ic, prm] for ic, prm in table]}, "random search on 2016-2018 only")
            print(f"{name}: tuned in {time.time() - t0:.0f}s -> {params}", flush=True)
            for win in a.windows.split(","):
                for seed in range(a.seeds):
                    t0 = time.time()
                    r = run(panel, log, "stage1", name, lambda make=make, seed=seed, params=params: make(seed, **params),
                            seed=seed, win=win, refit_every=252)
                    print(f"{name:24s} {win:11s} seed {seed} IC {r['ic']:+.4f} (t {r['ic_t']:+.1f}) spread "
                          f"{r['spread']:+.3%} (t {r['spread_t']:+.1f}) weekly {r['account']['total_return']:+.0%} "
                          f"monthly {r['account_slow']['total_return']:+.0%} vsEW {r['excess_slow_vs_ew']:+.0%} "
                          f"[{time.time() - t0:.0f}s]", flush=True)
        return 0
    if a.cmd == "controls":
        from .panel import Panel
        from .predictors import controls
        from .runner import run
        panel = Panel.from_db(db)
        log = experiments.ExperimentLog(os.path.join(a.dir, "lab.db"))
        print(f"{len(panel.symbols)} companies, {panel.dates[0]}..{panel.dates[-1]}, data {panel.data_hash()}")
        rows, rets = [], {}
        for name, mk in controls().items():
            t = time.time()
            r = run(panel, log, a.stage, name, mk, win=a.window)
            rets[name] = r["_returns"]
            rows.append((name, r))
            ac = r["account"]
            print(f"{name:18s} IC {r['ic']:+.4f} (t {r['ic_t']:+.1f})  top-quintile spread {r['spread']:+.4%} "
                  f"(t {r['spread_t']:+.1f})  account {ac['total_return']:+.1%} sharpe {ac['sharpe']:+.2f} "
                  f"fees {ac['fees']:.0f}  vs EW {r['excess_vs_ew']:+.1%}  DSR {r['dsr']:.2f}  [{time.time() - t:.0f}s]")
        import numpy as np
        from . import metrics
        n = min(len(v) for v in rets.values())
        pb = metrics.pbo(np.stack([v[:n] for v in rets.values()], axis=1), slices=10)
        print(f"probability of backtest overfitting across the controls: {pb['pbo']:.2f}")
        signal = [n_ for n_, r in rows if r["ic_t"] > 2 or r["spread_t"] > 2]
        print("verdict:", f"signal found in {signal}" if signal else
              "no control has a significant signal (all |IC t| and spread t below 2): the live 'no skill' verdict stands")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
