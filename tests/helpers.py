"""Shared fixtures. Trains ONE small model on SYNTHETIC data (mechanics only) and caches it for the whole run."""
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from machine_learning import data_loader as dl
from machine_learning import train_model as tm
from tests.synthetic import make_synthetic_traffic, populate_db

_CACHE = {}


def memory_db(days=14, **kw):
    segs, obs = make_synthetic_traffic(days=days, **kw)
    db = dl.Database(sqlite3.connect(":memory:"))
    populate_db(db, segs, obs)
    return db, segs, obs


def trained_env(days=14, horizons=(30,), fresh_until_now=False, seed=7):
    """Return dict(db_path, models_dir, now, last_obs) with a deployed model. Cached per argument set."""
    key = (days, tuple(horizons), fresh_until_now, seed)
    if key in _CACHE:
        return _CACHE[key]
    tmp = Path(tempfile.mkdtemp(prefix="mltest_"))
    if fresh_until_now:   # data that ends a few minutes ago, so the service sees it as fresh
        end = pd.Timestamp.now("UTC").tz_localize(None).floor("5min") - pd.Timedelta(minutes=5)
        start = (end - pd.Timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    else:
        start = "2026-01-05 00:00:00"
    segs, obs = make_synthetic_traffic(days=days, start=start, seed=seed)
    db = dl.Database(sqlite3.connect(str(tmp / "t.db")))
    populate_db(db, segs, obs)
    now = pd.Timestamp(obs["recorded_at"].max()) + pd.Timedelta(minutes=2)
    res = tm.train_all(db, list(horizons), str(tmp / "models"), fast=True, now=now if not fresh_until_now else None, plots=False, log=lambda *a: None)
    env = {"db_path": str(tmp / "t.db"), "models_dir": str(tmp / "models"), "now": now, "last_obs": pd.Timestamp(obs["recorded_at"].max()),
           "result": res, "segs": segs, "obs": obs, "tmp": tmp}
    db.conn.close()
    _CACHE[key] = env
    return env
