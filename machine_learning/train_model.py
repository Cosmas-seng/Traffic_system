
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone

from . import data_cleaning as dc
from . import data_loader as dl
from . import feature_engineering as fe
from . import model_config as cfg
from . import model_evaluation as ev
from . import model_registry as reg
from .congestion_classifier import MajorityClassifier, PersistenceClassifier, candidate_classifiers, label_ratio
from .traffic_predictor import PersistenceRegressor, SeasonalMeanRegressor, build_pipeline, candidate_regressors

MIN_CLASS_ROWS = 30     # a congestion class needs this many training rows to be learnable
MIN_VAL_ROWS = 100


def _ref(df, fcfg):
    return df["segment_id"].map(lambda s: fcfg["ref_speeds"][str(int(s))]).to_numpy(dtype=float)


def _ts_now(now):
    return pd.Timestamp(now) if now is not None else pd.Timestamp.now("UTC").tz_localize(None)


def train_all(db, horizons=None, models_dir=None, fast=False, deploy=True, force=False, now=None, plots=True, log=print):
    now = _ts_now(now)
    horizons = list(horizons or cfg.SUPPORTED_HORIZONS)
    segs, obs = dl.load_segments(db), dl.load_observations(db)
    summary = {"started": str(now), "status": None, "horizons": {}}
    if segs.empty or obs.empty:
        summary.update(status="no_data", message="No road segments or traffic records exist. Import a real dataset first "
                                                 "(python -m machine_learning.import_data --help).")
        return summary
    clean, qrep = dc.clean_observations(obs, segs, now=now)
    summary["data_quality"] = qrep
    if qrep["span_days"] < cfg.MIN_SPAN_DAYS:
        summary.update(status="insufficient_data", message=f"Only {qrep['span_days']} days of data; at least {cfg.MIN_SPAN_DAYS} "
                       "are needed to learn daily and weekly patterns.")
        return summary
    grid = dc.to_grid(clean)
    summary["grid_coverage"] = dc.grid_quality(grid)
    t_min, t_max = grid["ts"].min(), grid["ts"].max()
    train_end, val_end = ev.time_cutoffs(t_min, t_max)
    inc = dl.load_incidents(db, end=t_max)
    refs = fe.reference_speeds(grid[grid["ts"] <= train_end], segs)
    fcfg = fe.make_feature_config(grid[grid["ts"] <= train_end], segs, inc[pd.to_datetime(
        inc["created_at"]) <= train_end] if len(inc) else inc, refs)
    feats = fe.build_features(grid, fcfg, inc)
    summary["split"] = {"first": str(t_min), "train_end": str(
        train_end), "validation_end": str(val_end), "last": str(t_max)}
    log(f"data: {qrep['rows_out']} rows, {qrep['segments']} segments, {qrep['span_days']} days; "
        f"train<= {train_end:%Y-%m-%d %H:%M}, validation<= {val_end:%Y-%m-%d %H:%M}, test after")

    for h in horizons:
        log(f"--- horizon {h} min")
        summary["horizons"][str(h)] = _train_horizon(h, feats, grid, fcfg, inc, t_min, train_end, val_end, models_dir,
                                                     fast, deploy, force, plots, now, log)
    deployed = [h for h, r in summary["horizons"].items()
                if r.get("status") == "deployed"]
    summary["status"] = "ok" if deployed else "no_model_deployed"
    root = Path(models_dir or cfg.MODELS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    (root / "training_report.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _train_horizon(h, feats, grid, fcfg, inc, t_min, train_end, val_end, models_dir, fast, deploy, force, plots, now, log):
    out = {"horizon": h}
    if h not in cfg.SUPPORTED_HORIZONS or h % fcfg["bin_minutes"]:
        return {**out, "status": "skipped", "reason": f"horizon {h} min is not supported by the {fcfg['bin_minutes']}-minute grid / configuration"}
    sup = fe.make_supervised(feats, grid, h, fcfg)
    train, val, test = ev.chronological_split(sup, train_end, val_end)
    out["rows"] = {"train": len(
        train), "validation": len(val), "test": len(test)}
    if len(train) < cfg.MIN_TRAIN_ROWS or len(val) < MIN_VAL_ROWS or len(test) < cfg.MIN_TEST_ROWS:
        return {**out, "status": "skipped", "reason": f"not enough rows (train>={cfg.MIN_TRAIN_ROWS}, validation>={MIN_VAL_ROWS}, test>={cfg.MIN_TEST_ROWS} required)"}

    cols = fe.feature_columns(fcfg)
    def X(d): return d[cols + ["segment_id"]]
    trval = pd.concat([train, val], ignore_index=True)

    # ------------------------------------------------------------------ speed regression
    val_rows, test_rows, fit_train, fit_final, preds_test = [], [], {}, {}, {}

    def add(rows, name, kind, pred, df):
        rows.append({"model": name, "kind": kind, **
                    ev.regression_metrics(df["target_speed"], pred)})

    pers = PersistenceRegressor()
    seas_tr = SeasonalMeanRegressor().fit(train, train["target_speed"])
    seas_tv = SeasonalMeanRegressor().fit(trval, trval["target_speed"])
    add(val_rows, "Baseline: persistence", "baseline", pers.predict(val), val)
    add(val_rows, "Baseline: seasonal mean",
        "baseline", seas_tr.predict(val), val)
    add(test_rows, "Baseline: persistence",
        "baseline", pers.predict(test), test)
    add(test_rows, "Baseline: seasonal mean",
        "baseline", seas_tv.predict(test), test)
    regs = candidate_regressors(fast)
    for name, (est, scale) in regs.items():
        p1 = build_pipeline(clone(est), cols, scale).fit(
            X(train), train["target_speed"])
        add(val_rows, name, "model", p1.predict(X(val)), val)
        p2 = build_pipeline(clone(est), cols, scale).fit(
            X(trval), trval["target_speed"])   # final: train+validation
        preds_test[name] = p2.predict(X(test))
        add(test_rows, name, "model", preds_test[name], test)
        fit_train[name], fit_final[name] = p1, p2
    best = min((r for r in val_rows if r["kind"] == "model"), key=lambda r: r["mae"])[
        "model"]    # chosen on VALIDATION only
    est_best, scale_best = regs[best]

    def fit_fn(tr, te):
        return build_pipeline(clone(est_best), cols, scale_best).fit(X(tr), tr["target_speed"]).predict(X(te))
    rolling = ev.rolling_validation(trval, fit_fn, t_min, val_end)

    # ------------------------------------------------------------------ congestion classification
    cls = {"supported": False}
    counts = train["target_class"].value_counts()
    if (counts >= MIN_CLASS_ROWS).sum() >= 2:
        cls["supported"] = True
        cval, ctest, cfit_train, cfit_final = [], [], {}, {}

        def addc(rows, name, kind, pred, df):
            m = ev.classification_metrics(df["target_class"], pred)
            rows.append({"model": name, "kind": kind, "accuracy": m["accuracy"], "macro_precision": m["macro_precision"],
                         "macro_recall": m["macro_recall"], "macro_f1": m["macro_f1"], "n": m["n"], "details": m})

        maj_tr, maj_tv = MajorityClassifier().fit(
            train, train["target_class"]), MajorityClassifier().fit(trval, trval["target_class"])
        perc = PersistenceClassifier()
        addc(cval, "Baseline: persistence", "baseline", perc.predict(val), val)
        addc(cval, "Baseline: majority class",
             "baseline", maj_tr.predict(val), val)
        addc(ctest, "Baseline: persistence",
             "baseline", perc.predict(test), test)
        addc(ctest, "Baseline: majority class",
             "baseline", maj_tv.predict(test), test)
        addc(cval, "Derived from speed model", "model", label_ratio(
            fit_train[best].predict(X(val)) / _ref(val, fcfg), **fcfg["thresholds"]), val)
        addc(ctest, "Derived from speed model", "model", label_ratio(
            preds_test[best] / _ref(test, fcfg), **fcfg["thresholds"]), test)
        clfs = candidate_classifiers(fast)
        for name, (est, scale) in clfs.items():
            p1 = build_pipeline(clone(est), cols, scale).fit(
                X(train), train["target_class"])
            addc(cval, name, "model", p1.predict(X(val)), val)
            p2 = build_pipeline(clone(est), cols, scale).fit(
                X(trval), trval["target_class"])
            addc(ctest, name, "model", p2.predict(X(test)), test)
            cfit_final[name] = p2
        best_clf = max((r for r in cval if r["kind"] == "model" and r["model"]
                       != "Derived from speed model"), key=lambda r: r["macro_f1"])
        derived = next(
            r for r in cval if r["model"] == "Derived from speed model")
        cls.update(validation=cval, test=ctest, best_classifier=best_clf["model"],
                   source="derived" if derived["macro_f1"] >= best_clf["macro_f1"] else "classifier")
        cls["selected"] = "Derived from speed model" if cls["source"] == "derived" else best_clf["model"]
        final_clf = None if cls["source"] == "derived" else cfit_final[best_clf["model"]]
    else:
        final_clf = None
        cls["reason"] = f"fewer than 2 congestion classes have >= {MIN_CLASS_ROWS} training rows (class counts: {counts.to_dict()}); congestion is derived from predicted speed"

    # ------------------------------------------------------------------ acceptance rules
    t_best = next(r for r in test_rows if r["model"] == best)
    base_mae = min(r["mae"] for r in test_rows if r["kind"] == "baseline")
    improvement = (base_mae - t_best["mae"]) / \
        base_mae if base_mae > 0 else 0.0
    reasons = []
    if improvement < cfg.MIN_MAE_IMPROVEMENT_VS_PERSISTENCE:
        reasons.append(f"best model improves test MAE by {improvement:.1%} over the best baseline; "
                       f"{cfg.MIN_MAE_IMPROVEMENT_VS_PERSISTENCE:.0%} is required")
    deployed_cmp = None
    try:
        old = reg.load_bundle(h, models_dir)
    except reg.ModelError:
        old = None
    if old is not None:
        deployed_cmp = _compare_with_deployed(
            old, grid, inc, h, test, preds_test[best], val_end)
        if deployed_cmp and deployed_cmp["new_mae"] >= deployed_cmp["deployed_mae"]:
            reasons.append(f"not better than deployed version {deployed_cmp['deployed_version']} on the same test rows "
                           f"(new {deployed_cmp['new_mae']:.3f} vs deployed {deployed_cmp['deployed_mae']:.3f} km/h MAE)")
    accepted = not reasons or force
    out.update(validation=val_rows, test=test_rows, classification=cls, rolling_validation=rolling,
               selected={"regressor": best, "congestion": cls.get(
                   "selected", "derived from speed model")},
               acceptance={"accepted": bool(accepted), "forced": bool(force and reasons), "reasons": reasons,
                           "improvement_vs_best_baseline": improvement, "deployed_comparison": deployed_cmp},
               data={"train_end": str(train_end), "validation_end": str(val_end), "features": cols,
                     "optional_features": {k: fcfg[k] for k in ("use_vehicle_count", "use_density", "use_incidents", "use_month")}},
               notes=["Metrics are computed on the chronologically last, untouched test period.",
                      "Final models are refit on train+validation before test scoring; the model choice used validation only."])

    version = now.strftime("%Y%m%d%H%M%S")
    hdir = Path(models_dir or cfg.MODELS_DIR) / f"h{h}"
    while (hdir / version).exists() or (hdir / "rejected" / version).exists():
        version += "x"
    out["version"] = version
    if accepted and deploy:
        bundle = {"version": version, "horizon": h, "trained_at": str(now), "bin_minutes": fcfg["bin_minutes"],
                  "feature_config": fcfg, "feature_columns": cols, "regressor": fit_final[best], "regressor_name": best,
                  "classifier": final_clf, "classifier_name": cls.get("best_classifier") if final_clf is not None else None,
                  "metrics": {"test_mae_kmh": t_best["mae"], "test_rmse_kmh": t_best["rmse"], "test_r2": t_best["r2"],
                              "improvement_vs_best_baseline": improvement},
                  "library_versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__}}
        out["status"] = "deployed"
        vdir = reg.save_bundle(bundle, out, h, models_dir)
        log(
            f"deployed h{h} version {version}: {best}, test MAE {t_best['mae']:.2f} km/h ({improvement:.1%} better than best baseline)")
    else:
        out["status"] = "trained_not_deployed" if accepted else "rejected"
        vdir = hdir / "rejected" / version
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "metrics.json").write_text(json.dumps(out, indent=2, default=str))
        log(f"h{h} not deployed: {'; '.join(reasons) or '--no-deploy'}")
    (vdir / "comparison.md").write_text(comparison_markdown(out))
    if plots:
        tc = next((r for r in cls.get("test", [])
                  if r["model"] == cls.get("selected")), None)
        ev.save_diagnostic_plots(vdir, h, test["target_speed"], preds_test[best], tc["details"]["confusion_matrix"] if tc else None,
                                 list(cfg.CLASS_LABELS))
    return out


def _compare_with_deployed(old, grid, inc, h, test, new_pred, val_end):
    """Score the currently deployed model and the new one on the SAME test rows."""
    ofc = old["feature_config"]
    osup = fe.make_supervised(fe.build_features(grid, ofc, inc), grid, h, ofc)
    osup = osup[osup["ts"] > val_end]
    t = test[["segment_id", "ts", "target_speed"]].copy()
    t["_new"] = new_pred
    m = t.merge(osup.drop(columns="target_speed"), on=["segment_id", "ts"])
    if m.empty:
        return None
    old_pred = old["regressor"].predict(
        m[old["feature_columns"] + ["segment_id"]])
    return {"deployed_version": old["version"], "n": int(len(m)),
            "deployed_mae": float(np.abs(old_pred - m["target_speed"]).mean()), "new_mae": float(np.abs(m["_new"] - m["target_speed"]).mean())}


def comparison_markdown(rep):
    h = rep["horizon"]
    lines = [f"# Model comparison - {h} minutes ahead",
             f"Status: **{rep['status']}**", ""]
    for split in ("validation", "test"):
        reg_rows = {r["model"]: r for r in rep.get(split, [])}
        cls_rows = {r["model"]: r for r in rep.get(
            "classification", {}).get(split, [])}
        merged = []
        for name in list(dict.fromkeys(list(reg_rows) + list(cls_rows))):
            r, c = reg_rows.get(name, {}), cls_rows.get(name, {})
            merged.append({"Model": name, "MAE (km/h)": r.get("mae"), "RMSE (km/h)": r.get("rmse"), "R2": r.get("r2"),
                           "Accuracy": c.get("accuracy"), "Macro-F1": c.get("macro_f1")})
        lines += [f"## {split.capitalize()} period", ev.markdown_table(merged,
                                                                       ["Model", "MAE (km/h)", "RMSE (km/h)", "R2", "Accuracy", "Macro-F1"]), ""]
    acc = rep["acceptance"]
    lines += ["## Acceptance", f"Accepted: {acc['accepted']}", *[f"- {r}" for r in acc["reasons"]], "",
              f"Rows: {rep['rows']}", "", "_All values were produced by this training run._"]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--db", help="SQLite path (default: DATABASE_PATH or ./traffic.db). For MySQL set DB_ENGINE=mysql + DB_* variables.")
    ap.add_argument("--horizons", nargs="+", type=int,
                    default=list(cfg.SUPPORTED_HORIZONS))
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--fast", action="store_true",
                    help="smaller models (quick check only)")
    ap.add_argument("--no-deploy", action="store_true",
                    help="evaluate and write reports but do not deploy")
    ap.add_argument("--force", action="store_true",
                    help="deploy even if acceptance rules fail (not recommended)")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    db = dl.Database.from_env(a.db)
    db.ensure_schema() if db.engine == "sqlite" else None
    res = train_all(db, a.horizons, a.models_dir, a.fast,
                    not a.no_deploy, a.force, plots=not a.no_plots)
    print(f"\nstatus: {res['status']}" +
          (f" - {res.get('message')}" if res.get("message") else ""))
    for h, r in res.get("horizons", {}).items():
        print(f"  {h} min: {r.get('status')} {r.get('reason', '')}")
    return 0 if res["status"] in ("ok", "no_model_deployed") else 1


if __name__ == "__main__":
    sys.exit(main())
