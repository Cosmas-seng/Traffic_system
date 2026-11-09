
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error,
                             precision_recall_fscore_support, r2_score)

from . import model_config as cfg


def regression_metrics(y_true, y_pred):
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {"mae": float(mean_absolute_error(y, p)), "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p)) if len(y) > 1 else float("nan"), "n": int(len(y))}


def classification_metrics(y_true, y_pred, labels=cfg.CLASS_LABELS):
    labels = list(labels)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(prec.mean()), "macro_recall": float(rec.mean()), "macro_f1": float(f1.mean()),
            "per_class": {l: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(sup[i])}
                          for i, l in enumerate(labels)},
            "labels": labels, "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(), "n": int(len(y_true))}


# ---------------------------------------------------------------------------------- chronological split
def time_cutoffs(t_min, t_max, train_frac=cfg.TRAIN_FRAC, val_frac=cfg.VAL_FRAC):
    span = t_max - t_min
    return t_min + span * train_frac, t_min + span * (train_frac + val_frac)


def chronological_split(sup, train_end, val_end):
    """Train/validation/test by TIME, with purging: a training row's *target* must lie inside the training
    period, so no training label is measured after the point where validation inputs begin."""
    train = sup[sup["target_time"] <= train_end]
    val = sup[(sup["ts"] > train_end) & (sup["target_time"] <= val_end)]
    test = sup[sup["ts"] > val_end]
    return train.copy(), val.copy(), test.copy()


def rolling_validation(sup, fit_fn, t_min, t_end, n_folds=3, start_frac=0.5):
    """Expanding-window check of stability: train on everything up to a cut, test on the next slice."""
    out, span = [], t_end - t_min
    for i in range(n_folds):
        cut = t_min + span * (start_frac + (1 - start_frac) * i / n_folds)
        nxt = t_min + span * \
            (start_frac + (1 - start_frac) * (i + 1) / n_folds)
        tr = sup[sup["target_time"] <= cut]
        te = sup[(sup["ts"] > cut) & (sup["target_time"] <= nxt)]
        if len(tr) < cfg.MIN_TRAIN_ROWS // 2 or len(te) < 50:
            continue
        pred = fit_fn(tr, te)
        out.append({"fold": i + 1, "train_until": str(cut), "n_train": int(len(tr)), "n_test": int(len(te)),
                    "mae": float(mean_absolute_error(te["target_speed"], pred)),
                    "persistence_mae": float(mean_absolute_error(te["target_speed"], te["speed_now"]))})
    return out


# -------------------------------------------------------------------------------- stored predictions
def realized_error(db, horizon=None, bin_minutes=cfg.BIN_MINUTES, since=None):
    """Compare STORED predictions with what was later actually measured (when both exist)."""
    if not (db.has_table("traffic_predictions") and db.has_table("traffic_records")):
        return {"n": 0}
    sql, params = ("SELECT segment_id, horizon_min, predicted_speed_kmh, predicted_congestion, target_time, model_version "
                   "FROM traffic_predictions WHERE 1=1"), []
    if horizon:
        sql += " AND horizon_min=?"
        params.append(int(horizon))
    if since:
        sql += " AND target_time>=?"
        params.append(str(since))
    preds = db.frame(sql, params)
    if preds.empty:
        return {"n": 0}
    preds["target_time"] = pd.to_datetime(preds["target_time"])
    lo, hi = preds["target_time"].min() - pd.Timedelta(
        minutes=bin_minutes), preds["target_time"].max() + pd.Timedelta(minutes=bin_minutes)
    obs = db.frame("SELECT segment_id, recorded_at, speed_kmh FROM traffic_records WHERE recorded_at>=? AND recorded_at<=? ORDER BY recorded_at",
                   (lo.strftime("%Y-%m-%d %H:%M:%S"), hi.strftime("%Y-%m-%d %H:%M:%S")))
    if obs.empty:
        return {"n": 0}
    obs["recorded_at"] = pd.to_datetime(obs["recorded_at"])
    m = pd.merge_asof(preds.sort_values("target_time"), obs.sort_values("recorded_at"), left_on="target_time",
                      right_on="recorded_at", by="segment_id", direction="nearest",
                      tolerance=pd.Timedelta(minutes=bin_minutes / 2)).dropna(subset=["speed_kmh"])
    if m.empty:
        return {"n": 0}
    return {"n": int(len(m)), "mae_kmh": float(np.abs(m["predicted_speed_kmh"] - m["speed_kmh"]).mean()),
            "by_horizon": {int(h): {"n": int(len(g)), "mae_kmh": float(np.abs(g["predicted_speed_kmh"] - g["speed_kmh"]).mean())}
                           for h, g in m.groupby("horizon_min")}}


# ------------------------------------------------------------------------------------------- reporting
def markdown_table(rows, headers):
    """rows: list of dict. Missing/None -> '-'. Floats to 3 decimals."""
    def f(v):
        return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    lines = ["| " + " | ".join(headers) + " |", "|" +
             "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(f(r.get(h))
                                for h in headers) + " |" for r in rows]
    return "\n".join(lines)


def save_diagnostic_plots(out_dir, horizon, y_true, y_pred, cm, labels):
    """Predicted-vs-actual scatter and confusion-matrix heat-map (matplotlib/seaborn, headless)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    idx = np.random.default_rng(cfg.RANDOM_STATE).choice(
        len(y_true), size=min(3000, len(y_true)), replace=False)
    ax.scatter(np.asarray(y_true)[idx], np.asarray(y_pred)[idx], s=4, alpha=.4)
    lim = [0, max(np.max(y_true), np.max(y_pred)) * 1.05]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set(xlabel="Actual speed (km/h)", ylabel="Predicted speed (km/h)",
           title=f"Test period, +{horizon} min")
    fig.tight_layout()
    fig.savefig(out / "speed_pred_vs_actual.png", dpi=110)
    plt.close(fig)
    if cm is not None:
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        sns.heatmap(np.asarray(cm), annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
        ax.set(xlabel="Predicted", ylabel="Actual",
               title=f"Congestion class, +{horizon} min (test)")
        fig.tight_layout()
        fig.savefig(out / "congestion_confusion_matrix.png", dpi=110)
        plt.close(fig)
