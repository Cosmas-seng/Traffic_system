
import hashlib
import hmac
import json
import os
import shutil
from pathlib import Path

import joblib

from . import model_config as cfg


class ModelError(Exception):
    """Raised for missing / corrupt / untrusted model files. Messages are safe to log, not to show users."""


def _root(root=None):
    return Path(root or cfg.MODELS_DIR)


def _hdir(horizon, root=None):
    return _root(root) / f"h{int(horizon)}"


def _digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _signature(path):
    key = os.environ.get("ML_MODEL_SIGNING_KEY")
    if not key:
        return None
    return hmac.new(key.encode(), Path(path).read_bytes(), hashlib.sha256).hexdigest()


def _write_json_atomic(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    os.replace(tmp, path)


def save_bundle(bundle, report, horizon, root=None, make_current=True):
    hdir = _hdir(horizon, root)
    vdir = hdir / bundle["version"]
    vdir.mkdir(parents=True, exist_ok=False)
    path = vdir / "bundle.joblib"
    joblib.dump(bundle, path, compress=3)
    manifest = {"version": bundle["version"], "horizon": int(horizon), "trained_at": bundle["trained_at"],
                "sha256": _digest(path), "hmac_sha256": _signature(path), "library_versions": bundle.get("library_versions", {})}
    _write_json_atomic(vdir / "manifest.json", manifest)
    _write_json_atomic(vdir / "metrics.json", report)
    if make_current:
        _set_current(hdir, bundle["version"])
    _prune(hdir)
    return vdir


def _set_current(hdir, version):
    cur = hdir / "current.json"
    previous = None
    if cur.exists():
        old = json.loads(cur.read_text()).get("version")
        previous = old if old != version else json.loads(
            cur.read_text()).get("previous")
    _write_json_atomic(cur, {"version": version, "previous": previous})


def _prune(hdir):
    keep = {json.loads((hdir / "current.json").read_text()).get(k)
            for k in ("version", "previous")} if (hdir / "current.json").exists() else set()
    versions = sorted([p for p in hdir.iterdir() if p.is_dir() and (
        p / "manifest.json").exists()], key=lambda p: p.name, reverse=True)
    for p in versions[cfg.KEEP_VERSIONS:]:
        if p.name not in keep:
            shutil.rmtree(p, ignore_errors=True)


def current_version(horizon, root=None):
    cur = _hdir(horizon, root) / "current.json"
    return json.loads(cur.read_text()).get("version") if cur.exists() else None


def current_pointer_mtime(horizon, root=None):
    cur = _hdir(horizon, root) / "current.json"
    return cur.stat().st_mtime_ns if cur.exists() else None


def load_bundle(horizon, root=None, version=None):
    version = version or current_version(horizon, root)
    if not version:
        raise ModelError(f"no deployed model for horizon {horizon}")
    hdir = _hdir(horizon, root).resolve()
    vdir = (hdir / version).resolve()
    if hdir not in vdir.parents:                        # blocks '../' tricks in a version string
        raise ModelError("version path escapes the models directory")
    mpath, bpath = vdir / "manifest.json", vdir / "bundle.joblib"
    if not (mpath.exists() and bpath.exists()):
        raise ModelError(f"model files missing for h{horizon}/{version}")
    manifest = json.loads(mpath.read_text())
    if _digest(bpath) != manifest.get("sha256"):
        raise ModelError(
            "model file checksum mismatch (file changed or corrupt)")
    if os.environ.get("ML_MODEL_SIGNING_KEY"):
        if not manifest.get("hmac_sha256") or not hmac.compare_digest(manifest["hmac_sha256"], _signature(bpath)):
            raise ModelError("model signature invalid")
    bundle = joblib.load(bpath)
    if int(bundle.get("horizon", -1)) != int(horizon):
        raise ModelError("bundle horizon does not match its folder")
    return bundle


def list_versions(horizon, root=None):
    hdir = _hdir(horizon, root)
    return sorted([p.name for p in hdir.iterdir() if p.is_dir()], reverse=True) if hdir.exists() else []


def rollback(horizon, root=None):
    """Point 'current' back to the previously deployed version (kept on disk for this purpose)."""
    hdir = _hdir(horizon, root)
    cur = json.loads((hdir / "current.json").read_text())
    prev = cur.get("previous")
    if not prev or not (hdir / prev / "bundle.joblib").exists():
        raise ModelError("no previous version available to roll back to")
    _write_json_atomic(hdir / "current.json",
                       {"version": prev, "previous": None})
    return prev


def read_metrics(horizon, root=None, version=None):
    version = version or current_version(horizon, root)
    p = _hdir(horizon, root) / (version or "") / "metrics.json"
    return json.loads(p.read_text()) if version and p.exists() else None


def deployed_horizons(root=None):
    r = _root(root)
    return sorted(int(p.name[1:]) for p in r.glob("h*") if p.is_dir() and (p / "current.json").exists())
