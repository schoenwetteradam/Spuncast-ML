from __future__ import annotations

import argparse
import json
from typing import Any

from spuncast_ml.modeling import model_dir


def _metrics_from_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    sel = meta.get("selected_model")
    cand = meta.get("candidate_models") or {}
    if not sel or sel not in cand:
        return {}
    return cand[sel].get("validation_metrics") or {}


def _accuracy(m: dict[str, Any]) -> float | None:
    if "accuracy" in m:
        return float(m["accuracy"])
    cr = m.get("classification_report")
    if isinstance(cr, dict) and "accuracy" in cr:
        return float(cr["accuracy"])
    return None


def _pr_auc(m: dict[str, Any]) -> float | None:
    v = m.get("pr_auc")
    return float(v) if v is not None else None


def improved(old: float, new: float, min_relative: float) -> bool:
    delta = float(new) - float(old)
    if abs(float(old)) <= 1e-9:
        return delta >= min_relative
    return delta / float(old) >= min_relative


def promote_or_revert(
    feature_set: str,
    min_relative_improvement: float,
    *,
    dry_run: bool = False,
) -> int:
    """Compare the newest trained model to the previous artifact; delete the newest if it is not enough better."""
    root = model_dir()
    models = sorted(root.glob(f"scrap_baseline_{feature_set}_*.joblib"))
    if len(models) < 2:
        print(json.dumps({"status": "skip", "reason": "fewer_than_two_models", "model_count": len(models)}, indent=2))
        return 0

    new_path = models[-1]
    old_path = models[-2]
    new_meta_path = new_path.with_suffix(".json")
    old_meta_path = old_path.with_suffix(".json")
    if not new_meta_path.exists() or not old_meta_path.exists():
        print(json.dumps({"status": "error", "reason": "missing_metadata_sidecar"}, indent=2))
        return 1

    new_meta = json.loads(new_meta_path.read_text(encoding="utf-8"))
    old_meta = json.loads(old_meta_path.read_text(encoding="utf-8"))
    new_m = _metrics_from_metadata(new_meta)
    old_m = _metrics_from_metadata(old_meta)

    new_acc = _accuracy(new_m)
    old_acc = _accuracy(old_m)
    metric_name = "accuracy"
    if new_acc is None or old_acc is None:
        new_v = _pr_auc(new_m)
        old_v = _pr_auc(old_m)
        if new_v is None or old_v is None:
            print(json.dumps({"status": "skip", "reason": "no_comparable_metrics"}, indent=2))
            return 0
        new_acc, old_acc, metric_name = new_v, old_v, "pr_auc"

    rel = (float(new_acc) - float(old_acc)) / max(float(old_acc), 1e-9)
    passes = improved(float(old_acc), float(new_acc), min_relative_improvement)
    payload: dict[str, Any] = {
        "status": "keep" if passes else "revert",
        "metric": metric_name,
        "old": old_acc,
        "new": new_acc,
        "relative_improvement": rel,
        "threshold": min_relative_improvement,
        "old_model": str(old_path),
        "new_model": str(new_path),
        "dry_run": dry_run,
    }
    if not passes and not dry_run:
        new_path.unlink(missing_ok=True)
        new_meta_path.unlink(missing_ok=True)
        payload["deleted"] = [str(new_path), str(new_meta_path)]
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the newest model to the previous training artifact and delete the newest if it is not improved enough.",
    )
    parser.add_argument("--feature-set", default="early_remelt_decision")
    parser.add_argument("--min-relative-improvement", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return promote_or_revert(
        args.feature_set,
        args.min_relative_improvement,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
