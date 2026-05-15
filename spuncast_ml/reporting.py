from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spuncast_ml.modeling import model_dir, report_dir


def _latest_json(pattern: str) -> Path | None:
    matches = sorted(report_dir().glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _latest_model_metadata(feature_set: str) -> tuple[Path | None, dict[str, Any] | None]:
    models = sorted(model_dir().glob(f"scrap_baseline_{feature_set}_*.json"))
    if not models:
        return None, None
    latest = models[-1]
    return latest, json.loads(latest.read_text(encoding="utf-8"))


def generate_weekly_report(output_path: str | Path) -> Path:
    """Write a lightweight HTML summary from the latest evaluation JSON and model metadata."""
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    eval_path = _latest_json("evaluation_*.json")
    eval_blob: dict[str, Any] | None = None
    if eval_path and eval_path.exists():
        eval_blob = json.loads(eval_path.read_text(encoding="utf-8"))

    meta_path, meta = _latest_model_metadata("early_remelt_decision")
    if meta is None:
        meta_path, meta = _latest_model_metadata("pre_pour_in_process")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    parts: list[str] = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Weekly ML report</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:2rem;} table{border-collapse:collapse;} td,th{border:1px solid #ccc;padding:0.4rem 0.75rem;} code{background:#f4f4f4;padding:0.1rem 0.3rem;}</style>",
        "</head><body>",
        f"<h1>Spuncast-ML weekly summary</h1><p>Generated {html.escape(stamp)} UTC.</p>",
    ]

    if eval_blob:
        tm = eval_blob.get("test_metrics") or {}
        parts.append("<h2>Latest evaluation (hold-out)</h2><table>")
        for key in ("accuracy", "precision", "recall", "f1", "pr_auc", "roc_auc"):
            if key in tm:
                parts.append(f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(tm[key]))}</td></tr>")
        parts.append("</table>")
        parts.append(f"<p>Source: <code>{html.escape(str(eval_path))}</code></p>")
    else:
        parts.append("<p>No evaluation JSON found yet. Run <code>spuncast-ml evaluate</code> after training.</p>")

    if meta:
        parts.append("<h2>Latest trained model</h2><ul>")
        parts.append(f"<li>Selected: <code>{html.escape(str(meta.get('selected_model', '')))}</code></li>")
        parts.append(f"<li>Feature set: <code>{html.escape(str(meta.get('feature_set', '')))}</code></li>")
        gate = meta.get("promotion_gate") or {}
        parts.append(f"<li>Promotion gate passes: <strong>{html.escape(str(gate.get('passes')))}</strong></li>")
        parts.append("</ul>")
        parts.append(f"<p>Metadata: <code>{html.escape(str(meta_path))}</code></p>")
    else:
        parts.append("<p>No model metadata found yet.</p>")

    parts.append("</body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
