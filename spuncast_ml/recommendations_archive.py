from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import psycopg2

from spuncast_ml.db import get_conn


def archive_recommendations(days: int = 60) -> dict[str, Any]:
    """Move rows older than ``days`` from ``heat_recommendations`` to ``heat_recommendations_archive``."""
    if days < 1:
        raise ValueError("days must be >= 1")

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = ANY (current_schemas(false)) "
                    "  AND table_name = 'heat_recommendations_archive' "
                    "LIMIT 1"
                )
                if cur.fetchone() is None:
                    return {
                        "status": "skip",
                        "reason": "heat_recommendations_archive_missing",
                        "hint": "Apply sql/074_heat_recommendations_archive.sql in the operations database.",
                    }

                cur.execute(
                    "WITH deleted AS ( "
                    "  DELETE FROM heat_recommendations "
                    "  WHERE created_at < NOW() - make_interval(days => %s) "
                    "  RETURNING id, heat_number, decision_code, recommendation_text, primary_driver, "
                    "            scrap_probability, feature_set, created_at, resolved_at, resolved_by "
                    ") "
                    "INSERT INTO heat_recommendations_archive ( "
                    "  id, heat_number, decision_code, recommendation_text, primary_driver, "
                    "  scrap_probability, feature_set, created_at, resolved_at, resolved_by "
                    ") "
                    "SELECT * FROM deleted",
                    (days,),
                )
                moved = cur.rowcount
            conn.commit()

        return {"status": "ok", "rows_archived": int(moved), "days": days}
    except psycopg2.OperationalError as exc:
        return {"status": "error", "reason": "database_unreachable", "detail": str(exc).strip()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Archive aged heat_recommendations rows for audit retention.")
    parser.add_argument("--days", type=int, default=60, help="Age threshold in days (default 60).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = archive_recommendations(days=args.days)
    print(json.dumps(summary, indent=2))
    if summary.get("status") == "skip":
        print(summary.get("hint", ""), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
