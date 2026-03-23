from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONTRACTS_DIR = Path(__file__).resolve().parent / "contracts"
DEFAULT_CONTRACT_VERSION = "v_ml_heat_dataset_v1"


class DataContractError(ValueError):
    """Raised when the upstream dataset contract has changed unexpectedly."""


def load_contract(contract_version: str = DEFAULT_CONTRACT_VERSION) -> dict[str, Any]:
    contract_path = CONTRACTS_DIR / f"{contract_version}.json"
    if not contract_path.exists():
        raise FileNotFoundError(f"Dataset contract file not found: {contract_path}")
    return json.loads(contract_path.read_text(encoding="utf-8"))


def validate_contract_columns(actual_columns: list[str], contract_version: str = DEFAULT_CONTRACT_VERSION) -> dict[str, Any]:
    contract = load_contract(contract_version)
    expected_columns = contract["expected_columns"]

    unexpected = [column for column in actual_columns if column not in expected_columns]
    missing = [column for column in expected_columns if column not in actual_columns]

    if unexpected or missing:
        raise DataContractError(
            "Upstream dataset contract mismatch detected. "
            f"Missing columns: {missing or 'none'}. "
            f"Unexpected columns: {unexpected or 'none'}. "
            "Coordinate breaking schema changes with Spuncast-Operations and update the pinned contract file before rerunning."
        )

    return contract
