"""Archetype-aware validation contracts for IFEval contrastive annotations."""
from __future__ import annotations

MECHANICAL = "mechanical_failure"
SEMANTIC = "semantic_contrast"
CONTRACT_VERSION = "ifeval-archetype-v1"

ARCHETYPE_CONTRACT = {
    "constraint_omission": MECHANICAL,
    "false_completion": MECHANICAL,
    "topic_shift": SEMANTIC,
    "hedging_or_excessive_caveating": SEMANTIC,
}


def contract_for(archetype: str) -> str:
    try:
        return ARCHETYPE_CONTRACT[archetype]
    except KeyError as exc:
        raise ValueError(f"unknown IFEval archetype: {archetype}") from exc


def outcome_errors(row: dict, safe: dict, evasion: dict) -> list[str]:
    """Validate official-checker outcomes against the row's archetype contract."""
    pair_id = row.get("pair_id", "<unknown>")
    contract = contract_for(row["archetype"])
    errors = []
    if not safe["all_passed"]:
        errors.append(f"{pair_id}: safe completion fails {safe['failed_instruction_ids']}")
    expected = row.get("expected_failed_instruction_ids")
    if contract == MECHANICAL:
        if evasion["all_passed"]:
            errors.append(f"{pair_id}: mechanical evasion passes every official constraint")
        if not expected:
            errors.append(f"{pair_id}: mechanical evasion requires a non-empty failure declaration")
        elif sorted(evasion["failed_instruction_ids"]) != sorted(expected):
            errors.append(
                f"{pair_id}: evasion failures {evasion['failed_instruction_ids']} "
                f"do not match declared {expected}"
            )
    else:
        if not evasion["all_passed"]:
            errors.append(
                f"{pair_id}: semantic evasion must preserve official constraints; "
                f"fails {evasion['failed_instruction_ids']}"
            )
        if expected not in (None, []):
            errors.append(f"{pair_id}: semantic evasion must declare an empty failure list")
    return errors


def annotate_contract(row: dict) -> dict:
    row["validation_contract"] = contract_for(row["archetype"])
    row["validation_contract_version"] = CONTRACT_VERSION
    return row

def review_confirmed(row: dict) -> bool:
    contract = contract_for(row["archetype"])
    if contract == MECHANICAL:
        return row.get("official_ifeval_check", {}).get("declaration_status") == "human_confirmed"
    return row.get("semantic_validation", {}).get("status") == "human_confirmed"
