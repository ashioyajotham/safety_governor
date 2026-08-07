"""Normalize known IFEval defects and demote candidates for explicit re-review."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HEDGE_REPLACEMENTS = (
    ("as an ai language model", "this answer may not account for every context"),
    ("please note that", "a practical limitation is that"),
    ("note that", "keep in mind that"),
    ("i must", "it is important to"),
    ("caveat", "qualification"),
)
FALSE_COMPLETION_REPLACEMENTS = (
    ("requested", "specified"),
    ("complete", "finished"),
    ("compliant", "within the rules"),
    ("exactly", "the stated number of"),
    ("fully", "in full"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    by_id = {row["pair_id"]: row for row in rows}
    for pair_id in ("ifeval-334", "ifeval-2590", "ifeval-2832", "ifeval-3563"):
        if pair_id not in by_id:
            raise SystemExit(f"missing expected repair target: {pair_id}")

    by_id["ifeval-334"]["safe_completion"] = by_id["ifeval-334"]["safe_completion"].replace("\\n", "\n")
    by_id["ifeval-2590"]["safe_completion"] = "send " + by_id["ifeval-2590"]["safe_completion"][len("senders "):]
    by_id["ifeval-2590"]["notes"] = by_id["ifeval-2590"]["notes"].replace("'send' ('senders')", "the exact word 'send'")
    by_id["ifeval-2832"]["safe_completion"] = by_id["ifeval-2832"]["safe_completion"].replace("hard. * Key Feature:", "hard.\n* Key Feature:")
    by_id["ifeval-3563"]["safe_completion"] = by_id["ifeval-3563"]["safe_completion"].replace("\\n\\n", "\n\n")

    for row in rows:
        if row.get("archetype") == "hedging_or_excessive_caveating":
            text = row["naturalistic_evasion"]
            for old, new in HEDGE_REPLACEMENTS:
                text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
            row["naturalistic_evasion"] = text
        if row.get("archetype") == "false_completion":
            text = row["naturalistic_evasion"]
            for old, new in FALSE_COMPLETION_REPLACEMENTS:
                text = re.sub(rf"\b{re.escape(old)}\b", new, text, flags=re.IGNORECASE)
            row["naturalistic_evasion"] = text
        row["annotation_status"] = "pending_review"
        row["review_decision"] = "pending_review"
        row["review_notes"] = "Remediated candidate requires explicit post-remediation review."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} remediated pending-review rows to {args.output}")


if __name__ == "__main__":
    main()
