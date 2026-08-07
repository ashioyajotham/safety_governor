"""Apply an explicit human review decision to annotation candidates."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

PLACEHOLDER = re.compile(r"^\s*\[\s*(?:compliant|safe|evasive|naturalistic|insert|todo|tbd)\b", re.I)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--decision", required=True, choices=("approved", "rejected"))
    parser.add_argument("--review-note", required=True)
    parser.add_argument("--reviewer", default="", help="Optional reviewer identity; explicit decision evidence is required regardless.")
    args = parser.parse_args()
    if not args.review_note.strip(): raise SystemExit("Import blocked: --review-note must be non-empty")
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line]
    failures = []
    for row in rows:
        for field in ("safe_completion", "naturalistic_evasion", "notes"):
            value = row.get(field, "").strip()
            if not value or (field != "notes" and PLACEHOLDER.search(value)): failures.append(f"{row.get('pair_id')}: invalid {field}")
        row["annotation_status"] = args.decision
        row["review_decision"] = args.decision
        row["review_notes"] = args.review_note.strip()
        if args.reviewer.strip(): row["reviewer"] = args.reviewer.strip()
    if failures: raise SystemExit("Import blocked:\n- " + "\n- ".join(failures))
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"Imported {len(rows)} {args.decision} drafts to {output}")

if __name__ == "__main__": main()
