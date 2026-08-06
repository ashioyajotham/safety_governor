"""Import human-approved Gemini drafts into the restricted research working set."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

PLACEHOLDER = re.compile(r"^\s*\[\s*(?:compliant|safe|evasive|naturalistic|insert|todo|tbd)\b", re.I)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("input"); parser.add_argument("--output", default="data/raw/research_corpus/ifeval_gemini_reviewed.jsonl"); args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line]
    failures = []
    for row in rows:
        for field in ("safe_completion", "naturalistic_evasion", "notes"):
            value = row.get(field, "").strip()
            if not value or (field != "notes" and PLACEHOLDER.search(value)): failures.append(f"{row.get('pair_id')}: invalid {field}")
        row["annotation_status"] = "approved"
        row["review_notes"] = "Human-approved model-assisted draft; see notes for pair rationale."
    if failures: raise SystemExit("Import blocked:\n- " + "\n- ".join(failures))
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(f"Imported {len(rows)} approved drafts to {output}")

if __name__ == "__main__": main()
