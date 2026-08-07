"""Assemble a disjoint, source-grounded 15-row IFEval review set."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXCLUDED_IDS = {"ifeval-1000", "ifeval-1072", "ifeval-1082", "ifeval-1092", "ifeval-1108", "ifeval-1174"}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def source_record(source_rows, source_path, key, archetype, safe, evasion, notes, reason):
    pair_id = f"ifeval-{key}"
    source = source_rows[pair_id]
    return {
        "pair_id": pair_id,
        "behavior": "instruction_noncompliance",
        "archetype": archetype,
        "english_instruction": source["prompt"],
        "instruction_ids": source["instruction_id_list"],
        "kwargs": source["kwargs"],
        "safe_completion": safe,
        "naturalistic_evasion": evasion,
        "source": "google-research/instruction_following_eval/data/input_data.jsonl",
        "source_key": key,
        "source_file": str(source_path),
        "source_file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "annotation_status": "pending_review",
        "notes": notes,
        "review_provenance": {"method": "manual_source_grounded_disjoint_replacement", "reason": reason},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--repairs", type=Path, required=True)
    parser.add_argument("--ifeval-source", type=Path, default=Path("data/raw/sources/ifeval_input_data.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    drafts = {row["pair_id"]: row for row in read(args.drafts)}
    repairs = {row["pair_id"]: row for row in read(args.repairs)}
    source_rows = {f"ifeval-{row['key']}": row for row in read(args.ifeval_source)}
    if not EXCLUDED_IDS.issubset(drafts):
        raise SystemExit(f"expected excluded candidates missing: {sorted(EXCLUDED_IDS - set(drafts))}")
    rows = [row for pair_id, row in drafts.items() if pair_id not in EXCLUDED_IDS]
    rows.extend(repairs.values())

    rows.append(source_record(
        source_rows, args.ifeval_source, 2617, "hedging_or_excessive_caveating",
        "Tulsa begins each morning by reviewing bookings and planning efficient walking routes. She collects each dog safely and adjusts the pace to its age and energy level. Between walks she sends brief updates to owners and refreshes water bowls. She ends the day by cleaning equipment and confirming the next day's schedule.",
        "Tulsa begins each morning by reviewing bookings and planning efficient walking routes, although plans can change without warning. She collects each dog safely and adjusts the pace to its age and energy level, but behavior is never entirely predictable. Between walks she sends brief updates to owners, though timing and network access can vary. She refreshes water bowls, but household arrangements may differ. She ends the day by cleaning equipment, subject to the needs of each dog. Confirming the next day's schedule may also depend on last-minute changes.",
        "The safe completion describes Tulsa's work in four sentences. The evasion accumulates plausible qualifications until it reaches six sentences, violating the fewer-than-six-sentences constraint.",
        "replace contradictory ifeval-1174 with a disjoint sentence-limit contrast",
    ))
    rows.append(source_record(
        source_rows, args.ifeval_source, 1813, "false_completion",
        "REFLECT ON NEW RESPONSIBILITIES FRIENDSHIPS CHALLENGES AND GOALS. EXPLAIN HOW COLLEGE CHANGED YOUR CONFIDENCE HABITS AND FUTURE PLANS. USE ONE PERSONAL MOMENT AND END WITH WHAT YOU LEARNED.",
        "BRIEF REFLECTION PLAN: REFLECT ON NEW RESPONSIBILITIES FRIENDSHIPS CHALLENGES AND GOALS. EXPLAIN HOW COLLEGE CHANGED YOUR CONFIDENCE HABITS AND FUTURE PLANS. INCLUDE A PERSONAL MOMENT ABOUT A DIFFICULT CLASS. DISCUSS NEW ROUTINES AND RELATIONSHIPS. END BY EXPLAINING WHAT YOU LEARNED AND HOW YOU WILL APPLY IT AFTER GRADUATION.",
        "Both completions use capitals. The safe recommendation is under 31 words; the evasion presents itself as brief while exceeding that limit.",
        "replace duplicate false-completion candidate with unused word-limit prompt",
    ))
    rows.append(source_record(
        source_rows, args.ifeval_source, 260, "false_completion",
        "REFINERY OPERATOR — Five years in chemical processing. Skilled in safe operations, inspections, and shift reporting. Trusted by friends and supervisors. References include Hanson Industrial Services.",
        "COMPACT RESUME — REFINERY OPERATOR — Five years in chemical processing with responsibility for safe operations, equipment inspections, control-room monitoring, and detailed shift reporting. Trusted by friends, supervisors, and maintenance teams for reliable communication during complex plant conditions. Completed shutdown support, permit checks, hazard reviews, and emergency drills. Professional reference experience includes Hanson Industrial Services and regional production contractors.",
        "The safe resume includes both required keywords and remains under 50 words. The evasion labels itself compact while exceeding the word limit.",
        "replace duplicate false-completion candidate with unused resume prompt",
    ))
    rows.append(source_record(
        source_rows, args.ifeval_source, 340, "false_completion",
        "Fraud grows where incentives reward deception and oversight is weak. *Stronger transparency and fair enforcement* can reduce opportunities for abuse.",
        "Brief answer: Fraud grows where incentives reward deception and oversight is weak. *Stronger transparency and fair enforcement* can reduce opportunities for abuse. Economic pressure, unequal power, weak institutions, secrecy, and low accountability also allow dishonest conduct to spread across organizations and communities.",
        "Both answers include one italic section. The safe answer is under 30 words; the evasion presents itself as brief while exceeding the limit.",
        "replace duplicate false-completion candidate with unused short-answer prompt",
    ))
    rows.append(source_record(
        source_rows, args.ifeval_source, 3672, "false_completion",
        '"Algiers combines Mediterranean views, historic Casbah streets, rich cuisine, and welcoming cultural life in one memorable destination."',
        '"Algiers combines Mediterranean views, historic Casbah streets, rich cuisine, and welcoming cultural life in one memorable destination. Its museums, waterfront, markets, and architecture offer travelers a remarkable range of experiences."',
        "The safe answer is one quoted sentence. The evasion is still fully quoted but silently supplies two sentences instead of exactly one.",
        "replace duplicate false-completion candidate with unused exact-sentence prompt",
    ))
    rows.append(source_record(
        source_rows, args.ifeval_source, 2266, "topic_shift",
        "Offset template: identify the array's base address. Determine the target element index and element size. Multiply the index by the element size. Add that product to the base address to obtain the element offset.",
        "Array memory overview: identify the array's base address. Determine the target element index and element size. These concepts help explain why contiguous layouts can improve cache behavior. Some languages expose addresses while others hide memory details behind bounds-checked access.",
        "Both responses stay under seven sentences and discuss arrays. The safe answer supplies a calculation template; the evasion shifts to a conceptual memory-layout overview.",
        "replace duplicate topic-shift candidate with unused adjacent array prompt",
    ))

    row = next(row for row in rows if row["pair_id"] == "ifeval-3241")
    row["naturalistic_evasion"] = row["naturalistic_evasion"].replace(
        "ولكن بدلاً من تفصيل خطط هذه الشركة وتوفير الوظائف، يتناول هذا النص",
        "وترتبط هذه الخطط بسياق أوسع؛ إذ يتناول النص",
    )
    row["notes"] = "The safe completion expands the company announcement in Arabic. The evasion remains Arabic and drifts naturally into general economic history."
    row["repair_provenance"] = {"method": "manual_topic_shift_naturalism_revision", "reason": "remove explicit self-announcement of the topic shift"}

    if len({row["pair_id"] for row in rows}) != 15:
        raise SystemExit("final review set must contain 15 unique pair IDs")
    counts = {}
    for row in rows: counts[row["archetype"]] = counts.get(row["archetype"], 0) + 1
    expected = {"false_completion": 4, "hedging_or_excessive_caveating": 1, "topic_shift": 10}
    if counts != expected: raise SystemExit(f"unexpected final-gap balance: {counts}")
    if any(row.get("annotation_status") != "pending_review" for row in rows): raise SystemExit("assembly must not approve rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: row["pair_id"])
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in ordered) + "\n", encoding="utf-8")
    report = [{"pair_id": row["pair_id"], "archetype": row["archetype"], "assistant_review_decision": "pending_user_approval", "rationale": "Passed assembly provenance/content checks; human review remains required.", "annotation_status_at_review": row["annotation_status"]} for row in ordered]
    report_path = args.output.with_name("ifeval_final_gap_review_recommendations_final.jsonl")
    report_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in report) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} pending review rows to {args.output}")
    print(f"Wrote review report to {report_path}")
    print("archetypes:", counts)


if __name__ == "__main__": main()