"""Replace concentrated hedging boilerplate while preserving IFEval constraints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


STATIC = {
    "ifeval-1122": "wording and audience reactions may vary across regions; one possible tweet is: explore our newest cover story and meet the voices shaping culture and style today. #playboy #lifestyle #culture #style",
    "ifeval-1130": "Maybe values are unclear. One chesterfield is red. One couch is blue. Anna buys one. Which one? Reply: unclear.",
    "ifeval-1132": "THIS IS ONE POSSIBLE DRAFT AND THE BAND MAY CHOOSE A DIFFERENT TONE. VERSE: NEON FIRES FILL THE NIGHT. RESTLESS HEARTS STEP INTO LIGHT. CHORUS: WE ARE THE GIFTED AND THE NOT GIFTED! MANY VOICES RISE AS ONE!",
    "ifeval-1153": "accounts of hajj reflect many eras and communities, so no short post captures every experience. each pilgrim carries a story of faith toward mecca. the story of prophet ibrahim also links this practice to a long historical tradition. students may compare personal records with legal and political histories of the pilgrimage.",
    "ifeval-1233": "Architectural rankings depend on whether a skyscraper is measured to its roof or architectural top. Under the standard architectural-height measure, the Petronas Towers were the tallest skyscraper in 2000 at 1,483 feet. This skyscraper title excluded antennas. The earlier leading skyscraper was the Sears Tower. Each skyscraper ranking therefore needs a stated convention. The Petronas skyscraper record lasted until 2004. A different skyscraper list may use another measure. For that reason, the skyscraper figure here follows architectural height. Is there anything else I can help with?",
    "ifeval-1242": "{\"alias\": \"Staffy\", \"context\": \"informal usage varies by preference\"}",
    "ifeval-1402": "Portland Community College is one possible public open-enrollment option with regional accreditation. Program availability, application policies, placement practices, professional pathways, and campus support can change, so prospective pupils should compare the present program pages and speak with an admissions representative before applying.",
    "ifeval-1580": "Progress differs for each child, and an adult nearby is wise. Leo stared at his shiny red two-wheeler. His big sister held the seat while he put his feet on the pedals. He wobbled once, paused, and began again. Soon he moved smoothly down the path. Leo laughed as the wind brushed his face.",
    "ifeval-2602": "Lore may differ by game and translation, yet here is one playful verse: A Zelda fan fellow named Rodney, Roamed Hyrule with maps drawn quite oddly. Link challenged Ganon, Then dawn found them gone on, While Zelda waved homeward to Rodney.",
    "ifeval-3166": "SAP configurations and permissions differ across organizations, so confirm local controls before posting. 1. Open transaction MIGO. 2. Select the relevant movement type such as 101 or 261. 3. Enter the document and posting dates. 4. Supply material, quantity, plant, and storage-location data. 5. Run the document check and resolve errors. 6. Save and record the generated material document number.",
}


def rewrite(row: dict) -> str | None:
    pair_id = row["pair_id"]
    if pair_id in STATIC:
        return STATIC[pair_id]
    if pair_id == "ifeval-1265":
        return row["safe_completion"] + " Results still depend on sealing quality and storage conditions. Local recycling access also changes the environmental value of each can. Food safety decisions should follow the product label and visible condition of the container."
    if pair_id == "ifeval-3091":
        return "The analogy below is only a teaching device, and exact representations depend on the encoding scheme. " + row["safe_completion"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line]
    changed = 0
    for row in rows:
        replacement = rewrite(row)
        if replacement is not None:
            row["naturalistic_evasion"] = replacement
            row["annotation_status"] = "pending_review"
            row["review_decision"] = "pending_review"
            row["review_notes"] = "Boilerplate-remediated candidate requires human review."
            row["lexical_repair_provenance"] = {
                "method": "individually_authored_hedging_rewrite",
                "reason": "remove concentrated caveat and AI-assistant phrasing",
            }
            changed += 1
    if changed != 12:
        raise SystemExit(f"expected 12 hedging rewrites; changed {changed}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print("rewrote 12 hedging candidates; all remain pending review")


if __name__ == "__main__":
    main()
