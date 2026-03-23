"""Precision/recall scoring for knowledge-graph extraction.

Re-runs the extraction prompt from the guide on the two hand-labeled articles
and reports entity and relation P/R/F1 against data/sample_triples.json.
"""

import json
from pathlib import Path

import anthropic
import wikipedia

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5"

ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "ARTIFACT"]

EXTRACT_TOOL = {
    "name": "extract_graph",
    "description": "Record the entities and relations found in a document.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string", "enum": ENTITY_TYPES},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "type", "description"],
                },
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "predicate": {"type": "string"},
                        "target": {"type": "string"},
                    },
                    "required": ["source", "predicate", "target"],
                },
            },
        },
        "required": ["entities", "relations"],
    },
}

PROMPT = """Extract a knowledge graph from the document below.

<document>
{text}
</document>

Extract only entities that are central to what this document is about — skip \
incidental mentions. Every relation must connect two entities you extracted."""


def extract(text: str) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        tools=[EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "extract_graph"},
        messages=[{"role": "user", "content": PROMPT.format(text=text)}],
    )
    return next(b.input for b in response.content if b.type == "tool_use")


def prf(predicted: set, gold: set) -> tuple[float, float, float]:
    tp = len(predicted & gold)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def canonicalize(name: str, alias_map: dict[str, str]) -> str:
    lower = name.lower().strip()
    return alias_map.get(lower, lower)


def main() -> None:
    gold_path = Path(__file__).parent.parent / "data" / "sample_triples.json"
    with open(gold_path) as f:
        gold = json.load(f)

    # Gold names serve double duty as the canonical-form target for relation
    # endpoint matching — predicted "National Aeronautics and Space
    # Administration" should match gold "NASA" if the gold set lists it.
    alias_map: dict[str, str] = {}
    for labels in gold.values():
        for e in labels["entities"]:
            alias_map[e["name"].lower()] = e["name"].lower()

    ent_p_sum = ent_r_sum = ent_f_sum = 0.0
    rel_p_sum = rel_r_sum = rel_f_sum = 0.0

    for title, labels in gold.items():
        text = wikipedia.summary(title, sentences=8, auto_suggest=False)
        result = extract(text)

        pred_ents = {e["name"].lower() for e in result["entities"]}
        gold_ents = {e["name"].lower() for e in labels["entities"]}
        ep, er, ef = prf(pred_ents, gold_ents)
        ent_p_sum += ep
        ent_r_sum += er
        ent_f_sum += ef

        pred_rels = {
            (canonicalize(r["source"], alias_map), canonicalize(r["target"], alias_map))
            for r in result["relations"]
        }
        gold_rels = {(r["source"].lower(), r["target"].lower()) for r in labels["relations"]}
        rp, rr, rf = prf(pred_rels, gold_rels)
        rel_p_sum += rp
        rel_r_sum += rr
        rel_f_sum += rf

        print(f"\n{title}")
        print(f"  entities:  P={ep:.2f}  R={er:.2f}  F1={ef:.2f}")
        print(f"  relations: P={rp:.2f}  R={rr:.2f}  F1={rf:.2f}")
        missed_ents = gold_ents - pred_ents
        if missed_ents:
            print(f"  missed entities: {', '.join(sorted(missed_ents))}")
        missed_rels = gold_rels - pred_rels
        if missed_rels:
            print(f"  missed relations: {len(missed_rels)}")

    n = len(gold)
    print(f"\n{'=' * 50}")
    print(f"Macro-averaged over {n} documents:")
    print(f"  entities:  P={ent_p_sum / n:.2f}  R={ent_r_sum / n:.2f}  F1={ent_f_sum / n:.2f}")
    print(f"  relations: P={rel_p_sum / n:.2f}  R={rel_r_sum / n:.2f}  F1={rel_f_sum / n:.2f}")


if __name__ == "__main__":
    main()
