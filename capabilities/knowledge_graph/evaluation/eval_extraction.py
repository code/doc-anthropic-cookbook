"""Precision/recall scoring for knowledge-graph extraction.

Re-runs the extraction prompt from the guide on the two hand-labeled articles
and reports entity and relation P/R/F1 against data/sample_triples.json.
"""

import json
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv

MODEL = "claude-haiku-4-5"
WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
# Wikimedia policy rejects requests without an identifying User-Agent
HEADERS = {"User-Agent": "claude-cookbooks/1.0 (https://github.com/anthropics/claude-cookbooks)"}

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

# Maps known surface-form variants to the canonical names used in
# sample_triples.json. Without this a predicted "National Aeronautics and
# Space Administration" never matches gold "NASA" and relation recall is
# artificially low. Extend this when the extractor starts emitting new
# variants the gold set doesn't list verbatim.
ALIAS_MAP = {
    "national aeronautics and space administration": "nasa",
    "the moon": "moon",
    "edwin aldrin": "buzz aldrin",
    "edwin 'buzz' aldrin": "buzz aldrin",
    "neil a. armstrong": "neil armstrong",
    "apollo lunar module": "lunar module eagle",
    "eagle": "lunar module eagle",
    "command module columbia": "columbia",
    "u.s. navy": "united states navy",
    "us navy": "united states navy",
}


def fetch_summary(title: str) -> str:
    r = requests.get(WIKI_API + title.replace(" ", "_"), headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()["extract"]


def extract(client: anthropic.Anthropic, text: str) -> dict:
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        tools=[EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "extract_graph"},
        messages=[{"role": "user", "content": PROMPT.format(text=text)}],
    )
    block = next((b for b in response.content if b.type == "tool_use"), None)
    if block is None:
        raise ValueError(f"No tool_use block in response (stop_reason={response.stop_reason})")
    return block.input


def prf(
    predicted: set[str] | set[tuple[str, str]], gold: set[str] | set[tuple[str, str]]
) -> tuple[float, float, float]:
    tp = len(predicted & gold)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def canon(name: str) -> str:
    lower = name.lower().strip()
    return ALIAS_MAP.get(lower, lower)


def main() -> None:
    load_dotenv()
    client = anthropic.Anthropic()

    gold_path = Path(__file__).parent.parent / "data" / "sample_triples.json"
    with open(gold_path, encoding="utf-8") as f:
        gold = json.load(f)

    ent_p_sum = ent_r_sum = ent_f_sum = 0.0
    rel_p_sum = rel_r_sum = rel_f_sum = 0.0
    scored = 0

    for title, labels in gold.items():
        try:
            text = fetch_summary(title)
        except requests.RequestException as e:
            print(f"Skipping {title}: {e}")
            continue

        result = extract(client, text)
        scored += 1

        pred_ents = {canon(e["name"]) for e in result["entities"]}
        gold_ents = {canon(e["name"]) for e in labels["entities"]}
        ep, er, ef = prf(pred_ents, gold_ents)
        ent_p_sum += ep
        ent_r_sum += er
        ent_f_sum += ef

        pred_rels = {(canon(r["source"]), canon(r["target"])) for r in result["relations"]}
        gold_rels = {(canon(r["source"]), canon(r["target"])) for r in labels["relations"]}
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

    if not scored:
        print("No documents scored.")
        return

    print(f"\n{'=' * 50}")
    print(f"Macro-averaged over {scored} documents:")
    print(
        f"  entities:  P={ent_p_sum / scored:.2f}  "
        f"R={ent_r_sum / scored:.2f}  F1={ent_f_sum / scored:.2f}"
    )
    print(
        f"  relations: P={rel_p_sum / scored:.2f}  "
        f"R={rel_r_sum / scored:.2f}  F1={rel_f_sum / scored:.2f}"
    )


if __name__ == "__main__":
    main()
