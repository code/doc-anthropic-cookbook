# Knowledge Graph Extraction Evaluation

Scores entity and relation extraction against the hand-labeled gold set in `../data/sample_triples.json`.

## Running

```bash
export ANTHROPIC_API_KEY=sk-...
python eval_extraction.py
```

## Metrics

**Entity P/R/F1** — an extracted entity counts as a true positive if its lowercased name matches a gold entity in the same document. Type agreement is reported separately.

**Relation P/R/F1** — a relation counts as a true positive if both endpoints match gold endpoints (after canonicalization via the gold alias map). Predicate wording is ignored: "commanded" and "was commander of" both count.

## Expected baseline

With `claude-haiku-4-5` and the extraction prompt from the guide, expect roughly:

| Metric | P | R | F1 |
|---|---|---|---|
| Entities | 0.80–0.90 | 0.70–0.85 | 0.75–0.85 |
| Relations | 0.70–0.85 | 0.55–0.70 | 0.60–0.75 |

Recall on relations is the hard number — the extractor tends to be conservative, preferring fewer high-confidence edges over exhaustive coverage. Tuning the extraction prompt for higher recall (e.g. "extract every stated relationship, even minor ones") trades precision for recall.
