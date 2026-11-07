# Methods (plan, written before the sprint)

## Research question

When does similarity-based evidence checking get fooled? Specifically: how
often do cosine similarity, an NLI cross-encoder, and NLI + LLM catch
hallucinated sentences that closely copy the source but change one fact
(a number, date, name or negation)?

Cosine similarity measures how close two texts are in meaning, not whether
one supports the other. That gap is what this project measures.

## Data

- RAGTruth (https://github.com/ParticleMedia/RAGTruth), `response.jsonl` + `source_info.jsonl`
- Tasks: QA and Summary only. Data2txt is dropped because its sources are
  structured records, not prose.
- Train and test splits are pooled, because no method is trained. Only 11
  Subtle Conflict spans exist in the QA + Summary test split; pooled there are 138.
- A small dev subset (20%, fixed seed, split by source so the same source
  never appears in both) is used only to pick thresholds. All results are
  reported on the remaining 80%.

## Unit of evaluation

- Each response is split into sentences, keeping character start/end.
- A sentence is **unsupported** if it overlaps any annotated span.
- Sentence label: Conflict span → *contradicted*; Baseless span → *unsupported*;
  no span → *supported*. If a sentence overlaps both, contradicted wins.
- Main metric is binary (supported vs not). The three-way split is secondary.

## Evidence given to each method (same for all three)

- The source is split into chunks that fit the NLI model's 512-token limit.
- For each response sentence, the top-k source chunks are chosen once by
  embedding similarity, and every method sees exactly those chunks.
- All methods score the same sentences against the same chunks.

## Methods

1. **Cosine**: BAAI/bge-base-en-v1.5; score = max cosine similarity over the
   chunks; flag if below a threshold tuned on dev.
2. **NLI**: cross-encoder/nli-deberta-v3-base; keep entailment / neutral /
   contradiction probabilities; flag if max entailment is below a threshold
   tuned on dev. Label order is read from the model config.
3. **NLI + LLM**: run the LLM only on sentences where NLI is uncertain
   (entailment in a middle band set on dev); the LLM gives supported /
   contradicted / unsupported.

## Metrics

- Precision, recall and F1 for "unsupported"; false-positive rate
- Macro F1 over the three classes (secondary)
- Seconds per response for each method
- **Main analysis:** recall by word overlap between the hallucinated sentence
  and its closest source chunk (binned), and by RAGTruth label type
  (Evident/Subtle × Conflict/Baseless)

## Known limits (go in the report)

- Few Subtle Conflict examples, even pooled
- Sentence-level labels are coarser than RAGTruth's spans
- An NLI "entailment" is agreement with the given chunks, not proof of truth
- Chunking can split the evidence a sentence needs across two chunks
