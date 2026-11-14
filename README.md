# EvidenceGuard

**Similar but Wrong: Embedding Similarity Rewards Hallucinations That Copy Their Source**

When does similarity-based evidence checking get fooled? This project compares
cosine similarity (BAAI/bge-base-en-v1.5) with an NLI cross-encoder
(cross-encoder/nli-deberta-v3-base) as sentence-level support checks on
RAGTruth QA and summarization responses, with both methods given the same
evidence.

Apart Research AI x Epistemics Research Sprint, 13–15 November 2026
(Track 2: Trust Infrastructure).

## Main results (69,217 held-out sentences)

| Method | AUROC (95% CI) | Recall at 10% FPR |
| --- | --- | --- |
| Cosine | 0.569 (0.552–0.586) | 9.1% |
| NLI | 0.716 (0.703–0.730) | 26.8% |

- Cosine ranks Evident Conflict sentences below supported ones (AUROC 0.469, CI 0.451–0.487).
- NLI's lead grows with word overlap (+0.128 to +0.201 AUROC) but is not
  significant for near-copies with 85%+ overlap (+0.040, CI −0.013 to 0.095).

See the report for details and limitations.

## What was built when

**Before the sprint (7–12 Nov 2026), pre-existing:**

- `setup.sh`: environment setup
- `inspect_ragtruth.py`: RAGTruth schema inspection
- `check_models.py`: model loading check on 5 example sentences
- `METHODS.md`: written plan (the final method differs in places; the report is authoritative)

No experiments were run before the sprint.

**During the sprint (13–15 Nov 2026):**

- `prepare_data.py`, `test_prepare_data.py`: sentence splitting with offsets, labels, source-level dev/eval split
- `prepare_evidence.py`: 120-word source chunks with 25-word overlap
- `score_methods.py`: cosine and NLI scores for every sentence against the same top-3 chunks
- `evaluate.py`: AUROC, average precision, recall at a dev-set 10% FPR, breakdowns by task, length, overlap and label type
- `bootstrap_ci.py`: 95% confidence intervals from 500 source-level bootstrap resamples
- all experiments and the report

## How to reproduce

Data preparation runs on CPU; scoring needs a GPU (we used a free Colab T4, about 1 hour).

```bash
pip install sentence-transformers transformers torch scikit-learn pandas matplotlib
git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git data/RAGTruth

python -m unittest test_prepare_data          # splitter and labelling tests
python prepare_data.py --dataset data/RAGTruth/dataset --output prepared
python prepare_evidence.py                    # writes prepared/evidence.jsonl

python score_methods.py --dataset data/RAGTruth/dataset --prepared prepared --out results --resume
python evaluate.py --scores results/scores.jsonl --sentences prepared/sentences.jsonl --out results
python bootstrap_ci.py --scores results/scores.jsonl --sentences prepared/sentences.jsonl --out results
```

Expected `prepared/summary.json`: 86,186 sentences (dev 16,969, eval 69,217).
`--resume` lets the scoring continue after a disconnect.
Outputs: `results/report.json`, `results/bootstrap_ci.json`,
`results/auroc_by_overlap.png`, `results/timing.json`.

## Data and models

- RAGTruth: https://github.com/ParticleMedia/RAGTruth
- https://huggingface.co/BAAI/bge-base-en-v1.5
- https://huggingface.co/cross-encoder/nli-deberta-v3-base
