# EvidenceGuard

When does similarity-based evidence checking get fooled? A comparison of
cosine similarity, NLI and NLI + LLM for spotting unsupported claims in
RAG answers, using RAGTruth (QA and Summary tasks).

Apart Research AI x Epistemics sprint, 13–15 November 2026.

## Work done before the sprint (disclosed)

- Environment setup (`setup.sh`)
- RAGTruth download and schema inspection (`inspect_ragtruth.py`)
- Model load check on 5 example sentences (`check_models.py`)

Everything else (sentence-span aligner, the three methods, experiments,
analysis and report) was built during the sprint.

## Data

RAGTruth: https://github.com/ParticleMedia/RAGTruth
