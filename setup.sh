#!/usr/bin/env bash
# EvidenceGuard: one-time setup
#   git clone https://github.com/254Nicole-Nase/evidence-guard && cd evidence-guard && bash setup.sh
set -e

# 1. Python environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install sentence-transformers transformers torch datasets scipy \
            scikit-learn pandas matplotlib jupyter nltk

# 2. RAGTruth data (official repo)
mkdir -p data
git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git data/RAGTruth

# 3. Sentence splitter data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# 4. Sanity checks
python inspect_ragtruth.py data/RAGTruth/dataset
python check_models.py
