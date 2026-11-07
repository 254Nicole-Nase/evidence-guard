#!/usr/bin/env bash
# EvidenceGuard: one-time setup (run from inside your evidence-guard folder)
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

# 5. Git
git init
git add README.md setup.sh inspect_ragtruth.py check_models.py .gitignore
git commit -m "Pre-sprint setup: environment, data inspection, model load check"
