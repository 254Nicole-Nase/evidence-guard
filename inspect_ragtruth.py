"""Pre-sprint data inspection for RAGTruth. No modelling here.

Usage: python inspect_ragtruth.py data/RAGTruth/dataset
"""
import json
import sys
from collections import Counter

d = sys.argv[1] if len(sys.argv) > 1 else "data/RAGTruth/dataset"
R = [json.loads(l) for l in open(f"{d}/response.jsonl", encoding="utf-8")]
S = {}
for l in open(f"{d}/source_info.jsonl", encoding="utf-8"):
    s = json.loads(l)
    S[s["source_id"]] = s

task = lambda r: S[r["source_id"]]["task_type"]
print(f"responses: {len(R)}  sources: {len(S)}")
print("response keys:", list(R[0].keys()))
print("source keys:  ", list(next(iter(S.values())).keys()))

ex = next(r for r in R if r["labels"])
print("\nexample label:", json.dumps(ex["labels"][0], indent=1))

keep = [r for r in R if task(r) in ("QA", "Summary")]   # drop Data2txt
print(f"\nQA + Summary responses: {len(keep)}")

for split in ("train", "test", None):
    rows = [r for r in keep if split is None or r["split"] == split]
    c = Counter(l["label_type"] for r in rows for l in r["labels"])
    name = split or "train+test"
    print(f"\n[{name}] responses={len(rows)} "
          f"with_any_label={sum(1 for r in rows if r['labels'])}")
    for k, v in sorted(c.items()):
        print(f"   {k:<22} {v}")

# What the source looks like per task (QA = dict, Summary = str)
qa = next(r for r in keep if task(r) == "QA")
sm = next(r for r in keep if task(r) == "Summary")
print("\nQA source_info keys:", list(S[qa["source_id"]]["source_info"].keys()))
print("Summary source_info type:", type(S[sm["source_id"]]["source_info"]).__name__)

# Check offsets line up with the response text
bad = sum(1 for r in keep for l in r["labels"]
          if r["response"][l["start"]:l["end"]] != l["text"])
total = sum(len(r["labels"]) for r in keep)
print(f"\nspan offsets that don't match text: {bad}/{total}")
