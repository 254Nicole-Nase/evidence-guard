"""Sprint work (14 Nov). Score every response sentence with cosine and NLI.

Every method sees the same sentence and the same top-k source chunks.

Input:  prepared/sentences.jsonl (from prepare_data.py) + RAGTruth source_info.jsonl
Output: results/scores.jsonl, one row per sentence

Run on a GPU (Google Colab):
    python score_methods.py --dataset data/RAGTruth/dataset --prepared prepared --out results
Quick test without models:
    python score_methods.py ... --mock --limit 2000
"""
import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

EMB_MODEL = "BAAI/bge-base-en-v1.5"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
STOP = set("the a an of to in and or is are was were be been for on at by with as that this it its from".split())


def source_text(info):
    """QA sources are {question, passages}; Summary sources are plain text."""
    if isinstance(info, dict):
        return info.get("passages", "")
    return info


def chunk_source(text, max_words=180):
    """Split the source into chunks of whole sentences, about max_words each,
    so a chunk plus a response sentence fits the NLI model's 512 tokens."""
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(\[]|passage \d)", text.strip())
    chunks, cur, n = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and n + w > max_words:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [text[:2000]]


def content_words(text):
    return {w for w in WORD.findall(text.lower()) if w not in STOP}


def overlap(sentence, chunk):
    """Share of the sentence's content words that also appear in the chunk."""
    s = content_words(sentence)
    return len(s & content_words(chunk)) / len(s) if s else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--prepared", default="prepared")
    ap.add_argument("--out", default="results")
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="only first N sentences (testing)")
    ap.add_argument("--mock", action="store_true", help="TF-IDF + random NLI, no downloads")
    ap.add_argument("--resume", action="store_true", help="skip sources already in scores.jsonl")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(Path(a.prepared) / "sentences.jsonl", encoding="utf-8")]
    if a.limit:
        rows = rows[: a.limit]
    infos = {}
    for l in open(Path(a.dataset) / "source_info.jsonl", encoding="utf-8"):
        s = json.loads(l)
        infos[str(s["source_id"])] = s["source_info"]

    by_source = defaultdict(list)
    for i, r in enumerate(rows):
        by_source[str(r["source_id"])].append(i)
    ev_path = Path(a.prepared) / "evidence.jsonl"
    if ev_path.exists():   # use the chunks written by prepare_data.py
        chunks = defaultdict(list)
        for l in open(ev_path, encoding="utf-8"):
            e = json.loads(l)
            chunks[str(e["source_id"])].append(e["text"])
        print("chunks from", ev_path)
    else:
        chunks = {}
    for sid in by_source:
        if not chunks.get(sid):
            chunks[sid] = chunk_source(source_text(infos[sid]))
    print(f"{len(rows)} sentences, {len(by_source)} sources, "
          f"{sum(len(c) for c in chunks.values())} chunks")

    # ---- models ----
    if a.mock:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer().fit([c for cs in chunks.values() for c in cs] + [r["sentence"] for r in rows])

        def embed(texts):
            m = np.asarray(vec.transform(texts).todense())
            return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)

        rng = np.random.default_rng(0)

        def nli(pairs):
            p = rng.random((len(pairs), 3))
            return p / p.sum(1, keepdims=True)
        labels = ["contradiction", "entailment", "neutral"]
    else:
        import torch
        from sentence_transformers import CrossEncoder, SentenceTransformer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print("device:", dev)
        emb_model = SentenceTransformer(EMB_MODEL, device=dev)
        nli_model = CrossEncoder(NLI_MODEL, device=dev, max_length=512)
        id2label = nli_model.model.config.id2label
        labels = [id2label[i].lower() for i in range(len(id2label))]

        def embed(texts):
            return emb_model.encode(texts, batch_size=a.batch, normalize_embeddings=True,
                                    show_progress_bar=False)

        def nli(pairs):
            return nli_model.predict(pairs, batch_size=a.batch, apply_softmax=True,
                                     show_progress_bar=False)
    print("NLI label order:", labels)
    E, C, N = labels.index("entailment"), labels.index("contradiction"), labels.index("neutral")

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_cos = t_nli = 0.0
    scores_path = out_dir / "scores.jsonl"
    done = set()
    if a.resume and scores_path.exists():
        lines = scores_path.read_text(encoding="utf-8").splitlines()
        good = []
        for l in lines:            # drop a half-written last line, if any
            try:
                good.append(json.loads(l))
            except json.JSONDecodeError:
                break
        # a source is done only if all its sentences are there
        have = defaultdict(int)
        for g in good:
            have[g["source_id"]] += 1
        done = {sid for sid, idx in by_source.items() if have.get(sid, 0) == len(idx)}
        keep = [json.dumps(g) for g in good if g["source_id"] in done]
        scores_path.write_text("".join(k + "\n" for k in keep), encoding="utf-8")
        print(f"resuming: {len(done)} sources already done")
    with open(scores_path, "a" if a.resume else "w", encoding="utf-8") as f:
        for n, (sid, idx) in enumerate(by_source.items(), 1):
            if sid in done:
                continue
            cs = chunks[sid]
            sents = [rows[i]["sentence"] for i in idx]

            # Method 1: cosine. Also picks the top-k chunks used by every method.
            t0 = time.perf_counter()
            sim = embed(sents) @ embed(cs).T            # (sentences, chunks)
            t_cos += time.perf_counter() - t0
            k = min(a.top_k, len(cs))
            top = np.argsort(-sim, axis=1)[:, :k]

            # Method 2: NLI over the same top-k chunks (premise = chunk).
            pairs = [(cs[j], s) for s, js in zip(sents, top) for j in js]
            t0 = time.perf_counter()
            probs = np.asarray(nli(pairs)).reshape(len(sents), k, 3)
            t_nli += time.perf_counter() - t0

            for r_i, i in enumerate(idx):
                r = rows[i]
                best = top[r_i][int(np.argmax(probs[r_i, :, E]))]
                f.write(json.dumps({
                    "example_id": r["example_id"], "source_id": sid,
                    "task_type": r["task_type"], "split": r["split"],
                    "label": r["label"], "y": r["unsupported_binary"],
                    "label_types": sorted({x["label_type"] for x in r["overlapping_annotations"]}),
                    "cos_max": float(sim[r_i].max()),
                    "nli_entail": float(probs[r_i, :, E].max()),
                    "nli_contra": float(probs[r_i, :, C].max()),
                    "nli_neutral": float(probs[r_i, :, N].min()),
                    "top_chunks": [int(j) for j in top[r_i]],
                    "overlap_top1": overlap(r["sentence"], cs[top[r_i][0]]),
                    "overlap_best": max(overlap(r["sentence"], cs[j]) for j in top[r_i]),
                    "nli_best_chunk": int(best),
                }) + "\n")
            f.flush()
            if n % 200 == 0:
                print(f"  {n}/{len(by_source)} sources")

    n_scored = sum(len(idx) for sid, idx in by_source.items() if sid not in done)
    timing = {"sentences": n_scored, "cosine_sec": t_cos, "nli_sec": t_nli,
              "cosine_ms_per_sentence": 1000 * t_cos / max(n_scored, 1),
              "nli_ms_per_sentence": 1000 * t_nli / max(n_scored, 1),
              "top_k": a.top_k, "mock": a.mock,
              "resumed_run": bool(done), "note": "timings cover this run only"}
    json.dump(timing, open(out_dir / "timing.json", "w"), indent=2)
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
