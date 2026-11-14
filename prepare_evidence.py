import json
import re
from pathlib import Path

SOURCE_FILE = Path("data/RAGTruth/dataset/source_info.jsonl")
OUTPUT_FILE = Path("prepared/evidence.jsonl")
MAX_WORDS = 120
OVERLAP_WORDS = 25


def chunk_text(text, max_words=MAX_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    step = max_words - overlap

    if step <= 0:
        raise ValueError("Overlap must be smaller than chunk size")

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + max_words])
        if chunk:
            yield chunk
        if start + max_words >= len(words):
            break


def extract_documents(row):
    info = row["source_info"]
    task = row["task_type"]

    if task == "Summary":
        if not isinstance(info, str):
            raise ValueError("Expected Summary source_info to be text")
        return [info]

    if task == "QA":
        if not isinstance(info, dict):
            raise ValueError("Expected QA source_info to be a dictionary")

        passages = info.get("passages", "")
        if isinstance(passages, str):
            parts = re.split(
                r"(?=passage\s+\d+\s*:)",
                passages,
                flags=re.IGNORECASE
            )
            return [p.strip() for p in parts if p.strip()]

        if isinstance(passages, list):
            return [str(p) for p in passages if p]

        raise ValueError("Unsupported QA passage format")

    return []


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    sources = 0
    chunks = 0
    tasks = {}

    with SOURCE_FILE.open(encoding="utf-8") as inp, \
         OUTPUT_FILE.open("w", encoding="utf-8") as out:

        for line in inp:
            row = json.loads(line)
            task = row["task_type"]

            if task not in ("QA", "Summary"):
                continue

            sources += 1
            tasks[task] = tasks.get(task, 0) + 1

            for doc_index, document in enumerate(extract_documents(row)):
                for chunk_index, chunk in enumerate(chunk_text(document)):
                    record = {
                        "source_id": str(row["source_id"]),
                        "task_type": task,
                        "evidence_id": f"{row['source_id']}:{doc_index}:{chunk_index}",
                        "text": chunk
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    chunks += 1

    print(json.dumps({
        "sources": sources,
        "evidence_chunks": chunks,
        "tasks": tasks,
        "output": str(OUTPUT_FILE)
    }, indent=2))


if __name__ == "__main__":
    main()

