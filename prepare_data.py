"""Prepare sentence-level RAGTruth examples for EvidenceGuard.

Usage:
    python prepare_data.py --dataset data/RAGTruth/dataset --output prepared

Requires Python 3.9+; uses only the standard library.
"""
import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

SENTENCE_RE = re.compile(r'[^\n]+?(?:[.!?]+(?=\s+[^a-z\s]|\s*$)|(?=\n|$))', re.MULTILINE)


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{line_no}: {exc}') from exc


def sentences_with_offsets(text):
    """Conservative rule-based sentence splitter; review abbreviations manually."""
    for match in SENTENCE_RE.finditer(text):
        raw = match.group(0)
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        start, end = match.start() + left, match.start() + right
        if start < end:
            yield start, end, text[start:end]


def annotation_category(label):
    kind = str(label.get('label_type', '')).lower()
    if 'conflict' in kind:
        return 'contradicted'
    if 'baseless' in kind:
        return 'unsupported'
    raise ValueError(f'Unknown RAGTruth label_type: {label.get("label_type")!r}')


def label_sentence(start, end, labels):
    overlapping = [lab for lab in labels if int(lab['start']) < end and int(lab['end']) > start]
    categories = {annotation_category(lab) for lab in overlapping}
    category = ('contradicted' if 'contradicted' in categories else
                'unsupported' if 'unsupported' in categories else 'supported')
    return category, overlapping


def split_sources(source_ids, dev_fraction=0.2, seed=42):
    """Split source groups, not individual responses, to prevent source leakage."""
    ids = sorted(set(source_ids), key=str)
    if len(ids) < 2:
        raise ValueError('At least two distinct sources are needed for dev/eval splitting')
    rng = random.Random(seed)
    rng.shuffle(ids)
    dev_count = max(1, min(len(ids) - 1, round(len(ids) * dev_fraction)))
    dev = set(ids[:dev_count])
    return {source_id: ('dev' if source_id in dev else 'eval') for source_id in ids}


def prepare(dataset, output, dev_fraction=0.2, seed=42):
    dataset, output = Path(dataset), Path(output)
    sources = {row['source_id']: row for row in read_jsonl(dataset / 'source_info.jsonl')}
    responses = []
    for row in read_jsonl(dataset / 'response.jsonl'):
        source_id = row['source_id']
        if source_id not in sources:
            raise ValueError(f'Missing source metadata for {source_id!r}')
        task = sources[source_id]['task_type']
        if task in ('QA', 'Summary'):
            responses.append((row, task))
    if not responses:
        raise ValueError('No QA or Summary responses found')
    splits = split_sources((r['source_id'] for r, _ in responses), dev_fraction, seed)
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    with (output / 'sentences.jsonl').open('w', encoding='utf-8') as handle:
        for response_index, (row, task) in enumerate(responses):
            text = row['response']
            labels = row.get('labels') or []
            for lab in labels:
                a, b = int(lab['start']), int(lab['end'])
                if not (0 <= a < b <= len(text)):
                    raise ValueError(f'Invalid annotation offsets {a}:{b} in response {response_index}')
                if 'text' in lab and text[a:b] != lab['text']:
                    raise ValueError(f'Annotation text mismatch in response {response_index}, {a}:{b}')
            for sentence_index, (start, end, sentence) in enumerate(sentences_with_offsets(text)):
                category, overlaps = label_sentence(start, end, labels)
                example = {
                    'example_id': f'{response_index}:{sentence_index}',
                    'source_id': row['source_id'],
                    'task_type': task,
                    'original_split': row.get('split'),
                    'split': splits[row['source_id']],
                    'sentence': sentence,
                    'start': start,
                    'end': end,
                    'label': category,
                    'unsupported_binary': int(category != 'supported'),
                    'overlapping_annotations': overlaps,
                }
                handle.write(json.dumps(example, ensure_ascii=False) + '\n')
                counts[(example['split'], category)] += 1
    with (output / 'source_splits.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['source_id', 'split'])
        writer.writerows((sid, split) for sid, split in sorted(splits.items(), key=lambda x: str(x[0])))
    summary = {
        'seed': seed, 'dev_fraction_by_source': dev_fraction,
        'response_count': len(responses), 'source_count': len(splits),
        'sentence_count': sum(counts.values()),
        'label_counts': {f'{split}/{label}': n for (split, label), n in sorted(counts.items())},
        'note': 'Sentence-level proxy labels; unannotated does not guarantee factual support.'
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='data/RAGTruth/dataset')
    parser.add_argument('--output', default='prepared')
    parser.add_argument('--dev-fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.dev_fraction < 1:
        parser.error('--dev-fraction must be between 0 and 1')
    prepare(args.dataset, args.output, args.dev_fraction, args.seed)


if __name__ == '__main__':
    main()
