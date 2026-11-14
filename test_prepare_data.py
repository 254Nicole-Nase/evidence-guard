import json
import tempfile
import unittest
from pathlib import Path
from prepare_data import annotation_category, label_sentence, prepare, sentences_with_offsets

class PreparationTests(unittest.TestCase):
    def test_offsets(self):
        text = '  Paris is in France.\nThe moon is made of cheese!  '
        parts = list(sentences_with_offsets(text))
        self.assertEqual([x[2] for x in parts], ['Paris is in France.', 'The moon is made of cheese!'])
        self.assertTrue(all(text[a:b] == s for a, b, s in parts))

    def test_overlap_and_precedence(self):
        labels = [
            {'start': 0, 'end': 4, 'label_type': 'Baseless'},
            {'start': 2, 'end': 5, 'label_type': 'Subtle Conflict'},
        ]
        self.assertEqual(label_sentence(0, 10, labels)[0], 'contradicted')
        self.assertEqual(label_sentence(10, 20, labels)[0], 'supported')

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            dataset = root / 'dataset'
            dataset.mkdir()
            sources = [
                {'source_id': 's1', 'task_type': 'QA', 'source_info': {'question': 'Where?'}},
                {'source_id': 's2', 'task_type': 'Summary', 'source_info': 'A source'},
                {'source_id': 's3', 'task_type': 'Data2txt', 'source_info': {}},
            ]
            responses = [
                {'source_id': 's1', 'response': 'Paris is in France. The moon is cheese.', 'labels': [
                    {'start': 20, 'end': 38, 'text': 'The moon is cheese', 'label_type': 'Baseless'}], 'split': 'train'},
                {'source_id': 's2', 'response': 'A factual summary.', 'labels': [], 'split': 'test'},
                {'source_id': 's3', 'response': 'Ignored.', 'labels': [], 'split': 'train'},
            ]
            for name, data in [('source_info.jsonl', sources), ('response.jsonl', responses)]:
                (dataset / name).write_text(''.join(json.dumps(row) + '\n' for row in data))
            summary = prepare(dataset, root / 'output')
            self.assertEqual(summary['response_count'], 2)
            self.assertEqual(summary['sentence_count'], 3)
            rows = [json.loads(line) for line in (root / 'output' / 'sentences.jsonl').read_text().splitlines()]
            self.assertEqual([row['label'] for row in rows], ['supported', 'unsupported', 'supported'])
            self.assertNotEqual(rows[0]['split'], rows[2]['split'])

if __name__ == '__main__':
    unittest.main()
