import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from leakage_guard import audit, load_dataset, normalize, render_html, shingles, validate


def data(*records):
    return {'records': list(records)}


def record(key, text, **kwargs):
    return {'id': key, 'text': text, **kwargs}


class LeakageTests(unittest.TestCase):
    def test_normalization_unicode_case_whitespace(self):
        self.assertEqual(normalize('  CAFE\u0301\nDEL\tBOSQUE  '), 'café del bosque')
        r = audit(data(record('t','Café del bosque')), data(record('e',' CAFE\u0301 DEL BOSQUE ')))
        self.assertEqual(r['summary']['exact_pairs'], 1)

    def test_empty_datasets(self):
        self.assertFalse(audit(data(), data())['has_findings'])

    def test_disjoint(self):
        self.assertFalse(audit(data(record('t','llave azul del bosque')), data(record('e','mercado rojo junto al mar')))['has_findings'])

    def test_exact_not_also_near(self):
        r = audit(data(record('t','uno dos tres cuatro')), data(record('e','uno dos tres cuatro')))
        self.assertEqual((r['summary']['exact_pairs'], r['summary']['near_pairs']), (1,0))

    def test_known_jaccard_boundary(self):
        a,b = data(record('t','uno dos tres cuatro cinco seis')), data(record('e','uno dos tres cuatro cinco seis siete'))
        r = audit(a,b,.8)
        self.assertEqual(r['near_pairs'][0]['jaccard'], .8)
        self.assertEqual(audit(a,b,.81)['near_pairs'], [])

    def test_trigrams_are_sets(self):
        self.assertEqual(shingles('a a a a a'), {('a','a','a')})
        self.assertEqual(shingles('dos palabras'), set())

    def test_short_text_coverage_and_exact(self):
        r = audit(data(record('t','hola')), data(record('e','HOLA')))
        self.assertEqual(r['summary']['exact_pairs'], 1)
        self.assertEqual(r['coverage'], {'train_without_trigrams': 1, 'eval_without_trigrams': 1})

    def test_shared_group_without_text_match(self):
        r = audit(data(record('t','rojo', group='session-1')), data(record('e','azul', group='session-1')))
        self.assertEqual(r['summary']['shared_groups'], 1)
        self.assertEqual(r['shared_groups'][0]['eval_ids'], ['e'])

    def test_shared_id_and_flagged_records_deduplicated(self):
        r = audit(data(record('same','one two three',group='g')), data(record('same','one two three',group='g')))
        self.assertEqual(r['summary']['flagged_eval_records'], 1)
        self.assertEqual(r['shared_ids'], ['same'])

    def test_multiple_exact_pairs_not_lost(self):
        r = audit(data(record('a','texto'),record('b','texto')), data(record('x','texto'),record('y','texto')))
        self.assertEqual(r['summary']['exact_pairs'], 4)
        self.assertEqual(r['summary']['flagged_eval_records'], 2)

    def test_reordering_stable(self):
        a = [record('b','uno dos tres cuatro'), record('a','otro texto largo nuevo')]
        b = [record('y','uno dos tres cuatro'),record('x','otro texto largo nuevo')]
        self.assertEqual(audit(data(*a),data(*b)), audit(data(*reversed(a)),data(*reversed(b))))

    def test_texts_not_exported_and_html_escaped(self):
        text = 'contenido reservado ejemplo centinela'
        r = audit(data(record('<script>',text)),data(record('e',text)))
        self.assertNotIn(text, json.dumps(r))
        h = render_html(r)
        self.assertNotIn('<script>',h)
        self.assertIn('&lt;script&gt;',h)

    def test_invalid_inputs(self):
        for d in (data(record('x',' ')), data(record('x','a'),record('x','b')), data(record('x','a',group=5)), {'records':'oops'}):
            with self.assertRaises(ValueError):
                validate(d)
        for threshold in (True,0,float('nan'),1.1):
            with self.assertRaises(ValueError):
                audit(data(),data(),threshold)

    def test_duplicate_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.json'
            p.write_text('{"records":[],"records":[]}')
            with self.assertRaises(ValueError):
                load_dataset(p)

    def test_cli_status_export_and_input_protection(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b,j,h=[Path(tmp)/name for name in ('a.json','b.json','report.json','report.html')]
            original=json.dumps(data(record('t','uno dos tres cuatro')))
            a.write_text(original)
            b.write_text(json.dumps(data(record('e','uno dos tres cuatro'))))
            cli=[sys.executable,str(Path(__file__).with_name('leakage_guard.py')),str(a),str(b)]
            r=subprocess.run(cli+['--json',str(j),'--html',str(h)],capture_output=True)
            self.assertEqual(r.returncode,1,r.stderr)
            self.assertTrue(json.loads(j.read_text())['has_findings'])
            self.assertTrue(h.exists())
            self.assertEqual(subprocess.run(cli+['--json',str(a)],capture_output=True).returncode,2)
            self.assertEqual(a.read_text(),original)
            b.write_text('{"records":[]}')
            self.assertEqual(subprocess.run(cli,capture_output=True).returncode,0)


if __name__=='__main__':
    unittest.main()
