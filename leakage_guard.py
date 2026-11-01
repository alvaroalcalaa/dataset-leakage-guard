"""Offline overlap audit for two small AI text datasets."""
import argparse
from collections import defaultdict
import html
import json
import math
from pathlib import Path
import re
import sys
import unicodedata


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


def shingles(text):
    words = re.findall(r'\w+', normalize(text), flags=re.UNICODE)
    return {tuple(words[i:i+3]) for i in range(len(words)-2)}


def validate(data):
    if not isinstance(data, dict) or not isinstance(data.get('records'), list) or len(data['records']) > 300:
        raise ValueError('Se requiere records como lista de hasta 300 registros')
    result, seen = [], set()
    for record in data['records']:
        if not isinstance(record, dict):
            raise ValueError('Registro no válido')
        key, text = record.get('id'), record.get('text')
        if not isinstance(key, str) or not key.strip() or len(key) > 120 or key in seen:
            raise ValueError('ID vacío, demasiado largo o duplicado dentro del archivo')
        if not isinstance(text, str) or not text.strip() or len(text) > 20000:
            raise ValueError('text debe ser texto no vacío de hasta 20000 caracteres')
        group = record.get('group')
        if group is not None and (not isinstance(group, str) or not group.strip() or len(group) > 120):
            raise ValueError('group debe ser null o texto no vacío de hasta 120 caracteres')
        result.append({'id': key, 'text': text, 'group': group})
        seen.add(key)
    return sorted(result, key=lambda x: x['id'])


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Clave JSON repetida: ' + key)
        result[key] = value
    return result


def load_dataset(path):
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError('El archivo supera 2 MiB')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object)


def audit(train_data, eval_data, threshold=.8):
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not .1 <= threshold <= 1:
        raise ValueError('threshold debe estar entre 0.1 y 1')
    train, evaluation = validate(train_data), validate(eval_data)
    exact_index, shingle_index = defaultdict(list), defaultdict(set)
    train_sets = []
    train_groups, eval_groups = defaultdict(list), defaultdict(list)
    for i, record in enumerate(train):
        exact_index[normalize(record['text'])].append(i)
        parts = shingles(record['text'])
        train_sets.append(parts)
        for part in parts:
            shingle_index[part].add(i)
        if record['group'] is not None:
            train_groups[record['group']].append(record['id'])
    exact, near = [], []
    short_eval = 0
    for record in evaluation:
        text = normalize(record['text'])
        matches = set(exact_index.get(text, []))
        for i in sorted(matches):
            exact.append({'train_id': train[i]['id'], 'eval_id': record['id']})
        parts = shingles(text)
        short_eval += not bool(parts)
        candidates = set()
        for part in parts:
            candidates.update(shingle_index.get(part, ()))
        for i in sorted(candidates - matches):
            other = train_sets[i]
            similarity = len(parts & other) / len(parts | other)
            if similarity >= threshold:
                near.append({'train_id': train[i]['id'], 'eval_id': record['id'], 'jaccard': similarity})
        if record['group'] is not None:
            eval_groups[record['group']].append(record['id'])
    groups = [{'group': key, 'train_ids': train_groups[key], 'eval_ids': eval_groups[key]}
              for key in sorted(train_groups.keys() & eval_groups.keys())]
    shared_ids = sorted({r['id'] for r in train} & {r['id'] for r in evaluation})
    affected = {r['eval_id'] for r in exact + near} | set(shared_ids)
    for group in groups:
        affected.update(group['eval_ids'])
    return {'has_findings': bool(exact or near or groups or shared_ids),
            'summary': {'train_records': len(train), 'eval_records': len(evaluation),
                'exact_pairs': len(exact), 'near_pairs': len(near), 'shared_groups': len(groups),
                'shared_ids': len(shared_ids), 'flagged_eval_records': len(affected)},
            'method': {'normalization': 'NFC + casefold + whitespace collapse',
                'near_match': 'Jaccard of sets of three consecutive word tokens', 'threshold': threshold},
            'coverage': {'train_without_trigrams': sum(not s for s in train_sets),
                         'eval_without_trigrams': short_eval},
            'exact_pairs': exact, 'near_pairs': near, 'shared_groups': groups, 'shared_ids': shared_ids,
            'notice': 'Señales de solapamiento; no demuestra fuga causal ni certifica ausencia de contaminación.'}


def render_html(report):
    # Values included here are IDs, group labels and numeric metrics, never texts.
    body = html.escape(json.dumps(report, ensure_ascii=False, indent=2))
    return ('<!doctype html><html lang="es"><meta charset="utf-8"><title>Dataset Leakage Guard</title>'
            '<style>body{font:16px system-ui;background:#101827;color:#edf2ff;max-width:1000px;margin:40px auto;padding:20px}'
            'pre{white-space:pre-wrap;background:#1d2940;padding:20px;border-radius:12px}</style>'
            '<h1>Dataset Leakage Guard</h1><p>' + html.escape(report['notice']) + '</p><p>Registros de evaluación señalados: '
            + str(report['summary']['flagged_eval_records']) + '</p><pre>' + body + '</pre></html>')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train', type=Path)
    parser.add_argument('evaluation', type=Path)
    parser.add_argument('--threshold', type=float, default=.8)
    parser.add_argument('--json', dest='json_path', type=Path)
    parser.add_argument('--html', dest='html_path', type=Path)
    args = parser.parse_args(argv)
    try:
        inputs = {args.train.resolve(), args.evaluation.resolve()}
        outputs = [p.resolve() for p in (args.json_path, args.html_path) if p]
        if any(p in inputs for p in outputs) or len(set(outputs)) != len(outputs):
            raise ValueError('Las salidas deben ser distintas y no sobrescribir las entradas')
        report = audit(load_dataset(args.train), load_dataset(args.evaluation), args.threshold)
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if args.json_path:
            args.json_path.write_text(encoded, encoding='utf-8')
        if args.html_path:
            args.html_path.write_text(render_html(report), encoding='utf-8')
        print(encoded, end='')
        return 1 if report['has_findings'] else 0
    except (ValueError, OSError, RecursionError) as exc:
        print('Error: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
