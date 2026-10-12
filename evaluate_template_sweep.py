"""Training-development threshold sweep; never an inference entry point."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from evaluate import aggregate, load_manifest, metrics_from_counts, reviewed_domains
from hatch_texture import normalized_template_response
from inference import read_drawing


def score_response(response: np.ndarray, row: dict[str, Any], root: Path,
                   thresholds: list[float]) -> dict[str, dict[str, Any]]:
    positive, known, blank, metadata = reviewed_domains(row, root)
    positive_count = int(positive.sum())
    negative_count = int(known.sum())-positive_count
    blank_count = int(blank.sum())
    scores = {}
    for threshold in thresholds:
        selected = response >= threshold
        tp = int(np.count_nonzero(selected & positive))
        fp = int(np.count_nonzero(selected & known & ~positive))
        blank_fp = int(np.count_nonzero(selected & blank))
        metrics = metrics_from_counts(tp, fp, positive_count-tp, negative_count-fp,
                                      blank_fp, blank_count)
        scores[str(threshold)] = {'id':row['id'], 'document_id':row['document_id'],
                                 'kind':row['kind'], **metadata, 'metrics':metrics}
    return scores


def run_sweep(rows: list[dict[str, Any]], root: Path, output: Path,
              thresholds: list[float]) -> list[dict[str, Any]]:
    records = []
    cached_image = ''
    gray = np.zeros((1, 1), dtype=np.uint8)
    with (output/'query-records.jsonl').open('w') as log:
        for row in sorted(rows, key=lambda example:example['image']):
            start = time.monotonic()
            if row['image'] != cached_image:
                gray = read_drawing(root, row)
                cached_image = row['image']
            response = normalized_template_response(gray, tuple(row['query_box']))
            scores = score_response(response, row, root, thresholds)
            record = {'id':row['id'], 'seconds':time.monotonic()-start, 'scores':scores}
            records.append(record)
            log.write(json.dumps(record)+'\n')
            log.flush()
            print(json.dumps({'id':row['id'], 'seconds':record['seconds'],
                              'iou_at_0.65':scores['0.65']['metrics']['iou']}), flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    rows = load_manifest(args.manifest)
    if args.limit is not None:
        rows = rows[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [0.35,0.45,0.55,0.60,0.65,0.70,0.75,0.85,0.95]
    start = time.monotonic()
    records = run_sweep(rows, args.data_root, args.output_dir, thresholds)
    summaries = {str(threshold):aggregate([record['scores'][str(threshold)] for record in records])
                 for threshold in thresholds}
    report = {'method':'normalized-template-v1', 'partition':'public-training development only',
              'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'implementation_sha256':hashlib.sha256(Path('hatch_texture.py').read_bytes()).hexdigest(),
              'queries':len(rows), 'limited_pilot':args.limit is not None,
              'elapsed_seconds':time.monotonic()-start, 'thresholds':summaries}
    (args.output_dir/'sweep.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key:{name:value[name] for name in ('document_macro_iou','global_precision','global_recall')}
                      for key,value in summaries.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
