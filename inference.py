"""Label-free native-resolution inference for the hatch matching challenge."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

from evaluate import EvaluationError, data_path, load_manifest


def read_requests(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get('schema') != 'hatch-matching-inputs/v1':
        raise EvaluationError('Expected a label-free hatch-matching-inputs/v1 manifest')
    rows = load_manifest(path, require_labels=False)
    allowed = {'id', 'image', 'width', 'height', 'query_box', 'context_boxes'}
    for row in rows:
        if set(row) - allowed:
            raise EvaluationError('Inference request contains non-input fields: '+row['id'])
    return rows


def read_drawing(root: Path, row: dict[str, Any]) -> np.ndarray:
    path = data_path(root, row['image'])
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.shape != (row['height'], row['width']):
        raise EvaluationError('Cannot read drawing at its declared native dimensions: '+row['id'])
    return gray


def save_prediction(path: Path, mask: np.ndarray) -> None:
    image = Image.fromarray(mask.astype(np.uint8)*255)
    temporary = path.with_suffix('.partial.png')
    image.save(temporary, format='PNG')
    temporary.replace(path)


def run_template_inference(rows: list[dict[str, Any]], root: Path, output: Path,
                           threshold: float) -> list[dict[str, Any]]:
    from hatch_texture import normalized_template_response, template_response_to_mask

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row['image']].append(row)
    records = []
    for group in grouped.values():
        gray = read_drawing(root, group[0])
        for row in group:
            if gray.shape != (row['height'], row['width']):
                raise EvaluationError('Inconsistent dimensions for shared drawing')
            start = time.monotonic()
            response = normalized_template_response(gray, tuple(row['query_box']))
            mask = template_response_to_mask(response, threshold)
            save_prediction(output/(row['id']+'.png'), mask)
            record = {'id':row['id'], 'seconds':time.monotonic()-start,
                      'selected_pixels':int(mask.sum())}
            records.append(record)
            print(json.dumps(record), flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--threshold', type=float, default=0.65)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error('threshold must lie in [0,1]')
    rows = read_requests(args.inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    records = run_template_inference(rows, args.data_root, args.output_dir, args.threshold)
    report = {'method':'normalized-template-v1', 'threshold':args.threshold,
              'elapsed_seconds':time.monotonic()-start, 'queries':len(records),
              'timing_coverage':'image loading, preprocessing, inference, and PNG writing; imports excluded',
              'examples':records}
    (args.output_dir/'runtime.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
