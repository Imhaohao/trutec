"""Create document-disjoint development partitions from public training only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def development_document(document_id: str) -> bool:
    return int(hashlib.sha256(document_id.encode()).hexdigest()[:8], 16) % 5 == 0


def main() -> None:
    source = Path('dataset/train.json')
    rows = json.loads(source.read_text())['examples']
    fit = [row for row in rows if not development_document(row['document_id'])]
    development = [row for row in rows if development_document(row['document_id'])]
    fit_documents = {row['document_id'] for row in fit}
    development_documents = {row['document_id'] for row in development}
    if fit_documents & development_documents:
        raise RuntimeError('Development documents overlap training documents')
    output = Path('artifacts/splits')
    output.mkdir(parents=True, exist_ok=True)
    partitions = {'fit':fit, 'development':development,
                  'development-real':[row for row in development if row['kind']=='real']}
    for name, examples in partitions.items():
        payload = {'schema':'hatch-matching-challenge/v1', 'examples':examples}
        (output/(name+'.json')).write_text(json.dumps(payload, indent=2))
        print(name, len(examples), 'queries,', len({row['document_id'] for row in examples}), 'documents')
    info = {'source':'public train only', 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'selection':'sha256(document_id) first 8 hex digits mod 5 == 0',
            'fit_documents':sorted(fit_documents), 'development_documents':sorted(development_documents)}
    (output/'split-info.json').write_text(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
