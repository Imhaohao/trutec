"""Test script to verify synthetic manifest with evaluate.py functions."""

import sys
from pathlib import Path
from evaluate import load_manifest, reviewed_domains


def test_synthetic_manifest(manifest_path, data_root):
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        print(f"Error: manifest file {manifest_file} does not exist.")
        return False

    rows = load_manifest(manifest_file, require_labels=True)
    print(f"Loaded {len(rows)} examples from {manifest_path}")

    for idx, row in enumerate(rows):
        pos, known, blank, meta = reviewed_domains(row, data_root)
        if pos.sum() == 0:
            print(f"Error on {row['id']}: 0 positive pixels in reviewed domain.")
            return False
        if not known.any():
            print(f"Error on {row['id']}: 0 known pixels.")
            return False

    print("SUCCESS: All examples verified successfully with evaluate.py!")
    return True


if __name__ == "__main__":
    m_path = "research_v5/synthetic/data/manifest.json"
    d_root = "research_v5/synthetic/data"
    ok = test_synthetic_manifest(m_path, d_root)
    sys.exit(0 if ok else 1)
