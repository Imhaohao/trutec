"""Download the pinned Apache-2.0 HatchFinder model and verify its bytes."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tempfile
import urllib.request

MODEL_URL = 'https://huggingface.co/GreenMap/hatch-finder-3.5m/resolve/762f842d1e1f05b14cab8f19cbc2ddb749a8385f/model.pt'
MODEL_SHA256 = '47fcc870796d5e3f68dbdef1b3069911105585643350e66fdd6d2026eb9a762b'
MODEL_BYTES = 14104743


def verify_model(path: Path) -> None:
    if path.stat().st_size != MODEL_BYTES:
        raise ValueError('Model byte length does not match the pinned release')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError('Model SHA-256 does not match the pinned release')


def download_model(path: Path) -> None:
    if path.exists():
        verify_model(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hatch-model-', dir=path.parent) as folder:
        stage = Path(folder)/'model.pt'
        request = urllib.request.Request(MODEL_URL, headers={'User-Agent':'TruTec-hatch-research/1'})
        with urllib.request.urlopen(request, timeout=120) as response, stage.open('wb') as output:
            while chunk := response.read(1024*1024):
                output.write(chunk)
        verify_model(stage)
        stage.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('models/hatchfinder-3.5m.pt'))
    args = parser.parse_args()
    download_model(args.output)
    print('Verified model:', args.output, MODEL_SHA256)


if __name__ == '__main__':
    main()
