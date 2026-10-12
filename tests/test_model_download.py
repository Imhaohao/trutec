"""Reject damaged or substituted model artifacts before loading weights."""
import hashlib
from pathlib import Path

import pytest

import download_model


def test_model_integrity_rejects_wrong_length(tmp_path: Path):
    model = tmp_path/'model.pt'
    model.write_bytes(b'incomplete')
    with pytest.raises(ValueError, match='byte length'):
        download_model.verify_model(model)


def test_model_integrity_checks_content_even_when_length_matches(tmp_path: Path, monkeypatch):
    model = tmp_path/'model.pt'
    original = b'weights'
    monkeypatch.setattr(download_model, 'MODEL_BYTES', len(original))
    monkeypatch.setattr(download_model, 'MODEL_SHA256', hashlib.sha256(original).hexdigest())
    model.write_bytes(original)
    download_model.verify_model(model)
    model.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='SHA-256'):
        download_model.verify_model(model)


def test_existing_damaged_model_is_not_silently_replaced(tmp_path: Path):
    model = tmp_path/'model.pt'
    model.write_bytes(b'wrong artifact')
    with pytest.raises(ValueError, match='byte length'):
        download_model.download_model(model)
    assert model.read_bytes() == b'wrong artifact'
