import subprocess
import zipfile
from pathlib import Path

import pytest

from tools.build_research_delivery import (
    create_verified_bundle,
    QUERY_RECORDS,
    RESEARCH_FILES,
    require_commit,
    require_committed_file,
    sha256_file,
    write_deterministic_zip,
)


def make_repository(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "--quiet", str(path)], check=True)
    (path / "source.txt").write_text("frozen source\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "source.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "source"],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_source_bundle_clones_and_checks_out_exact_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    commit = make_repository(repo)
    first_bundle = tmp_path / "source-first.bundle"
    second_bundle = tmp_path / "source-second.bundle"
    proof = create_verified_bundle(repo, commit, first_bundle)
    second_proof = create_verified_bundle(repo, commit, second_bundle)

    assert proof["commit"] == commit
    assert proof["cloned_commit"] == commit
    assert proof["temporary_clone_verified"] is True
    assert proof["bytes"] > 0
    assert sha256_file(first_bundle) == sha256_file(second_bundle)
    assert proof["sha256"] == second_proof["sha256"]
    refs = subprocess.run(
        ["git", "-C", str(repo), "for-each-ref", "--format=%(refname)", "refs/codex-delivery-tmp"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert refs == ""


def test_commit_requires_full_exact_sha(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    make_repository(repo)

    with pytest.raises(ValueError, match="full 40-character"):
        require_commit(repo, "abcd1234")


def test_packaged_documents_must_match_source_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    commit = make_repository(repo)

    require_committed_file(repo, commit, "source.txt")
    (repo / "source.txt").write_text("uncommitted edit\n", encoding="utf-8")

    with pytest.raises(ValueError, match="differs from the requested source commit"):
        require_committed_file(repo, commit, "source.txt")


def test_zip_has_fixed_order_timestamps_and_bytes(tmp_path: Path) -> None:
    payload = {"z.txt": b"last", "a.txt": b"first"}
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    write_deterministic_zip(payload, first)
    write_deterministic_zip(payload, second)

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["a.txt", "z.txt"]
        assert archive.getinfo("a.txt").date_time == (1980, 1, 1, 0, 0, 0)


def test_research_inventory_keeps_resource_proof_and_excludes_large_sweep() -> None:
    required_seggpt = {
        "seggpt-resource-freeze.json",
        "seggpt-runtime-environment.txt",
        "seggpt-resource-probe.json",
        "seggpt-resource-proof.json",
        "seggpt-resource-baseline.json",
        "seggpt-resource-end.json",
    }

    assert required_seggpt.issubset(RESEARCH_FILES)
    assert "cadence-development/sweep.json" not in RESEARCH_FILES
    assert "cadence-development/query-records.jsonl" in QUERY_RECORDS
