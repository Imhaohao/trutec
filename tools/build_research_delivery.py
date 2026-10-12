#!/usr/bin/env python3
"""Build a deterministic local research delivery after its source commit exists."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V2_ARCHIVE_SHA256 = "cb37d5d5e1c70884f366c636c8ba206532161595a64cd5b932aa301ba9874e34"
REJECTED_CADENCE_MODEL_SHA256 = "49b0a2fd390c09f26f51dfdfeba41743780344b72bbff794c6a45f4c8a02a0f4"
RESEARCH_FILES = (
    "research-state.json",
    "paired-development-review.json",
    "paired-evaluation-plan.json",
    "paired-training-protocol.json",
    "negative-weight-development-review.json",
    "negative-weight-evaluation-plan.json",
    "negative-weight-training-protocol.json",
    "dino-development-review.json",
    "dino-evaluation-plan.json",
    "dino-classifier-protocol.json",
    "dino-final-review.json",
    "external-dino-license.json",
    "cadence-development-review.json",
    "cadence-development-proof.json",
    "cadence-evaluation-plan.json",
    "cadence-feature-protocol.json",
    "cadence-integration-plan.json",
    "cadence-fit-verification-provenance.json",
    "cadence-training-proof.json",
    "cadence-vs-original-cnn.json",
    "p0-training-proof.json",
    "p1-training-proof.json",
    "p2-training-proof.json",
    "seggpt-interface-review.json",
    "seggpt-resource-protocol.json",
    "seggpt-resource-freeze.json",
    "seggpt-runtime-environment.txt",
    "seggpt-resource-probe.json",
    "seggpt-resource-proof.json",
    "seggpt-resource-baseline.json",
    "seggpt-resource-end.json",
    "seggpt-resource.log",
    "seggpt-weight-provisioning.json",
)
QUERY_RECORDS = (
    "P0-development/query-records.jsonl",
    "P1-development/query-records.jsonl",
    "P2-development/query-records.jsonl",
    "DINO-development/query-records.jsonl",
    "cadence-development/query-records.jsonl",
)
REPRODUCTION_TEXT = """Research delivery v4

The unchanged public-validation result is the preserved v2 archive. All v4
comparisons are development-only and were not promoted. Reproducing development
comparisons requires the original challenge dataset and the source revision
recorded in source-bundle-proof.json. Probability caches, dataset files, raw
provider logs, the full cadence threshold sweep, external pretrained DINO
weights, and external SegGPT weights are excluded. The cadence sweep SHA-256 is
686a82a18e9399fd68e6c90c71d0707ed0b90316d692002a70c733eddf0e60db; regenerate
it from the original dataset to inspect the complete threshold grid. The cadence
model is included as rejected research only and must not be used as a promoted
submission. The DINO checkpoint must be obtained separately under its license.

The delivery manifest and checksums cover each included payload. The v2 archive
is included byte-for-byte and must have SHA-256
cb37d5d5e1c70884f366c636c8ba206532161595a64cd5b932aa301ba9874e34.
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def require_commit(repo: Path, source_commit: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source commit must be a full 40-character lowercase SHA")
    resolved = git_output(repo, "rev-parse", "--verify", f"{source_commit}^{{commit}}")
    if resolved != source_commit:
        raise ValueError("source commit did not resolve to the exact requested SHA")
    return resolved


def require_committed_file(repo: Path, source_commit: str, relative_path: str) -> None:
    committed = subprocess.run(
        ["git", "-C", str(repo), "show", f"{source_commit}:{relative_path}"],
        check=True,
        capture_output=True,
    ).stdout
    if committed != (repo / relative_path).read_bytes():
        raise ValueError(f"working file differs from the requested source commit: {relative_path}")


def create_verified_bundle(repo: Path, source_commit: str, bundle_path: Path) -> dict:
    commit = require_commit(repo, source_commit)
    temporary_ref = f"refs/codex-delivery-tmp/{commit}"
    empty_sha = "0" * len(commit)
    git_output(repo, "update-ref", temporary_ref, commit, empty_sha)
    try:
        subprocess.run(
            ["git", "-C", str(repo), "bundle", "create", str(bundle_path), temporary_ref],
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        git_output(repo, "update-ref", "-d", temporary_ref, commit)
    with tempfile.TemporaryDirectory(prefix="research-delivery-source-") as temporary:
        clone = Path(temporary) / "clone"
        subprocess.run(
            ["git", "clone", "--quiet", str(bundle_path), str(clone)],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(clone), "fetch", "--quiet", str(bundle_path), temporary_ref],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(clone), "checkout", "--quiet", "--detach", commit],
            check=True,
            capture_output=True,
            text=True,
        )
        cloned_commit = git_output(clone, "rev-parse", "HEAD")
    if cloned_commit != commit:
        raise ValueError("temporary bundle clone checked out a different commit")
    return {
        "schema": "source-bundle-proof/v1",
        "commit": commit,
        "bytes": bundle_path.stat().st_size,
        "sha256": sha256_file(bundle_path),
        "temporary_clone_verified": True,
        "cloned_commit": cloned_commit,
    }


def add_payload(payload: dict[str, bytes], archive_path: str, source: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"required delivery input is missing: {source}")
    payload[archive_path] = source.read_bytes()


def collect_payload(repo: Path, bundle_proof: dict, bundle_path: Path) -> dict[str, bytes]:
    payload: dict[str, bytes] = {}
    for relative in (
        "LICENSE",
        "DATA_TERMS.md",
        "environment.txt",
        "requirements-lock.txt",
        "requirements-inference.txt",
        "requirements-neural.txt",
        "technical-report.txt",
        "submission-email-draft.txt",
    ):
        if relative in {"technical-report.txt", "submission-email-draft.txt"}:
            require_committed_file(repo, bundle_proof["commit"], relative)
        add_payload(payload, relative, repo / relative)
    add_payload(payload, "hatch-matching-submission-v2.zip", repo / "artifacts/delivery-v2/hatch-matching-submission-v2.zip")
    if sha256_bytes(payload["hatch-matching-submission-v2.zip"]) != V2_ARCHIVE_SHA256:
        raise ValueError("the preserved v2 archive checksum changed")
    for relative in ("checksums.txt", "delivery-manifest.json", "source-bundle-proof.json"):
        add_payload(payload, f"v2/{relative}", repo / "artifacts/delivery-v2" / relative)
    for relative in RESEARCH_FILES:
        add_payload(payload, f"research-v4/{relative}", repo / "artifacts/research-v4" / relative)
    for relative in QUERY_RECORDS:
        add_payload(payload, f"research-v4/{relative}", repo / "artifacts/research-v4" / relative)
    add_rejected_cadence_model(repo, payload)
    payload["source.bundle"] = bundle_path.read_bytes()
    payload["source-bundle-proof.json"] = json_bytes(bundle_proof)
    payload["reproduction.txt"] = REPRODUCTION_TEXT.encode("utf-8")
    payload["checksums.txt"] = checksums_bytes(payload)
    payload["delivery-manifest.json"] = json_bytes(manifest_for(payload))
    return payload


def add_rejected_cadence_model(repo: Path, payload: dict[str, bytes]) -> None:
    model = repo / "models/cadence-hgb-v4.joblib"
    metadata = repo / "models/cadence-hgb-v4.json"
    checksum = repo / "models/cadence-hgb-v4.joblib.sha256"
    sidecars = (model, metadata, checksum)
    if not any(path.exists() for path in sidecars):
        return
    if not all(path.is_file() for path in sidecars):
        raise FileNotFoundError("the rejected cadence model must include its JSON and SHA-256 sidecars")
    model_bytes = model.read_bytes()
    metadata_bytes = metadata.read_bytes()
    checksum_fields = checksum.read_text(encoding="utf-8").strip().split()
    if sha256_bytes(model_bytes) != REJECTED_CADENCE_MODEL_SHA256:
        raise ValueError("rejected cadence model checksum differs from the reviewed artifact")
    if not checksum_fields or checksum_fields[0] != REJECTED_CADENCE_MODEL_SHA256:
        raise ValueError("rejected cadence model checksum sidecar does not match")
    proof_path = repo / "artifacts/research-v4/cadence-training-proof.json"
    training_proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if sha256_bytes(metadata_bytes) != training_proof.get("model_metadata_sha256"):
        raise ValueError("rejected cadence model metadata differs from its training proof")
    payload["rejected-research-only/cadence-hgb-v4.joblib"] = model_bytes
    payload["rejected-research-only/cadence-hgb-v4.json"] = metadata_bytes
    payload["rejected-research-only/cadence-hgb-v4.joblib.sha256"] = checksum.read_bytes()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def manifest_for(payload: dict[str, bytes]) -> dict:
    return {
        "schema": "hatch-matching-research-delivery/v4",
        "source_commit": json.loads(payload["source-bundle-proof.json"])["commit"],
        "primary_submission": "preserved_v2",
        "publication_complete": False,
        "email_sent": False,
        "dataset_included": False,
        "probability_caches_included": False,
        "external_dino_weights_included": False,
        "external_seggpt_weights_included": False,
        "cadence_model_role": "rejected_research_only",
        "files": [
            {"path": name, "bytes": len(data), "sha256": sha256_bytes(data)}
            for name, data in sorted(payload.items())
        ],
    }


def checksums_bytes(payload: dict[str, bytes]) -> bytes:
    rows = [f"{sha256_bytes(data)}  {name}" for name, data in sorted(payload.items())]
    return ("\n".join(rows) + "\n").encode("utf-8")


def write_deterministic_zip(payload: dict[str, bytes], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)


def verify_archive(payload: dict[str, bytes], archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("delivery archive failed CRC verification")
        if archive.namelist() != sorted(payload):
            raise ValueError("delivery archive members differ from the payload manifest")
        for name, expected in payload.items():
            if sha256_bytes(archive.read(name)) != sha256_bytes(expected):
                raise ValueError(f"delivery member hash mismatch: {name}")


def build_delivery(repo: Path, source_commit: str, output: Path) -> dict:
    commit = require_commit(repo, source_commit)
    with tempfile.TemporaryDirectory(prefix="research-delivery-build-") as temporary:
        bundle_path = Path(temporary) / "source.bundle"
        proof = create_verified_bundle(repo, commit, bundle_path)
        payload = collect_payload(repo, proof, bundle_path)
        write_deterministic_zip(payload, output)
        verify_archive(payload, output)
    archive_digest = sha256_file(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{archive_digest}  {output.name}\n", encoding="utf-8"
    )
    proof_path = output.parent / "package-proof.json"
    proof_path.write_bytes(json_bytes({
        "schema": "hatch-delivery-proof/v4",
        "archive": {"path": output.name, "bytes": output.stat().st_size, "sha256": archive_digest},
        "preserved_v2_sha256": V2_ARCHIVE_SHA256,
        "source_commit": commit,
        "source_bundle_temporary_clone_verified": True,
        "outer_crc_and_all_member_hashes_verified": True,
        "publication_complete": False,
        "email_sent": False,
    }))
    return {"archive": str(output), "sha256": archive_digest, "source_commit": commit}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True, help="full frozen source commit SHA")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/delivery-v4/hatch-matching-research-delivery-v4.zip",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_delivery(ROOT, args.source_commit, args.output.resolve())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
