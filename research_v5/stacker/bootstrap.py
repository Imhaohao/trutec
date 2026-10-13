"""Process limits and import paths. Import this before numpy."""
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research_v5"
for entry in (str(ROOT), str(RESEARCH)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

# dataset/assets is a symlink to the image store. evaluate.data_path resolves
# that link and then rejects the file for leaving the dataset directory.
# Keep the logical path under dataset/ and let open() follow the link.
import evaluate  # noqa: E402


def _logical_data_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise evaluate.EvaluationError("Dataset paths must be nonempty relative paths")
    parts = Path(relative).parts
    if ".." in parts:
        raise evaluate.EvaluationError("Dataset path escapes --data-root: " + relative)
    base = Path(root)
    path = base.joinpath(*parts)
    if not path.is_relative_to(base):
        raise evaluate.EvaluationError("Dataset path escapes --data-root: " + relative)
    return path


evaluate.data_path = _logical_data_path


def atomic_savez(path, **arrays) -> None:
    """Write npz via a file handle so numpy cannot rename the destination."""
    import os

    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.is_file():
            temporary.unlink()


def atomic_save_npy(path, values) -> None:
    import os

    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.is_file():
            temporary.unlink()
