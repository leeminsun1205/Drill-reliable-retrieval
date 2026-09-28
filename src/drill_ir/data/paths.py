"""Resolve DRiLL data paths consistently across local and Kaggle environments."""

from pathlib import Path

LOCAL_DATA_DIR = Path("data")
KAGGLE_DATASET_DIR = Path("/kaggle/input/datasets/lmnhnht125/wedev-vlsp")
KAGGLE_INPUT_DIR = Path("/kaggle/input")


def resolve_data_file(path: str | Path | None, *, filename: str) -> Path:
    """Resolve an explicit path, then known local/Kaggle locations.

    A recursive Kaggle lookup is accepted only when exactly one matching file
    exists, preventing silent selection of the wrong dataset version.
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path).expanduser())
    candidates.extend([LOCAL_DATA_DIR / filename, KAGGLE_DATASET_DIR / filename])

    unique_candidates: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)
            if candidate.is_file():
                return candidate.resolve()

    discovered = (
        sorted(KAGGLE_INPUT_DIR.glob(f"**/{filename}")) if KAGGLE_INPUT_DIR.is_dir() else []
    )
    discovered = [candidate for candidate in discovered if candidate.is_file()]
    if len(discovered) == 1:
        return discovered[0].resolve()
    if len(discovered) > 1:
        matches = "\n  - ".join(str(item) for item in discovered)
        raise FileNotFoundError(
            f"Multiple Kaggle files named {filename!r} were found; pass an explicit path:\n"
            f"  - {matches}"
        )

    attempted = "\n  - ".join(str(item) for item in unique_candidates)
    raise FileNotFoundError(
        f"Could not find {filename!r}. Tried:\n  - {attempted}\n"
        f"No unique match was found under {KAGGLE_INPUT_DIR}."
    )


def resolve_corpus_path(path: str | Path | None = None) -> Path:
    return resolve_data_file(path, filename="legal-corpus.parquet")


def resolve_train_path(path: str | Path | None = None) -> Path:
    return resolve_data_file(path, filename="train.parquet")
