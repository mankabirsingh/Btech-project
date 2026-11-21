"""Guard for reading the locked held-out test sets with checksum validation and access logging."""

import hashlib
import inspect
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import pandas as pd

EXPECTED_SHA256 = "ebd95dfc837f12264dd0fd8126563d00ede27908bca38fe46611e25d7bb45bbc"
EXPECTED_SHA256_TEST2 = "c52a6976429ec795265503155ca4d9663838d98bd53a8f360061cf175baead94"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_PARQUET_PATH = PROJECT_ROOT / "data" / "splits" / "vulnerability_test_heldout_projects_clean.parquet"
TEST2_PARQUET_PATH = PROJECT_ROOT / "data" / "p10" / "test2.parquet"
ACCESS_LOG_PATH = PROJECT_ROOT / "data" / "TEST_ACCESS_LOG.md"


def compute_file_sha256(path: Path) -> str:
    """Compute the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def _get_caller_info() -> str:
    """Identify the external calling script and line number."""
    stack = inspect.stack()
    current_file = Path(__file__).resolve()
    for frame_info in stack[1:]:
        frame_file = Path(frame_info.filename).resolve()
        if frame_file != current_file:
            try:
                rel = frame_file.relative_to(PROJECT_ROOT)
                return f"{rel}:{frame_info.lineno}"
            except ValueError:
                return f"{frame_file.name}:{frame_info.lineno}"
    return "unknown"


def _log_access(caller: str, dataset_name: str, reason: str, sha256_verified: bool) -> None:
    """Append access record to data/TEST_ACCESS_LOG.md."""
    ACCESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ACCESS_LOG_PATH.exists():
        ACCESS_LOG_PATH.write_text(
            "# Test Set Access Log\n\n"
            "This log records every access to locked test sets.\n\n"
            "| Timestamp (UTC) | Dataset | Caller | Reason | SHA-256 Verified |\n"
            "| :--- | :--- | :--- | :--- | :---: |\n",
            encoding="utf-8",
        )

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    clean_reason = reason.replace("|", "\\|").replace("\n", " ")
    entry = f"| {now_iso} | `{dataset_name}` | `{caller}` | {clean_reason} | {sha256_verified} |\n"

    with open(ACCESS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry)


def load_locked_test(reason: str, custom_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the locked held-out test parquet file after verifying its SHA-256 checksum.
    """
    if not reason or not isinstance(reason, str) or not reason.strip():
        raise ValueError("Must provide a non-empty reason for loading the locked test set.")

    target_path = custom_path if custom_path is not None else TEST_PARQUET_PATH

    if not target_path.exists():
        raise FileNotFoundError(f"Locked test dataset not found at {target_path}")

    actual_hash = compute_file_sha256(target_path)
    caller = _get_caller_info()

    if actual_hash != EXPECTED_SHA256:
        _log_access(caller, "vulnerability_test_heldout_projects_clean.parquet", f"FAILED_HASH_MISMATCH: {reason}", sha256_verified=False)
        raise RuntimeError(
            f"SHA-256 checksum mismatch for {target_path}!\n"
            f"Expected: {EXPECTED_SHA256}\n"
            f"Actual:   {actual_hash}"
        )

    _log_access(caller, "vulnerability_test_heldout_projects_clean.parquet", reason.strip(), sha256_verified=True)
    return pd.read_parquet(target_path)


def load_locked_test2(reason: str, custom_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the new locked test2 parquet file after verifying its SHA-256 checksum.

    Appends an access entry to data/TEST_ACCESS_LOG.md.
    NEVER call before A1.8 final evaluation!
    """
    if not reason or not isinstance(reason, str) or not reason.strip():
        raise ValueError("Must provide a non-empty reason for loading the locked test2 set.")

    target_path = custom_path if custom_path is not None else TEST2_PARQUET_PATH

    if not target_path.exists():
        raise FileNotFoundError(f"Locked test2 dataset not found at {target_path}")

    actual_hash = compute_file_sha256(target_path)
    caller = _get_caller_info()

    if actual_hash != EXPECTED_SHA256_TEST2:
        _log_access(caller, "test2.parquet", f"FAILED_HASH_MISMATCH: {reason}", sha256_verified=False)
        raise RuntimeError(
            f"SHA-256 checksum mismatch for {target_path}!\n"
            f"Expected: {EXPECTED_SHA256_TEST2}\n"
            f"Actual:   {actual_hash}"
        )

    _log_access(caller, "test2.parquet", reason.strip(), sha256_verified=True)
    return pd.read_parquet(target_path)
