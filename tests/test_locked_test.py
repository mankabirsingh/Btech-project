"""Unit tests for src/eval/locked_test.py."""

import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.eval.locked_test import (
    EXPECTED_SHA256,
    EXPECTED_SHA256_TEST2,
    ACCESS_LOG_PATH,
    compute_file_sha256,
    load_locked_test,
    load_locked_test2,
)


class TestLockedTest(unittest.TestCase):
    def test_empty_reason_raises(self):
        """Empty reason should raise ValueError."""
        with self.assertRaises(ValueError):
            load_locked_test(reason="")
        with self.assertRaises(ValueError):
            load_locked_test(reason="   ")
        with self.assertRaises(ValueError):
            load_locked_test2(reason="")
        with self.assertRaises(ValueError):
            load_locked_test2(reason="   ")

    def test_tampered_file_raises(self):
        """Tampered file should fail checksum and raise RuntimeError."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_path = Path(f.name)
            temp_path.write_bytes(b"tampered content")

        try:
            with self.assertRaises(RuntimeError):
                load_locked_test(reason="Testing tampered file", custom_path=temp_path)
            with self.assertRaises(RuntimeError):
                load_locked_test2(reason="Testing tampered file2", custom_path=temp_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def test_nonexistent_file_raises(self):
        """Nonexistent file should raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_locked_test(reason="Testing missing file", custom_path=Path("/tmp/nonexistent_xyz.parquet"))
        with self.assertRaises(FileNotFoundError):
            load_locked_test2(reason="Testing missing file", custom_path=Path("/tmp/nonexistent_xyz.parquet"))


if __name__ == "__main__":
    unittest.main()
