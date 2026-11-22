# Test Set Access Log

This log records every access to `vulnerability_test_heldout_projects_clean.parquet`.

| Timestamp (UTC) | Caller | Reason | SHA-256 Verified |
| :--- | :--- | :--- | :---: |
| 2026-11-19 10:52:33 UTC | `tests/test_locked_test.py:18` | Unit test verification of load_locked_test | True |
| 2026-11-19 10:52:33 UTC | `tests/test_locked_test.py:41` | FAILED_HASH_MISMATCH: Testing tampered file | False |
| 2026-11-19 10:54:30 UTC | `tests/test_locked_test.py:18` | Unit test verification of load_locked_test | True |
| 2026-11-19 10:54:30 UTC | `tests/test_locked_test.py:41` | FAILED_HASH_MISMATCH: Testing tampered file | False |
| 2026-11-19 10:54:50 UTC | `<string>:6` | Inspection for Step 3/4 | True |
| 2026-11-19 10:55:12 UTC | `<string>:7` | Projects inspection for Step 3 | True |
| 2026-11-19 10:59:35 UTC | `scripts/p2/run_validation_diagnosis.py:27` | Step 3: Validation Split Diagnosis | True |
| 2026-11-19 11:02:20 UTC | `scripts/p2/run_near_duplicate_analysis.py:80` | Step 4: Near-duplicate leakage check on clean test set | True |
| 2026-11-19 11:03:52 UTC | `scripts/p2/run_near_duplicate_analysis.py:80` | Step 4: Near-duplicate leakage check on clean test set | True |
| 2026-11-19 11:05:04 UTC | `scripts/p2/run_near_duplicate_analysis.py:80` | Step 4: Near-duplicate leakage check on clean test set | True |
| 2026-11-19 12:03:06 UTC | `<string>:3` | check test lengths | True |
| 2026-11-19 12:05:48 UTC | `scripts/p2/run_near_duplicate_analysis.py:117` | Step 4: Near-duplicate leakage check on clean test set | True |
| 2026-11-19 12:10:51 UTC | `scripts/p2/cache_embeddings_p2.py:195` | Step 6: Compute frozen embeddings and structural features for locked test cache | True |
| 2026-11-19 12:18:20 UTC | `scripts/p2/cache_embeddings_p2.py:196` | Step 6: Compute frozen embeddings and structural features for locked test cache | True |
| 2026-11-19 12:22:04 UTC | `scripts/p2/cache_embeddings_p2.py:206` | Step 6: Compute frozen embeddings and structural features for locked test cache | True |
| 2026-11-19 15:39:52 UTC | `tests/test_locked_test.py:18` | Unit test verification of load_locked_test | True |
| 2026-11-19 15:39:52 UTC | `tests/test_locked_test.py:41` | FAILED_HASH_MISMATCH: Testing tampered file | False |
| 2026-11-19 15:39:59 UTC | `tests/test_locked_test.py:18` | Unit test verification of load_locked_test | True |
| 2026-11-19 15:39:59 UTC | `tests/test_locked_test.py:41` | FAILED_HASH_MISMATCH: Testing tampered file | False |
| 2026-11-19 15:51:16 UTC | `scripts/p2/evaluate_reference_p2.py:43` | Step 7: Reference evaluation of length-bucketed predictions | True |
| 2026-11-20 04:53:51 UTC | `tests/test_locked_test.py:18` | Unit test verification of load_locked_test | True |
| 2026-11-20 04:53:51 UTC | `tests/test_locked_test.py:41` | FAILED_HASH_MISMATCH: Testing tampered file | False |
| 2026-11-20 06:44:40 UTC | `tests/test_locked_test.py:18` | Unit test verification of load_locked_test | True |
| 2026-11-20 06:44:40 UTC | `tests/test_locked_test.py:41` | FAILED_HASH_MISMATCH: Testing tampered file | False |
| 2026-11-20 16:47:08 UTC | `tests/test_locked_test.py:18` | Unit test verification of load_locked_test | True |
| 2026-11-20 16:47:08 UTC | `tests/test_locked_test.py:41` | FAILED_HASH_MISMATCH: Testing tampered file | False |
| 2026-11-21 14:17:02 UTC | `scripts/p6/evaluate_locked_test.py:219` | P6 single final evaluation of frozen Model A | True |
| 2026-11-21 15:46:37 UTC | `vulnerability_test_heldout_projects_clean.parquet` | `tests/test_locked_test.py:37` | FAILED_HASH_MISMATCH: Testing tampered file | False |
| 2026-11-21 15:46:37 UTC | `test2.parquet` | `tests/test_locked_test.py:39` | FAILED_HASH_MISMATCH: Testing tampered file2 | False |
| 2026-11-21 17:15:35 UTC | `test2.parquet` | `scripts/p10/train_model_a_v2.py:852` | P10 final | True |
| 2026-11-21 17:15:35 UTC | `vulnerability_test_heldout_projects_clean.parquet` | `scripts/p10/train_model_a_v2.py:857` | P10 secondary, viewed in P6 | True |
| 2026-11-21 17:24:02 UTC | `test2.parquet` | `scripts/p10/train_model_a_v2.py:853` | P10 final | True |
| 2026-11-21 17:24:02 UTC | `vulnerability_test_heldout_projects_clean.parquet` | `scripts/p10/train_model_a_v2.py:858` | P10 secondary, viewed in P6 | True |
| 2026-11-21 21:49:20 UTC | `vulnerability_test_heldout_projects_clean.parquet` | `tests/test_locked_test.py:37` | FAILED_HASH_MISMATCH: Testing tampered file | False |
| 2026-11-21 21:49:20 UTC | `test2.parquet` | `tests/test_locked_test.py:39` | FAILED_HASH_MISMATCH: Testing tampered file2 | False |
| 2026-11-21 23:04:36 UTC | `vulnerability_test_heldout_projects_clean.parquet` | `tests/test_locked_test.py:37` | FAILED_HASH_MISMATCH: Testing tampered file | False |
| 2026-11-21 23:04:36 UTC | `test2.parquet` | `tests/test_locked_test.py:39` | FAILED_HASH_MISMATCH: Testing tampered file2 | False |
| 2026-11-21 23:05:38 UTC | `vulnerability_test_heldout_projects_clean.parquet` | `tests/test_locked_test.py:37` | FAILED_HASH_MISMATCH: Testing tampered file | False |
| 2026-11-21 23:05:38 UTC | `test2.parquet` | `tests/test_locked_test.py:39` | FAILED_HASH_MISMATCH: Testing tampered file2 | False |
