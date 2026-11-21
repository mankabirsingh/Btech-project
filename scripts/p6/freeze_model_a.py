"""Step 1 of P6: Compute and record SHA-256 hashes of all model_a_final files.

Must be executed BEFORE touching the locked test set.
Saves to results/p6/freeze.json.
"""

import hashlib
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_A_FINAL_DIR = PROJECT_ROOT / "models" / "checkpoints" / "p5" / "model_a_final"
RESULTS_P6_DIR = PROJECT_ROOT / "results" / "p6"
RESULTS_P6_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 80)
    print("FREEZING MODEL A FINAL: COMPUTING SHA-256 DIGESTS PRIOR TO TEST ACCESS")
    print("=" * 80)

    frozen_files = {}
    for p in sorted(MODEL_A_FINAL_DIR.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(MODEL_A_FINAL_DIR))
            sha = sha256_file(p)
            size_bytes = p.stat().st_size
            frozen_files[rel] = {
                "sha256": sha,
                "size_bytes": size_bytes,
            }
            print(f"  {rel} ({size_bytes:,} bytes): {sha}")

    freeze_record = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model_dir": str(MODEL_A_FINAL_DIR),
        "status": "FROZEN_BEFORE_TEST_EVALUATION",
        "file_count": len(frozen_files),
        "files": frozen_files,
    }

    out_path = RESULTS_P6_DIR / "freeze.json"
    with open(out_path, "w") as f:
        json.dump(freeze_record, f, indent=2)

    print(f"\nSuccessfully wrote freeze record to {out_path}")


if __name__ == "__main__":
    main()
