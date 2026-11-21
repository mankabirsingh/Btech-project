#!/usr/bin/env python3
"""
scripts/p7/generate_real_llm.py
===============================
Phase 16 Stage 2 / Task 10: Part B Real LLM Versions.
Checks if GEMINI_API_KEY or OPENAI_API_KEY is present in environment (without printing key value).
If neither is set, writes 'skipped' to results/p7/part_b_real_llm.json.
If set, would generate pure-AI versions for up to 100 tasks at temp 0.2 and 0.8.
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_P7_DIR = PROJECT_ROOT / "results" / "p7"
RESULTS_P7_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RESULTS_P7_DIR / "part_b_real_llm.json"


def main():
    print("=" * 80)
    print("PART B: REAL LLM GENERATION CHECK")
    print("=" * 80)

    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    has_gemini = bool(gemini_key and gemini_key.strip())
    has_openai = bool(openai_key and openai_key.strip())

    print(f"GEMINI_API_KEY present: {has_gemini}")
    print(f"OPENAI_API_KEY present: {has_openai}")

    if not (has_gemini or has_openai):
        res = {
            "status": "skipped",
            "reason": "Neither GEMINI_API_KEY nor OPENAI_API_KEY is set in environment",
            "n_tasks_attempted": 0,
            "n_generated": 0,
            "temperatures": [0.2, 0.8]
        }
        with open(OUT_PATH, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Part B status: skipped. Written to {OUT_PATH}")
        return

    # If keys were present, generation logic would proceed here
    # (Keys are not present in the current execution environment)


if __name__ == "__main__":
    main()
