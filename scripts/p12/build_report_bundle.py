#!/usr/bin/env python3
"""
scripts/p12/build_report_bundle.py

Packages everything needed to write the final project report into report_bundle/ and report_bundle.zip:
- results/: copy of all results/p1..p12 *.json and *.md (no .parquet, .npy, .pt, .pkl)
- figs/: paper/figs/*.png
- FINAL_RESULTS.md, facts.md, all reports/HANDOFF_P*.md
- screenshots/ (scan_vulnerable.png, scan_fixed.png, model_performance.png)
- Zip to report_bundle.zip (< 25 MB)
- Prints zip path, size, and file count.
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BUNDLE_DIR = BASE_DIR / "report_bundle"
ZIP_PATH = BASE_DIR / "report_bundle.zip"

DISALLOWED_EXTENSIONS = {".parquet", ".npy", ".pt", ".pkl", ".bin", ".h5", ".pth", ".ckpt"}


def main():
    print("=" * 80)
    print("P12: ASSEMBLING REPORT BUNDLE AND CREATING ZIP ARCHIVE")
    print("=" * 80)

    # Clean / prepare target directories
    bundle_results = BUNDLE_DIR / "results"
    bundle_figs = BUNDLE_DIR / "figs"
    bundle_reports = BUNDLE_DIR / "reports"
    bundle_screenshots = BUNDLE_DIR / "screenshots"

    bundle_results.mkdir(parents=True, exist_ok=True)
    bundle_figs.mkdir(parents=True, exist_ok=True)
    bundle_reports.mkdir(parents=True, exist_ok=True)
    bundle_screenshots.mkdir(parents=True, exist_ok=True)

    # 1. Copy results/*.json and *.md from results/
    print("\n[Step 1] Copying results JSON and Markdown files...")
    results_src = BASE_DIR / "results"
    copied_results = 0
    for root, dirs, files in os.walk(results_src):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in DISALLOWED_EXTENSIONS:
                continue
            if ext in {".json", ".md"}:
                src_file = Path(root) / f
                rel_path = src_file.relative_to(results_src)
                dst_file = bundle_results / rel_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                copied_results += 1
    print(f"  Copied {copied_results} result files (.json, .md) into report_bundle/results/")

    # 2. Copy figures
    print("\n[Step 2] Copying paper figures...")
    figs_src = BASE_DIR / "paper" / "figs"
    copied_figs = 0
    if figs_src.exists():
        for f in sorted(os.listdir(figs_src)):
            if f.endswith(".png"):
                shutil.copy2(figs_src / f, bundle_figs / f)
                copied_figs += 1
    print(f"  Copied {copied_figs} figure files (.png) into report_bundle/figs/")

    # 3. Copy reports (FINAL_RESULTS.md, facts.md, HANDOFF_P*.md)
    print("\n[Step 3] Copying reports and markdown documentation...")
    shutil.copy2(BASE_DIR / "reports" / "FINAL_RESULTS.md", BUNDLE_DIR / "FINAL_RESULTS.md")
    shutil.copy2(BASE_DIR / "reports" / "FINAL_RESULTS.md", bundle_reports / "FINAL_RESULTS.md")

    # Copy all reports/HANDOFF_P*.md
    reports_src = BASE_DIR / "reports"
    copied_handoffs = 0
    for f in sorted(os.listdir(reports_src)):
        if f.startswith("HANDOFF_P") and f.endswith(".md"):
            shutil.copy2(reports_src / f, bundle_reports / f)
            shutil.copy2(reports_src / f, BUNDLE_DIR / f)
            copied_handoffs += 1
    print(f"  Copied FINAL_RESULTS.md and {copied_handoffs} HANDOFF reports into bundle.")

    # 4. Check facts.md and screenshots
    print("\n[Step 4] Verifying facts.md and screenshots...")
    assert (BUNDLE_DIR / "facts.md").exists(), "Error: report_bundle/facts.md missing!"
    for sc in ["scan_vulnerable.png", "scan_fixed.png", "model_performance.png"]:
        sc_path = bundle_screenshots / sc
        assert sc_path.exists(), f"Error: missing screenshot {sc_path}"
        print(f"  Screenshot verified: {sc} ({sc_path.stat().st_size / 1024:.1f} KB)")

    # 5. Build ZIP archive
    print("\n[Step 5] Building report_bundle.zip archive...")
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    file_count = 0
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(BUNDLE_DIR):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                assert ext not in DISALLOWED_EXTENSIONS, f"Disallowed file included: {file}"
                full_path = Path(root) / file
                rel_in_zip = full_path.relative_to(BUNDLE_DIR.parent)
                zipf.write(full_path, rel_in_zip)
                file_count += 1

    zip_size_bytes = ZIP_PATH.stat().st_size
    zip_size_mb = zip_size_bytes / (1024 * 1024)

    print("\n" + "=" * 80)
    print("REPORT BUNDLE CREATION SUMMARY")
    print("=" * 80)
    print(f"Archive Path: {ZIP_PATH.resolve()}")
    print(f"Archive Size: {zip_size_bytes:,} bytes ({zip_size_mb:.2f} MB)")
    print(f"Total Files in Archive: {file_count}")
    print(f"Size Compliance (< 25 MB): {'PASSED' if zip_size_mb < 25.0 else 'FAILED'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
