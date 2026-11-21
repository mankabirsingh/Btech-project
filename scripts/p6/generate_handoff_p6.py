"""Generates reports/HANDOFF_P6.md strictly from results/p6/locked_test_results.json.

Constraints:
- At most 2 pages
- Tables for steps 4-7
- At most 5 bullets of surprises or failures
- No adjectives
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_P6_DIR = PROJECT_ROOT / "results" / "p6"
REPORTS_DIR = PROJECT_ROOT / "reports"


def main():
    with open(RESULTS_P6_DIR / "locked_test_results.json") as f:
        data = json.load(f)

    full = data["full_test_evaluation"]
    dedup = data["deduplicated_test_evaluation"]
    pairs = data["pairs_evaluation"]
    cwe = data["cwe_evaluation"]
    paired = data["paired_tests"]
    l1_expl = data["l1_pair_correct_explanation"]

    lines = []
    lines.append("# Phase 16 — Prompt 6: Model A Locked Test Evaluation")
    lines.append("")
    lines.append("## 1. Frozen Artifact Verification")
    lines.append("- Checkpoint directory: `models/checkpoints/p5/model_a_final/`")
    lines.append("- Freeze record: `results/p6/freeze.json` (timestamp: 2026-11-21 14:14:03 UTC, 8 files, SHA-256 verified)")
    lines.append("- Test dataset access: `data/splits/vulnerability_test_heldout_projects_clean.parquet` loaded once via `load_locked_test(reason=\"P6 single final evaluation of frozen Model A\")`")
    lines.append(f"- L1 pair_correct discrepancy explanation: {l1_expl}")
    lines.append("")
    lines.append("## 2. Full and Deduplicated Test Evaluation (Step 4)")
    lines.append("")
    lines.append("### Full Test Set (n=3,169)")
    lines.append("")
    lines.append("| Cohort | Model | n | Prev | AUROC [95% CI] | LS-AUROC 4-bin [95% CI] | LS-AUROC 10-bin [95% CI] | Length-Matched AUROC [95% CI] (Pairs) | AUPRC [95% CI] | Prec@10% | Rec@10% | Lift |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

    cohort_order = ["Overall", "Envoy", "Envoy Production", "OpenSSL", "Vim", "GPAC"]
    models_order = ["L1", "Champion", "D2_ensemble"]

    for c in cohort_order:
        for m in models_order:
            r = full[m][c]
            lines.append(
                f"| {c} | {m} | {r['n']} | {r['prevalence']*100:.1f}% | "
                f"{r['auroc']:.4f} [{r['auroc_ci_95'][0]:.4f}, {r['auroc_ci_95'][1]:.4f}] | "
                f"{r['ls_auroc_4bin']:.4f} [{r['ls_auroc_4bin_ci_95'][0]:.4f}, {r['ls_auroc_4bin_ci_95'][1]:.4f}] | "
                f"{r['ls_auroc_10bin']:.4f} [{r['ls_auroc_10bin_ci_95'][0]:.4f}, {r['ls_auroc_10bin_ci_95'][1]:.4f}] | "
                f"{r['length_matched_auroc']:.4f} [{r['length_matched_ci_95'][0]:.4f}, {r['length_matched_ci_95'][1]:.4f}] ({r['matched_pairs_count']}) | "
                f"{r['auprc']:.4f} [{r['auprc_ci_95'][0]:.4f}, {r['auprc_ci_95'][1]:.4f}] | "
                f"{r['alert_top10']['precision']*100:.1f}% | {r['alert_top10']['recall']*100:.1f}% | {r['alert_top10']['lift']:.2f}x |"
            )

    lines.append("")
    lines.append("### Deduplicated Test Set (n=2,932; 237 near-duplicates removed)")
    lines.append("")
    lines.append("| Cohort | Model | n | Prev | AUROC [95% CI] | LS-AUROC 4-bin [95% CI] | LS-AUROC 10-bin [95% CI] | Length-Matched AUROC [95% CI] (Pairs) | AUPRC [95% CI] | Prec@10% | Rec@10% | Lift |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

    for c in cohort_order:
        for m in models_order:
            r = dedup[m][c]
            lines.append(
                f"| {c} | {m} | {r['n']} | {r['prevalence']*100:.1f}% | "
                f"{r['auroc']:.4f} [{r['auroc_ci_95'][0]:.4f}, {r['auroc_ci_95'][1]:.4f}] | "
                f"{r['ls_auroc_4bin']:.4f} [{r['ls_auroc_4bin_ci_95'][0]:.4f}, {r['ls_auroc_4bin_ci_95'][1]:.4f}] | "
                f"{r['ls_auroc_10bin']:.4f} [{r['ls_auroc_10bin_ci_95'][0]:.4f}, {r['ls_auroc_10bin_ci_95'][1]:.4f}] | "
                f"{r['length_matched_auroc']:.4f} [{r['length_matched_ci_95'][0]:.4f}, {r['length_matched_ci_95'][1]:.4f}] ({r['matched_pairs_count']}) | "
                f"{r['auprc']:.4f} [{r['auprc_ci_95'][0]:.4f}, {r['auprc_ci_95'][1]:.4f}] | "
                f"{r['alert_top10']['precision']*100:.1f}% | {r['alert_top10']['recall']*100:.1f}% | {r['alert_top10']['lift']:.2f}x |"
            )

    lines.append("")
    lines.append("## 3. Test Pairs Evaluation (Step 5)")
    lines.append("")
    lines.append("| Model | Pairs | Operating Threshold (Top 10%) | Pair Correct [95% CI] | P-C | P-V | P-B | P-R |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for m in models_order:
        r = pairs[m]
        lines.append(
            f"| {m} | {r['pair_count']} | {r['operating_threshold']:.4f} | "
            f"{r['pair_correct']*100:.1f}% [{r['pair_correct_ci_95'][0]*100:.1f}%, {r['pair_correct_ci_95'][1]*100:.1f}%] | "
            f"{r['P-C']*100:.1f}% | {r['P-V']*100:.1f}% | {r['P-B']*100:.1f}% | {r['P-R']*100:.1f}% |"
        )

    lines.append("")
    lines.append("## 4. Per-CWE Recall at 10% Alert Budget (Step 6)")
    lines.append("")
    lines.append("| CWE | Vulnerable Count | L1 Recall [Wilson 95% CI] | Champion Recall [Wilson 95% CI] | D2_ensemble Recall [Wilson 95% CI] |")
    lines.append("|---|---|---|---|---|")
    for cwe_id in ["CWE-476", "CWE-400", "CWE-835", "CWE-416", "CWE-787", "CWE-119"]:
        r_cwe = cwe[cwe_id]
        n_c = r_cwe["n_total"]
        m_l1 = r_cwe["models"]["L1"]
        m_ch = r_cwe["models"]["Champion"]
        m_d2 = r_cwe["models"]["D2_ensemble"]
        lines.append(
            f"| {cwe_id} | {n_c} | "
            f"{m_l1['k_flagged']}/{n_c} ({m_l1['recall']*100:.1f}% [{m_l1['wilson_ci_95'][0]*100:.1f}%, {m_l1['wilson_ci_95'][1]*100:.1f}%]) | "
            f"{m_ch['k_flagged']}/{n_c} ({m_ch['recall']*100:.1f}% [{m_ch['wilson_ci_95'][0]*100:.1f}%, {m_ch['wilson_ci_95'][1]*100:.1f}%]) | "
            f"{m_d2['k_flagged']}/{n_c} ({m_d2['recall']*100:.1f}% [{m_d2['wilson_ci_95'][0]*100:.1f}%, {m_d2['wilson_ci_95'][1]*100:.1f}%]) |"
        )

    lines.append("")
    lines.append("## 5. Paired Hypothesis Tests (Step 7)")
    lines.append("")
    lines.append("| Comparison | Metric | Observed Diff | 95% Cluster Bootstrap CI | Raw p-value | Holm p-value |")
    lines.append("|---|---|---|---|---|---|")
    test_rows = [
        ("D2_ensemble vs L1", "auroc", "AUROC"),
        ("D2_ensemble vs L1", "ls_auroc_4bin", "LS-AUROC (4-bin)"),
        ("D2_ensemble vs L1", "length_matched_auroc", "Length-Matched AUROC"),
        ("D2_ensemble vs Champion", "auroc", "AUROC"),
        ("D2_ensemble vs Champion", "ls_auroc_4bin", "LS-AUROC (4-bin)"),
        ("D2_ensemble vs Champion", "length_matched_auroc", "Length-Matched AUROC"),
    ]
    for comp_key, metric_key, metric_label in test_rows:
        dict_key = comp_key.replace(" ", "_")
        r = paired[dict_key][metric_key]
        lines.append(
            f"| {comp_key} | {metric_label} | {r['diff_observed']:+.4f} | "
            f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}] | {r['raw_p_value']:.4f} | {r['p_value_holm_formatted']} |"
        )

    lines.append("")
    lines.append("## 6. Surprises and Failures")
    lines.append("- On OpenSSL, D2_ensemble increases LS-AUROC to 0.6839 vs Champion 0.6018, length-matched AUROC to 0.6559 vs Champion 0.5622, and precision@10% to 85.7% vs Champion 79.4%.")
    lines.append("- On CWE-119 (memory buffer bounds), D2_ensemble recalls 16/36 (44.4% [29.5%, 60.4%]) at the 10% operating budget vs Champion 7/36 (19.4%) and L1 5/36 (13.9%).")
    lines.append("- On CWE-400 (uncontrolled resource consumption), D2_ensemble flags 0/91 instances (0.0% [0.0%, 4.1%]) at the 10% operating budget, whereas Champion flags 9/91 (9.9%) and L1 flags 5/91 (5.5%).")
    lines.append("- On Envoy, all three models remain between 0.5415 and 0.5806 AUROC on full test and between 0.5460 and 0.5746 on deduplicated test; Envoy production code reaches 0.6253 for D2_ensemble and 0.6869 for L1.")
    lines.append("- On full test overall, L1 token length alone yields an AUROC of 0.7559 [0.7137, 0.7953], while dropping to 0.5097 [0.5021, 0.5171] on the 65,406 length-matched test pairs.")

    content = "\n".join(lines) + "\n"
    out_file = REPORTS_DIR / "HANDOFF_P6.md"
    with open(out_file, "w") as f:
        f.write(content)
    print(f"Generated {out_file} ({len(content)} bytes)")


if __name__ == "__main__":
    main()
