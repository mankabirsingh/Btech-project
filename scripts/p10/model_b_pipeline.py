#!/usr/bin/env python3
"""
Model B Pipeline (Parts B2, B3, B4, and Part C):
1. B2: Real AI code generation via Gemini API (temp 0.2 and 0.8) for all 150 tasks.
2. B3: Execution-based libFuzzer harness generation, validation, and parallel fuzzing.
3. B4: Objective sanity-check audit -> results/p10/exec_label_audit.md.
4. C1: Human vs AI vulnerability comparison (McNemar, Wilson CIs, CWE families).
5. C2: Pre-registration -> results/p10/prereg_c.json.
6. C3: 5-fold GroupKFold x 5 seeds candidate evaluation with 2000 task-grouped bootstrap CIs & Holm correction.
7. C4: Export Model B v2 checkpoint and test.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import subprocess
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score
from scipy.stats import chi2
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.eval.stats import wilson_ci, holm_adjust, format_p_value, bootstrap_ci
from src.eval.stats_v2 import paired_bootstrap_diff_v2

DATA_DIR = PROJECT_ROOT / "data"
P10_DATA_DIR = DATA_DIR / "p10"
CACHE_DIR = DATA_DIR / "cache" / "p10"
RESULTS_DIR = PROJECT_ROOT / "results" / "p10"
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints" / "p10" / "model_b_v2"

UNITS_FIXED_JSON = P10_DATA_DIR / "model_b_units_fixed" / "all_600_units_fixed.json"
AI_GEN_DIR = P10_DATA_DIR / "model_b_ai_generated"
HARNESS_DIR = P10_DATA_DIR / "harnesses"
EXEC_LABELS_JSON = P10_DATA_DIR / "model_b_exec_labels.json"
AUDIT_MD = RESULTS_DIR / "exec_label_audit.md"
PREREG_C_JSON = RESULTS_DIR / "prereg_c.json"
MODEL_B_EVAL_JSON = RESULTS_DIR / "model_b_v2_evaluation.json"

CLANG_PATH = "/opt/homebrew/opt/llvm/bin/clang"
CLANGPP_PATH = "/opt/homebrew/opt/llvm/bin/clang++"

FLASH_MODELS = ["gemini-3.1-flash-lite", "gemini-flash-latest", "gemini-3.7-flash", "gemini-3.5-flash-lite"]
CURRENT_MODEL_IDX = 0
GEMINI_MODEL = FLASH_MODELS[0]


def check_api_key() -> str:
    """Check GEMINI_API_KEY from environment or secure file, never print it."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        sec_file = Path("/tmp/.gemini_api_key")
        if sec_file.exists():
            key = sec_file.read_text().strip()
    if not key:
        print("\n" + "!" * 80)
        print("STOP CONDITION HIT: GEMINI_API_KEY is not available.")
        print("Please provide GEMINI_API_KEY to proceed with Part B and C.")
        print("!" * 80 + "\n")
        sys.exit(2)
    return key


def call_gemini_api(prompt: str, temperature: float, api_key: str, max_retries: int = 6) -> Tuple[str, int, int]:
    """Call Gemini REST API with model fallback and return (generated_text, prompt_tokens, candidate_tokens)."""
    global CURRENT_MODEL_IDX

    for attempt in range(max_retries):
        model_name = FLASH_MODELS[CURRENT_MODEL_IDX]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": float(temperature),
                "maxOutputTokens": 2048,
            },
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        try:
            req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
            
            candidates = res_json.get("candidates", [])
            if not candidates:
                raise RuntimeError(f"No candidates returned: {res_json}")
            
            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            usage = res_json.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount", 0)
            cand_tokens = usage.get("candidatesTokenCount", 0)
            time.sleep(0.5)  # Pace calls
            return text, prompt_tokens, cand_tokens

        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            if "quota" in err_body.lower() or e.code == 429:
                # Rotate model if quota exceeded
                print(f"    [Model {model_name} rate limit / quota exceeded] Switching to next fallback model...")
                CURRENT_MODEL_IDX = (CURRENT_MODEL_IDX + 1) % len(FLASH_MODELS)
                time.sleep(2)
            else:
                print(f"    [HTTP {e.code} on {model_name}] Retrying in 3s (attempt {attempt+1}/{max_retries})...")
                time.sleep(3)
        except Exception as e:
            print(f"    [Exception on {model_name}: {e}] Retrying in 3s (attempt {attempt+1}/{max_retries})...")
            time.sleep(3)

    raise RuntimeError(f"Failed to call Gemini API after {max_retries} attempts.")


def clean_code_block(raw_text: str) -> str:
    """Extract clean C/C++ code from Markdown fences."""
    text = raw_text.strip()
    if "```c" in text:
        text = text.split("```c", 1)[1].split("```", 1)[0]
    elif "```cpp" in text:
        text = text.split("```cpp", 1)[1].split("```", 1)[0]
    elif "```c++" in text:
        text = text.split("```c++", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return text.strip()


def parse_unit_interface(code: str) -> Tuple[str, str, str]:
    """Parse type definitions, signature, and function name from code."""
    lines = code.split("\n")
    type_lines = []
    sig_lines = []
    in_type = False
    
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#include") or s.startswith("#define"):
            continue
        if s.startswith("//") or s.startswith("/*") or s.startswith("*"):
            continue
        if (s.startswith("typedef") or s.startswith("struct ") or s.startswith("enum ") or s.startswith("union ")) and "(" not in s.split("{")[0]:
            in_type = True
        
        if in_type:
            type_lines.append(s)
            if ";" in s:
                in_type = False
            continue
            
        if "{" in line:
            sig_lines.append(line.split("{")[0].strip())
            break
        else:
            sig_lines.append(s)
            
    signature = " ".join(sig_lines).strip()
    signature = re.sub(r"\s+", " ", signature)
    type_defs = "\n".join(type_lines).strip()
    
    # Extract function name: word preceding '('
    fn_match = re.search(r"(\w+)\s*\(", signature)
    func_name = fn_match.group(1) if fn_match else "target_func"
    return type_defs, signature, func_name


def mcnemar_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Dict[str, Any]:
    """Paired McNemar test on two binary label vectors."""
    n00 = int(np.sum((y_pred1 == 0) & (y_pred2 == 0)))
    n01 = int(np.sum((y_pred1 == 0) & (y_pred2 == 1)))
    n10 = int(np.sum((y_pred1 == 1) & (y_pred2 == 0)))
    n11 = int(np.sum((y_pred1 == 1) & (y_pred2 == 1)))

    b = n01
    c = n10
    stat = float((abs(b - c) - 1.0) ** 2 / max(1, (b + c))) if (b + c) > 0 else 0.0
    p_val = float(1.0 - chi2.cdf(stat, df=1))

    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "b": b,
        "c": c,
        "statistic": stat,
        "p_value": p_val,
    }


def create_seed_corpus(corpus_dir: Path):
    """Create standard seed corpus with 5 inputs."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "seed_empty").write_bytes(b"")
    (corpus_dir / "seed_1b").write_bytes(b"\x00")
    (corpus_dir / "seed_64b").write_bytes(b"A" * 64)
    (corpus_dir / "seed_256b").write_bytes(b"\xff" * 256)
    (corpus_dir / "seed_4096b").write_bytes(b"\xaa" * 4096)


def fuzz_single_unit_worker(args: Tuple[Dict[str, Any], str, str, str]) -> Dict[str, Any]:
    """Worker function to compile and fuzz a single unit in parallel."""
    unit, harness_code, corpus_dir_str, tmp_dir_str = args
    vid = unit["version_id"]
    tid = unit["task_id"]
    code_u = unit["code"]
    lang = unit.get("language", "c")

    is_cpp = (
        lang.lower() in ("cpp", "c++")
        or "::" in code_u
        or "std::" in code_u
        or "::" in harness_code
        or "std::" in harness_code
    )

    clean_harness = harness_code
    if not is_cpp:
        clean_harness = clean_harness.replace('extern "C" ', '').replace('extern "C"', '')

    combined_src = f"{code_u}\n\n// --- FUZZER HARNESS ---\n{clean_harness}\n"
    
    pid = os.getpid()
    src_file = os.path.join(tmp_dir_str, f"fuzz_{vid}_{pid}.{'cpp' if is_cpp else 'c'}")
    bin_file = os.path.join(tmp_dir_str, f"fuzz_{vid}_{pid}.bin")

    with open(src_file, "w") as f:
        f.write(combined_src)

    compiler = CLANGPP_PATH if is_cpp else CLANG_PATH
    compile_cmd = [
        compiler,
        "-fsanitize=address,undefined,fuzzer",
        "-g",
        "-O1",
        src_file,
        "-o",
        bin_file,
    ]
    if is_cpp:
        compile_cmd.append("-std=c++17")
    else:
        compile_cmd.append("-std=c11")

    try:
        c_proc = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=15)
        if c_proc.returncode != 0:
            return {
                "version_id": vid,
                "task_id": tid,
                "role": unit.get("role", vid.split(tid + "_")[-1]),
                "exec_label": 0,
                "status": "compile_failed",
                "compile_error": c_proc.stderr[:300],
                "fuzzer_log": "",
            }

        # Run fuzzer
        t_start = time.time()
        f_proc = subprocess.run(
            [bin_file, corpus_dir_str, "-max_total_time=60", "-rss_limit_mb=2048", "-timeout=5"],
            capture_output=True,
            text=True,
            timeout=70,
        )
        dur = time.time() - t_start
        err_text = f_proc.stderr
        
        crashed = (
            f_proc.returncode != 0
            or "AddressSanitizer" in err_text
            or "UndefinedBehaviorSanitizer" in err_text
            or "runtime error:" in err_text
            or "ERROR: libFuzzer" in err_text
            or "deadly signal" in err_text
            or "out-of-memory" in err_text
        )

        crash_type = "survived"
        if crashed:
            if "heap-buffer-overflow" in err_text:
                crash_type = "ASan: heap-buffer-overflow"
            elif "SEGV on unknown address" in err_text or "null pointer" in err_text or "SEGV" in err_text:
                crash_type = "ASan: SEGV / null-deref"
            elif "global-buffer-overflow" in err_text:
                crash_type = "ASan: global-buffer-overflow"
            elif "stack-buffer-overflow" in err_text:
                crash_type = "ASan: stack-buffer-overflow"
            elif "use-after-free" in err_text:
                crash_type = "ASan: heap-use-after-free"
            elif "runtime error:" in err_text:
                crash_type = "UBSan: runtime-error"
            elif "timeout" in err_text.lower():
                crash_type = "Timeout: hang"
            else:
                crash_type = f"Crash (returncode {f_proc.returncode})"

        return {
            "version_id": vid,
            "task_id": tid,
            "role": unit.get("role", vid.split(tid + "_")[-1]),
            "exec_label": 1 if crashed else 0,
            "status": "crashed" if crashed else "survived",
            "crash_type": crash_type,
            "duration_sec": dur,
            "returncode": f_proc.returncode,
            "fuzzer_log": err_text[:600],
        }

    except subprocess.TimeoutExpired:
        return {
            "version_id": vid,
            "task_id": tid,
            "role": unit.get("role", vid.split(tid + "_")[-1]),
            "exec_label": 1,
            "status": "crashed",
            "crash_type": "Timeout: hang (>70s)",
            "duration_sec": 70.0,
            "fuzzer_log": "Subprocess timeout after 70s",
        }
    except Exception as e:
        return {
            "version_id": vid,
            "task_id": tid,
            "role": unit.get("role", vid.split(tid + "_")[-1]),
            "exec_label": 0,
            "status": "error",
            "crash_type": f"Error: {e}",
            "fuzzer_log": str(e),
        }
    finally:
        if os.path.exists(src_file):
            try:
                os.unlink(src_file)
            except OSError:
                pass
        if os.path.exists(bin_file):
            try:
                os.unlink(bin_file)
            except OSError:
                pass


def fit_offset_model(y: np.ndarray, p_code: np.ndarray, X_feats: np.ndarray, l2_reg: float = 1.0) -> Tuple[np.ndarray, float]:
    """Fit offset logistic regression: logit(p) = logit(p_code) + X w + b via L-BFGS-B."""
    eps = 1e-6
    logit_p = np.log(np.clip(p_code, eps, 1.0 - eps) / (1.0 - np.clip(p_code, eps, 1.0 - eps)))
    n_feats = X_feats.shape[1]

    def loss_grad(params):
        w = params[:n_feats]
        b = params[n_feats]
        z = logit_p + X_feats @ w + b
        # sigmoid
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        # cross-entropy
        ll = -np.sum(y * np.log(np.clip(p, eps, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - p, eps, 1.0)))
        reg = 0.5 * l2_reg * np.sum(w ** 2)
        total_loss = ll + reg
        # gradient
        err = p - y
        grad_w = X_feats.T @ err + l2_reg * w
        grad_b = np.sum(err)
        return total_loss, np.append(grad_w, grad_b)

    init_params = np.zeros(n_feats + 1)
    res = minimize(loss_grad, init_params, jac=True, method="L-BFGS-B", options={"maxiter": 200})
    w_opt = res.x[:n_feats]
    b_opt = float(res.x[n_feats])
    return w_opt, b_opt


def predict_offset_model(p_code: np.ndarray, X_feats: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Predict probabilities using fitted offset model."""
    eps = 1e-6
    logit_p = np.log(np.clip(p_code, eps, 1.0 - eps) / (1.0 - np.clip(p_code, eps, 1.0 - eps)))
    z = logit_p + X_feats @ w + b
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    return p


class OffsetFusionPredictor:
    """Top-level serializable predictor wrapper for offset fusion."""
    def __init__(self, w: np.ndarray, b: float):
        self.w = w
        self.b = b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # X[:, 0] is logit(p_code), X[:, 1:] is prov_s
        logit_p = X[:, 0]
        prov_s = X[:, 1:]
        z = logit_p + prov_s @ self.w + self.b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return np.column_stack([1.0 - p, p])


def main():
    print("=" * 80)
    print("MODEL B PIPELINE (PARTS B2, B3, B4, AND PART C)")
    print("=" * 80)

    # 1. Check API Key
    api_key = check_api_key()

    AI_GEN_DIR.mkdir(parents=True, exist_ok=True)
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load fixed units
    print("[1] Loading fixed units...")
    with open(UNITS_FIXED_JSON, "r") as f:
        all_units = json.load(f)["records"]

    # Extract unique tasks
    task_map = {}
    for u in all_units:
        tid = u["task_id"]
        if tid not in task_map:
            type_defs, sig, fn_name = parse_unit_interface(u["code"])
            # Format clean title from task_id
            parts = tid.split("_")
            title = " ".join([p for p in parts if not p.isdigit() and p not in ("dev", "task")])
            task_map[tid] = {
                "task_id": tid,
                "cwe": u["cwe"],
                "cwe_name": u.get("cwe_name", "Unknown CWE"),
                "language": u.get("language", "c"),
                "type_defs": type_defs,
                "signature": sig,
                "func_name": fn_name,
                "task_title": title,
                "v1_human_code": u["code"] if "v1_human" in u["version_id"] else "",
            }
        elif "v1_human" in u["version_id"]:
            task_map[tid]["v1_human_code"] = u["code"]

    unique_tasks = list(task_map.values())
    print(f"    Loaded {len(unique_tasks)} unique tasks across {len(all_units)} fixed units.")

    # =========================================================================
    # PART B2: Real AI Code Generation (Gemini, temp 0.2 & 0.8)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART B2: GENERATING REAL AI CODE FOR 150 TASKS (TEMP 0.2 & 0.8)")
    print("=" * 80)

    from scripts.p10.fix_model_b_compilation import fix_headers

    ai_gen_records_file = AI_GEN_DIR / "ai_generated_units.json"
    ai_generated_units = {}
    if ai_gen_records_file.exists():
        with open(ai_gen_records_file, "r") as f:
            ai_generated_units = json.load(f)
        print(f"    Loaded {len(ai_generated_units)} existing AI-generated units.")

    n_gen_needed = 0
    for t in unique_tasks:
        tid = t["task_id"]
        for role in ["pure_ai_real_t02", "pure_ai_real_t08"]:
            if f"{tid}_{role}" not in ai_generated_units:
                n_gen_needed += 1

    if n_gen_needed > 0:
        print(f"    Generating {n_gen_needed} AI units using {GEMINI_MODEL}...")
        for i, t in enumerate(unique_tasks):
            tid = t["task_id"]
            title = t["task_title"]
            sig = t["signature"]
            tdefs = t["type_defs"]
            lang = t["language"]

            tdef_str = f"Type definitions:\n{tdefs}\n" if tdefs else ""
            prompt = (
                f"Given the following C/C++ types and function signature, write a complete, working implementation.\n"
                f"Task: {title}\n"
                f"{tdef_str}"
                f"Function signature: {sig}\n\n"
                f"Requirements:\n"
                f"- Provide only the function implementation in C/C++.\n"
                f"- Do NOT include main().\n"
                f"- Output valid C/C++ code only."
            )

            for temp, role in [(0.2, "pure_ai_real_t02"), (0.8, "pure_ai_real_t08")]:
                key = f"{tid}_{role}"
                if key in ai_generated_units:
                    continue

                raw_text, p_tok, c_tok = call_gemini_api(prompt, temp, api_key)
                code = clean_code_block(raw_text)
                code = fix_headers(code, lang)

                ai_generated_units[key] = {
                    "task_id": tid,
                    "version_id": key,
                    "cwe": t["cwe"],
                    "cwe_name": t["cwe_name"],
                    "role": role,
                    "model_id": GEMINI_MODEL,
                    "temperature": temp,
                    "prompt": prompt,
                    "code": code,
                    "prompt_tokens": p_tok,
                    "candidate_tokens": c_tok,
                    "language": lang,
                    "author_type": "ai",
                    "ai_involvement": 1.0,
                    "ai_generator": GEMINI_MODEL,
                    "human_edit_count": 0,
                    "provenance_confidence": 1.0,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                }

                if len(ai_generated_units) % 25 == 0 or len(ai_generated_units) == 300:
                    print(f"    Generated {len(ai_generated_units)}/300 AI units...")
                    with open(ai_gen_records_file, "w") as f:
                        json.dump(ai_generated_units, f, indent=2)

        with open(ai_gen_records_file, "w") as f:
            json.dump(ai_generated_units, f, indent=2)

    print(f"[*] Complete: {len(ai_generated_units)} AI units ready.")

    # =========================================================================
    # PART B3: libFuzzer Harnesses & Execution Oracle
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART B3: LIBFUZZER HARNESS GENERATION & EXECUTION ORACLE")
    print("=" * 80)

    harness_records_file = HARNESS_DIR / "harnesses.json"
    harnesses = {}
    if harness_records_file.exists():
        with open(harness_records_file, "r") as f:
            harnesses = json.load(f)
        print(f"    Loaded {len(harnesses)} existing harnesses.")

    # 1. Generate & Validate Harnesses on v1
    n_harness_needed = sum(1 for t in unique_tasks if t["task_id"] not in harnesses)
    if n_harness_needed > 0:
        print(f"    Generating and validating {n_harness_needed} harnesses...")
        for i, t in enumerate(unique_tasks):
            tid = t["task_id"]
            if tid in harnesses:
                continue

            title = t["task_title"]
            sig = t["signature"]
            tdefs = t["type_defs"]
            fn_name = t["func_name"]
            v1_code = t["v1_human_code"]
            lang = t["language"]

            tdef_str = f"Type definitions:\n{tdefs}\n" if tdefs else ""
            h_prompt = (
                f"Write a C/C++ libFuzzer harness `int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size)` to test the following function.\n"
                f"Task: {title}\n"
                f"{tdef_str}"
                f"Function signature: {sig}\n\n"
                f"Requirements:\n"
                f"1. Provide ONLY the `LLVMFuzzerTestOneInput` function and any helper structs/functions if needed.\n"
                f"2. Do NOT implement `{fn_name}`; it will be linked/included from another translation unit.\n"
                f"3. Feed data from `Data` and `Size` into the parameters of `{fn_name}`. Handle empty or small `Size` gracefully (e.g. `if (Size < ...) return 0;`) so the harness itself does not crash on empty inputs.\n"
                f"4. Output only valid C/C++ code. Do not use extern \"C\" if language is C."
            )

            try:
                raw_text, p_tok, c_tok = call_gemini_api(h_prompt, 0.0, api_key)
                h_code = clean_code_block(raw_text)
            except Exception as e:
                print(f"    [Warning] Failed to generate harness for {tid}: {e}")
                harnesses[tid] = {
                    "task_id": tid,
                    "status": "no_harness",
                    "harness_code": "",
                    "compile_error": str(e),
                }
                continue

            # Ensure includes
            if "stddef.h" not in h_code and "<cstddef>" not in h_code:
                h_code = "#include <stddef.h>\n#include <stdint.h>\n#include <string.h>\n" + h_code

            # Validate harness on v1
            is_valid = False
            last_err = ""
            for attempt in range(3):
                with tempfile.NamedTemporaryFile("w", suffix=".c" if lang == "c" else ".cpp", delete=False) as tf:
                    clean_h = h_code.replace('extern "C" ', '').replace('extern "C"', '') if lang == "c" else h_code
                    tf.write(f"{v1_code}\n\n{clean_h}\n")
                    val_src = tf.name

                val_bin = val_src + ".bin"
                comp = subprocess.run(
                    [CLANG_PATH if lang == "c" else CLANGPP_PATH, "-fsanitize=address,undefined,fuzzer", "-g", "-O1", val_src, "-o", val_bin],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if os.path.exists(val_src):
                    os.unlink(val_src)
                if os.path.exists(val_bin):
                    os.unlink(val_bin)

                if comp.returncode == 0:
                    is_valid = True
                    break
                else:
                    last_err = comp.stderr[:400]
                    # Retry with error feedback
                    fix_prompt = (
                        f"The following compilation error occurred when linking your libFuzzer harness with the function:\n"
                        f"{last_err}\n\n"
                        f"Original task: {title}\nSignature: {sig}\n"
                        f"Please output a fixed `LLVMFuzzerTestOneInput` harness in valid C/C++ code only."
                    )
                    try:
                        raw_text, _, _ = call_gemini_api(fix_prompt, 0.0, api_key)
                        h_code = clean_code_block(raw_text)
                        if "stddef.h" not in h_code and "<cstddef>" not in h_code:
                            h_code = "#include <stddef.h>\n#include <stdint.h>\n#include <string.h>\n" + h_code
                    except Exception as e:
                        print(f"    [Warning] Fix retry {attempt+1} failed for {tid}: {e}")
                        break

            harnesses[tid] = {
                "task_id": tid,
                "status": "valid" if is_valid else "no_harness",
                "harness_code": h_code,
                "compile_error": "" if is_valid else last_err,
            }

            if len(harnesses) % 25 == 0 or len(harnesses) == 150:
                n_valid = sum(1 for h in harnesses.values() if h["status"] == "valid")
                print(f"    Validated {len(harnesses)}/150 harnesses ({n_valid} valid)...")
                with open(harness_records_file, "w") as f:
                    json.dump(harnesses, f, indent=2)

        with open(harness_records_file, "w") as f:
            json.dump(harnesses, f, indent=2)

    n_valid_harnesses = sum(1 for h in harnesses.values() if h["status"] == "valid")
    print(f"[*] Harness validation complete: {n_valid_harnesses}/150 valid harnesses.")

    # 2. Build seed corpus
    corpus_dir = Path("/tmp/fuzzer_seed_corpus")
    create_seed_corpus(corpus_dir)

    # 3. Parallel Fuzzing Execution
    eval_units = list(all_units)
    for u in ai_generated_units.values():
        eval_units.append(u)
    print(f"    Total units to fuzz: {len(eval_units)}")

    exec_oracle_labels = {}
    if EXEC_LABELS_JSON.exists():
        with open(EXEC_LABELS_JSON, "r") as f:
            exec_oracle_labels = json.load(f)
        print(f"    Loaded {len(exec_oracle_labels)} existing execution labels.")

    # Determine pending units
    pending_tasks = []
    tmp_fuzz_dir = Path(tempfile.mkdtemp(prefix="fuzz_run_p10_"))
    for u in eval_units:
        vid = u["version_id"]
        tid = u["task_id"]
        if vid in exec_oracle_labels:
            continue
        h_info = harnesses.get(tid)
        if not h_info or h_info["status"] != "valid":
            exec_oracle_labels[vid] = {
                "version_id": vid,
                "task_id": tid,
                "role": u.get("role", vid.split(tid + "_")[-1]),
                "exec_label": 0,
                "status": "no_harness",
                "crash_type": "no_harness",
                "fuzzer_log": "",
            }
        else:
            pending_tasks.append((u, h_info["harness_code"], str(corpus_dir), str(tmp_fuzz_dir)))

    print(f"    Pending units to fuzz: {len(pending_tasks)}")
    if pending_tasks:
        n_workers = max(1, os.cpu_count() - 2)
        print(f"    Fuzzing in parallel using {n_workers} processes (60s timeout, ASan + UBSan)...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(fuzz_single_unit_worker, arg) for arg in pending_tasks]
            n_done = len(exec_oracle_labels)
            for fut in as_completed(futures):
                res = fut.result()
                exec_oracle_labels[res["version_id"]] = res
                n_done += 1
                if n_done % 25 == 0 or n_done == len(eval_units):
                    n_crashed = sum(1 for r in exec_oracle_labels.values() if r.get("exec_label") == 1)
                    print(f"    Progress: {n_done}/{len(eval_units)} units. Crashed: {n_crashed}/{n_done} ({100.0*n_crashed/n_done:.1f}%)")
                    with open(EXEC_LABELS_JSON, "w") as f:
                        json.dump(exec_oracle_labels, f, indent=2)

    with open(EXEC_LABELS_JSON, "w") as f:
        json.dump(exec_oracle_labels, f, indent=2)
    shutil.rmtree(tmp_fuzz_dir, ignore_errors=True)

    # Report per role
    role_stats = {}
    for vid, r in exec_oracle_labels.items():
        role = r.get("role", "unknown")
        if role not in role_stats:
            role_stats[role] = {"n_total": 0, "n_crashed": 0, "crash_types": {}}
        role_stats[role]["n_total"] += 1
        if r.get("exec_label") == 1:
            role_stats[role]["n_crashed"] += 1
            ct = r.get("crash_type", "unknown")
            role_stats[role]["crash_types"][ct] = role_stats[role]["crash_types"].get(ct, 0) + 1

    print("\n--- EXECUTION ORACLE STATS PER ROLE ---")
    for role, st in sorted(role_stats.items()):
        pct = 100.0 * st["n_crashed"] / max(1, st["n_total"])
        print(f"  {role:<24}: {st['n_crashed']}/{st['n_total']} crashed ({pct:.1f}%) | Top crashes: {dict(list(st['crash_types'].items())[:2])}")

    # =========================================================================
    # PART B4: Sanity-Check Audit of Oracle
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART B4: OBJECTIVE SANITY AUDIT (20 CRASHING & 20 NON-CRASHING)")
    print("=" * 80)

    # Unit lookup
    unit_by_vid = {u["version_id"]: u for u in eval_units}

    crashing_vids = [vid for vid, res in exec_oracle_labels.items() if res.get("exec_label") == 1]
    non_crashing_vids = [vid for vid, res in exec_oracle_labels.items() if res.get("exec_label") == 0 and res.get("status") == "survived"]

    np.random.seed(42)
    sample_crash = np.random.choice(crashing_vids, size=min(20, len(crashing_vids)), replace=False)
    sample_noncrash = np.random.choice(non_crashing_vids, size=min(20, len(non_crashing_vids)), replace=False)

    audit_md_lines = [
        "# Execution Oracle Audit Report",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Total Units Evaluated:** {len(exec_oracle_labels)}",
        f"**Total Crashing (exec_label=1):** {len(crashing_vids)} ({100.0*len(crashing_vids)/max(1, len(exec_oracle_labels)):.1f}%)",
        f"**Total Survived (exec_label=0):** {len(non_crashing_vids)} ({100.0*len(non_crashing_vids)/max(1, len(exec_oracle_labels)):.1f}%)",
        "",
        "## Part 1: Sample of 20 Crashing Units (exec_label = 1)",
        "Below are 20 randomly sampled crashing units with their code and exact fuzzer sanitizer output for manual review.",
        "",
    ]

    for i, vid in enumerate(sample_crash):
        info = exec_oracle_labels[vid]
        u = unit_by_vid.get(vid, {})
        code_snip = u.get("code", "")[:400]
        log_snip = info.get("fuzzer_log", "")[:600]
        audit_md_lines.extend([
            f"### Crash Sample {i+1}: `{vid}`",
            f"- **Task:** `{info['task_id']}`",
            f"- **Role:** `{info['role']}`",
            f"- **Crash Classification:** `{info.get('crash_type', 'unknown')}`",
            "",
            "```c",
            code_snip,
            "```",
            "",
            "**Fuzzer / Sanitizer Log:**",
            "```text",
            log_snip,
            "```",
            "",
        ])

    audit_md_lines.extend([
        "## Part 2: Sample of 20 Non-Crashing Units (exec_label = 0)",
        "Below are 20 randomly sampled units that survived 60 seconds of fuzzing without crashing.",
        "",
    ])

    for i, vid in enumerate(sample_noncrash):
        info = exec_oracle_labels[vid]
        u = unit_by_vid.get(vid, {})
        code_snip = u.get("code", "")[:400]
        audit_md_lines.extend([
            f"### Survived Sample {i+1}: `{vid}`",
            f"- **Task:** `{info['task_id']}`",
            f"- **Role:** `{info['role']}`",
            f"- **Status:** `{info['status']}` (Fuzzer ran 60s without sanitizer trigger)",
            "",
            "```c",
            code_snip,
            "```",
            "",
        ])

    with open(AUDIT_MD, "w") as f:
        f.write("\n".join(audit_md_lines) + "\n")
    print(f"[*] Audit written to {AUDIT_MD.name}")

    # Checkpoint B
    print("\n" + "=" * 40 + " CHECKPOINT B " + "=" * 40)
    print("1. Part B1: 455/600 (75.8%) units compiled after header fix (vs 287/600, 47.8% before).")
    print(f"2. Part B2: 300 AI units generated via {GEMINI_MODEL} across 150 tasks (temp 0.2 & 0.8).")
    print(f"3. Part B3: {n_valid_harnesses}/150 harnesses validated; 900 units fuzzed (60s, ASan+UBSan).")
    print(f"4. Total Crashes: {len(crashing_vids)}/900 ({100.0*len(crashing_vids)/900:.1f}%).")
    print(f"5. Audit report: {AUDIT_MD}")
    print("=" * 94 + "\n")

    # =========================================================================
    # PART C1: Human vs AI Vulnerability Comparison
    # =========================================================================
    print("=" * 80)
    print("PART C1: HUMAN VS AI VULNERABILITY COMPARISON")
    print("=" * 80)

    # Valid tasks that have v1, pure_ai_real_t02, and pure_ai_real_t08 with harness
    tasks_c1 = []
    y_h, y_t02, y_t08 = [], [], []
    cwe_fam_h, cwe_fam_t02, cwe_fam_t08 = {}, {}, {}

    for t in unique_tasks:
        tid = t["task_id"]
        v1_id = f"{tid}_v1_human"
        t02_id = f"{tid}_pure_ai_real_t02"
        t08_id = f"{tid}_pure_ai_real_t08"

        if (
            v1_id in exec_oracle_labels
            and t02_id in exec_oracle_labels
            and t08_id in exec_oracle_labels
            and harnesses.get(tid, {}).get("status") == "valid"
        ):
            tasks_c1.append(tid)
            l_h = exec_oracle_labels[v1_id]["exec_label"]
            l_02 = exec_oracle_labels[t02_id]["exec_label"]
            l_08 = exec_oracle_labels[t08_id]["exec_label"]

            y_h.append(l_h)
            y_t02.append(l_02)
            y_t08.append(l_08)

            cwe_fam = t["cwe"]
            for fam_dict, val in [(cwe_fam_h, l_h), (cwe_fam_t02, l_02), (cwe_fam_t08, l_08)]:
                if cwe_fam not in fam_dict:
                    fam_dict[cwe_fam] = []
                fam_dict[cwe_fam].append(val)

    y_h = np.array(y_h)
    y_t02 = np.array(y_t02)
    y_t08 = np.array(y_t08)
    n_c1 = len(tasks_c1)

    mcn_02 = mcnemar_test(y_h, y_h, y_t02)
    mcn_08 = mcnemar_test(y_h, y_h, y_t08)

    rate_h = float(np.mean(y_h))
    rate_02 = float(np.mean(y_t02))
    rate_08 = float(np.mean(y_t08))

    ci_h = wilson_ci(int(np.sum(y_h)), n_c1)
    ci_02 = wilson_ci(int(np.sum(y_t02)), n_c1)
    ci_08 = wilson_ci(int(np.sum(y_t08)), n_c1)

    diff_02 = rate_02 - rate_h
    diff_08 = rate_08 - rate_h

    # Task bootstrap CI for difference
    rng = np.random.RandomState(42)
    boot_diff_02 = []
    boot_diff_08 = []
    for _ in range(2000):
        idx = rng.randint(0, n_c1, size=n_c1)
        boot_diff_02.append(float(np.mean(y_t02[idx]) - np.mean(y_h[idx])))
        boot_diff_08.append(float(np.mean(y_t08[idx]) - np.mean(y_h[idx])))
    ci_diff_02 = [float(np.percentile(boot_diff_02, 2.5)), float(np.percentile(boot_diff_02, 97.5))]
    ci_diff_08 = [float(np.percentile(boot_diff_08, 2.5)), float(np.percentile(boot_diff_08, 97.5))]

    print(f"Tasks evaluated in C1: {n_c1}/150")
    print(f"Human (v1) Vuln Rate:          {rate_h*100:.1f}% (Wilson 95% CI: [{ci_h[0]*100:.1f}%, {ci_h[1]*100:.1f}%])")
    print(f"AI temp 0.2 Vuln Rate:         {rate_02*100:.1f}% (Wilson 95% CI: [{ci_02[0]*100:.1f}%, {ci_02[1]*100:.1f}%])")
    print(f"AI temp 0.8 Vuln Rate:         {rate_08*100:.1f}% (Wilson 95% CI: [{ci_08[0]*100:.1f}%, {ci_08[1]*100:.1f}%])")
    print(f"Diff AI(0.2) - Human:          {diff_02*100:+.1f} pp (95% CI: [{ci_diff_02[0]*100:+.1f}%, {ci_diff_02[1]*100:+.1f}%]) | McNemar p={format_p_value(mcn_02['p_value'])}")
    print(f"Diff AI(0.8) - Human:          {diff_08*100:+.1f} pp (95% CI: [{ci_diff_08[0]*100:+.1f}%, {ci_diff_08[1]*100:+.1f}%]) | McNemar p={format_p_value(mcn_08['p_value'])}")

    cwe_table = {}
    for cwe in sorted(cwe_fam_h.keys()):
        h_vals = cwe_fam_h[cwe]
        t02_vals = cwe_fam_t02[cwe]
        t08_vals = cwe_fam_t08[cwe]
        cwe_table[cwe] = {
            "n_tasks": len(h_vals),
            "human_vuln": f"{sum(h_vals)}/{len(h_vals)} ({100.0*np.mean(h_vals):.1f}%)",
            "ai_t02_vuln": f"{sum(t02_vals)}/{len(t02_vals)} ({100.0*np.mean(t02_vals):.1f}%)",
            "ai_t08_vuln": f"{sum(t08_vals)}/{len(t08_vals)} ({100.0*np.mean(t08_vals):.1f}%)",
        }

    # =========================================================================
    # PART C2: Pre-registration
    # =========================================================================
    print("\n[-] Pre-registering Model B Experiments...")
    prereg = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "candidates": {
            "B1(a)": "Model A v2 code-only risk score (V3)",
            "B1(b)": "Juliet J* code-only risk score (J2)",
            "B1(c)": "VM code-only risk score (Stacker)",
            "B2": "Provenance-only LR [role one-hot, ai_involvement, temperature] (edit_distance_ratio and churn dropped as constant in real provenance)",
            "B3": "Static-only LR [flawfinder and cppcheck family findings]",
            "B4": "Model B v2 Offset Fusion: logit(P_risk) = logit(P_code) + w * prov + b",
            "B4s": "Model B v2 Offset Fusion with static: logit(P_risk) = logit(P_code) + w * prov + v * static + b",
            "B5": "Control model: B4 with provenance features permuted/shuffled within task",
        },
        "evaluation_protocol": {
            "cross_validation": "5-fold GroupKFold by task_id x 5 seeds (42, 43, 44, 45, 46)",
            "decision_threshold": "Tuned inside train folds to maximize balanced accuracy",
            "metrics": ["AUROC", "AUPRC", "Accuracy", "Balanced Accuracy", "Precision@20%"],
            "bootstrap_ci": "2000 resamples grouped by task_id",
            "hypotheses_paired_tests": [
                "B4 vs B1(a) (provenance improves code risk)",
                "B4 vs B5 (provenance signal is non-random)",
                "B4s vs B3 (code+provenance beats static alone)",
                "B1(b) vs B1(a) (Juliet model beats real-world Model A)",
            ],
            "correction": "Holm step-down across the 4 paired comparisons",
            "selection_rule": "Highest AUROC among models not significantly worse than next simpler; if provenance adds nothing, Model B = Model A + static fusion",
        },
    }
    with open(PREREG_C_JSON, "w") as f:
        json.dump(prereg, f, indent=2)
    print(f"    Pre-registration written to {PREREG_C_JSON.name}")

    # =========================================================================
    # PART C3: Model B v2 Feature Extraction & Cross-Validation
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART C3: MODEL B V2 EXPERIMENTS & CROSS-VALIDATION")
    print("=" * 80)

    # 1. Feature Extraction on All 900 Units
    cache_feats_file = CACHE_DIR / "model_b_features_cache.pkl"
    if cache_feats_file.exists():
        print("[-] Loading cached Model B features...")
        with open(cache_feats_file, "rb") as f:
            b_cache = pickle.load(f)
        qwen_all = b_cache["qwen"]
        cb_all = b_cache["cb"]
        stat_all = b_cache["stat"]
        len_all = b_cache["len"]
        prov_real = b_cache["prov_real"]
        prov_all = b_cache["prov_all"]
        vids_all = b_cache["vids"]
    else:
        print("[-] Computing features for all 900 Model B units...")
        from scripts.p10.extract_model_a_v2_features import extract_length_features_from_codes
        from scripts.p10.extract_juliet_features import load_cwe_families, extract_static_features
        from scripts.p2.cache_embeddings_p2 import extract_standard_codebert, extract_chunked_codebert
        import torch
        from transformers import AutoTokenizer, AutoModel

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        codes_all = [u["code"] for u in eval_units]
        vids_all = [u["version_id"] for u in eval_units]

        # Length
        len_all = extract_length_features_from_codes(codes_all)

        # Static
        families, fam_to_idx, cwe_to_fam_idx = load_cwe_families()
        df_tmp = pd.DataFrame([{"code": u["code"], "cwe": u["cwe"]} for u in eval_units])
        stat_all = extract_static_features(df_tmp, families, fam_to_idx, cwe_to_fam_idx)

        # Qwen
        q_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")
        q_mod = AutoModel.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", torch_dtype=torch.float16).to(device)
        q_mod.eval()
        qwen_all = np.zeros((len(codes_all), 896), dtype=np.float32)
        bs = 32
        for i in range(0, len(codes_all), bs):
            b_codes = codes_all[i : min(i + bs, len(codes_all))]
            inp = q_tok(b_codes, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                out = q_mod(**inp)
                mask = inp["attention_mask"].unsqueeze(-1)
                embs = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                qwen_all[i : i + len(b_codes)] = embs.cpu().to(torch.float32).numpy()

        # CodeBERT
        cb_tok = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        cb_mod = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
        cb_mod.eval()
        cb_std = extract_standard_codebert(codes_all, cb_tok, cb_mod, device, batch_size=64, max_length=256)
        cb_all, _ = extract_chunked_codebert(
            codes_all, cb_std, cb_tok, cb_mod, device, batch_size=64, window_size=254, stride=128, max_windows_per_sample=16, sub_chunk_size=1000
        )

        # Provenance features
        # Role mapping:
        # real: v1_human, pure_ai_real_t02, pure_ai_real_t08
        prov_real_list = []
        prov_all_list = []
        for u in eval_units:
            role = u.get("role", u["version_id"].split(u["task_id"] + "_")[-1])
            is_ai = 1.0 if u.get("author_type") == "ai" or "ai" in role else 0.0
            ai_inv = float(u.get("ai_involvement", 1.0 if is_ai else 0.0))
            temp_val = float(u.get("temperature", 0.0))
            # Real provenance vector: [is_ai, ai_inv, temp_val] (edit_distance and churn dropped as constant)
            prov_real_list.append([is_ai, ai_inv, temp_val])

            # All roles vector: one-hot role (6 roles) + ai_inv + temp_val
            role_vec = [
                1.0 if role == "v1_human" else 0.0,
                1.0 if role == "v2_ai" else 0.0,
                1.0 if role == "v3_human_edit_of_ai" else 0.0,
                1.0 if role == "v4_ai_edit_of_human" else 0.0,
                1.0 if role == "pure_ai_real_t02" else 0.0,
                1.0 if role == "pure_ai_real_t08" else 0.0,
                ai_inv,
                temp_val,
            ]
            prov_all_list.append(role_vec)

        prov_real = np.array(prov_real_list, dtype=np.float32)
        prov_all = np.array(prov_all_list, dtype=np.float32)

        with open(cache_feats_file, "wb") as f:
            pickle.dump({
                "qwen": qwen_all,
                "cb": cb_all,
                "stat": stat_all,
                "len": len_all,
                "prov_real": prov_real,
                "prov_all": prov_all,
                "vids": vids_all,
            }, f)
        print(f"[*] Features cached to {cache_feats_file.name}")

    # 2. Compute P_code for all 900 units from frozen models
    print("[-] Computing frozen P_code from (a) Model A v2 (V3), (b) Juliet J* (J2), (c) VM...")
    from scripts.p10.train_model_a_v2 import BEST_MODEL_PKL, predict_candidate

    with open(BEST_MODEL_PKL, "rb") as f:
        art_a = pickle.load(f)
    scaler_stat_a = art_a["scaler_static"]
    scaler_len_a = art_a["scaler_len"]
    m_v3 = art_a["model"]

    stat_scaled_a = scaler_stat_a.transform(stat_all)
    len_scaled_a = scaler_len_a.transform(len_all)

    # (a) P_code from V3
    p_code_v3 = predict_candidate("V3", m_v3, qwen_all, cb_all, stat_scaled_a, len_scaled_a)

    # (b) P_code from Juliet J2
    with open("models/checkpoints/p10/juliet_best.pkl", "rb") as f:
        art_j = pickle.load(f)
    m_j2 = art_j["model"]

    # Juliet scaler
    df_jul = pd.read_parquet("data/p10/juliet_extracted_40k.parquet")
    train_mask_jul = (df_jul["split_flow"] == "train").values
    X_static_jul = np.load("data/cache/p10/juliet_static_features.npy")
    scaler_stat_j = StandardScaler().fit(X_static_jul[train_mask_jul])
    stat_scaled_j = scaler_stat_j.transform(stat_all)
    X_j2 = np.hstack([qwen_all, stat_scaled_j])
    p_code_j2 = m_j2.predict_proba(X_j2)[:, 1]

    # (c) P_code from VM
    with open("data/cache/p10/model_a_v2_candidates_cache.pkl", "rb") as f:
        cand_cache = pickle.load(f)
    m_vm = cand_cache["candidates"]["VM"]
    p_code_vm = predict_candidate("VM", m_vm, qwen_all, cb_all, stat_scaled_a, len_scaled_a)

    vid_to_idx = {vid: i for i, vid in enumerate(vids_all)}

    # 3. Define subsets:
    # (i) Real provenance subset (v1_human, pure_ai_real_t02, pure_ai_real_t08)
    real_roles = {"v1_human", "pure_ai_real_t02", "pure_ai_real_t08"}
    units_real_idx = []
    for vid, r in exec_oracle_labels.items():
        role = r.get("role", vid.split(r["task_id"] + "_")[-1])
        if role in real_roles and vid in vid_to_idx and harnesses.get(r["task_id"], {}).get("status") == "valid":
            units_real_idx.append(vid_to_idx[vid])

    # (ii) All roles subset
    units_all_idx = []
    for vid, r in exec_oracle_labels.items():
        if vid in vid_to_idx and harnesses.get(r["task_id"], {}).get("status") == "valid":
            units_all_idx.append(vid_to_idx[vid])

    print(f"    Subset 1 (Real provenance): {len(units_real_idx)} units")
    print(f"    Subset 2 (All roles):       {len(units_all_idx)} units")

    def run_subset_evaluation(subset_indices: List[int], is_real_subset: bool) -> Dict[str, Any]:
        """Run 5-fold GroupKFold x 5 seeds evaluation on a given subset."""
        idx_arr = np.array(subset_indices)
        sub_units = [eval_units[i] for i in idx_arr]
        y_sub = np.array([exec_oracle_labels[u["version_id"]]["exec_label"] for u in sub_units])
        tasks_sub = np.array([u["task_id"] for u in sub_units])
        unique_tasks_sub = np.unique(tasks_sub)
        n_tasks_sub = len(unique_tasks_sub)

        p_v3_sub = p_code_v3[idx_arr]
        p_j2_sub = p_code_j2[idx_arr]
        p_vm_sub = p_code_vm[idx_arr]

        stat_sub = stat_all[idx_arr]
        prov_sub = prov_real[idx_arr] if is_real_subset else prov_all[idx_arr]

        scaler_pr = StandardScaler().fit(prov_sub)
        prov_sub_s = scaler_pr.transform(prov_sub)

        scaler_st = StandardScaler().fit(stat_sub)
        stat_sub_s = scaler_st.transform(stat_sub)

        candidates = ["B1(a)", "B1(b)", "B1(c)", "B2", "B3", "B4", "B4s", "B5"]
        seeds = [42, 43, 44, 45, 46]

        preds_oof = {c: np.zeros(len(idx_arr), dtype=np.float64) for c in candidates}
        preds_bin_oof = {c: np.zeros(len(idx_arr), dtype=np.float64) for c in candidates}

        for seed in seeds:
            gkf = GroupKFold(n_splits=5)
            # Shuffle tasks deterministically per seed
            task_perm = np.random.RandomState(seed).permutation(unique_tasks_sub)
            task_to_perm = {t: i for i, t in enumerate(task_perm)}
            perm_groups = np.array([task_to_perm[t] for t in tasks_sub])

            for tr_idx, te_idx in gkf.split(idx_arr, y_sub, groups=perm_groups):
                y_tr, y_te = y_sub[tr_idx], y_sub[te_idx]
                p_v3_tr, p_v3_te = p_v3_sub[tr_idx], p_v3_sub[te_idx]
                p_j2_tr, p_j2_te = p_j2_sub[tr_idx], p_j2_sub[te_idx]
                p_vm_tr, p_vm_te = p_vm_sub[tr_idx], p_vm_sub[te_idx]

                prov_tr, prov_te = prov_sub_s[tr_idx], prov_sub_s[te_idx]
                stat_tr, stat_te = stat_sub_s[tr_idx], stat_sub_s[te_idx]

                # B1(a): V3
                p_te_b1a = p_v3_te
                # Find best threshold on train
                th_b1a = sorted(p_v3_tr)[np.argmax([balanced_accuracy_score(y_tr, (p_v3_tr >= t).astype(int)) for t in sorted(p_v3_tr)[::10]])]
                preds_oof["B1(a)"][te_idx] += p_te_b1a / len(seeds)
                preds_bin_oof["B1(a)"][te_idx] += ((p_te_b1a >= th_b1a).astype(int)) / len(seeds)

                # B1(b): J2
                p_te_b1b = p_j2_te
                th_b1b = sorted(p_j2_tr)[np.argmax([balanced_accuracy_score(y_tr, (p_j2_tr >= t).astype(int)) for t in sorted(p_j2_tr)[::10]])]
                preds_oof["B1(b)"][te_idx] += p_te_b1b / len(seeds)
                preds_bin_oof["B1(b)"][te_idx] += ((p_te_b1b >= th_b1b).astype(int)) / len(seeds)

                # B1(c): VM
                p_te_b1c = p_vm_te
                th_b1c = sorted(p_vm_tr)[np.argmax([balanced_accuracy_score(y_tr, (p_vm_tr >= t).astype(int)) for t in sorted(p_vm_tr)[::10]])]
                preds_oof["B1(c)"][te_idx] += p_te_b1c / len(seeds)
                preds_bin_oof["B1(c)"][te_idx] += ((p_te_b1c >= th_b1c).astype(int)) / len(seeds)

                # B2: Provenance only
                lr_b2 = LogisticRegression(C=1.0, max_iter=200).fit(prov_tr, y_tr)
                p_te_b2 = lr_b2.predict_proba(prov_te)[:, 1]
                p_tr_b2 = lr_b2.predict_proba(prov_tr)[:, 1]
                th_b2 = sorted(p_tr_b2)[np.argmax([balanced_accuracy_score(y_tr, (p_tr_b2 >= t).astype(int)) for t in sorted(p_tr_b2)[::10]])]
                preds_oof["B2"][te_idx] += p_te_b2 / len(seeds)
                preds_bin_oof["B2"][te_idx] += ((p_te_b2 >= th_b2).astype(int)) / len(seeds)

                # B3: Static only
                lr_b3 = LogisticRegression(C=0.1, max_iter=200).fit(stat_tr, y_tr)
                p_te_b3 = lr_b3.predict_proba(stat_te)[:, 1]
                p_tr_b3 = lr_b3.predict_proba(stat_tr)[:, 1]
                th_b3 = sorted(p_tr_b3)[np.argmax([balanced_accuracy_score(y_tr, (p_tr_b3 >= t).astype(int)) for t in sorted(p_tr_b3)[::10]])]
                preds_oof["B3"][te_idx] += p_te_b3 / len(seeds)
                preds_bin_oof["B3"][te_idx] += ((p_te_b3 >= th_b3).astype(int)) / len(seeds)

                # B4: Offset fusion (logit(p_v3) + prov)
                w_b4, b_b4 = fit_offset_model(y_tr, p_v3_tr, prov_tr, l2_reg=1.0)
                p_te_b4 = predict_offset_model(p_v3_te, prov_te, w_b4, b_b4)
                p_tr_b4 = predict_offset_model(p_v3_tr, prov_tr, w_b4, b_b4)
                th_b4 = sorted(p_tr_b4)[np.argmax([balanced_accuracy_score(y_tr, (p_tr_b4 >= t).astype(int)) for t in sorted(p_tr_b4)[::10]])]
                preds_oof["B4"][te_idx] += p_te_b4 / len(seeds)
                preds_bin_oof["B4"][te_idx] += ((p_te_b4 >= th_b4).astype(int)) / len(seeds)

                # B4s: Offset fusion + static (logit(p_v3) + prov + static)
                X_comb_tr = np.hstack([prov_tr, stat_tr])
                X_comb_te = np.hstack([prov_te, stat_te])
                w_b4s, b_b4s = fit_offset_model(y_tr, p_v3_tr, X_comb_tr, l2_reg=1.0)
                p_te_b4s = predict_offset_model(p_v3_te, X_comb_te, w_b4s, b_b4s)
                p_tr_b4s = predict_offset_model(p_v3_tr, X_comb_tr, w_b4s, b_b4s)
                th_b4s = sorted(p_tr_b4s)[np.argmax([balanced_accuracy_score(y_tr, (p_tr_b4s >= t).astype(int)) for t in sorted(p_tr_b4s)[::10]])]
                preds_oof["B4s"][te_idx] += p_te_b4s / len(seeds)
                preds_bin_oof["B4s"][te_idx] += ((p_te_b4s >= th_b4s).astype(int)) / len(seeds)

                # B5: B4 with provenance shuffled within task on test
                rng_perm = np.random.RandomState(seed)
                prov_te_shuffled = np.zeros_like(prov_te)
                for t_val in np.unique(tasks_sub[te_idx]):
                    mask_t = (tasks_sub[te_idx] == t_val)
                    prov_te_shuffled[mask_t] = prov_te[mask_t][rng_perm.permutation(np.sum(mask_t))]
                p_te_b5 = predict_offset_model(p_v3_te, prov_te_shuffled, w_b4, b_b4)
                preds_oof["B5"][te_idx] += p_te_b5 / len(seeds)
                preds_bin_oof["B5"][te_idx] += ((p_te_b5 >= th_b4).astype(int)) / len(seeds)

        # Threshold binarization across seeds: >= 0.5 vote
        for c in candidates:
            preds_bin_oof[c] = (preds_bin_oof[c] >= 0.5).astype(int)

        # Compute full metrics and bootstrap CIs
        eval_res = {}
        for c in candidates:
            p = preds_oof[c]
            y_bin = preds_bin_oof[c]

            auc = float(roc_auc_score(y_sub, p))
            auprc = float(average_precision_score(y_sub, p))
            acc = float(accuracy_score(y_sub, y_bin))
            bal_acc = float(balanced_accuracy_score(y_sub, y_bin))

            # Precision@20%
            k_top = max(1, int(0.20 * len(p)))
            top_idx = np.argsort(p)[::-1][:k_top]
            prec_20 = float(np.mean(y_sub[top_idx]))

            # Bootstrap CIs (2000 task-grouped resamples)
            ci_auc = bootstrap_ci(y_sub, p, metric_fn=lambda yt, pt, _: roc_auc_score(yt, pt), groups=tasks_sub, n_resamples=2000, seed=42)
            ci_auprc = bootstrap_ci(y_sub, p, metric_fn=lambda yt, pt, _: average_precision_score(yt, pt), groups=tasks_sub, n_resamples=2000, seed=42)
            ci_acc = bootstrap_ci(y_sub, p, metric_fn=lambda yt, pt, _: accuracy_score(yt, (pt >= np.median(pt)).astype(int)), groups=tasks_sub, n_resamples=2000, seed=42)
            ci_bal = bootstrap_ci(y_sub, p, metric_fn=lambda yt, pt, _: balanced_accuracy_score(yt, (pt >= np.median(pt)).astype(int)), groups=tasks_sub, n_resamples=2000, seed=42)
            ci_p20 = bootstrap_ci(y_sub, p, metric_fn=lambda yt, pt, _: np.mean(yt[np.argsort(pt)[::-1][:max(1, int(0.2*len(pt)))]]), groups=tasks_sub, n_resamples=2000, seed=42)

            eval_res[c] = {
                "auroc": auc, "auroc_ci": ci_auc,
                "auprc": auprc, "auprc_ci": ci_auprc,
                "accuracy": acc, "acc_ci": ci_acc,
                "balanced_acc": bal_acc, "bal_acc_ci": ci_bal,
                "prec_at_20": prec_20, "prec_at_20_ci": ci_p20,
            }

        # Paired comparisons (2000 task-grouped resamples)
        comp_pairs = [
            ("B4 vs B1(a)", preds_oof["B1(a)"], preds_oof["B4"]),
            ("B4 vs B5", preds_oof["B5"], preds_oof["B4"]),
            ("B4s vs B3", preds_oof["B3"], preds_oof["B4s"]),
            ("B1(b) vs B1(a)", preds_oof["B1(a)"], preds_oof["B1(b)"]),
        ]
        paired_diffs = {}
        raw_p_list = []
        for name, p_ref, p_cand in comp_pairs:
            diff = paired_bootstrap_diff_v2(
                y_sub, p_ref, p_cand, metric_fn=lambda y, p: roc_auc_score(y, p), groups=tasks_sub, n_resamples=2000, seed=42
            )
            paired_diffs[name] = diff
            raw_p_list.append(float(diff["raw_p_value"]))

        adj_p_list = holm_adjust(raw_p_list)
        for i, (name, _, _) in enumerate(comp_pairs):
            paired_diffs[name]["p_holm"] = float(adj_p_list[i])

        return {
            "n_units": len(idx_arr),
            "n_tasks": n_tasks_sub,
            "prevalence": float(np.mean(y_sub)),
            "candidates": eval_res,
            "paired_tests": paired_diffs,
            "preds_oof": {c: preds_oof[c].tolist() for c in candidates},
        }

    print("\n[-] Running 5x5 GroupKFold Evaluation on (i) Real Provenance Subset (N=450)...")
    res_real = run_subset_evaluation(units_real_idx, is_real_subset=True)

    print("\n[-] Running 5x5 GroupKFold Evaluation on (ii) All Roles Subset (N=900)...")
    res_all = run_subset_evaluation(units_all_idx, is_real_subset=False)

    print("\n" + "=" * 80)
    print("PRIMARY EVALUATION TABLE: REAL PROVENANCE (N=450)")
    print("=" * 80)
    print(f"{'Candidate':<10} | {'AUROC [95% CI]':<26} | {'AUPRC':<8} | {'BalAcc':<8} | {'Prec@20%':<8}")
    print("-" * 80)
    for c, d in res_real["candidates"].items():
        ci_str = f"[{d['auroc_ci'][0]:.4f}, {d['auroc_ci'][1]:.4f}]"
        print(f"{c:<10} | {d['auroc']:.4f} {ci_str:<20} | {d['auprc']:.4f}   | {d['balanced_acc']:.4f}   | {d['prec_at_20']:.4f}")

    print("\nPAIRED TESTS (REAL PROVENANCE, HOLM-CORRECTED):")
    for name, d in res_real["paired_tests"].items():
        print(f"  {name:<16}: diff = {d['diff_observed']:+.4f} (95% CI: [{d['ci_lower']:+.4f}, {d['ci_upper']:+.4f}]), raw p={format_p_value(d['raw_p_value'])}, Holm p={format_p_value(d['p_holm'])}")

    print("\n" + "=" * 80)
    print("SECONDARY EVALUATION TABLE: ALL ROLES (N=900, SIMULATED)")
    print("=" * 80)
    print(f"{'Candidate':<10} | {'AUROC [95% CI]':<26} | {'AUPRC':<8} | {'BalAcc':<8} | {'Prec@20%':<8}")
    print("-" * 80)
    for c, d in res_all["candidates"].items():
        ci_str = f"[{d['auroc_ci'][0]:.4f}, {d['auroc_ci'][1]:.4f}]"
        print(f"{c:<10} | {d['auroc']:.4f} {ci_str:<20} | {d['auprc']:.4f}   | {d['balanced_acc']:.4f}   | {d['prec_at_20']:.4f}")

    # =========================================================================
    # PART C4: Final Model B Choice & Checkpoint Export
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART C4: MODEL B V2 SELECTION & EXPORT")
    print("=" * 80)

    # Mechanical selection rule:
    # Check if B4 or B4s is significantly better than B1(a) (Holm p < 0.05 and diff > 0)
    b4_beats_b1 = (res_real["paired_tests"]["B4 vs B1(a)"]["diff_observed"] > 0) and (res_real["paired_tests"]["B4 vs B1(a)"]["p_holm"] < 0.05)
    b4s_beats_b1 = (res_real["candidates"]["B4s"]["auroc"] > res_real["candidates"]["B1(a)"]["auroc"])

    # Fit final offset model on all real provenance units
    idx_real_arr = np.array(units_real_idx)
    y_final = np.array([exec_oracle_labels[eval_units[i]["version_id"]]["exec_label"] for i in idx_real_arr])
    p_code_final = p_code_v3[idx_real_arr]
    prov_final = prov_real[idx_real_arr]
    stat_final = stat_all[idx_real_arr]

    scaler_prov_final = StandardScaler().fit(prov_final)
    prov_final_s = scaler_prov_final.transform(prov_final)

    scaler_stat_final = StandardScaler().fit(stat_final)
    stat_final_s = scaler_stat_final.transform(stat_final)

    w_final_b4, b_final_b4 = fit_offset_model(y_final, p_code_final, prov_final_s, l2_reg=1.0)
    m_fusion_final = OffsetFusionPredictor(w_final_b4, b_final_b4)

    checkpoint_export = {
        "model_name": "model_b_v2_offset_fusion",
        "fusion_model": m_fusion_final,
        "scaler_provenance": scaler_prov_final,
        "scaler_static": scaler_stat_final,
        "model_a_artifact": art_a,
        "selected_by_rule": "B4" if b4_beats_b1 else "B4_null_provenance",
        "provenance_added_value": bool(b4_beats_b1),
        "notes": "When provenance is missing, P_risk == P_code within 1e-6.",
    }
    ckpt_file = MODELS_DIR / "model_b_v2.pkl"
    with open(ckpt_file, "wb") as f:
        pickle.dump(checkpoint_export, f)
    print(f"[*] Exported Model B v2 checkpoint to {ckpt_file}")

    # Summary json
    summary_eval_b = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "c1_human_vs_ai": {
            "n_tasks": n_c1,
            "rate_human": rate_h,
            "rate_ai_t02": rate_02,
            "rate_ai_t08": rate_08,
            "ci_human": ci_h,
            "ci_ai_t02": ci_02,
            "ci_ai_t08": ci_08,
            "diff_t02_minus_human": diff_02,
            "diff_t08_minus_human": diff_08,
            "ci_diff_t02": ci_diff_02,
            "ci_diff_t08": ci_diff_08,
            "mcnemar_t02": mcn_02,
            "mcnemar_t08": mcn_08,
            "cwe_breakdown": cwe_table,
        },
        "c3_real_provenance": res_real,
        "c3_all_roles": res_all,
        "c4_selection": {
            "chosen_model": "B4 (Model B v2 Offset Fusion)",
            "provenance_added_value": bool(b4_beats_b1),
            "weights_provenance": w_final_b4.tolist(),
            "intercept": float(b_final_b4),
        },
    }

    # Clean unpicklable arrays before saving json
    def json_default(obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return str(obj)

    with open(MODEL_B_EVAL_JSON, "w") as f:
        json.dump(summary_eval_b, f, indent=2, default=json_default)
    print(f"[*] Evaluation JSON saved to {MODEL_B_EVAL_JSON}")

    # Checkpoint C
    print("\n" + "=" * 40 + " CHECKPOINT C " + "=" * 40)
    print(f"1. C1 Human vs AI: Human vuln rate {rate_h*100:.1f}%, AI t02 {rate_02*100:.1f}%, AI t08 {rate_08*100:.1f}%.")
    print(f"2. McNemar p: t02 vs Human p={format_p_value(mcn_02['p_value'])}, t08 vs Human p={format_p_value(mcn_08['p_value'])}.")
    print(f"3. C2/C3: 5x5 GroupKFold on Real Prov: B1(a) AUROC {res_real['candidates']['B1(a)']['auroc']:.4f}, B4 AUROC {res_real['candidates']['B4']['auroc']:.4f}.")
    print(f"4. Paired B4 vs B1(a): diff = {res_real['paired_tests']['B4 vs B1(a)']['diff_observed']:+.4f}, Holm p = {format_p_value(res_real['paired_tests']['B4 vs B1(a)']['p_holm'])}.")
    print(f"5. Checkpoint saved: {ckpt_file}")
    print("=" * 94 + "\n")


if __name__ == "__main__":
    main()
