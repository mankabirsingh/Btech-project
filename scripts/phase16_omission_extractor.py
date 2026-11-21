#!/usr/bin/env python3
"""
Phase 16 — Omission-Aware Modern C/C++ Structural Feature Extractor
===================================================================
Expands the Phase 15 28-dimensional structural representation to 38 dimensions
by explicitly capturing safeguard presence, missing checks, and omission patterns.

Targets specifically:
- CWE-476: NULL Pointer Dereference (79.8% missed in Phase 15)
- CWE-400: Uncontrolled Resource Consumption (74.7% missed)
- CWE-835: Loop with Unreachable Exit Condition (90.9% missed)
- CWE-416 / CWE-401: Allocation-Deallocation Imbalance and Resource Cleanup
"""

import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

DANGEROUS_C_FUNCS = {
    'strcpy', 'strcat', 'sprintf', 'vsprintf', 'gets', 'scanf', 'sscanf',
    'fscanf', 'vfscanf', 'vscanf', 'vsscanf', 'realpath', 'getwd'
}

BOUNDED_SAFE_FUNCS = {
    'strncpy', 'strncat', 'snprintf', 'vsnprintf', 'fgets', 'memcpy',
    'memmove', 'memset', 'strlcpy', 'strlcat', 'bcopy'
}

CONTROL_KEYWORDS = {
    'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default',
    'goto', 'break', 'continue', 'return', 'throw', 'try', 'catch'
}

C_TOKEN_RE = re.compile(
    r'//.*?$|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|'
    r'0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|'
    r'->|\+\+|--|::|<<|>>|<=|>=|==|!=|&&|\|\||'
    r'[A-Za-z_][A-Za-z0-9_]*|'
    r'[+\-*/%=&|^~<>!?:;,.]|'
    r'[{}()\[\]]',
    re.MULTILINE
)


def cxx_tokenize(code: str):
    return [t for t in C_TOKEN_RE.findall(code) if t and not t.startswith('/*') and not t.startswith('//')]


class Phase16OmissionAwareCxxExtractor(BaseEstimator, TransformerMixin):
    """
    38-Dimensional Modern C/C++ & Omission-Aware Structural Feature Extractor.
    Dimensions 1–28: Phase 15 Modern C++ & GoogleTest Scaffolding.
    Dimensions 29–38: Omission-Aware Safeguards (CWE-476, CWE-400, CWE-835, CWE-416).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for code in X:
            if not isinstance(code, str):
                code = str(code) if code is not None else ""

            tokens = cxx_tokenize(code)
            n_tokens = max(1.0, float(len(tokens)))
            n_lines = max(1.0, float(code.count('\n') + 1))
            n_chars = max(1.0, float(len(code)))

            # --- Base Phase 15 Features (1..24) ---
            n_dang = sum(1 for t in tokens if t in DANGEROUS_C_FUNCS)
            n_safe = sum(1 for t in tokens if t in BOUNDED_SAFE_FUNCS)
            n_ctrl = sum(1 for t in tokens if t in CONTROL_KEYWORDS)

            n_ptrs = sum(1 for t in tokens if t in ('->', '*'))
            n_refs = sum(1 for t in tokens if t == '&')

            n_c_mem_alloc = sum(1 for t in tokens if t in ('malloc', 'calloc', 'realloc', 'alloca'))
            n_c_mem_free = sum(1 for t in tokens if t == 'free')
            n_c_mem = n_c_mem_alloc + n_c_mem_free

            n_smart_ptrs = len(re.findall(r'\b(?:unique_ptr|shared_ptr|weak_ptr|make_unique|make_shared)\b', code))
            n_new = len(re.findall(r'\bnew\b', code))
            n_delete = len(re.findall(r'\bdelete\b', code))
            n_new_del = n_new + n_delete

            n_casts = len(re.findall(r'\b(?:static_cast|dynamic_cast|reinterpret_cast|const_cast)\b', code))
            n_risky_casts = len(re.findall(r'\b(?:reinterpret_cast|const_cast)\b', code))

            n_templates = len(re.findall(r'\b(?:template|typename|decltype)\b', code))
            n_scopes = code.count('::')
            n_namespaces = len(re.findall(r'\bnamespace\b', code))

            n_virtual = len(re.findall(r'\b(?:virtual|override|final)\b', code))
            n_exceptions = len(re.findall(r'\b(?:try|catch|throw|noexcept)\b', code))
            n_locks = len(re.findall(r'\b(?:lock_guard|unique_lock|scoped_lock|mutex)\b', code))

            n_gtest = len(re.findall(r'\b(?:EXPECT_|ASSERT_|MOCK_METHOD|TEST_F|TEST_P|TEST|NiceMock)\b', code))
            is_test = 1.0 if n_gtest > 0 else 0.0

            # Max nesting depth
            depth, max_depth = 0, 0
            for t in tokens:
                if t == '{':
                    depth += 1
                    if depth > max_depth:
                        max_depth = depth
                elif t == '}':
                    depth = max(0, depth - 1)
            n_comments = len(re.findall(r'(//|/\*|#)', code))

            is_cpp = 1.0 if (n_scopes > 0 or n_smart_ptrs > 0 or n_templates > 0 or n_namespaces > 0 or 'std::' in code) else 0.0
            smart_ratio = float(n_smart_ptrs) / (n_c_mem + n_new_del + 1.0)

            # Phase 15 Modern C++ Additions (25..28)
            n_smart_escape = len(re.findall(r'\.(?:get|release)\s*\(\)', code))
            n_moves = len(re.findall(r'\b(?:std::move|std::forward)\b', code))
            n_ptr_arith = len(re.findall(r'(?:->|\*|\w+)\s*(?:\+\+|--|\+\s*\d+|\-\s*\d+)', code))
            n_int_casts = len(re.findall(r'(?:static_cast\s*<\s*(?:int|short|unsigned|uint\w*)\s*>|\((?:int|short|unsigned int)\))', code))

            # --- NEW Phase 16 Omission-Aware Safeguard Features (29..38) ---
            # 29. Null Check Count: explicit validation of pointer validity
            null_checks = len(re.findall(r'\b(?:NULL|nullptr)\b|\bif\s*\(\s*!\s*\w+|\bif\s*\(\s*\w+\s*==\s*(?:NULL|nullptr|0)\b|\bif\s*\(\s*\w+\s*!=\s*(?:NULL|nullptr|0)\b|\bif\s*\(\s*\w+\s*\)', code))
            
            # 30. Explicit Pointer Arrow Dereferences (ptr->field)
            n_arrow_derefs = code.count('->')
            
            # 31. Null Check Density per Dereference: high = safeguarded, 0 = unchecked!
            null_check_density = float(null_checks) / (float(n_arrow_derefs) + 1.0)
            
            # 32. Unchecked Dereference Heuristic: pointer deref present with 0 null checks
            unchecked_deref_heuristic = 1.0 if (n_arrow_derefs > 2 and null_checks == 0) else 0.0

            # 33. Suspicious / Unbounded Loops (CWE-835 / CWE-400)
            suspicious_loops = len(re.findall(r'\bwhile\s*\(\s*(?:1|true|!0)\s*\)|\bfor\s*\(\s*;\s*;\s*\)', code))

            # 34. Loop Exit Guards: loops with break/return escape paths
            loop_exit_guards = len(re.findall(r'\b(?:break|return|goto)\b', code))
            
            # 35. Loop Bound Checks: presence of upper limit / length checks
            loop_bound_checks = len(re.findall(r'\b(?:sizeof|length\s*\(\)|size\s*\(\)|count|limit|max_len|max_size)\b', code))

            # 36. Allocation-Free Imbalance (CWE-401 / CWE-416)
            total_allocs = n_c_mem_alloc + n_new
            total_frees = n_c_mem_free + n_delete
            alloc_free_imbalance = abs(total_allocs - total_frees)

            # 37. Multiple Early Return Exit Points (Resource leak risk in complex branches)
            early_returns = len(re.findall(r'\breturn\b', code))

            # 38. Array Bounds Validation Checks
            array_accesses = len(re.findall(r'\[[^\]]+\]', code))
            bounds_checks = len(re.findall(r'(?:<\s*\w+|>\s*0|len\s*-\s*1|\bassert\s*\(|\bASSERT\s*\(|\bCHECK\s*\()', code))
            bounds_check_density = float(bounds_checks) / (float(array_accesses) + 1.0)

            feat_row = [
                # 1..3: Size
                np.log1p(n_chars),
                np.log1p(n_lines),
                np.log1p(len(tokens)),
                # 4..8: Ratios
                n_dang / n_tokens,
                n_safe / n_tokens,
                n_ctrl / n_tokens,
                n_ptrs / n_tokens,
                n_refs / n_tokens,
                # 9..13: C/C++ Alloc & Casts
                float(n_c_mem),
                float(n_smart_ptrs),
                float(n_new_del),
                float(n_casts),
                float(n_risky_casts),
                # 14..18: OOP & Concurrency
                float(n_templates),
                float(n_scopes),
                float(n_virtual),
                float(n_exceptions),
                float(n_locks),
                # 19..20: Testing
                float(n_gtest),
                is_test,
                # 21..24: Complexity & Style
                float(max_depth),
                n_comments / n_lines,
                is_cpp,
                smart_ratio,
                # 25..28: Phase 15 Modern C++ Additions
                float(n_smart_escape),
                float(n_moves),
                float(n_ptr_arith),
                float(n_int_casts),
                # 29..38: NEW Phase 16 Omission-Aware Features
                float(null_checks),
                float(n_arrow_derefs),
                float(null_check_density),
                float(unchecked_deref_heuristic),
                float(suspicious_loops),
                float(loop_exit_guards),
                float(loop_bound_checks),
                float(alloc_free_imbalance),
                float(early_returns),
                float(bounds_check_density)
            ]
            features.append(feat_row)

        return np.array(features, dtype=np.float32)

    def get_feature_names(self):
        return [
            # 1..28 (Phase 15)
            "log1p_chars", "log1p_lines", "log1p_tokens", "dang_ratio", "safe_ratio",
            "ctrl_ratio", "ptr_ratio", "ref_ratio", "c_mem_count", "smart_ptrs_count",
            "new_del_count", "casts_count", "risky_casts_count", "templates_count",
            "scopes_count", "virtual_count", "exceptions_count", "locks_count",
            "gtest_count", "is_test", "max_depth", "comment_density",
            "is_cpp", "smart_ratio", "smart_escape_count", "moves_count",
            "ptr_arith_count", "int_casts_count",
            # 29..38 (NEW Phase 16 Omission Features)
            "null_checks_count", "arrow_derefs_count", "null_check_density",
            "unchecked_deref_heuristic", "suspicious_loops_count",
            "loop_exit_guards_count", "loop_bound_checks_count",
            "alloc_free_imbalance", "early_returns_count", "bounds_check_density"
        ]
