"""
Phase 14 Modern C++ Structural Feature Extractor
================================================
Extracts 24-dimensional AST and lexical security features tailored for
both classical C and modern C++17 constructs (Envoy Proxy, Chrome, TensorFlow).

Captures:
- Dangerous libc APIs vs. modern bounded alternatives
- Modern C++ RAII and smart pointers (std::unique_ptr, std::shared_ptr) vs raw new/delete
- GoogleTest / unit test fixtures (suppresses test false positives)
- C++ template, namespace, and virtual inheritance depth
- C++ cast risks (reinterpret_cast vs static_cast)
- Concurrency locks and exception safety
"""

import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

DANGEROUS_C_FUNCS = set([
    'strcpy', 'strcat', 'sprintf', 'vsprintf', 'gets', 'scanf', 'sscanf', 'fscanf',
    'alloca', 'system', 'popen', 'execve', 'execl', 'execvp', 'open', 'read', 'write',
    'close', 'chown', 'chmod', 'unlink', 'realpath', 'getenv', 'setuid', 'seteuid'
])

BOUNDED_SAFE_FUNCS = set([
    'snprintf', 'strncpy', 'strncat', 'strlcpy', 'strlcat', 'fgets', 'getline',
    'memcpy_s', 'memmove_s', 'memset_s'
])

CONTROL_KEYWORDS = set(['if', 'else', 'while', 'for', 'switch', 'case', 'goto', 'return', 'break', 'continue'])


def cxx_tokenize(code: str):
    """
    Tokenizer that preserves C++ identifiers, scope resolution (::),
    and arrow operators without over-fragmentation.
    """
    if not isinstance(code, str):
        return []
    # Strip comments
    clean = re.sub(r'//.*$|/\*.*?\*/', ' ', code, flags=re.MULTILINE | re.DOTALL)
    # Match C++ scope tokens, arrows, operators, and identifiers
    pattern = r'::|->|<=|>=|==|!=|\+\+|--|&&|\|\||<<|>>|\b[A-Za-z_][A-Za-z0-9_]*\b|[{}()\[\];,\*\&<>]'
    return re.findall(pattern, clean)


class EnhancedModernCxxStructuralExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts 24 quantitative security and structural features for C and C++ code.
    Compatible with scikit-learn pipelines and FeatureUnion.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for code in X:
            if not isinstance(code, str):
                code = ""
            
            tokens = cxx_tokenize(code)
            n_tokens = len(tokens) + 1e-6
            n_chars = len(code) + 1.0
            n_lines = code.count('\n') + 1.0

            # 1. Dangerous C Functions
            n_dang = sum(1 for t in tokens if t in DANGEROUS_C_FUNCS)
            
            # 2. Bounded Safe Functions
            n_safe = sum(1 for t in tokens if t in BOUNDED_SAFE_FUNCS)

            # 3. Control flow
            n_ctrl = sum(1 for t in tokens if t in CONTROL_KEYWORDS)

            # 4. Raw pointers vs references
            n_ptrs = sum(1 for t in tokens if t in ('->', '*'))
            n_refs = sum(1 for t in tokens if t == '&')

            # 5. C Memory management
            n_c_mem = sum(1 for t in tokens if t in ('malloc', 'calloc', 'realloc', 'free', 'alloca'))

            # 6. C++ Smart pointers
            n_smart_ptrs = len(re.findall(r'\b(?:unique_ptr|shared_ptr|weak_ptr|make_unique|make_shared)\b', code))

            # 7. C++ Raw new / delete
            n_new_del = len(re.findall(r'\b(?:new|delete)\b', code))

            # 8. C++ Casts
            n_casts = len(re.findall(r'\b(?:static_cast|dynamic_cast|reinterpret_cast|const_cast)\b', code))
            n_risky_casts = len(re.findall(r'\b(?:reinterpret_cast|const_cast)\b', code))

            # 9. Templates
            n_templates = len(re.findall(r'\b(?:template|typename)\b', code))

            # 10. Scope resolution & Namespaces
            n_scopes = code.count('::')
            n_namespaces = len(re.findall(r'\bnamespace\b', code))

            # 11. Virtual & Override
            n_virtual = len(re.findall(r'\b(?:virtual|override|final)\b', code))

            # 12. Exception handling
            n_exceptions = len(re.findall(r'\b(?:try|catch|throw|noexcept)\b', code))

            # 13. Concurrency / RAII locks
            n_locks = len(re.findall(r'\b(?:lock_guard|unique_lock|scoped_lock|mutex)\b', code))

            # 14. GoogleTest & Mock Macros
            n_gtest = len(re.findall(r'\b(?:EXPECT_|ASSERT_|MOCK_METHOD|TEST_F|TEST_P|TEST|NiceMock)\b', code))
            is_test = 1.0 if n_gtest > 0 else 0.0

            # 15. Lambda Expressions
            n_lambdas = len(re.findall(r'\[[^\]]*\]\s*\([^\)]*\)', code))

            # 16. Maximum nesting depth
            depth, max_depth = 0, 0
            for t in tokens:
                if t == '{':
                    depth += 1
                    if depth > max_depth:
                        max_depth = depth
                elif t == '}':
                    depth = max(0, depth - 1)

            # 17. Comment density
            n_comments = len(re.findall(r'(//|/\*|#)', code))

            # 18. Is Modern C++ flag
            is_cpp = 1.0 if (n_scopes > 0 or n_smart_ptrs > 0 or n_templates > 0 or n_namespaces > 0 or 'std::' in code) else 0.0

            # 19. Smart to Raw Memory Ratio
            smart_ratio = float(n_smart_ptrs) / (n_c_mem + n_new_del + 1.0)

            # 20. C++ standard library containers
            n_containers = len(re.findall(r'\b(?:vector|string|map|unordered_map|set|array|deque|list)\b', code))

            # 21. NULL / bounds checks
            n_null_checks = len(re.findall(r'\b(?:NULL|nullptr|nullptr_t)\b', code)) + code.count('!ptr') + code.count('== 0')

            # 22. Buffer copy calls
            n_cpy = sum(1 for t in tokens if t in ('strcpy', 'strncpy', 'strcat', 'strncat', 'memcpy', 'memmove'))

            feat_row = [
                np.log1p(n_chars),                       # 1
                np.log1p(n_lines),                       # 2
                np.log1p(len(tokens)),                   # 3
                n_dang / n_tokens,                       # 4
                n_safe / n_tokens,                       # 5
                n_ctrl / n_tokens,                       # 6
                n_ptrs / n_tokens,                       # 7
                n_refs / n_tokens,                       # 8
                float(n_c_mem),                          # 9
                float(n_smart_ptrs),                     # 10
                float(n_new_del),                        # 11
                float(n_casts),                          # 12
                float(n_risky_casts),                    # 13
                float(n_templates),                      # 14
                float(n_scopes),                         # 15
                float(n_virtual),                        # 16
                float(n_exceptions),                     # 17
                float(n_locks),                          # 18
                float(n_gtest),                          # 19
                is_test,                                 # 20
                float(max_depth),                        # 21
                n_comments / n_lines,                    # 22
                is_cpp,                                  # 23
                smart_ratio,                             # 24
            ]
            features.append(feat_row)

        return np.array(features, dtype=np.float32)
