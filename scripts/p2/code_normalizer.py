"""Code normalization and MinHash feature extraction for duplicate detection."""

import hashlib
import re
from typing import List, Set
import numpy as np
from datasketch import MinHash

CPP_KEYWORDS: Set[str] = {
    "alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit",
    "atomic_noexcept", "auto", "bitand", "bitor", "bool", "break", "case", "catch",
    "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept", "const",
    "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await",
    "co_return", "co_yield", "decltype", "default", "delete", "do", "double",
    "dynamic_cast", "else", "enum", "explicit", "export", "extern", "false", "float",
    "for", "friend", "goto", "if", "inline", "int", "long", "mutable", "namespace",
    "new", "noexcept", "not", "not_eq", "nullptr", "NULL", "operator", "or", "or_eq",
    "private", "protected", "public", "reflexpr", "register", "reinterpret_cast",
    "requires", "return", "short", "signed", "sizeof", "static", "static_assert",
    "static_cast", "struct", "switch", "synchronized", "template", "this",
    "thread_local", "throw", "true", "try", "typedef", "typeid", "typename",
    "union", "unsigned", "using", "virtual", "void", "volatile", "wchar_t",
    "while", "xor", "xor_eq"
}

_STR_DQUOTE = re.compile(r'"(?:\\.|[^"\\])*"')
_STR_SQUOTE = re.compile(r"'(?:\\.|[^'\\])*'")
_COMMENT_MULTI = re.compile(r"/\*[\s\S]*?\*/")
_COMMENT_SINGLE = re.compile(r"//.*")
_COMMENT_HASH = re.compile(r"#.*")
_NUM_HEX = re.compile(r"\b0x[0-9a-fA-F]+\b")
_NUM_DEC = re.compile(r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b")
_TOKEN_PATTERN = re.compile(r"[a-zA-Z_]\w*|[^\w\s]")
_IDENT_PATTERN = re.compile(r"^[a-zA-Z_]\w*$")


def tokenize_normalized(code: str) -> List[str]:
    """
    Normalizes code snippet by:
    - Stripping string literals and replacing with STR
    - Stripping multi-line and single-line comments
    - Normalizing hexadecimal, float, and integer literals to NUM
    - Mapping user identifiers to ID while preserving language keywords
    - Retaining structural punctuation and operators

    Returns list of normalized string tokens.
    """
    if not isinstance(code, str) or not code:
        return []

    # 1. Replace string literals with STR
    c = _STR_DQUOTE.sub(" STR ", code)
    c = _STR_SQUOTE.sub(" STR ", c)

    # 2. Strip comments
    c = _COMMENT_MULTI.sub(" ", c)
    c = _COMMENT_SINGLE.sub(" ", c)
    c = _COMMENT_HASH.sub(" ", c)

    # 3. Numbers -> NUM
    c = _NUM_HEX.sub(" NUM ", c)
    c = _NUM_DEC.sub(" NUM ", c)

    # 4. Tokenize
    raw_tokens = _TOKEN_PATTERN.findall(c)

    # 5. Map identifiers
    tokens: List[str] = []
    for t in raw_tokens:
        if t in ("STR", "NUM") or t in CPP_KEYWORDS or not _IDENT_PATTERN.match(t):
            tokens.append(t)
        else:
            tokens.append("ID")

    return tokens


def normalize_code_string(code: str) -> str:
    """Returns normalized code as a single whitespace-separated string."""
    return " ".join(tokenize_normalized(code))


def get_5grams(tokens: List[str]) -> List[str]:
    """Generates 5-gram token shingles."""
    if len(tokens) < 5:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + 5]) for i in range(len(tokens) - 4)]


def compute_minhash(tokens: List[str], num_perm: int = 128) -> MinHash:
    """Computes a 128-permutation MinHash from 5-gram tokens."""
    mh = MinHash(num_perm=num_perm)
    shingles = get_5grams(tokens)
    for s in shingles:
        mh.update(s.encode("utf-8"))
    return mh


def compute_minhash_slice(args: tuple) -> tuple:
    """
    Worker function to process a slice of code snippets and save hashvalues and MD5s to disk.
    args: (slice_idx, codes_slice, output_npy_path, output_md5_path)
    Returns: (output_npy_path, output_md5_path)
    """
    slice_idx, codes_slice, output_npy_path, output_md5_path = args
    n = len(codes_slice)
    arr = np.zeros((n, 128), dtype=np.uint32)
    md5_hashes = []
    for i, code in enumerate(codes_slice):
        if not isinstance(code, str) or not code:
            md5_hashes.append("")
            continue
        if len(code) > 25000:
            code = code[:25000]
        toks = tokenize_normalized(code)
        norm_str = " ".join(toks)
        md5_hashes.append(hashlib.md5(norm_str.encode("utf-8")).hexdigest())
        mh = compute_minhash(toks)
        arr[i] = mh.hashvalues

    np.save(output_npy_path, arr)
    np.save(output_md5_path, np.array(md5_hashes))
    return (output_npy_path, output_md5_path)

