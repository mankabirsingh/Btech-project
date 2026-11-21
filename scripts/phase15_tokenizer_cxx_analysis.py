#!/usr/bin/env python3
"""
Phase 15 — Step 2: Modern C++ Tokenizer Fragmentation Analysis
==============================================================
Analyzes BPE token fragmentation of CodeBERT (microsoft/codebert-base)
across modern C++ constructs vs traditional C syntax.
Inspects whether modifying the vocabulary is beneficial or if tokenization
artefacts can be effectively resolved via structural representation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    print("=" * 80)
    print("PHASE 15 — MODERN C++ TOKENIZER FRAGMENTATION AUDIT")
    print("=" * 80)
    print(f"Tokenizer type: {type(tokenizer).__name__}, Vocab size: {tokenizer.vocab_size}")

    cxx_snippets = [
        ("Smart Pointer Allocation", "auto ptr = std::make_unique<NiceMock<Ssl::MockConnectionInfo>>();"),
        ("Smart Pointer Access", "std::shared_ptr<Network::Address::Instance> addr = socket->connectionInfoProvider().remoteAddress();"),
        ("Modern C++ Cast", "auto* derived = dynamic_cast<DerivedClass*>(base_ptr);"),
        ("Reinterpret Cast", "uint64_t raw_val = reinterpret_cast<uint64_t>(ptr);"),
        ("Lambda Capture", "auto callback = [&state, count](Buffer::Instance& data) -> bool { return state.process(data); };"),
        ("GoogleTest Mock Scaffolding", "EXPECT_CALL(runtime_.snapshot_, featureEnabled(\"health_check.verify_cluster\", 100)).WillRepeatedly(Return(false));"),
        ("RAII Lock Guard", "std::lock_guard<std::mutex> guard(mutex_);"),
        ("Move Semantics", "headers.addViaMove(std::move(key), std::move(value));"),
        ("C Traditional Malloc", "char* buf = (char*)malloc(sizeof(char) * 1024); if (!buf) return -1; strcpy(buf, input);"),
        ("C Traditional Free", "if (ctx->buffer) { free(ctx->buffer); ctx->buffer = NULL; }")
    ]

    print("\nSnippet Subword Breakdown:")
    rows = []
    for category, code in cxx_snippets:
        tokens = tokenizer.tokenize(code)
        subwords = len(tokens)
        chars = len(code)
        frag_ratio = subwords / (len(code.split()) + 1e-6)
        rows.append({
            "category": category,
            "code": code,
            "num_tokens": subwords,
            "tokens": tokens[:12],  # first 12 tokens
            "frag_ratio": frag_ratio
        })
        print(f"\n[{category}]")
        print(f"  Code: {code}")
        print(f"  Tokens ({subwords}): {tokens[:10]} ...")
        print(f"  Fragmentation Ratio (Tokens/Word): {frag_ratio:.2f}")

    # Inspect tokenization on Envoy dataset
    test_df = pd.read_parquet(PROJECT_ROOT / "data" / "splits" / "vulnerability_test_heldout_projects_clean.parquet")
    envoy_df = test_df[test_df['project'] == 'envoy']
    openssl_df = test_df[test_df['project'] == 'openssl']

    print("\n--- Project Token Length Distribution ---")
    for name, df in [("Envoy Proxy (C++)", envoy_df), ("OpenSSL (C)", openssl_df)]:
        lengths = [len(tokenizer.tokenize(str(c)[:2000])) for c in df['code'].values[:200]]
        print(f"{name:20}: Mean Tokens={np.mean(lengths):.1f}, Median={np.median(lengths):.1f}, 90th Pct={np.percentile(lengths, 90):.1f}")


if __name__ == "__main__":
    main()
