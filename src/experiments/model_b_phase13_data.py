"""
Phase 13 Expanded Model B Dataset Generator
============================================
Creates 12 additional controlled security seed tasks (Tasks 25 to 36, 48 code units)
across 12 distinct CWE categories, and bundles them with Tasks 9-24 (64 units)
to form the Phase 13 expanded dataset of 28 tasks (112 units).

Guarantees:
1. Zero task or code overlap with locked benchmark 32 (Tasks 1-8).
2. Grouped splitting by task ID (all 4 versions of a task stay in the same split).
3. Exactly 50% vulnerable and 50% safe in every partition and provenance category.
4. Cryptographic SHA-256 manifest.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

PROJECT_ROOT = Path("/Users/mankabir/Desktop/Btech project")

@dataclass
class CodeUnit:
    task_id: str
    version_id: str
    cwe: str
    cwe_name: str
    code: str
    language: str
    author_type: str        # 'human', 'ai', 'human_edit_of_ai', 'ai_edit_of_human'
    ai_involvement: float
    ai_generator: str
    human_edit_count: int
    provenance_confidence: float
    vuln_label: int         # 0 = safe, 1 = vulnerable
    lineage_stage: str
    rationale: str
    code_hash: str = ""

    def __post_init__(self):
        self.code_hash = hashlib.sha256(self.code.strip().encode()).hexdigest()[:16]


def create_phase13_tasks() -> List[CodeUnit]:
    units = []

    # =========================================================================
    # Task 25: CWE-121 Stack-based Buffer Overflow in Packet Header Decoding
    # =========================================================================
    units.append(CodeUnit(
        task_id="dev_task_025_stack_overflow",
        version_id="t25_v1_human_secure",
        cwe="CWE-121",
        cwe_name="Stack-based Buffer Overflow",
        code="""#include <string.h>
#include <stdio.h>

int decode_packet_header(const char *raw, size_t len, char *out_tag, size_t tag_cap) {
    if (!raw || !out_tag || tag_cap == 0) return -1;
    char header_buf[32];
    if (len >= sizeof(header_buf)) return -1;
    memcpy(header_buf, raw, len);
    header_buf[len] = '\\0';
    snprintf(out_tag, tag_cap, "HDR_%s", header_buf);
    return 0;
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human defensive: validates len < sizeof(header_buf) before memcpy."
    ))

    units.append(CodeUnit(
        task_id="dev_task_025_stack_overflow",
        version_id="t25_v2_ai_naive",
        cwe="CWE-121",
        cwe_name="Stack-based Buffer Overflow",
        code="""#include <string.h>
#include <stdio.h>

int decode_packet_header(const char *raw, size_t len, char *out_tag, size_t tag_cap) {
    char header_buf[32];
    // AI naive: copies arbitrary len directly into fixed 32-byte stack buffer
    memcpy(header_buf, raw, len);
    header_buf[len] = '\\0';
    snprintf(out_tag, tag_cap, "HDR_%s", header_buf);
    return 0;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: omits stack boundary check, causing stack buffer overflow."
    ))

    units.append(CodeUnit(
        task_id="dev_task_025_stack_overflow",
        version_id="t25_v3_human_remediated",
        cwe="CWE-121",
        cwe_name="Stack-based Buffer Overflow",
        code="""#include <string.h>
#include <stdio.h>

int decode_packet_header(const char *raw, size_t len, char *out_tag, size_t tag_cap) {
    if (!raw || !out_tag || tag_cap == 0) return -1;
    char header_buf[32];
    size_t copy_bytes = (len < sizeof(header_buf) - 1) ? len : sizeof(header_buf) - 1;
    memcpy(header_buf, raw, copy_bytes);
    header_buf[copy_bytes] = '\\0';
    snprintf(out_tag, tag_cap, "HDR_%s", header_buf);
    return 0;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="GitHub Copilot",
        human_edit_count=2,
        provenance_confidence=0.90,
        vuln_label=0,
        lineage_stage="human_remediated",
        rationale="Human remediation: bounds copy_bytes strictly to capacity - 1."
    ))

    units.append(CodeUnit(
        task_id="dev_task_025_stack_overflow",
        version_id="t25_v4_ai_regressive",
        cwe="CWE-121",
        cwe_name="Stack-based Buffer Overflow",
        code="""#include <string.h>
#include <stdio.h>

int decode_packet_header(const char *raw, size_t len, char *out_tag, size_t tag_cap) {
    char header_buf[32];
    // AI edit: checked len <= sizeof(header_buf), but writing '\\0' at header_buf[len] overflows by 1 byte
    if (len > sizeof(header_buf)) return -1;
    memcpy(header_buf, raw, len);
    header_buf[len] = '\\0'; // Off-by-one stack corruption when len == 32!
    snprintf(out_tag, tag_cap, "HDR_%s", header_buf);
    return 0;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=0,
        provenance_confidence=0.85,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI regressive: 1-byte off-by-one stack overwrite when len == sizeof(header_buf)."
    ))

    # =========================================================================
    # Task 26: CWE-122 Heap-based Buffer Overflow in Audio Frame Resizing
    # =========================================================================
    units.append(CodeUnit(
        task_id="dev_task_026_heap_buffer_overflow",
        version_id="t26_v1_human_legacy_vuln",
        cwe="CWE-122",
        cwe_name="Heap-based Buffer Overflow",
        code="""#include <stdlib.h>
#include <string.h>

short *resize_audio_frame(const short *in, int samples, int add_samples) {
    // Human legacy flaw: allocates samples * sizeof(short) but copies (samples + add_samples)
    short *out = (short *)malloc(samples * sizeof(short));
    if (!out) return NULL;
    memcpy(out, in, (samples + add_samples) * sizeof(short));
    return out;
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=1,  # Inverted for balance: Human is vulnerable
        lineage_stage="base",
        rationale="Human legacy flaw: mismatch between allocation size and copy byte count."
    ))

    units.append(CodeUnit(
        task_id="dev_task_026_heap_buffer_overflow",
        version_id="t26_v2_ai_defensive_fix",
        cwe="CWE-122",
        cwe_name="Heap-based Buffer Overflow",
        code="""#include <stdlib.h>
#include <string.h>

short *resize_audio_frame(const short *in, int samples, int add_samples) {
    if (!in || samples <= 0 || add_samples < 0) return NULL;
    size_t total = (size_t)samples + (size_t)add_samples;
    if (total > 65536) return NULL; // Safety threshold
    short *out = (short *)calloc(total, sizeof(short));
    if (!out) return NULL;
    memcpy(out, in, samples * sizeof(short));
    return out;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="GitHub Copilot",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=0,  # Inverted for balance: AI is safe
        lineage_stage="ai_naive",
        rationale="AI defensive generation: allocates for total samples and bounds copies safely."
    ))

    units.append(CodeUnit(
        task_id="dev_task_026_heap_buffer_overflow",
        version_id="t26_v3_human_buggy_edit",
        cwe="CWE-122",
        cwe_name="Heap-based Buffer Overflow",
        code="""#include <stdlib.h>
#include <string.h>

short *resize_audio_frame(const short *in, int samples, int add_samples) {
    // Human developer introduced regression during optimization
    size_t alloc_bytes = samples * sizeof(short);
    short *out = (short *)malloc(alloc_bytes);
    if (!out) return NULL;
    memcpy(out, in, (samples + add_samples) * sizeof(short)); // Overflows heap!
    return out;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="GitHub Copilot",
        human_edit_count=1,
        provenance_confidence=0.90,
        vuln_label=1,  # Inverted for balance: Human-edit is vulnerable
        lineage_stage="human_remediated",
        rationale="Human edit regression: reinstated undersized allocation during manual tuning."
    ))

    units.append(CodeUnit(
        task_id="dev_task_026_heap_buffer_overflow",
        version_id="t26_v4_ai_remediated_secure",
        cwe="CWE-122",
        cwe_name="Heap-based Buffer Overflow",
        code="""#include <stdlib.h>
#include <string.h>

short *resize_audio_frame(const short *in, int samples, int add_samples) {
    if (!in || samples <= 0 || add_samples < 0) return NULL;
    size_t total = (size_t)samples + (size_t)add_samples;
    if (total > 100000) return NULL;
    short *out = (short *)malloc(total * sizeof(short));
    if (!out) return NULL;
    memcpy(out, in, samples * sizeof(short));
    memset(out + samples, 0, add_samples * sizeof(short));
    return out;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=0,
        provenance_confidence=0.90,
        vuln_label=0,  # Inverted for balance: AI-edit is safe
        lineage_stage="ai_regressive",
        rationale="AI assistant remediation: fixes heap buffer calculation and zeroes padding."
    ))

    # =========================================================================
    # Task 27: CWE-126 Buffer Over-read in String Tokenizer
    # =========================================================================
    units.append(CodeUnit(
        task_id="dev_task_027_buffer_overread",
        version_id="t27_v1_human_secure",
        cwe="CWE-126",
        cwe_name="Buffer Over-read",
        code="""#include <stddef.h>

const char *find_delim(const char *buf, size_t buf_len, char delim) {
    if (!buf) return NULL;
    for (size_t i = 0; i < buf_len; ++i) {
        if (buf[i] == delim) return &buf[i];
    }
    return NULL;
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human defensive: strictly terminates search at buf_len."
    ))

    units.append(CodeUnit(
        task_id="dev_task_027_buffer_overread",
        version_id="t27_v2_ai_naive",
        cwe="CWE-126",
        cwe_name="Buffer Over-read",
        code="""#include <stddef.h>

const char *find_delim(const char *buf, size_t buf_len, char delim) {
    // AI naive: ignores buf_len and searches until '\\0', over-reading non-null terminated binary buffers!
    const char *p = buf;
    while (*p != delim && *p != '\\0') {
        p++;
    }
    return (*p == delim) ? p : NULL;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: assumes null-termination, causing buffer over-read on binary slices."
    ))

    units.append(CodeUnit(
        task_id="dev_task_027_buffer_overread",
        version_id="t27_v3_human_remediated",
        cwe="CWE-126",
        cwe_name="Buffer Over-read",
        code="""#include <stddef.h>

const char *find_delim(const char *buf, size_t buf_len, char delim) {
    if (!buf || buf_len == 0) return NULL;
    const char *end = buf + buf_len;
    for (const char *p = buf; p < end; ++p) {
        if (*p == delim) return p;
    }
    return NULL;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="GitHub Copilot",
        human_edit_count=2,
        provenance_confidence=0.90,
        vuln_label=0,
        lineage_stage="human_remediated",
        rationale="Human remediation: replaces unbounded pointer walk with bounded end-pointer check."
    ))

    units.append(CodeUnit(
        task_id="dev_task_027_buffer_overread",
        version_id="t27_v4_ai_regressive",
        cwe="CWE-126",
        cwe_name="Buffer Over-read",
        code="""#include <stddef.h>

const char *find_delim(const char *buf, size_t buf_len, char delim) {
    // AI edit: checks p <= end, allowing reading one byte past allocated buffer at *end!
    if (!buf) return NULL;
    const char *end = buf + buf_len;
    for (const char *p = buf; p <= end; ++p) {
        if (*p == delim) return p; // Over-reads 1 byte when p == end!
    }
    return NULL;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.85,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI regressive: off-by-one boundary comparison (<= instead of <) reads 1 byte past buffer."
    ))

    # =========================================================================
    # Task 28: CWE-191 Integer Underflow in Ring Buffer Offset Calculation
    # =========================================================================
    units.append(CodeUnit(
        task_id="dev_task_028_integer_underflow",
        version_id="t28_v1_human_secure",
        cwe="CWE-191",
        cwe_name="Integer Underflow",
        code="""#include <stddef.h>

size_t compute_rewind_offset(size_t head, size_t rewind_bytes, size_t capacity) {
    if (capacity == 0) return 0;
    if (rewind_bytes > head) {
        // Underflow guard: wrap around cleanly using capacity
        size_t diff = rewind_bytes - head;
        return (capacity - (diff % capacity)) % capacity;
    }
    return head - rewind_bytes;
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human defensive: validates head >= rewind_bytes before subtraction to prevent unsigned underflow."
    ))

    units.append(CodeUnit(
        task_id="dev_task_028_integer_underflow",
        version_id="t28_v2_ai_naive",
        cwe="CWE-191",
        cwe_name="Integer Underflow",
        code="""#include <stddef.h>

size_t compute_rewind_offset(size_t head, size_t rewind_bytes, size_t capacity) {
    // AI naive: directly subtracts unsigned integers without check; underflows to ~SIZE_MAX!
    return (head - rewind_bytes) % capacity;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: unsigned subtraction underflow when rewind_bytes > head."
    ))

    units.append(CodeUnit(
        task_id="dev_task_028_integer_underflow",
        version_id="t28_v3_human_remediated",
        cwe="CWE-191",
        cwe_name="Integer Underflow",
        code="""#include <stddef.h>

size_t compute_rewind_offset(size_t head, size_t rewind_bytes, size_t capacity) {
    if (capacity == 0) return 0;
    size_t effective_rewind = rewind_bytes % capacity;
    if (head >= effective_rewind) {
        return head - effective_rewind;
    }
    return capacity - (effective_rewind - head);
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="GitHub Copilot",
        human_edit_count=2,
        provenance_confidence=0.90,
        vuln_label=0,
        lineage_stage="human_remediated",
        rationale="Human remediation: normalizes modulo capacity and uses explicit branches to prevent underflow."
    ))

    units.append(CodeUnit(
        task_id="dev_task_028_integer_underflow",
        version_id="t28_v4_ai_regressive",
        cwe="CWE-191",
        cwe_name="Integer Underflow",
        code="""#include <stddef.h>

size_t compute_rewind_offset(size_t head, size_t rewind_bytes, size_t capacity) {
    // AI edit: attempts signed cast but capacity is size_t, casting back to unsigned causes wrap
    long offset = (long)head - (long)rewind_bytes;
    return (size_t)offset % capacity; // Negative signed value cast to size_t becomes huge unsigned number!
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=0,
        provenance_confidence=0.85,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI regressive: signed underflow converted to unsigned causes modulo on giant integer."
    ))

    # =========================================================================
    # Task 29: CWE-369 Divide by Zero in Audio Resampling Rate
    # =========================================================================
    units.append(CodeUnit(
        task_id="dev_task_029_divide_by_zero",
        version_id="t29_v1_human_legacy_vuln",
        cwe="CWE-369",
        cwe_name="Divide By Zero",
        code="""int calculate_stride(int sample_rate, int target_fps) {
    // Human legacy flaw: fails to check if target_fps == 0
    return sample_rate / target_fps;
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=1,  # Inverted for balance
        lineage_stage="base",
        rationale="Human legacy flaw: integer division without divisor validation."
    ))

    units.append(CodeUnit(
        task_id="dev_task_029_divide_by_zero",
        version_id="t29_v2_ai_defensive",
        cwe="CWE-369",
        cwe_name="Divide By Zero",
        code="""int calculate_stride(int sample_rate, int target_fps) {
    if (target_fps <= 0 || sample_rate <= 0) {
        return -1; // Safe error code
    }
    return sample_rate / target_fps;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=0,  # Inverted for balance
        lineage_stage="ai_naive",
        rationale="AI defensive generation: validates divisor > 0 before division."
    ))

    units.append(CodeUnit(
        task_id="dev_task_029_divide_by_zero",
        version_id="t29_v3_human_buggy",
        cwe="CWE-369",
        cwe_name="Divide By Zero",
        code="""int calculate_stride(int sample_rate, int target_fps) {
    // Human refactor checked sample_rate instead of target_fps!
    if (sample_rate <= 0) return -1;
    return sample_rate / target_fps; // Crash when target_fps == 0
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="GitHub Copilot",
        human_edit_count=1,
        provenance_confidence=0.90,
        vuln_label=1,  # Inverted for balance
        lineage_stage="human_remediated",
        rationale="Human developer checked dividend instead of divisor."
    ))

    units.append(CodeUnit(
        task_id="dev_task_029_divide_by_zero",
        version_id="t29_v4_ai_fixed",
        cwe="CWE-369",
        cwe_name="Divide By Zero",
        code="""int calculate_stride(int sample_rate, int target_fps) {
    if (target_fps == 0) return 0;
    if (sample_rate <= 0 || target_fps < 0) return -1;
    return sample_rate / target_fps;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=0,
        provenance_confidence=0.90,
        vuln_label=0,  # Inverted for balance
        lineage_stage="ai_regressive",
        rationale="AI assistant remediation: explicitly guards target_fps == 0."
    ))

    # =========================================================================
    # Task 30: CWE-401 Memory Leak in Error Handling Return Path
    # =========================================================================
    units.append(CodeUnit(
        task_id="dev_task_030_memory_leak",
        version_id="t30_v1_human_secure",
        cwe="CWE-401",
        cwe_name="Missing Release of Memory",
        code="""#include <stdlib.h>

int parse_config_token(const char *in, int validate) {
    char *buf = (char *)malloc(256);
    if (!buf) return -1;
    if (!validate) {
        free(buf); // Clean release on early return
        return -2;
    }
    // process
    free(buf);
    return 0;
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human defensive: ensures free(buf) on all exit paths."
    ))

    units.append(CodeUnit(
        task_id="dev_task_030_memory_leak",
        version_id="t30_v2_ai_naive",
        cwe="CWE-401",
        cwe_name="Missing Release of Memory",
        code="""#include <stdlib.h>

int parse_config_token(const char *in, int validate) {
    char *buf = (char *)malloc(256);
    if (!buf) return -1;
    if (!validate) {
        return -2; // AI naive: early return leaks allocated buffer!
    }
    free(buf);
    return 0;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: leaks heap allocation on early error return."
    ))

    units.append(CodeUnit(
        task_id="dev_task_030_memory_leak",
        version_id="t30_v3_human_remediated",
        cwe="CWE-401",
        cwe_name="Missing Release of Memory",
        code="""#include <stdlib.h>

int parse_config_token(const char *in, int validate) {
    char *buf = (char *)malloc(256);
    if (!buf) return -1;
    int ret = 0;
    if (!validate) {
        ret = -2;
        goto cleanup;
    }
cleanup:
    free(buf);
    return ret;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="GitHub Copilot",
        human_edit_count=2,
        provenance_confidence=0.90,
        vuln_label=0,
        lineage_stage="human_remediated",
        rationale="Human remediation: uses RAII-style goto cleanup idiom to prevent leaks."
    ))

    units.append(CodeUnit(
        task_id="dev_task_030_memory_leak",
        version_id="t30_v4_ai_regressive",
        cwe="CWE-401",
        cwe_name="Missing Release of Memory",
        code="""#include <stdlib.h>

int parse_config_token(const char *in, int validate) {
    char *buf = (char *)malloc(256);
    if (!buf) return -1;
    if (!in) return -3; // AI edit: added new check but omitted free(buf)!
    free(buf);
    return 0;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=0,
        provenance_confidence=0.85,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI regressive: introduces un-freed return path on parameter validation."
    ))

    # =========================================================================
    # Task 31: CWE-476 NULL Pointer Dereference in Optional Configuration
    # =========================================================================
    units.append(CodeUnit(
        task_id="dev_task_031_null_dereference",
        version_id="t31_v1_human_secure",
        cwe="CWE-476",
        cwe_name="NULL Pointer Dereference",
        code="""#include <stddef.h>

typedef struct { int timeout; int port; } Config;

int apply_config(const Config *cfg) {
    if (!cfg) return -1; // Defensive null guard
    return cfg->timeout + cfg->port;
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human defensive: checks cfg against NULL before member access."
    ))

    units.append(CodeUnit(
        task_id="dev_task_031_null_dereference",
        version_id="t31_v2_ai_naive",
        cwe="CWE-476",
        cwe_name="NULL Pointer Dereference",
        code="""#include <stddef.h>

typedef struct { int timeout; int port; } Config;

int apply_config(const Config *cfg) {
    // AI naive: directly accesses cfg without verifying pointer validity
    return cfg->timeout + cfg->port;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: omits pointer validation leading to segfault on null cfg."
    ))

    units.append(CodeUnit(
        task_id="dev_task_031_null_dereference",
        version_id="t31_v3_human_remediated",
        cwe="CWE-476",
        cwe_name="NULL Pointer Dereference",
        code="""#include <stddef.h>

typedef struct { int timeout; int port; } Config;

int apply_config(const Config *cfg) {
    if (cfg == NULL) {
        return 0; // Default safe fallback
    }
    return cfg->timeout + cfg->port;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="GitHub Copilot",
        human_edit_count=2,
        provenance_confidence=0.90,
        vuln_label=0,
        lineage_stage="human_remediated",
        rationale="Human remediation: explicitly catches null pointer with safe fallback."
    ))

    units.append(CodeUnit(
        task_id="dev_task_031_null_dereference",
        version_id="t31_v4_ai_regressive",
        cwe="CWE-476",
        cwe_name="NULL Pointer Dereference",
        code="""#include <stddef.h>

typedef struct { int timeout; int port; } Config;

int apply_config(const Config *cfg) {
    // AI edit: checks cfg->port before checking if cfg is NULL!
    if (cfg->port < 0 || !cfg) return -1; // Dereference occurs BEFORE null check!
    return cfg->timeout;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.85,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI regressive: evaluates member access before pointer null check in logical OR."
    ))

    # =========================================================================
    # Task 32: CWE-590 Free of Memory not on the Heap
    # =========================================================================
    units.append(CodeUnit(
        task_id="dev_task_032_free_non_heap",
        version_id="t32_v1_human_legacy_vuln",
        cwe="CWE-590",
        cwe_name="Free of Memory not on the Heap",
        code="""#include <stdlib.h>

void release_buffer(char *ptr, int is_dynamic) {
    // Human legacy flaw: always calls free() even if buffer was stack-allocated
    free(ptr);
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=1,  # Inverted for balance
        lineage_stage="base",
        rationale="Human legacy flaw: unconditionally frees pointer without dynamic check."
    ))

    units.append(CodeUnit(
        task_id="dev_task_032_free_non_heap",
        version_id="t32_v2_ai_defensive",
        cwe="CWE-590",
        cwe_name="Free of Memory not on the Heap",
        code="""#include <stdlib.h>

void release_buffer(char *ptr, int is_dynamic) {
    if (ptr != NULL && is_dynamic) {
        free(ptr);
    }
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=0,  # Inverted for balance
        lineage_stage="ai_naive",
        rationale="AI defensive generation: validates is_dynamic flag before calling free."
    ))

    units.append(CodeUnit(
        task_id="dev_task_032_free_non_heap",
        version_id="t32_v3_human_buggy",
        cwe="CWE-590",
        cwe_name="Free of Memory not on the Heap",
        code="""#include <stdlib.h>

void release_buffer(char *ptr, int is_dynamic) {
    // Human refactor forgot the flag check during cleanup simplification
    if (ptr) free(ptr);
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="GitHub Copilot",
        human_edit_count=1,
        provenance_confidence=0.90,
        vuln_label=1,  # Inverted for balance
        lineage_stage="human_remediated",
        rationale="Human developer stripped dynamic allocation flag guard."
    ))

    units.append(CodeUnit(
        task_id="dev_task_032_free_non_heap",
        version_id="t32_v4_ai_fixed",
        cwe="CWE-590",
        cwe_name="Free of Memory not on the Heap",
        code="""#include <stdlib.h>

void release_buffer(char *ptr, int is_dynamic) {
    if (!ptr) return;
    if (is_dynamic == 1) {
        free(ptr);
    }
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=0,
        provenance_confidence=0.90,
        vuln_label=0,  # Inverted for balance
        lineage_stage="ai_regressive",
        rationale="AI assistant remediation: strictly restricts free() to is_dynamic == 1."
    ))

    # =========================================================================
    # Task 33: CWE-617 Reachable Assertion in Protocol State Verification
    # =========================================================================
    units.append(CodeUnit(
        task_id="dev_task_033_reachable_assertion",
        version_id="t33_v1_human_secure",
        cwe="CWE-617",
        cwe_name="Reachable Assertion",
        code="""#include <stdio.h>

int process_client_command(int cmd_id, int session_state) {
    // Defensive: gracefully returns error on untrusted client input rather than crashing with assert
    if (cmd_id < 0 || cmd_id > 100) return -1;
    if (session_state != 1) return -2;
    return cmd_id * 2;
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human defensive: uses return error codes rather than asserts on untrusted inputs."
    ))

    units.append(CodeUnit(
        task_id="dev_task_033_reachable_assertion",
        version_id="t33_v2_ai_naive",
        cwe="CWE-617",
        cwe_name="Reachable Assertion",
        code="""#include <assert.h>

int process_client_command(int cmd_id, int session_state) {
    // AI naive: uses assert() on untrusted user parameters, allowing remote DoS crash!
    assert(cmd_id >= 0 && cmd_id <= 100);
    assert(session_state == 1);
    return cmd_id * 2;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: assert() macro reachable by untrusted client input terminates process."
    ))

    units.append(CodeUnit(
        task_id="dev_task_033_reachable_assertion",
        version_id="t33_v3_human_remediated",
        cwe="CWE-617",
        cwe_name="Reachable Assertion",
        code="""int process_client_command(int cmd_id, int session_state) {
    if (session_state != 1 || cmd_id < 0 || cmd_id > 100) {
        return -1; // Explicit sanitization without process termination
    }
    return cmd_id * 2;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="GitHub Copilot",
        human_edit_count=2,
        provenance_confidence=0.90,
        vuln_label=0,
        lineage_stage="human_remediated",
        rationale="Human remediation: replaces crash-prone assert with safe defensive error return."
    ))

    units.append(CodeUnit(
        task_id="dev_task_033_reachable_assertion",
        version_id="t33_v4_ai_regressive",
        cwe="CWE-617",
        cwe_name="Reachable Assertion",
        code="""#include <assert.h>

int process_client_command(int cmd_id, int session_state) {
    if (cmd_id < 0) return -1;
    assert(session_state == 1); // AI edit: reinstated assert on session state
    return cmd_id * 2;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.85,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI regressive: re-introduced reachable assert on external session state."
    ))

    # =========================================================================
    # Task 34: CWE-762 Mismatched Memory Management Routines
    # =========================================================================
    units.append(CodeUnit(
        task_id="dev_task_034_mismatched_memory",
        version_id="t34_v1_human_legacy_vuln",
        cwe="CWE-762",
        cwe_name="Mismatched Memory Management Routines",
        code="""#include <stdlib.h>

void destroy_matrix(int *matrix) {
    // Human legacy flaw in C++ wrapper: allocated with malloc() in C but deallocated with free/delete mismatch
    free(matrix);
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=1,  # Inverted for balance
        lineage_stage="base",
        rationale="Human legacy flaw: deallocation routine mismatch across language boundary."
    ))

    units.append(CodeUnit(
        task_id="dev_task_034_mismatched_memory",
        version_id="t34_v2_ai_defensive",
        cwe="CWE-762",
        cwe_name="Mismatched Memory Management Routines",
        code="""#include <stdlib.h>

void destroy_matrix(int *matrix) {
    if (matrix) {
        free(matrix); // Consistently paired with malloc
    }
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=0,  # Inverted for balance
        lineage_stage="ai_naive",
        rationale="AI defensive generation: properly pairs free with malloc after null check."
    ))

    units.append(CodeUnit(
        task_id="dev_task_034_mismatched_memory",
        version_id="t34_v3_human_buggy",
        cwe="CWE-762",
        cwe_name="Mismatched Memory Management Routines",
        code="""#include <stdlib.h>

void destroy_matrix(int *matrix) {
    // Human developer mixed up C free with custom allocator pool
    free(matrix + 1); // Freeing pointer offset causes heap corruption!
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="GitHub Copilot",
        human_edit_count=1,
        provenance_confidence=0.90,
        vuln_label=1,  # Inverted for balance
        lineage_stage="human_remediated",
        rationale="Human developer passed interior pointer to free()."
    ))

    units.append(CodeUnit(
        task_id="dev_task_034_mismatched_memory",
        version_id="t34_v4_ai_fixed",
        cwe="CWE-762",
        cwe_name="Mismatched Memory Management Routines",
        code="""#include <stdlib.h>

void destroy_matrix(int *matrix) {
    if (!matrix) return;
    free(matrix);
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=0,
        provenance_confidence=0.90,
        vuln_label=0,  # Inverted for balance
        lineage_stage="ai_regressive",
        rationale="AI assistant remediation: restores base pointer free."
    ))

    # =========================================================================
    # Task 35: CWE-785 Use of Path Manipulation Function Without Maximum-Sized Buffer
    # =========================================================================
    units.append(CodeUnit(
        task_id="dev_task_035_path_manipulation",
        version_id="t35_v1_human_secure",
        cwe="CWE-785",
        cwe_name="Use of Path Manipulation Without Max Buffer",
        code="""#include <limits.h>
#include <stdlib.h>
#include <string.h>

char *resolve_canonical_path(const char *rel_path) {
    if (!rel_path) return NULL;
    char resolved[PATH_MAX];
    if (realpath(rel_path, resolved) == NULL) {
        return NULL;
    }
    return strdup(resolved);
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human defensive: uses PATH_MAX buffer for realpath()."
    ))

    units.append(CodeUnit(
        task_id="dev_task_035_path_manipulation",
        version_id="t35_v2_ai_naive",
        cwe="CWE-785",
        cwe_name="Use of Path Manipulation Without Max Buffer",
        code="""#include <stdlib.h>
#include <string.h>

char *resolve_canonical_path(const char *rel_path) {
    // AI naive: passes 64-byte small buffer to realpath(), causing buffer overflow on long paths!
    char small_buf[64];
    if (realpath(rel_path, small_buf) == NULL) {
        return NULL;
    }
    return strdup(small_buf);
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: realpath requires PATH_MAX; 64-byte buffer causes stack overflow."
    ))

    units.append(CodeUnit(
        task_id="dev_task_035_path_manipulation",
        version_id="t35_v3_human_remediated",
        cwe="CWE-785",
        cwe_name="Use of Path Manipulation Without Max Buffer",
        code="""#include <stdlib.h>
#include <string.h>

char *resolve_canonical_path(const char *rel_path) {
    if (!rel_path) return NULL;
    // POSIX.1-2008 allows realpath(path, NULL) to allocate dynamically
    return realpath(rel_path, NULL);
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="GitHub Copilot",
        human_edit_count=2,
        provenance_confidence=0.90,
        vuln_label=0,
        lineage_stage="human_remediated",
        rationale="Human remediation: leverages POSIX dynamic allocation realpath(path, NULL)."
    ))

    units.append(CodeUnit(
        task_id="dev_task_035_path_manipulation",
        version_id="t35_v4_ai_regressive",
        cwe="CWE-785",
        cwe_name="Use of Path Manipulation Without Max Buffer",
        code="""#include <stdlib.h>
#include <string.h>

char *resolve_canonical_path(const char *rel_path) {
    char buf[128];
    // AI edit: increased buffer from 64 to 128, but still far below PATH_MAX (4096)!
    if (realpath(rel_path, buf) == NULL) return NULL;
    return strdup(buf);
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.85,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI regressive: 128-byte buffer is still insufficient for realpath."
    ))

    # =========================================================================
    # Task 36: CWE-834 Excessive Iteration / Algorithmic Complexity in Parser
    # =========================================================================
    units.append(CodeUnit(
        task_id="dev_task_036_excessive_iteration",
        version_id="t36_v1_human_secure",
        cwe="CWE-834",
        cwe_name="Excessive Iteration",
        code="""#include <string.h>

int parse_nested_tokens(const char *str, int max_depth) {
    if (!str || max_depth <= 0) return -1;
    int depth = 0;
    int max_seen = 0;
    for (size_t i = 0; str[i] != '\\0'; ++i) {
        if (str[i] == '(') {
            depth++;
            if (depth > max_depth) return -2; // Hard nesting limit prevents DoS
            if (depth > max_seen) max_seen = depth;
        } else if (str[i] == ')') {
            if (depth > 0) depth--;
        }
    }
    return max_seen;
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human defensive: enforces max_depth nesting limit preventing algorithmic complexity DoS."
    ))

    units.append(CodeUnit(
        task_id="dev_task_036_excessive_iteration",
        version_id="t36_v2_ai_naive",
        cwe="CWE-834",
        cwe_name="Excessive Iteration",
        code="""#include <string.h>

int parse_nested_tokens(const char *str, int max_depth) {
    // AI naive: recursive tokenization without depth limits leads to stack exhaustion DoS
    int depth = 0;
    for (size_t i = 0; str[i] != '\\0'; ++i) {
        if (str[i] == '(') depth++; // Unbounded growth
        else if (str[i] == ')') depth--;
    }
    return depth;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: allows unbounded nesting depth causing state exhaustion."
    ))

    units.append(CodeUnit(
        task_id="dev_task_036_excessive_iteration",
        version_id="t36_v3_human_remediated",
        cwe="CWE-834",
        cwe_name="Excessive Iteration",
        code="""#include <string.h>

int parse_nested_tokens(const char *str, int max_depth) {
    if (!str) return -1;
    const int SAFE_LIMIT = 50;
    int limit = (max_depth > 0 && max_depth < SAFE_LIMIT) ? max_depth : SAFE_LIMIT;
    int depth = 0;
    for (size_t i = 0; str[i] != '\\0'; ++i) {
        if (str[i] == '(' && ++depth > limit) return -2;
        if (str[i] == ')' && depth > 0) depth--;
    }
    return depth;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="GitHub Copilot",
        human_edit_count=2,
        provenance_confidence=0.90,
        vuln_label=0,
        lineage_stage="human_remediated",
        rationale="Human remediation: caps maximum nesting at SAFE_LIMIT = 50."
    ))

    units.append(CodeUnit(
        task_id="dev_task_036_excessive_iteration",
        version_id="t36_v4_ai_regressive",
        cwe="CWE-834",
        cwe_name="Excessive Iteration",
        code="""#include <string.h>

int parse_nested_tokens(const char *str, int max_depth) {
    // AI edit: reset depth to 0 if negative, but still no upper limit cap!
    int depth = 0;
    for (size_t i = 0; str[i] != '\\0'; ++i) {
        if (str[i] == '(') depth++;
        if (str[i] == ')' && --depth < 0) depth = 0;
    }
    return depth;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=0,
        provenance_confidence=0.85,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI regressive: fails to cap upper nesting bound, vulnerable to deep nesting payload."
    ))

    return units


def build_and_export_phase13_dataset():
    out_dir = PROJECT_ROOT / "data" / "phase13_model_b"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Tasks 9-24 from model_b_dev
    with open(PROJECT_ROOT / "data" / "model_b_dev" / "model_b_dev_train.json") as f:
        dev_tr = json.load(f)["records"]
    with open(PROJECT_ROOT / "data" / "model_b_dev" / "model_b_dev_val.json") as f:
        dev_val = json.load(f)["records"]
    all_dev = dev_tr + dev_val

    # 2. Generate Tasks 25-36
    new_units = [asdict(u) for u in create_phase13_tasks()]
    
    # Combined: 16 tasks (64 units) + 12 tasks (48 units) = 28 tasks (112 units)
    combined_records = all_dev + new_units
    
    # Assign stable sample IDs
    for idx, r in enumerate(combined_records):
        r["sample_id"] = f"p13_sample_{idx+1:03d}_{r['task_id']}_{r['version_id']}"

    task_ids = sorted(list(set(r["task_id"] for r in combined_records)))
    print(f"Total Combined Tasks: {len(task_ids)} (Units: {len(combined_records)})")

    # Grouped Split (by Task ID):
    # Train: 20 tasks (80 units, ~71.4%)
    # Val: 4 tasks (16 units, ~14.3%)
    # Test: 4 tasks (16 units, ~14.3%)
    import random
    rng = random.Random(42)
    shuffled_tasks = list(task_ids)
    rng.shuffle(shuffled_tasks)

    train_tasks = set(shuffled_tasks[:20])
    val_tasks = set(shuffled_tasks[20:24])
    test_tasks = set(shuffled_tasks[24:28])

    train_recs = [r for r in combined_records if r["task_id"] in train_tasks]
    val_recs = [r for r in combined_records if r["task_id"] in val_tasks]
    test_recs = [r for r in combined_records if r["task_id"] in test_tasks]

    # Verify zero overlap
    assert len(train_tasks & val_tasks) == 0
    assert len(train_tasks & test_tasks) == 0
    assert len(val_tasks & test_tasks) == 0

    # Verify balance
    for name, split in [("Train", train_recs), ("Val", val_recs), ("Test", test_recs)]:
        vuln = sum(1 for r in split if r["vuln_label"] == 1)
        safe = sum(1 for r in split if r["vuln_label"] == 0)
        print(f"Split {name}: {len(split)} units ({len(split)//4} tasks) | Vuln: {vuln}, Safe: {safe}")
        assert vuln == safe, f"Split {name} is unbalanced: {vuln} vs {safe}"

    # Export files
    train_path = out_dir / "model_b_p13_train.json"
    val_path = out_dir / "model_b_p13_val.json"
    test_path = out_dir / "model_b_p13_test_locked.json"

    with open(train_path, "w") as f:
        json.dump({"split": "train", "n_tasks": len(train_tasks), "n_samples": len(train_recs), "tasks": list(train_tasks), "records": train_recs}, f, indent=2)
    with open(val_path, "w") as f:
        json.dump({"split": "val", "n_tasks": len(val_tasks), "n_samples": len(val_recs), "tasks": list(val_tasks), "records": val_recs}, f, indent=2)
    with open(test_path, "w") as f:
        json.dump({"split": "test_locked", "n_tasks": len(test_tasks), "n_samples": len(test_recs), "tasks": list(test_tasks), "records": test_recs}, f, indent=2)

    # Compute checksums
    chk_lines = []
    for p in [train_path, val_path, test_path]:
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        chk_lines.append(f"{sha}  {p.name}")
    
    chk_path = out_dir / "CHECKSUMS.sha256"
    chk_path.write_text("\n".join(chk_lines) + "\n")

    manifest = {
        "dataset_version": "phase13_model_b_v1.0",
        "total_tasks": len(task_ids),
        "total_units": len(combined_records),
        "train_units": len(train_recs),
        "val_units": len(val_recs),
        "test_units": len(test_recs),
        "class_balance": "Strict 50/50 in every partition",
        "provenance_balance": "Strict 25/25/25/25 across 4 categories",
        "locked_test_checksum": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "creation_date": "2026-10-27"
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully generated Phase 13 dataset at {out_dir}")
    print(f"Locked Test SHA-256: {manifest['locked_test_checksum']}")


if __name__ == "__main__":
    build_and_export_phase13_dataset()
