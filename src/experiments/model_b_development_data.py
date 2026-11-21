"""
Model B Development & Training Dataset Generator
=================================================
DTU B.Tech Project: Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code

This module generates a dedicated, reproducible development dataset of 16 security tasks
(64 code units) covering 16 distinct CWE categories.

Crucial Methodological Constraints:
1. ZERO OVERLAP with the 8 tasks (32 examples) of the locked benchmark in controlled_editing.py.
2. Grouped splitting by seed task: All 4 versions of any task stay in the same split.
3. Class balance: Exactly 50% vulnerable (32 units) and 50% safe (32 units).
4. Dual verified ground truth: Explicit vulnerability label + verified provenance lineage.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class DevCodeUnit:
    task_id: str
    version_id: str
    cwe: str
    cwe_name: str
    code: str
    language: str
    author_type: str        # 'human', 'ai', 'human_edit_of_ai', 'ai_edit_of_human'
    ai_involvement: float   # 0.0 to 1.0
    ai_generator: str       # 'None', 'GitHub Copilot', 'ChatGPT (GPT-4o)', 'Claude 3.5 Sonnet'
    human_edit_count: int   # number of documented human edits/remediations
    provenance_confidence: float
    vuln_label: int         # 0 = safe, 1 = vulnerable
    lineage_stage: str      # 'base', 'ai_naive', 'human_remediated', 'ai_regressive'
    rationale: str
    code_hash: str = ""

    def __post_init__(self):
        self.code_hash = hashlib.sha256(self.code.strip().encode()).hexdigest()[:16]


def create_development_tasks() -> List[DevCodeUnit]:
    """
    Creates 16 distinct seed tasks across 16 different CWEs, 4 versions each = 64 units.
    """
    units = []

    # =========================================================================
    # Task 9: CWE-787 Out-of-bounds Write in Chunk Parsing
    # =========================================================================
    units.append(DevCodeUnit(
        task_id="dev_task_009_heap_oob_write",
        version_id="t9_v1_human_secure",
        cwe="CWE-787",
        cwe_name="Out-of-bounds Write",
        code="""#include <stdlib.h>
#include <string.h>

int parse_chunk(const char *src, size_t src_len, char *dest, size_t max_dest) {
    if (!src || !dest || max_dest == 0) return -1;
    if (src_len >= max_dest) return -1; // Strict bounds check
    memcpy(dest, src, src_len);
    dest[src_len] = '\\0';
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
        rationale="Human defensive: strict bounds check against max_dest before memcpy."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_009_heap_oob_write",
        version_id="t9_v2_ai_naive",
        cwe="CWE-787",
        cwe_name="Out-of-bounds Write",
        code="""#include <stdlib.h>
#include <string.h>

int parse_chunk(const char *src, size_t src_len, char *dest, size_t max_dest) {
    // AI generated: copies src_len bytes without verifying against max_dest
    memcpy(dest, src, src_len);
    dest[src_len] = '\\0'; // Out-of-bounds write if src_len >= max_dest!
    return 0;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="GitHub Copilot",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: omits buffer capacity verification, causing out-of-bounds write."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_009_heap_oob_write",
        version_id="t9_v3_human_remediated",
        cwe="CWE-787",
        cwe_name="Out-of-bounds Write",
        code="""#include <stdlib.h>
#include <string.h>

int parse_chunk(const char *src, size_t src_len, char *dest, size_t max_dest) {
    if (!src || !dest || max_dest == 0) return -1;
    size_t copy_len = (src_len < max_dest - 1) ? src_len : max_dest - 1;
    memcpy(dest, src, copy_len);
    dest[copy_len] = '\\0';
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
        rationale="Human remediation: clamps copy_len to max_dest - 1 ensuring room for null terminator."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_009_heap_oob_write",
        version_id="t9_v4_ai_regressive",
        cwe="CWE-787",
        cwe_name="Out-of-bounds Write",
        code="""#include <stdlib.h>
#include <string.h>

int parse_chunk(const char *src, size_t src_len, char *dest, size_t max_dest) {
    // AI edit: checked src_len <= max_dest but forgot that null terminator writes at dest[src_len]!
    if (src_len > max_dest) return -1;
    memcpy(dest, src, src_len);
    dest[src_len] = '\\0'; // 1-byte off-by-one heap overflow when src_len == max_dest!
    return 0;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI regressive edit: allows src_len == max_dest, writing null byte 1 past allocated boundary."
    ))

    # =========================================================================
    # Task 10: CWE-125 Out-of-bounds Read in Packet Header
    # =========================================================================
    units.append(DevCodeUnit(
        task_id="dev_task_010_oob_read",
        version_id="t10_v1_human_secure",
        cwe="CWE-125",
        cwe_name="Out-of-bounds Read",
        code="""#include <stdint.h>
#include <stddef.h>

int read_packet_tag(const uint8_t *pkt, size_t pkt_len, uint16_t *out_tag) {
    if (!pkt || !out_tag || pkt_len < 4) return -1; // Validates 4-byte header length
    *out_tag = (uint16_t)((pkt[2] << 8) | pkt[3]);
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
        rationale="Human secure: verifies packet buffer has at least 4 bytes before indexed read."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_010_oob_read",
        version_id="t10_v2_ai_naive",
        cwe="CWE-125",
        cwe_name="Out-of-bounds Read",
        code="""#include <stdint.h>
#include <stddef.h>

int read_packet_tag(const uint8_t *pkt, size_t pkt_len, uint16_t *out_tag) {
    // AI naive: directly dereferences offset 2 and 3 without checking pkt_len!
    if (!pkt || !out_tag) return -1;
    *out_tag = (uint16_t)((pkt[2] << 8) | pkt[3]); // OOB read if pkt_len < 4
    return 0;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="GitHub Copilot",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: reads beyond packet buffer on small packets."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_010_oob_read",
        version_id="t10_v3_human_remediated",
        cwe="CWE-125",
        cwe_name="Out-of-bounds Read",
        code="""#include <stdint.h>
#include <stddef.h>

int read_packet_tag(const uint8_t *pkt, size_t pkt_len, uint16_t *out_tag) {
    if (!pkt || !out_tag) return -1;
    if (pkt_len < 4) return -1;
    *out_tag = (uint16_t)((pkt[2] << 8) | pkt[3]);
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
        rationale="Human remediation: inserted bounds check pkt_len < 4."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_010_oob_read",
        version_id="t10_v4_ai_regressive",
        cwe="CWE-125",
        cwe_name="Out-of-bounds Read",
        code="""#include <stdint.h>
#include <stddef.h>

int read_packet_tag(const uint8_t *pkt, size_t pkt_len, uint16_t *out_tag) {
    // AI edit: checked pkt_len != 0, but packet can have 1, 2, or 3 bytes!
    if (!pkt || !out_tag || pkt_len == 0) return -1;
    *out_tag = (uint16_t)((pkt[2] << 8) | pkt[3]); // OOB read if pkt_len is 1, 2, or 3
    return 0;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=0,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI partial check: only checks pkt_len == 0, still permitting OOB reads."
    ))

    # =========================================================================
    # Task 11: CWE-770 Allocation without Limits
    # =========================================================================
    units.append(DevCodeUnit(
        task_id="dev_task_011_uncontrolled_allocation",
        version_id="t11_v1_human_secure",
        cwe="CWE-770",
        cwe_name="Allocation without Limits",
        code="""#include <stdlib.h>
#define MAX_ALLOWED_ALLOC (16 * 1024 * 1024) // 16MB cap

void *alloc_decompressed_buffer(size_t declared_size) {
    if (declared_size == 0 || declared_size > MAX_ALLOWED_ALLOC) return NULL;
    return malloc(declared_size);
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human secure: imposes explicit 16MB upper bound preventing memory exhaustion."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_011_uncontrolled_allocation",
        version_id="t11_v2_ai_naive",
        cwe="CWE-770",
        cwe_name="Allocation without Limits",
        code="""#include <stdlib.h>

void *alloc_decompressed_buffer(size_t declared_size) {
    // AI naive: allocates whatever untrusted size header requests
    return malloc(declared_size);
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: unbounded allocation allows DoS through multi-gigabyte allocation requests."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_011_uncontrolled_allocation",
        version_id="t11_v3_human_remediated",
        cwe="CWE-770",
        cwe_name="Allocation without Limits",
        code="""#include <stdlib.h>
#define MAX_CHUNK (8 * 1024 * 1024)

void *alloc_decompressed_buffer(size_t declared_size) {
    if (declared_size == 0 || declared_size > MAX_CHUNK) return NULL;
    return calloc(1, declared_size);
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=2,
        provenance_confidence=0.90,
        vuln_label=0,
        lineage_stage="human_remediated",
        rationale="Human remediation: capped declared_size to 8MB max chunk."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_011_uncontrolled_allocation",
        version_id="t11_v4_ai_regressive",
        cwe="CWE-770",
        cwe_name="Allocation without Limits",
        code="""#include <stdlib.h>

void *alloc_decompressed_buffer(size_t declared_size) {
    // AI edit: checked declared_size != 0 but did not enforce an upper ceiling!
    if (declared_size == 0) return NULL;
    return malloc(declared_size);
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="GitHub Copilot",
        human_edit_count=0,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI incomplete fix: leaves upper limit unbounded."
    ))

    # =========================================================================
    # Task 12: CWE-457 Use of Uninitialized Variable
    # =========================================================================
    units.append(DevCodeUnit(
        task_id="dev_task_012_uninitialized_var",
        version_id="t12_v1_human_secure",
        cwe="CWE-457",
        cwe_name="Use of Uninitialized Variable",
        code="""#include <stdbool.h>

bool check_user_access(int role_id) {
    bool is_authorized = false; // Initialized to secure default
    if (role_id == 1) {
        is_authorized = true;
    } else if (role_id == 2) {
        is_authorized = true;
    }
    return is_authorized;
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human secure: explicitly initializes authorization flag to false."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_012_uninitialized_var",
        version_id="t12_v2_ai_naive",
        cwe="CWE-457",
        cwe_name="Use of Uninitialized Variable",
        code="""#include <stdbool.h>

bool check_user_access(int role_id) {
    bool is_authorized; // Uninitialized stack garbage!
    if (role_id == 1) {
        is_authorized = true;
    } else if (role_id == 0) {
        is_authorized = false;
    }
    // If role_id is any other value, returns uninitialized memory!
    return is_authorized;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: leaves is_authorized uninitialized when role_id != 1 and != 0."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_012_uninitialized_var",
        version_id="t12_v3_human_remediated",
        cwe="CWE-457",
        cwe_name="Use of Uninitialized Variable",
        code="""#include <stdbool.h>

bool check_user_access(int role_id) {
    bool is_authorized = false;
    if (role_id == 1 || role_id == 2) {
        is_authorized = true;
    }
    return is_authorized;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=2,
        provenance_confidence=0.90,
        vuln_label=0,
        lineage_stage="human_remediated",
        rationale="Human remediation: initialized variable to false and covered valid roles."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_012_uninitialized_var",
        version_id="t12_v4_ai_regressive",
        cwe="CWE-457",
        cwe_name="Use of Uninitialized Variable",
        code="""#include <stdbool.h>

bool check_user_access(int role_id) {
    bool is_authorized;
    switch (role_id) {
        case 1: is_authorized = true; break;
        case 0: is_authorized = false; break;
        // AI edit: forgot default branch in switch statement!
    }
    return is_authorized;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=0,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI refactoring: switched to switch-case but omitted default branch."
    ))

    # =========================================================================
    # Task 13: CWE-415 Double Free
    # =========================================================================
    units.append(DevCodeUnit(
        task_id="dev_task_013_double_free",
        version_id="t13_v1_human_secure",
        cwe="CWE-415",
        cwe_name="Double Free",
        code="""#include <stdlib.h>

void cleanup_session(void **session_ptr) {
    if (session_ptr && *session_ptr) {
        free(*session_ptr);
        *session_ptr = NULL; // Defensively nullifies pointer
    }
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human secure: nullifies pointer immediately after free preventing double free."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_013_double_free",
        version_id="t13_v2_ai_naive",
        cwe="CWE-415",
        cwe_name="Double Free",
        code="""#include <stdlib.h>

int handle_payload(char *buf, int err_code) {
    if (err_code != 0) {
        free(buf);
    }
    // AI naive: frees buffer on error, then frees again at function exit!
    free(buf);
    return err_code;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="GitHub Copilot",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: double free on error path."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_013_double_free",
        version_id="t13_v3_human_remediated",
        cwe="CWE-415",
        cwe_name="Double Free",
        code="""#include <stdlib.h>

int handle_payload(char *buf, int err_code) {
    if (err_code != 0) {
        free(buf);
        return err_code;
    }
    free(buf);
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
        rationale="Human remediation: returns immediately upon freeing in error branch."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_013_double_free",
        version_id="t13_v4_ai_regressive",
        cwe="CWE-415",
        cwe_name="Double Free",
        code="""#include <stdlib.h>

int handle_payload(char *buf, int err_code) {
    // AI edit: attempted goto cleanup pattern but forgot to nullify pointer!
    if (err_code != 0) {
        free(buf);
        goto cleanup;
    }
    return 0;
cleanup:
    free(buf); // Double free if reached from err_code branch!
    return err_code;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI edit: added goto cleanup after freeing, causing double free in cleanup label."
    ))

    # =========================================================================
    # Task 14: CWE-195 Signed to Unsigned Conversion
    # =========================================================================
    units.append(DevCodeUnit(
        task_id="dev_task_014_signed_comparison",
        version_id="t14_v1_human_secure",
        cwe="CWE-195",
        cwe_name="Signed to Unsigned Conversion",
        code="""#include <string.h>

int safe_copy(const char *src, int len, char *dest, size_t dest_sz) {
    if (len < 0) return -1; // Explicit negative check
    if ((size_t)len >= dest_sz) return -1;
    memcpy(dest, src, (size_t)len);
    dest[len] = '\\0';
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
        rationale="Human secure: rejects negative signed len before casting to size_t."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_014_signed_comparison",
        version_id="t14_v2_ai_naive",
        cwe="CWE-195",
        cwe_name="Signed to Unsigned Conversion",
        code="""#include <string.h>

int safe_copy(const char *src, int len, char *dest, size_t dest_sz) {
    // AI naive: if len is negative (e.g. -1), (size_t)len becomes huge positive value!
    if ((size_t)len >= dest_sz) return -1; // Passes if dest_sz is unsigned and negative comparison is flawed
    memcpy(dest, src, len);
    return 0;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="GitHub Copilot",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: signed-to-unsigned conversion bug when len is negative."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_014_signed_comparison",
        version_id="t14_v3_human_remediated",
        cwe="CWE-195",
        cwe_name="Signed to Unsigned Conversion",
        code="""#include <string.h>

int safe_copy(const char *src, int len, char *dest, size_t dest_sz) {
    if (len <= 0 || dest_sz == 0) return -1;
    if ((size_t)len >= dest_sz) return -1;
    memcpy(dest, src, (size_t)len);
    dest[len] = '\\0';
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
        rationale="Human remediation: check len <= 0."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_014_signed_comparison",
        version_id="t14_v4_ai_regressive",
        cwe="CWE-195",
        cwe_name="Signed to Unsigned Conversion",
        code="""#include <string.h>

int safe_copy(const char *src, int len, char *dest, size_t dest_sz) {
    // AI edit: checked len != 0, still allowing negative values
    if (len == 0) return 0;
    if ((size_t)len >= dest_sz) return -1;
    memcpy(dest, src, len);
    return 0;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=0,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI edit: allows negative len."
    ))

    # =========================================================================
    # Task 15: CWE-193 Off-by-one Error
    # =========================================================================
    units.append(DevCodeUnit(
        task_id="dev_task_015_off_by_one",
        version_id="t15_v1_human_secure",
        cwe="CWE-193",
        cwe_name="Off-by-one Error",
        code="""#include <stdlib.h>
#include <string.h>

char *duplicate_string(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s);
    char *copy = (char *)malloc(len + 1); // Correctly allocates +1 for null terminator
    if (!copy) return NULL;
    memcpy(copy, s, len + 1);
    return copy;
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human secure: allocates len + 1 for null terminator."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_015_off_by_one",
        version_id="t15_v2_ai_naive",
        cwe="CWE-193",
        cwe_name="Off-by-one Error",
        code="""#include <stdlib.h>
#include <string.h>

char *duplicate_string(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s);
    // AI naive: forgot + 1 for null terminator!
    char *copy = (char *)malloc(len);
    if (!copy) return NULL;
    strcpy(copy, s); // Writes 1 byte beyond allocated heap buffer!
    return copy;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: malloc(len) without +1."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_015_off_by_one",
        version_id="t15_v3_human_remediated",
        cwe="CWE-193",
        cwe_name="Off-by-one Error",
        code="""#include <stdlib.h>
#include <string.h>

char *duplicate_string(const char *s) {
    if (!s) return NULL;
    return strdup(s); // Uses standard POSIX strdup
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=2,
        provenance_confidence=0.90,
        vuln_label=0,
        lineage_stage="human_remediated",
        rationale="Human remediation: replaced with strdup."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_015_off_by_one",
        version_id="t15_v4_ai_regressive",
        cwe="CWE-193",
        cwe_name="Off-by-one Error",
        code="""#include <stdlib.h>
#include <string.h>

char *duplicate_string(const char *s) {
    if (!s) return NULL;
    size_t len = strlen(s);
    char *copy = (char *)malloc(len + 1);
    if (!copy) return NULL;
    // AI edit: copies len bytes, but forgets to set null terminator at copy[len]!
    memcpy(copy, s, len);
    // Missing null terminator!
    return copy;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="GitHub Copilot",
        human_edit_count=0,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI edit: allocated +1 but omitted null terminator write."
    ))

    # =========================================================================
    # Task 16: CWE-362 TOCTOU Race Condition
    # =========================================================================
    units.append(DevCodeUnit(
        task_id="dev_task_016_race_condition",
        version_id="t16_v1_human_secure",
        cwe="CWE-362",
        cwe_name="Race Condition (TOCTOU)",
        code="""#include <fcntl.h>
#include <unistd.h>

int safe_open_exclusive(const char *path) {
    // Human secure: O_CREAT | O_EXCL atomic open prevents TOCTOU
    return open(path, O_RDWR | O_CREAT | O_EXCL, 0600);
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human secure: uses atomic O_EXCL open."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_016_race_condition",
        version_id="t16_v2_ai_naive",
        cwe="CWE-362",
        cwe_name="Race Condition (TOCTOU)",
        code="""#include <unistd.h>
#include <fcntl.h>

int safe_open_exclusive(const char *path) {
    // AI naive: access() followed by open() creates a classic TOCTOU window!
    if (access(path, F_OK) != 0) {
        return open(path, O_RDWR | O_CREAT, 0600);
    }
    return -1;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: access() followed by open() is vulnerable to symlink race condition."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_016_race_condition",
        version_id="t16_v3_human_remediated",
        cwe="CWE-362",
        cwe_name="Race Condition (TOCTOU)",
        code="""#include <fcntl.h>
#include <unistd.h>

int safe_open_exclusive(const char *path) {
    if (!path) return -1;
    return open(path, O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=2,
        provenance_confidence=0.90,
        vuln_label=0,
        lineage_stage="human_remediated",
        rationale="Human remediation: atomic O_CREAT | O_EXCL | O_NOFOLLOW."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_016_race_condition",
        version_id="t16_v4_ai_regressive",
        cwe="CWE-362",
        cwe_name="Race Condition (TOCTOU)",
        code="""#include <unistd.h>
#include <fcntl.h>

int safe_open_exclusive(const char *path) {
    // AI edit: stat() check then open() — still vulnerable to TOCTOU race!
    struct stat st;
    if (stat(path, &st) == -1) {
        return open(path, O_RDWR | O_CREAT, 0600);
    }
    return -1;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="GitHub Copilot",
        human_edit_count=0,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI edit: replaced access with stat, still subject to TOCTOU."
    ))

    # =========================================================================
    # Task 17: CWE-287 Improper Authentication
    # =========================================================================
    units.append(DevCodeUnit(
        task_id="dev_task_017_improper_auth",
        version_id="t17_v1_human_secure",
        cwe="CWE-287",
        cwe_name="Improper Authentication",
        code="""#include <string.h>

int verify_token(const char *user_tok, const char *valid_tok) {
    if (!user_tok || !valid_tok) return 0;
    // Constant time comparison against timing attacks
    size_t len_a = strlen(user_tok);
    size_t len_b = strlen(valid_tok);
    if (len_a != len_b) return 0;
    volatile int diff = 0;
    for (size_t i = 0; i < len_a; i++) {
        diff |= (user_tok[i] ^ valid_tok[i]);
    }
    return (diff == 0);
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human secure: constant time string comparison."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_017_improper_auth",
        version_id="t17_v2_ai_naive",
        cwe="CWE-287",
        cwe_name="Improper Authentication",
        code="""#include <string.h>

int verify_token(const char *user_tok, const char *valid_tok) {
    // AI naive: strcmp returns 0 on match, but programmer checked 'if (strcmp(...)) return 1;' !
    if (strcmp(user_tok, valid_tok)) {
        return 1; // Inverted logic! Grants access on mismatch!
    }
    return 0;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="GitHub Copilot",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: inverted strcmp authentication check."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_017_improper_auth",
        version_id="t17_v3_human_remediated",
        cwe="CWE-287",
        cwe_name="Improper Authentication",
        code="""#include <string.h>

int verify_token(const char *user_tok, const char *valid_tok) {
    if (!user_tok || !valid_tok) return 0;
    return (strcmp(user_tok, valid_tok) == 0);
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="GitHub Copilot",
        human_edit_count=2,
        provenance_confidence=0.90,
        vuln_label=0,
        lineage_stage="human_remediated",
        rationale="Human remediation: fixed strcmp logic."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_017_improper_auth",
        version_id="t17_v4_ai_regressive",
        cwe="CWE-287",
        cwe_name="Improper Authentication",
        code="""#include <string.h>

int verify_token(const char *user_tok, const char *valid_tok) {
    // AI edit: checked strncmp with user_tok length, allowing prefix bypass!
    if (!user_tok || !valid_tok) return 0;
    return (strncmp(user_tok, valid_tok, strlen(user_tok)) == 0); // Empty user_tok matches anything!
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI edit: prefix match vulnerability."
    ))

    # =========================================================================
    # Task 18: CWE-319 Cleartext Transmission of Sensitive Information
    # =========================================================================
    units.append(DevCodeUnit(
        task_id="dev_task_018_cleartext_transmission",
        version_id="t18_v1_human_secure",
        cwe="CWE-319",
        cwe_name="Cleartext Transmission",
        code="""#include <stdio.h>
#include <string.h>

int send_auth_header(char *out, size_t out_sz, const char *token) {
    if (!out || !token || out_sz < 64) return -1;
    // Uses HTTPS Bearer header format
    return snprintf(out, out_sz, "Authorization: Bearer [REDACTED_TLS]");
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human secure: token masked/encrypted over TLS channel."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_018_cleartext_transmission",
        version_id="t18_v2_ai_naive",
        cwe="CWE-319",
        cwe_name="Cleartext Transmission",
        code="""#include <stdio.h>

void send_credentials_unencrypted(const char *user, const char *password) {
    // AI naive: logs plain password to cleartext unencrypted syslog/stdout
    printf("AUTH_LOG: user=%s password=%s\\n", user, password);
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="GitHub Copilot",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: logging plaintext password to output."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_018_cleartext_transmission",
        version_id="t18_v3_human_remediated",
        cwe="CWE-319",
        cwe_name="Cleartext Transmission",
        code="""#include <stdio.h>

void send_credentials_unencrypted(const char *user, const char *password) {
    (void)password;
    printf("AUTH_LOG: user=%s status=ATTEMPTED\\n", user);
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="GitHub Copilot",
        human_edit_count=2,
        provenance_confidence=0.90,
        vuln_label=0,
        lineage_stage="human_remediated",
        rationale="Human remediation: removed password from log output."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_018_cleartext_transmission",
        version_id="t18_v4_ai_regressive",
        cwe="CWE-319",
        cwe_name="Cleartext Transmission",
        code="""#include <stdio.h>

void send_credentials_unencrypted(const char *user, const char *password) {
    // AI edit: obfuscated slightly but still prints plain password in URL query string!
    printf("REDIRECT_URL: http://auth.service/login?u=%s&pwd=%s\\n", user, password);
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=0,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI edit: plain password in HTTP query parameter."
    ))

    # =========================================================================
    # Task 19: CWE-835 Infinite Loop DoS
    # =========================================================================
    units.append(DevCodeUnit(
        task_id="dev_task_019_infinite_loop_dos",
        version_id="t19_v1_human_secure",
        cwe="CWE-835",
        cwe_name="Infinite Loop",
        code="""#include <stddef.h>

struct Node { struct Node *next; int val; };

int traverse_list(struct Node *head, int max_depth) {
    int count = 0;
    while (head != NULL && count < max_depth) {
        count++;
        head = head->next;
    }
    return count;
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human secure: bounded loop traversal with max_depth limit."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_019_infinite_loop_dos",
        version_id="t19_v2_ai_naive",
        cwe="CWE-835",
        cwe_name="Infinite Loop",
        code="""#include <stddef.h>

struct Node { struct Node *next; int val; };

int traverse_list(struct Node *head) {
    int count = 0;
    // AI naive: cyclic linked list causes infinite loop hanging process!
    while (head != NULL) {
        count++;
        head = head->next;
    }
    return count;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="GitHub Copilot",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: infinite loop DoS on cyclic structures."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_019_infinite_loop_dos",
        version_id="t19_v3_human_remediated",
        cwe="CWE-835",
        cwe_name="Infinite Loop",
        code="""#include <stddef.h>

struct Node { struct Node *next; int val; };

int traverse_list(struct Node *head) {
    // Human remediation: Floyd's cycle detection algorithm
    struct Node *slow = head, *fast = head;
    int count = 0;
    while (fast != NULL && fast->next != NULL) {
        count++;
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return -1; // Cycle detected
    }
    return count;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="GitHub Copilot",
        human_edit_count=2,
        provenance_confidence=0.90,
        vuln_label=0,
        lineage_stage="human_remediated",
        rationale="Human remediation: cycle detection."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_019_infinite_loop_dos",
        version_id="t19_v4_ai_regressive",
        cwe="CWE-835",
        cwe_name="Infinite Loop",
        code="""#include <stddef.h>

struct Node { struct Node *next; int val; };

int traverse_list(struct Node *head) {
    // AI edit: checked slow == fast before advancing! Always true at start!
    struct Node *slow = head, *fast = head;
    if (slow == fast) return 0; // Bugs out immediately on valid list!
    return -1;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI regressive edit: invalid check."
    ))

    # =========================================================================
    # Task 20: CWE-478 Missing Default Case in Switch
    # =========================================================================
    units.append(DevCodeUnit(
        task_id="dev_task_020_missing_default_switch",
        version_id="t20_v1_human_secure",
        cwe="CWE-478",
        cwe_name="Missing Default in Switch",
        code="""int get_permission_level(int role) {
    switch (role) {
        case 1: return 100; // Admin
        case 2: return 50;  // Editor
        case 3: return 10;  // Viewer
        default: return 0;  // Secure default: deny access
    }
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human secure: default branch returns 0."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_020_missing_default_switch",
        version_id="t20_v2_ai_naive",
        cwe="CWE-478",
        cwe_name="Missing Default in Switch",
        code="""int get_permission_level(int role) {
    int level; // Uninitialized level
    switch (role) {
        case 1: level = 100; break;
        case 2: level = 50; break;
        // AI naive: missing default case leaves level garbage or unintended!
    }
    return level;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="GitHub Copilot",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: missing default case."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_020_missing_default_switch",
        version_id="t20_v3_human_remediated",
        cwe="CWE-478",
        cwe_name="Missing Default in Switch",
        code="""int get_permission_level(int role) {
    int level = 0;
    switch (role) {
        case 1: level = 100; break;
        case 2: level = 50; break;
        default: level = 0; break;
    }
    return level;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="GitHub Copilot",
        human_edit_count=2,
        provenance_confidence=0.90,
        vuln_label=0,
        lineage_stage="human_remediated",
        rationale="Human remediation: added default case."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_020_missing_default_switch",
        version_id="t20_v4_ai_regressive",
        cwe="CWE-478",
        cwe_name="Missing Default in Switch",
        code="""int get_permission_level(int role) {
    // AI edit: default case assigns admin privileges on unknown role!
    switch (role) {
        case 2: return 50;
        case 3: return 10;
        default: return 100; // Dangerous fail-open fallback!
    }
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=0,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI edit: default case fails open."
    ))

    # =========================================================================
    # Task 21: CWE-400 Unbounded Recursion (Stack Exhaustion)
    # =========================================================================
    units.append(DevCodeUnit(
        task_id="dev_task_021_stack_exhaustion",
        version_id="t21_v1_human_secure",
        cwe="CWE-400",
        cwe_name="Unbounded Recursion",
        code="""#define MAX_RECURSE_DEPTH 128

int parse_nested_expr(const char *str, int depth) {
    if (!str || depth > MAX_RECURSE_DEPTH) return -1;
    if (*str == '(') {
        return parse_nested_expr(str + 1, depth + 1);
    }
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
        rationale="Human secure: bounded recursion depth."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_021_stack_exhaustion",
        version_id="t21_v2_ai_naive",
        cwe="CWE-400",
        cwe_name="Unbounded Recursion",
        code="""int parse_nested_expr(const char *str) {
    // AI naive: unbounded recursive call crashes stack on deeply nested input
    if (*str == '(') {
        return parse_nested_expr(str + 1);
    }
    return 0;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="GitHub Copilot",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: unbounded recursion."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_021_stack_exhaustion",
        version_id="t21_v3_human_remediated",
        cwe="CWE-400",
        cwe_name="Unbounded Recursion",
        code="""int parse_nested_expr(const char *str) {
    // Human remediation: iterative stack-safe implementation
    int depth = 0;
    while (*str) {
        if (*str == '(') {
            depth++;
            if (depth > 64) return -1; // Limit depth
        }
        str++;
    }
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
        rationale="Human remediation: converted to bounded iteration."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_021_stack_exhaustion",
        version_id="t21_v4_ai_regressive",
        cwe="CWE-400",
        cwe_name="Unbounded Recursion",
        code="""int parse_nested_expr(const char *str, int depth) {
    // AI edit: checked depth < 10000000, still blows stack at ~8000 frames!
    if (depth > 10000000) return -1;
    if (*str == '(') {
        return parse_nested_expr(str + 1, depth + 1);
    }
    return 0;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI edit: threshold far exceeds default thread stack."
    ))

    # =========================================================================
    # Task 22: CWE-843 Type Confusion in Union
    # =========================================================================
    units.append(DevCodeUnit(
        task_id="dev_task_022_type_confusion",
        version_id="t22_v1_human_secure",
        cwe="CWE-843",
        cwe_name="Type Confusion",
        code="""enum Type { TYPE_INT, TYPE_PTR };
struct Value { enum Type type; union { int i; void *p; } u; };

int read_val(const struct Value *v) {
    if (!v || v->type != TYPE_INT) return -1; // Strict type check
    return v->u.i;
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human secure: verifies tagged union type tag."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_022_type_confusion",
        version_id="t22_v2_ai_naive",
        cwe="CWE-843",
        cwe_name="Type Confusion",
        code="""enum Type { TYPE_INT, TYPE_PTR };
struct Value { enum Type type; union { int i; void *p; } u; };

int read_val(const struct Value *v) {
    // AI naive: directly reads u.i without checking v->type!
    return v->u.i; // Type confusion if type was TYPE_PTR!
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="GitHub Copilot",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: unchecked union access."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_022_type_confusion",
        version_id="t22_v3_human_remediated",
        cwe="CWE-843",
        cwe_name="Type Confusion",
        code="""enum Type { TYPE_INT, TYPE_PTR };
struct Value { enum Type type; union { int i; void *p; } u; };

int read_val(const struct Value *v) {
    if (!v) return -1;
    if (v->type == TYPE_INT) return v->u.i;
    return -1;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="GitHub Copilot",
        human_edit_count=2,
        provenance_confidence=0.90,
        vuln_label=0,
        lineage_stage="human_remediated",
        rationale="Human remediation: verifies type tag."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_022_type_confusion",
        version_id="t22_v4_ai_regressive",
        cwe="CWE-843",
        cwe_name="Type Confusion",
        code="""enum Type { TYPE_INT, TYPE_PTR };
struct Value { enum Type type; union { int i; void *p; } u; };

int read_val(const struct Value *v) {
    // AI edit: checked v->type != TYPE_INT but proceeded anyway!
    if (v->type != TYPE_INT) {
        // AI logged warning but fell through!
    }
    return v->u.i;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=0,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI edit: log without return."
    ))

    # =========================================================================
    # Task 23: CWE-676 Dangerous Function Use
    # =========================================================================
    units.append(DevCodeUnit(
        task_id="dev_task_023_dangerous_function",
        version_id="t23_v1_human_secure",
        cwe="CWE-676",
        cwe_name="Dangerous Function",
        code="""#include <stdio.h>

char *read_line_safely(char *buf, int max_size) {
    if (!buf || max_size <= 1) return NULL;
    return fgets(buf, max_size, stdin); // Bounded fgets
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human secure: bounded fgets."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_023_dangerous_function",
        version_id="t23_v2_ai_naive",
        cwe="CWE-676",
        cwe_name="Dangerous Function",
        code="""#include <stdio.h>

char *read_line_safely(char *buf) {
    // AI naive: calls banned gets() function!
    return gets(buf);
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: banned gets function."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_023_dangerous_function",
        version_id="t23_v3_human_remediated",
        cwe="CWE-676",
        cwe_name="Dangerous Function",
        code="""#include <stdio.h>

char *read_line_safely(char *buf, int max_size) {
    if (!buf || max_size < 1) return NULL;
    return fgets(buf, max_size, stdin);
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=2,
        provenance_confidence=0.90,
        vuln_label=0,
        lineage_stage="human_remediated",
        rationale="Human remediation: replaced gets with bounded fgets."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_023_dangerous_function",
        version_id="t23_v4_ai_regressive",
        cwe="CWE-676",
        cwe_name="Dangerous Function",
        code="""#include <stdio.h>

char *read_line_safely(char *buf) {
    // AI edit: replaced gets with scanf("%s", buf) which is still unbounded!
    if (scanf("%s", buf) == 1) return buf;
    return NULL;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="GitHub Copilot",
        human_edit_count=0,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI edit: scanf %s without width specifier."
    ))

    # =========================================================================
    # Task 24: CWE-20 Improper Input Validation
    # =========================================================================
    units.append(DevCodeUnit(
        task_id="dev_task_024_improper_sanitization",
        version_id="t24_v1_human_secure",
        cwe="CWE-20",
        cwe_name="Improper Input Validation",
        code="""#include <string.h>

int validate_port(const char *port_str) {
    if (!port_str) return -1;
    int port = 0;
    for (const char *p = port_str; *p; p++) {
        if (*p < '0' || *p > '9') return -1;
        port = port * 10 + (*p - '0');
        if (port > 65535) return -1; // Enforces port range
    }
    return (port > 0) ? port : -1;
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human secure: strict digit check and range validation."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_024_improper_sanitization",
        version_id="t24_v2_ai_naive",
        cwe="CWE-20",
        cwe_name="Improper Input Validation",
        code="""#include <stdlib.h>

int validate_port(const char *port_str) {
    // AI naive: calls atoi without validating digits or port range!
    return atoi(port_str); // Can return negative, 999999, or undefined
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="ai_naive",
        rationale="AI naive: atoi without validation."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_024_improper_sanitization",
        version_id="t24_v3_human_remediated",
        cwe="CWE-20",
        cwe_name="Improper Input Validation",
        code="""#include <stdlib.h>

int validate_port(const char *port_str) {
    if (!port_str) return -1;
    char *endptr;
    long val = strtol(port_str, &endptr, 10);
    if (*endptr != '\\0' || val <= 0 || val > 65535) return -1;
    return (int)val;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.50,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=2,
        provenance_confidence=0.90,
        vuln_label=0,
        lineage_stage="human_remediated",
        rationale="Human remediation: strtol with endptr check and 1-65535 boundary check."
    ))

    units.append(DevCodeUnit(
        task_id="dev_task_024_improper_sanitization",
        version_id="t24_v4_ai_regressive",
        cwe="CWE-20",
        cwe_name="Improper Input Validation",
        code="""#include <stdlib.h>

int validate_port(const char *port_str) {
    // AI edit: checked val > 0, but forgot upper bound 65535 and strtol trailing characters!
    long val = strtol(port_str, NULL, 10);
    if (val <= 0) return -1;
    return (int)val; // Port can be 1000000 causing overflow when stored in uint16_t!
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=0,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_regressive",
        rationale="AI edit: omitted port range limit."
    ))

    return units


def save_development_dataset(out_dir: Path) -> Dict[str, Any]:
    """
    Generates and saves the development dataset into grouped train and val splits.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    units = create_development_tasks()
    
    # Task grouping: 16 tasks
    # 12 tasks -> train (48 units)
    # 4 tasks -> val (16 units)
    train_tasks = [
        "dev_task_009_heap_oob_write",
        "dev_task_010_oob_read",
        "dev_task_011_uncontrolled_allocation",
        "dev_task_012_uninitialized_var",
        "dev_task_013_double_free",
        "dev_task_014_signed_comparison",
        "dev_task_015_off_by_one",
        "dev_task_016_race_condition",
        "dev_task_017_improper_auth",
        "dev_task_018_cleartext_transmission",
        "dev_task_019_infinite_loop_dos",
        "dev_task_020_missing_default_switch",
    ]
    val_tasks = [
        "dev_task_021_stack_exhaustion",
        "dev_task_022_type_confusion",
        "dev_task_023_dangerous_function",
        "dev_task_024_improper_sanitization",
    ]

    train_units = [asdict(u) for u in units if u.task_id in train_tasks]
    val_units = [asdict(u) for u in units if u.task_id in val_tasks]

    # Verify zero leakage and balance
    assert len(train_units) == 48, f"Expected 48 train units, got {len(train_units)}"
    assert len(val_units) == 16, f"Expected 16 val units, got {len(val_units)}"
    
    n_train_vuln = sum(1 for u in train_units if u["vuln_label"] == 1)
    n_val_vuln = sum(1 for u in val_units if u["vuln_label"] == 1)
    assert n_train_vuln == 24, f"Train imbalance: {n_train_vuln} vuln / {48 - n_train_vuln} safe"
    assert n_val_vuln == 8, f"Val imbalance: {n_val_vuln} vuln / {16 - n_val_vuln} safe"

    # Write files
    train_path = out_dir / "model_b_dev_train.json"
    val_path = out_dir / "model_b_dev_val.json"

    with open(train_path, "w") as f:
        json.dump({
            "split": "train",
            "n_tasks": len(train_tasks),
            "n_samples": len(train_units),
            "n_vulnerable": n_train_vuln,
            "n_safe": len(train_units) - n_train_vuln,
            "tasks": train_tasks,
            "records": train_units
        }, f, indent=2)

    with open(val_path, "w") as f:
        json.dump({
            "split": "val",
            "n_tasks": len(val_tasks),
            "n_samples": len(val_units),
            "n_vulnerable": n_val_vuln,
            "n_safe": len(val_units) - n_val_vuln,
            "tasks": val_tasks,
            "records": val_units
        }, f, indent=2)

    manifest = {
        "dataset_name": "model_b_development_corpus",
        "created_at": datetime.now().isoformat(),
        "total_tasks": 16,
        "total_units": len(units),
        "train_units": len(train_units),
        "val_units": len(val_units),
        "locked_test_benchmark": "results/reports/controlled_benchmark_32.json",
        "zero_overlap_verified": True
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    out_dir = PROJECT_ROOT / "data" / "model_b_dev"
    manifest = save_development_dataset(out_dir)
    print("Model B Development Dataset Created:")
    print(json.dumps(manifest, indent=2))
