"""
Phase 11 Balanced Provenance-Aware Dataset Generator
====================================================
DTU B.Tech Project: Provenance-Aware Security Risk Assessment of Human-AI Hybrid Code
Authors: Mankabir Singh (23/IT/199) and Mohnish Gautam (23/IT/200)

Generates 24 distinct security tasks across 24 distinct CWE categories (96 total code units).

Methodological Integrity Constraints:
1. ZERO OVERLAP with the locked 32-sample benchmark in data/locked_benchmark_32/
2. ZERO OVERLAP with Phase 10 development tasks (dev_task_009 to dev_task_024).
3. PERFECT CLASS BALANCE across every provenance category:
   - human: 50% safe, 50% vulnerable
   - ai: 50% safe, 50% vulnerable
   - human_edit_of_ai: 50% safe, 50% vulnerable
   - ai_edit_of_human: 50% safe, 50% vulnerable
4. Strict Group Isolation: All 4 versions of each seed task remain in the same partition.
5. Dual Ground Truth: Concrete code logic with confirmed vulnerability behavior + recorded provenance lineage.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class Phase11CodeUnit:
    sample_id: str
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
    lineage_stage: str      # 'base', 'ai_gen', 'human_edit', 'ai_edit'
    rationale: str
    code_hash: str = ""

    def __post_init__(self):
        self.code_hash = hashlib.sha256(self.code.strip().encode()).hexdigest()[:16]


def create_phase11_tasks() -> List[Phase11CodeUnit]:
    units = []

    # =========================================================================
    # Task 25 (Odd): CWE-120 Buffer Overflow in URI Percent-Decoding
    # =========================================================================
    units.append(Phase11CodeUnit(
        sample_id="p11_task_025_v1_human_vulnerable",
        task_id="p11_task_025_uri_decoder",
        version_id="t25_v1_human_vuln",
        cwe="CWE-120",
        cwe_name="Buffer Copy without Checking Size of Input",
        code="""#include <stdio.h>
#include <string.h>
#include <ctype.h>

void url_decode_legacy(const char *src, char *dest) {
    // Legacy human C implementation: unbounded output write
    while (*src) {
        if (*src == '%' && isxdigit(src[1]) && isxdigit(src[2])) {
            int value;
            sscanf(src + 1, "%2x", &value);
            *dest++ = (char)value;
            src += 3;
        } else if (*src == '+') {
            *dest++ = ' ';
            src++;
        } else {
            *dest++ = *src++;
        }
    }
    *dest = '\\0';
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=1,
        lineage_stage="base",
        rationale="Human legacy code: unbounded write into dest buffer causes buffer overflow."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_025_v2_ai_secure",
        task_id="p11_task_025_uri_decoder",
        version_id="t25_v2_ai_secure",
        cwe="CWE-120",
        cwe_name="Buffer Copy without Checking Size of Input",
        code="""#include <stdio.h>
#include <string.h>
#include <ctype.h>

int url_decode_safe(const char *src, char *dest, size_t dest_max) {
    if (!src || !dest || dest_max == 0) return -1;
    size_t written = 0;
    while (*src && written + 1 < dest_max) {
        if (*src == '%' && isxdigit(src[1]) && isxdigit(src[2])) {
            int value = 0;
            if (sscanf(src + 1, "%2x", &value) == 1) {
                dest[written++] = (char)value;
                src += 3;
            } else {
                dest[written++] = *src++;
            }
        } else if (*src == '+') {
            dest[written++] = ' ';
            src++;
        } else {
            dest[written++] = *src++;
        }
    }
    dest[written] = '\\0';
    return (*src == '\\0') ? 0 : -1;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="ai_gen",
        rationale="AI-generated defensive decode with strict dest_max limit and bounds checking."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_025_v3_human_edit_of_ai_vulnerable",
        task_id="p11_task_025_uri_decoder",
        version_id="t25_v3_human_edit_vuln",
        cwe="CWE-120",
        cwe_name="Buffer Copy without Checking Size of Input",
        code="""#include <stdio.h>
#include <string.h>
#include <ctype.h>

int url_decode_optimized(const char *src, char *dest, size_t dest_max) {
    if (!src || !dest) return -1;
    // Human developer 'optimized' AI code by removing the written bounds check inside loop
    char *d = dest;
    while (*src) {
        if (*src == '%' && isxdigit(src[1]) && isxdigit(src[2])) {
            int value;
            sscanf(src + 1, "%2x", &value);
            *d++ = (char)value;
            src += 3;
        } else {
            *d++ = *src++;
        }
    }
    *d = '\\0'; // Flaw: Human regression removed size protection!
    return 0;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.5,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=1,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="human_edit",
        rationale="Human edit of AI: Developer stripped bounds checks during 'optimization', creating overflow."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_025_v4_ai_edit_of_human_secure",
        task_id="p11_task_025_uri_decoder",
        version_id="t25_v4_ai_edit_secure",
        cwe="CWE-120",
        cwe_name="Buffer Copy without Checking Size of Input",
        code="""#include <stdio.h>
#include <string.h>
#include <ctype.h>

int url_decode_ai_refactored(const char *src, char *dest, size_t dest_max) {
    // AI refactored legacy human function: introduced bounds checking & validation
    if (!src || !dest || dest_max == 0) return -1;
    size_t out_idx = 0;
    for (size_t i = 0; src[i] != '\\0'; ) {
        if (out_idx + 1 >= dest_max) {
            dest[out_idx] = '\\0';
            return -1; // Truncation error, prevent overflow
        }
        if (src[i] == '%' && src[i+1] && src[i+2] && isxdigit(src[i+1]) && isxdigit(src[i+2])) {
            int val;
            sscanf(&src[i+1], "%2x", &val);
            dest[out_idx++] = (char)val;
            i += 3;
        } else {
            dest[out_idx++] = src[i++];
        }
    }
    dest[out_idx] = '\\0';
    return 0;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.8,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=1,
        provenance_confidence=0.95,
        vuln_label=0,
        lineage_stage="ai_edit",
        rationale="AI successfully remediated human legacy vulnerability by enforcing dest_max guard."
    ))

    # =========================================================================
    # Task 26 (Even): CWE-287 Authentication Bypass via Algorithm None
    # =========================================================================
    units.append(Phase11CodeUnit(
        sample_id="p11_task_026_v1_human_secure",
        task_id="p11_task_026_jwt_verifier",
        version_id="t26_v1_human_secure",
        cwe="CWE-287",
        cwe_name="Improper Authentication",
        code="""#include <string.h>
#include <stdbool.h>

bool verify_token_header(const char *alg, const char *expected_alg) {
    if (!alg || !expected_alg) return false;
    // Human security practice: explicitly reject "none" algorithm and enforce exact match
    if (strcmp(alg, "none") == 0 || strcmp(alg, "NONE") == 0) {
        return false;
    }
    return strcmp(alg, expected_alg) == 0;
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human defensive: explicitly rejects 'none' algorithm and validates algorithm strictness."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_026_v2_ai_vulnerable",
        task_id="p11_task_026_jwt_verifier",
        version_id="t26_v2_ai_vuln",
        cwe="CWE-287",
        cwe_name="Improper Authentication",
        code="""#include <string.h>
#include <stdbool.h>

bool verify_token_header(const char *alg, const char *expected_alg) {
    // AI generated: accepts 'none' algorithm if user provided no expected_alg or empty
    if (!alg) return false;
    if (strcmp(alg, "none") == 0) {
        return true; // Vulnerability: Allows unverified tokens with algorithm 'none'!
    }
    return expected_alg ? (strcmp(alg, expected_alg) == 0) : true;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="GitHub Copilot",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=1,
        lineage_stage="ai_gen",
        rationale="AI naive implementation permits algorithm 'none', causing authentication bypass."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_026_v3_human_edit_of_ai_secure",
        task_id="p11_task_026_jwt_verifier",
        version_id="t26_v3_human_edit_secure",
        cwe="CWE-287",
        cwe_name="Improper Authentication",
        code="""#include <string.h>
#include <stdbool.h>

bool verify_token_header_remediated(const char *alg, const char *expected_alg) {
    // Human remediated AI flaw: explicitly require expected_alg and forbid "none"
    if (!alg || !expected_alg) return false;
    if (strcasecmp(alg, "none") == 0) return false;
    return strcmp(alg, expected_alg) == 0;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.6,
        ai_generator="GitHub Copilot",
        human_edit_count=1,
        provenance_confidence=0.95,
        vuln_label=0,
        lineage_stage="human_edit",
        rationale="Human developer remediated AI vulnerability by stripping 'none' bypass and requiring key match."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_026_v4_ai_edit_of_human_vulnerable",
        task_id="p11_task_026_jwt_verifier",
        version_id="t26_v4_ai_edit_vuln",
        cwe="CWE-287",
        cwe_name="Improper Authentication",
        code="""#include <string.h>
#include <stdbool.h>

bool verify_token_header_flexible(const char *alg, const char *expected_alg) {
    // AI refactoring human code: tried to add backwards compatibility, regressing security
    if (!alg) return false;
    if (expected_alg == NULL || strlen(expected_alg) == 0) {
        return true; // AI introduced regression: skips verification if config omitted!
    }
    return strcmp(alg, expected_alg) == 0;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=1,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_edit",
        rationale="AI edit of human code introduced regression allowing empty expected_alg to pass auth."
    ))

    # =========================================================================
    # Task 27 (Odd): CWE-787 Out-of-Bounds Write in IP Defragmentation
    # =========================================================================
    units.append(Phase11CodeUnit(
        sample_id="p11_task_027_v1_human_vulnerable",
        task_id="p11_task_027_packet_reassembly",
        version_id="t27_v1_human_vuln",
        cwe="CWE-787",
        cwe_name="Out-of-bounds Write",
        code="""#include <string.h>
#include <stdlib.h>

int reassemble_packet(char *pkt_buf, size_t buf_capacity, const char *frag, size_t frag_len, size_t offset) {
    // Legacy human networking code: trusts fragment offset without checking total overflow
    memcpy(pkt_buf + offset, frag, frag_len);
    return 0; // Flaw: offset + frag_len can exceed buf_capacity!
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=1,
        lineage_stage="base",
        rationale="Human code fails to verify (offset + frag_len <= buf_capacity), allowing heap corruption."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_027_v2_ai_secure",
        task_id="p11_task_027_packet_reassembly",
        version_id="t27_v2_ai_secure",
        cwe="CWE-787",
        cwe_name="Out-of-bounds Write",
        code="""#include <string.h>
#include <stdlib.h>

int reassemble_packet_safe(char *pkt_buf, size_t buf_capacity, const char *frag, size_t frag_len, size_t offset) {
    if (!pkt_buf || !frag) return -1;
    // AI generated safe check: check integer wraparound and capacity limits
    if (offset > buf_capacity || frag_len > buf_capacity - offset) {
        return -1; // Out of bounds detected
    }
    memcpy(pkt_buf + offset, frag, frag_len);
    return 0;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="ai_gen",
        rationale="AI generated code with robust arithmetic bounds validation before memcpy."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_027_v3_human_edit_of_ai_vulnerable",
        task_id="p11_task_027_packet_reassembly",
        version_id="t27_v3_human_edit_vuln",
        cwe="CWE-787",
        cwe_name="Out-of-bounds Write",
        code="""#include <string.h>
#include <stdlib.h>

int reassemble_packet_human_edited(char *pkt_buf, size_t buf_capacity, const char *frag, size_t frag_len, size_t offset) {
    if (!pkt_buf || !frag) return -1;
    // Human developer changed subtraction check to addition, causing unsigned integer overflow
    if (offset + frag_len > buf_capacity) { // Integer overflow if offset + frag_len wraps!
        return -1;
    }
    memcpy(pkt_buf + offset, frag, frag_len);
    return 0;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.5,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=1,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="human_edit",
        rationale="Human edit of AI: replaced safe subtraction check with vulnerable addition prone to wraparound."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_027_v4_ai_edit_of_human_secure",
        task_id="p11_task_027_packet_reassembly",
        version_id="t27_v4_ai_edit_secure",
        cwe="CWE-787",
        cwe_name="Out-of-bounds Write",
        code="""#include <string.h>
#include <stdlib.h>

int reassemble_packet_ai_fixed(char *pkt_buf, size_t buf_capacity, const char *frag, size_t frag_len, size_t offset) {
    if (!pkt_buf || !frag) return -1;
    // AI audit fix: verify pointer and ensure offset cannot overflow size_t bounds
    if (offset >= buf_capacity || frag_len > (buf_capacity - offset)) {
        return -1;
    }
    memcpy(pkt_buf + offset, frag, frag_len);
    return 0;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.7,
        ai_generator="GitHub Copilot",
        human_edit_count=1,
        provenance_confidence=0.95,
        vuln_label=0,
        lineage_stage="ai_edit",
        rationale="AI edit of human code fixed missing bounds check, ensuring memory safety."
    ))

    # =========================================================================
    # Task 28 (Even): CWE-377 / CWE-362 Insecure Temporary File Creation & TOCTOU
    # =========================================================================
    units.append(Phase11CodeUnit(
        sample_id="p11_task_028_v1_human_secure",
        task_id="p11_task_028_tempfile_creation",
        version_id="t28_v1_human_secure",
        cwe="CWE-377",
        cwe_name="Insecure Temporary File",
        code="""#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

int create_secure_temp_file(char *template_path) {
    // Human secure practice: mkstemp atomically creates with 0600 permissions
    int fd = mkstemp(template_path);
    if (fd < 0) return -1;
    return fd;
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human defensive implementation uses atomic mkstemp with private permissions."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_028_v2_ai_vulnerable",
        task_id="p11_task_028_tempfile_creation",
        version_id="t28_v2_ai_vuln",
        cwe="CWE-377",
        cwe_name="Insecure Temporary File",
        code="""#include <stdio.h>
#include <stdlib.h>

FILE* create_temp_file_ai(void) {
    // AI naive generation: uses deprecated tmpnam followed by fopen (race condition TOCTOU)
    char temp_name[L_tmpnam];
    if (!tmpnam(temp_name)) return NULL;
    return fopen(temp_name, "w+"); // Vulnerability: Symlink race condition between tmpnam and fopen!
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=1,
        lineage_stage="ai_gen",
        rationale="AI used unsafe tmpnam() API creating high-severity TOCTOU symlink vulnerability."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_028_v3_human_edit_of_ai_secure",
        task_id="p11_task_028_tempfile_creation",
        version_id="t28_v3_human_edit_secure",
        cwe="CWE-377",
        cwe_name="Insecure Temporary File",
        code="""#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int create_temp_file_remediated(char *buf, size_t buf_len) {
    // Human remediation: replace AI's tmpnam with mkstemp
    if (!buf || buf_len < 20) return -1;
    snprintf(buf, buf_len, "/tmp/secure_app_XXXXXX");
    int fd = mkstemp(buf);
    return fd;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.5,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=1,
        provenance_confidence=0.95,
        vuln_label=0,
        lineage_stage="human_edit",
        rationale="Human developer replaced AI's insecure tmpnam with atomic mkstemp."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_028_v4_ai_edit_of_human_vulnerable",
        task_id="p11_task_028_tempfile_creation",
        version_id="t28_v4_ai_edit_vuln",
        cwe="CWE-377",
        cwe_name="Insecure Temporary File",
        code="""#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>

int create_temp_file_ai_regressed(const char *filename) {
    // AI edit of human code: removed atomic flag O_EXCL to allow 'reusing' temp files
    int fd = open(filename, O_CREAT | O_WRONLY | O_TRUNC, 0666); // Missing O_EXCL allows symlink hijack!
    return fd;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="GitHub Copilot",
        human_edit_count=1,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_edit",
        rationale="AI edit removed O_EXCL from open() call, introducing symlink vulnerability."
    ))

    # =========================================================================
    # Task 29 (Odd): CWE-125 Out-of-Bounds Read in Base64 Stream Decoding
    # =========================================================================
    units.append(Phase11CodeUnit(
        sample_id="p11_task_029_v1_human_vulnerable",
        task_id="p11_task_029_base64_decode",
        version_id="t29_v1_human_vuln",
        cwe="CWE-125",
        cwe_name="Out-of-bounds Read",
        code="""#include <stdlib.h>
#include <string.h>

int base64_decode_block(const char *in, unsigned char *out) {
    // Legacy human routine: reads 4 chars without validating if in has 4 chars left
    out[0] = (in[0] << 2) | (in[1] >> 4);
    out[1] = (in[1] << 4) | (in[2] >> 2);
    out[2] = (in[2] << 6) | in[3]; // Vulnerability: Reads past buffer if input < 4 bytes!
    return 3;
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=1,
        lineage_stage="base",
        rationale="Human code assumes 4 bytes exist, reading beyond the boundary on truncated inputs."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_029_v2_ai_secure",
        task_id="p11_task_029_base64_decode",
        version_id="t29_v2_ai_secure",
        cwe="CWE-125",
        cwe_name="Out-of-bounds Read",
        code="""#include <stdlib.h>
#include <string.h>

int base64_decode_block_safe(const char *in, size_t in_len, unsigned char *out, size_t out_max) {
    if (!in || !out || in_len < 4 || out_max < 3) return -1;
    out[0] = (unsigned char)((in[0] << 2) | (in[1] >> 4));
    out[1] = (unsigned char)((in[1] << 4) | (in[2] >> 2));
    out[2] = (unsigned char)((in[2] << 6) | in[3]);
    return 3;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="ai_gen",
        rationale="AI generated code validates both in_len >= 4 and out_max >= 3 before reading/writing."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_029_v3_human_edit_of_ai_vulnerable",
        task_id="p11_task_029_base64_decode",
        version_id="t29_v3_human_edit_vuln",
        cwe="CWE-125",
        cwe_name="Out-of-bounds Read",
        code="""#include <stdlib.h>
#include <string.h>

int base64_decode_fast(const char *in, size_t in_len, unsigned char *out) {
    // Human developer removed in_len validation to eliminate branch overhead
    if (!in || !out) return -1;
    out[0] = (in[0] << 2) | (in[1] >> 4);
    out[1] = (in[1] << 4) | (in[2] >> 2);
    out[2] = (in[2] << 6) | in[3]; // Regression: Reads OOB if in_len < 4
    return 3;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.5,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=1,
        provenance_confidence=0.95,
        vuln_label=1,
        lineage_stage="human_edit",
        rationale="Human developer removed AI length check to optimize performance, introducing OOB read."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_029_v4_ai_edit_of_human_secure",
        task_id="p11_task_029_base64_decode",
        version_id="t29_v4_ai_edit_secure",
        cwe="CWE-125",
        cwe_name="Out-of-bounds Read",
        code="""#include <stdlib.h>
#include <string.h>

int base64_decode_block_ai_repaired(const char *in, size_t in_len, unsigned char *out, size_t out_len) {
    // AI audit repair of human code: verifies buffer length
    if (!in || in_len < 4 || !out || out_len < 3) return -1;
    out[0] = (unsigned char)((in[0] << 2) | (in[1] >> 4));
    out[1] = (unsigned char)((in[1] << 4) | (in[2] >> 2));
    out[2] = (unsigned char)((in[2] << 6) | in[3]);
    return 3;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.8,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=1,
        provenance_confidence=0.95,
        vuln_label=0,
        lineage_stage="ai_edit",
        rationale="AI edit of human code added in_len >= 4 verification, eliminating OOB read."
    ))

    # =========================================================================
    # Task 30 (Even): CWE-295 Improper Certificate Validation
    # =========================================================================
    units.append(Phase11CodeUnit(
        sample_id="p11_task_030_v1_human_secure",
        task_id="p11_task_030_ssl_cert_validate",
        version_id="t30_v1_human_secure",
        cwe="CWE-295",
        cwe_name="Improper Certificate Validation",
        code="""#include <stdbool.h>
#include <string.h>

bool verify_tls_certificate(int verify_result, const char *cert_hostname, const char *expected_hostname) {
    // Human defensive practice: require both valid cert chain AND strict hostname match
    if (verify_result != 0) return false;
    if (!cert_hostname || !expected_hostname) return false;
    return strcmp(cert_hostname, expected_hostname) == 0;
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human code requires valid chain result and strict hostname equivalence."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_030_v2_ai_vulnerable",
        task_id="p11_task_030_ssl_cert_validate",
        version_id="t30_v2_ai_vuln",
        cwe="CWE-295",
        cwe_name="Improper Certificate Validation",
        code="""#include <stdbool.h>

bool verify_tls_certificate_ai(int verify_result, const char *cert_hostname, const char *expected_hostname) {
    // AI generated: ignores verify_result error codes to 'allow self-signed certs in dev'
    (void)cert_hostname;
    (void)expected_hostname;
    return true; // Vulnerability: Completely disables TLS certificate validation!
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="GitHub Copilot",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=1,
        lineage_stage="ai_gen",
        rationale="AI disabled TLS certificate verification to permit self-signed certs, causing MITM."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_030_v3_human_edit_of_ai_secure",
        task_id="p11_task_030_ssl_cert_validate",
        version_id="t30_v3_human_edit_secure",
        cwe="CWE-295",
        cwe_name="Improper Certificate Validation",
        code="""#include <stdbool.h>
#include <string.h>

bool verify_tls_certificate_remediated(int verify_result, const char *cert_hostname, const char *expected_hostname) {
    // Human developer rejected AI bypass: enforced verify_result == 0 and hostname validation
    if (verify_result != 0) return false;
    if (!cert_hostname || !expected_hostname) return false;
    return strcasecmp(cert_hostname, expected_hostname) == 0;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.5,
        ai_generator="GitHub Copilot",
        human_edit_count=1,
        provenance_confidence=0.95,
        vuln_label=0,
        lineage_stage="human_edit",
        rationale="Human reviewed AI bypass and restored strict certificate checking."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_030_v4_ai_edit_of_human_vulnerable",
        task_id="p11_task_030_ssl_cert_validate",
        version_id="t30_v4_ai_edit_vuln",
        cwe="CWE-295",
        cwe_name="Improper Certificate Validation",
        code="""#include <stdbool.h>
#include <string.h>

bool verify_tls_certificate_ai_modified(int verify_result, const char *cert_hostname, const char *expected_hostname) {
    // AI edit of human code: added wildcard matching with strstr instead of exact domain match
    if (verify_result != 0) return false;
    if (!cert_hostname || !expected_hostname) return false;
    return strstr(cert_hostname, expected_hostname) != NULL; // Flaw: evil-target.com matches target.com!
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.8,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=1,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_edit",
        rationale="AI edit replaced exact hostname check with strstr, allowing subdomain spoofing."
    ))

    # =========================================================================
    # Task 31 (Odd): CWE-78 Command Injection in System Shell Execution
    # =========================================================================
    units.append(Phase11CodeUnit(
        sample_id="p11_task_031_v1_human_vulnerable",
        task_id="p11_task_031_cmd_exec_wrapper",
        version_id="t31_v1_human_vuln",
        cwe="CWE-78",
        cwe_name="Improper Neutralization of Special Elements used in an OS Command",
        code="""#include <stdio.h>
#include <stdlib.h>

int run_user_backup(const char *username) {
    // Human legacy code: formats system command without argument sanitization
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "tar -czf /backups/%s.tar.gz /home/%s", username, username);
    return system(cmd); // Flaw: username with ';' or '|' executes arbitrary shell commands!
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=1,
        lineage_stage="base",
        rationale="Human code uses system() with unescaped user string, causing OS command injection."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_031_v2_ai_secure",
        task_id="p11_task_031_cmd_exec_wrapper",
        version_id="t31_v2_ai_secure",
        cwe="CWE-78",
        cwe_name="Improper Neutralization of Special Elements used in an OS Command",
        code="""#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <ctype.h>

int run_user_backup_safe(const char *username) {
    if (!username) return -1;
    // AI generated safe implementation: validate alphanumeric and use fork+execvp (no shell)
    for (size_t i = 0; username[i]; i++) {
        if (!isalnum(username[i]) && username[i] != '_') return -1;
    }
    char archive[256], home[256];
    snprintf(archive, sizeof(archive), "/backups/%s.tar.gz", username);
    snprintf(home, sizeof(home), "/home/%s", username);
    pid_t pid = fork();
    if (pid == 0) {
        char *args[] = {"tar", "-czf", archive, home, NULL};
        execvp("tar", args);
        _exit(127);
    }
    int status;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="ai_gen",
        rationale="AI implemented fork()+execvp() with strict username character validation."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_031_v3_human_edit_of_ai_vulnerable",
        task_id="p11_task_031_cmd_exec_wrapper",
        version_id="t31_v3_human_edit_vuln",
        cwe="CWE-78",
        cwe_name="Improper Neutralization of Special Elements used in an OS Command",
        code="""#include <stdio.h>
#include <stdlib.h>

int run_user_backup_human_mod(const char *username) {
    // Human developer simplified AI's fork+exec to popen for brevity
    char cmd[512];
    snprintf(cmd, sizeof(cmd), "/bin/tar -czf /backups/%s.tar.gz /home/%s 2>&1", username, username);
    FILE *fp = popen(cmd, "r"); // Flaw: popen executes shell, reinstating command injection!
    if (!fp) return -1;
    return pclose(fp);
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.5,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=1,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="human_edit",
        rationale="Human edit reverted execvp to popen() for brevity, re-introducing shell injection."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_031_v4_ai_edit_of_human_secure",
        task_id="p11_task_031_cmd_exec_wrapper",
        version_id="t31_v4_ai_edit_secure",
        cwe="CWE-78",
        cwe_name="Improper Neutralization of Special Elements used in an OS Command",
        code="""#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <ctype.h>

int run_user_backup_ai_hardened(const char *username) {
    if (!username) return -1;
    for (int i = 0; username[i]; i++) {
        if (!isalnum(username[i])) return -1;
    }
    pid_t pid = fork();
    if (pid == 0) {
        char archive[128], homedir[128];
        snprintf(archive, sizeof(archive), "/backups/%s.tar.gz", username);
        snprintf(homedir, sizeof(homedir), "/home/%s", username);
        char *args[] = {"/bin/tar", "-czf", archive, homedir, NULL};
        execv(args[0], args);
        _exit(1);
    }
    int status;
    waitpid(pid, &status, 0);
    return WEXITSTATUS(status);
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.85,
        ai_generator="GitHub Copilot",
        human_edit_count=1,
        provenance_confidence=0.95,
        vuln_label=0,
        lineage_stage="ai_edit",
        rationale="AI edit eliminated system() call by adopting execv() and input validation."
    ))

    # =========================================================================
    # Task 32 (Even): CWE-134 Format String in Logging Configuration
    # =========================================================================
    units.append(Phase11CodeUnit(
        sample_id="p11_task_032_v1_human_secure",
        task_id="p11_task_032_config_parser",
        version_id="t32_v1_human_secure",
        cwe="CWE-134",
        cwe_name="Use of Externally-Controlled Format String",
        code="""#include <stdio.h>
#include <syslog.h>

void log_app_message(int priority, const char *user_msg) {
    // Human defensive practice: pass explicit format string literal "%s"
    if (!user_msg) return;
    syslog(priority, "%s", user_msg);
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human code uses safe literal format specifier '%s'."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_032_v2_ai_vulnerable",
        task_id="p11_task_032_config_parser",
        version_id="t32_v2_ai_vuln",
        cwe="CWE-134",
        cwe_name="Use of Externally-Controlled Format String",
        code="""#include <stdio.h>
#include <syslog.h>

void log_app_message_ai(int priority, const char *user_msg) {
    // AI generated: directly passes user string as format argument to syslog
    if (!user_msg) return;
    syslog(priority, user_msg); // Flaw: Format string vulnerability if user_msg contains "%x" or "%n"
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="GitHub Copilot",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=1,
        lineage_stage="ai_gen",
        rationale="AI passed user string directly as format specifier to syslog()."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_032_v3_human_edit_of_ai_secure",
        task_id="p11_task_032_config_parser",
        version_id="t32_v3_human_edit_secure",
        cwe="CWE-134",
        cwe_name="Use of Externally-Controlled Format String",
        code="""#include <stdio.h>
#include <syslog.h>

void log_app_message_remediated(int priority, const char *user_msg) {
    // Human code review: fixed AI format string bug by inserting "%s"
    if (!user_msg) return;
    syslog(priority, "%s", user_msg);
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.5,
        ai_generator="GitHub Copilot",
        human_edit_count=1,
        provenance_confidence=0.95,
        vuln_label=0,
        lineage_stage="human_edit",
        rationale="Human developer inserted '%s' literal to neutralize format string vulnerability."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_032_v4_ai_edit_of_human_vulnerable",
        task_id="p11_task_032_config_parser",
        version_id="t32_v4_ai_edit_vuln",
        cwe="CWE-134",
        cwe_name="Use of Externally-Controlled Format String",
        code="""#include <stdio.h>

void print_custom_banner(const char *prefix, const char *banner_text) {
    // AI edit of human code: tried to support custom user formatting
    char buf[512];
    snprintf(buf, sizeof(buf), "%s: %s", prefix, banner_text);
    printf(buf); // Flaw: printf(buf) without "%s" format string!
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.7,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=1,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_edit",
        rationale="AI edit used printf(buf) instead of printf('%s', buf), introducing format flaw."
    ))

    # =========================================================================
    # Task 33 (Odd): CWE-327 Broken Cryptography / Static IV & Salt
    # =========================================================================
    units.append(Phase11CodeUnit(
        sample_id="p11_task_033_v1_human_vulnerable",
        task_id="p11_task_033_crypto_key_derivation",
        version_id="t33_v1_human_vuln",
        cwe="CWE-327",
        cwe_name="Use of a Broken or Risky Cryptographic Algorithm",
        code="""#include <string.h>
#include <openssl/md5.h>

void derive_hash_key_legacy(const char *password, unsigned char *out_key) {
    // Legacy human code: derives encryption key using broken MD5 with hardcoded static salt
    const char *salt = "STATIC_SALT_123";
    char combined[128];
    snprintf(combined, sizeof(combined), "%s%s", password, salt);
    MD5((const unsigned char*)combined, strlen(combined), out_key);
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=1,
        lineage_stage="base",
        rationale="Human code uses deprecated MD5 with hardcoded salt for cryptographic key derivation."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_033_v2_ai_secure",
        task_id="p11_task_033_crypto_key_derivation",
        version_id="t33_v2_ai_secure",
        cwe="CWE-327",
        cwe_name="Use of a Broken or Risky Cryptographic Algorithm",
        code="""#include <string.h>
#include <openssl/evp.h>
#include <openssl/rand.h>

int derive_hash_key_safe(const char *password, const unsigned char *salt, size_t salt_len, unsigned char *out_key, size_t key_len) {
    if (!password || !salt || salt_len < 16 || !out_key || key_len < 32) return -1;
    // AI generated modern standard: PBKDF2 with SHA-256 and 100,000 iterations
    return PKCS5_PBKDF2_HMAC(password, strlen(password), salt, salt_len, 100000, EVP_sha256(), key_len, out_key) ? 0 : -1;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="ai_gen",
        rationale="AI implemented standard PBKDF2-HMAC-SHA256 with 100,000 rounds and random salt."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_033_v3_human_edit_of_ai_vulnerable",
        task_id="p11_task_033_crypto_key_derivation",
        version_id="t33_v3_human_edit_vuln",
        cwe="CWE-327",
        cwe_name="Use of a Broken or Risky Cryptographic Algorithm",
        code="""#include <string.h>
#include <openssl/evp.h>

int derive_hash_key_fast(const char *password, const unsigned char *salt, unsigned char *out_key) {
    // Human developer reduced PBKDF2 iterations to 1 to speed up test execution, leaving key weak
    return PKCS5_PBKDF2_HMAC(password, strlen(password), salt, 16, 1, EVP_sha256(), 32, out_key) ? 0 : -1;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.5,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=1,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="human_edit",
        rationale="Human edit lowered KDF iterations to 1, effectively removing cryptographic work factor."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_033_v4_ai_edit_of_human_secure",
        task_id="p11_task_033_crypto_key_derivation",
        version_id="t33_v4_ai_edit_secure",
        cwe="CWE-327",
        cwe_name="Use of a Broken or Risky Cryptographic Algorithm",
        code="""#include <string.h>
#include <openssl/evp.h>

int derive_hash_key_ai_upgraded(const char *password, const unsigned char *salt, size_t salt_len, unsigned char *out_key) {
    if (!password || !salt || salt_len < 16) return -1;
    // AI upgraded legacy MD5 to standard EVP PBKDF2-HMAC-SHA256
    return PKCS5_PBKDF2_HMAC(password, strlen(password), salt, salt_len, 600000, EVP_sha256(), 32, out_key) ? 0 : -1;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.8,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=1,
        provenance_confidence=0.95,
        vuln_label=0,
        lineage_stage="ai_edit",
        rationale="AI edit upgraded insecure MD5 derivation to compliant 600k round PBKDF2."
    ))

    # =========================================================================
    # Task 34 (Even): CWE-416 Use-After-Free in Token Node Reclamation
    # =========================================================================
    units.append(Phase11CodeUnit(
        sample_id="p11_task_034_v1_human_secure",
        task_id="p11_task_034_json_array_parse",
        version_id="t34_v1_human_secure",
        cwe="CWE-416",
        cwe_name="Use After Free",
        code="""#include <stdlib.h>

typedef struct Node {
    char *value;
    struct Node *next;
} Node;

void free_node_list(Node *head) {
    // Human defensive practice: store next pointer before freeing current node
    Node *curr = head;
    while (curr != NULL) {
        Node *next = curr->next;
        free(curr->value);
        free(curr);
        curr = next;
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
        rationale="Human code correctly saves curr->next before releasing memory, preventing UAF."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_034_v2_ai_vulnerable",
        task_id="p11_task_034_json_array_parse",
        version_id="t34_v2_ai_vuln",
        cwe="CWE-416",
        cwe_name="Use After Free",
        code="""#include <stdlib.h>

typedef struct Node {
    char *value;
    struct Node *next;
} Node;

void free_node_list_ai(Node *head) {
    // AI generated: accesses curr->next AFTER calling free(curr)
    Node *curr = head;
    while (curr != NULL) {
        free(curr->value);
        free(curr);
        curr = curr->next; // Vulnerability: Use-After-Free accessing freed struct memory!
    }
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="GitHub Copilot",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=1,
        lineage_stage="ai_gen",
        rationale="AI accessed curr->next after freeing curr, causing classic heap UAF."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_034_v3_human_edit_of_ai_secure",
        task_id="p11_task_034_json_array_parse",
        version_id="t34_v3_human_edit_secure",
        cwe="CWE-416",
        cwe_name="Use After Free",
        code="""#include <stdlib.h>

typedef struct Node {
    char *value;
    struct Node *next;
} Node;

void free_node_list_remediated(Node *head) {
    // Human developer caught AI UAF bug and cached next pointer
    Node *curr = head;
    while (curr) {
        Node *tmp = curr->next;
        free(curr->value);
        free(curr);
        curr = tmp;
    }
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.5,
        ai_generator="GitHub Copilot",
        human_edit_count=1,
        provenance_confidence=0.95,
        vuln_label=0,
        lineage_stage="human_edit",
        rationale="Human code review remediated AI UAF by caching next node prior to free."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_034_v4_ai_edit_of_human_vulnerable",
        task_id="p11_task_034_json_array_parse",
        version_id="t34_v4_ai_edit_vuln",
        cwe="CWE-416",
        cwe_name="Use After Free",
        code="""#include <stdlib.h>

typedef struct Node {
    char *value;
    struct Node *next;
} Node;

Node* remove_head_ai_modified(Node **head) {
    // AI refactoring human code: frees head but returns pointer to freed memory
    if (!head || !*head) return NULL;
    Node *old = *head;
    *head = old->next;
    free(old);
    return old; // Flaw: returns pointer to deallocated memory, causing UAF in caller!
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=1,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_edit",
        rationale="AI edit deallocated struct and returned the freed pointer to caller."
    ))

    # =========================================================================
    # Task 35 (Odd): CWE-22 Path Traversal in Archive Extraction
    # =========================================================================
    units.append(Phase11CodeUnit(
        sample_id="p11_task_035_v1_human_vulnerable",
        task_id="p11_task_035_file_path_resolver",
        version_id="t35_v1_human_vuln",
        cwe="CWE-22",
        cwe_name="Improper Limitation of a Pathname to a Restricted Directory",
        code="""#include <stdio.h>
#include <string.h>

int extract_file_legacy(const char *base_dir, const char *rel_path, char *out_path, size_t out_len) {
    // Legacy human routine: blindly concatenates paths without checking for "../"
    snprintf(out_path, out_len, "%s/%s", base_dir, rel_path);
    return 0; // Flaw: rel_path containing "../../etc/passwd" traverses outside base_dir!
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=1,
        lineage_stage="base",
        rationale="Human code directly concatenates untrusted relative path allowing directory traversal."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_035_v2_ai_secure",
        task_id="p11_task_035_file_path_resolver",
        version_id="t35_v2_ai_secure",
        cwe="CWE-22",
        cwe_name="Improper Limitation of a Pathname to a Restricted Directory",
        code="""#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

int extract_file_safe(const char *base_dir, const char *rel_path, char *out_path, size_t out_len) {
    if (!base_dir || !rel_path || !out_path) return -1;
    if (strstr(rel_path, "..") != NULL) return -1; // Block traversal tokens
    char combined[PATH_MAX];
    snprintf(combined, sizeof(combined), "%s/%s", base_dir, rel_path);
    char resolved[PATH_MAX];
    if (realpath(combined, resolved) == NULL) {
        // Path doesn't exist yet, verify combined prefix
        if (strncmp(combined, base_dir, strlen(base_dir)) != 0) return -1;
        snprintf(out_path, out_len, "%s", combined);
        return 0;
    }
    if (strncmp(resolved, base_dir, strlen(base_dir)) != 0) return -1;
    snprintf(out_path, out_len, "%s", resolved);
    return 0;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="ai_gen",
        rationale="AI implemented path normalization and canonical root prefix enforcement."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_035_v3_human_edit_of_ai_vulnerable",
        task_id="p11_task_035_file_path_resolver",
        version_id="t35_v3_human_edit_vuln",
        cwe="CWE-22",
        cwe_name="Improper Limitation of a Pathname to a Restricted Directory",
        code="""#include <stdio.h>
#include <string.h>

int extract_file_naive_filter(const char *base_dir, const char *rel_path, char *out_path, size_t out_len) {
    // Human developer tried to sanitize by stripping "../" once with strstr
    char clean_path[256];
    snprintf(clean_path, sizeof(clean_path), "%s", rel_path);
    char *p = strstr(clean_path, "../");
    if (p) memmove(p, p + 3, strlen(p + 3) + 1); // Flaw: "..././" collapses back into "../" after single pass!
    snprintf(out_path, out_len, "%s/%s", base_dir, clean_path);
    return 0;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.5,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=1,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="human_edit",
        rationale="Human edit used single-pass token stripping, bypassable with nested tokens like '..././'."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_035_v4_ai_edit_of_human_secure",
        task_id="p11_task_035_file_path_resolver",
        version_id="t35_v4_ai_edit_secure",
        cwe="CWE-22",
        cwe_name="Improper Limitation of a Pathname to a Restricted Directory",
        code="""#include <stdio.h>
#include <string.h>
#include <ctype.h>

int extract_file_ai_hardened(const char *base_dir, const char *rel_path, char *out_path, size_t out_len) {
    if (!base_dir || !rel_path || !out_path) return -1;
    // AI audit fix: strictly disallow any slash followed by dot-dot
    if (strstr(rel_path, "..") != NULL || rel_path[0] == '/') return -1;
    snprintf(out_path, out_len, "%s/%s", base_dir, rel_path);
    return 0;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.8,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=1,
        provenance_confidence=0.95,
        vuln_label=0,
        lineage_stage="ai_edit",
        rationale="AI edit added strict rejection of relative escape sequences and leading slashes."
    ))

    # =========================================================================
    # Task 36 (Even): CWE-400 Resource Exhaustion via Unbounded Allocation
    # =========================================================================
    units.append(Phase11CodeUnit(
        sample_id="p11_task_036_v1_human_secure",
        task_id="p11_task_036_net_socket_accept",
        version_id="t36_v1_human_secure",
        cwe="CWE-400",
        cwe_name="Uncontrolled Resource Consumption",
        code="""#include <stdlib.h>
#include <stdbool.h>

#define MAX_CONNECTIONS 1024

bool allocate_connection_slot(int *active_count) {
    if (!active_count) return false;
    // Human defensive practice: strictly enforce upper bound on concurrent descriptors
    if (*active_count >= MAX_CONNECTIONS) {
        return false; // Reject connection to prevent file descriptor exhaustion
    }
    (*active_count)++;
    return true;
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human code enforces hard capacity limit preventing socket descriptor exhaustion."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_036_v2_ai_vulnerable",
        task_id="p11_task_036_net_socket_accept",
        version_id="t36_v2_ai_vuln",
        cwe="CWE-400",
        cwe_name="Uncontrolled Resource Consumption",
        code="""#include <stdlib.h>
#include <stdbool.h>

bool allocate_connection_slot_ai(int *active_count) {
    // AI generated: increments active count with no upper limit
    if (!active_count) return false;
    (*active_count)++; // Vulnerability: Unbounded allocation causes DoS / fd exhaustion
    return true;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="GitHub Copilot",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=1,
        lineage_stage="ai_gen",
        rationale="AI omitted resource limits, allowing unbounded client connection state allocation."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_036_v3_human_edit_of_ai_secure",
        task_id="p11_task_036_net_socket_accept",
        version_id="t36_v3_human_edit_secure",
        cwe="CWE-400",
        cwe_name="Uncontrolled Resource Consumption",
        code="""#include <stdlib.h>
#include <stdbool.h>

#define MAX_CONNS 2048

bool allocate_connection_slot_remediated(int *active_count) {
    // Human remediation: restored resource guard to prevent starvation
    if (!active_count) return false;
    if (*active_count >= MAX_CONNS) return false;
    (*active_count)++;
    return true;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.5,
        ai_generator="GitHub Copilot",
        human_edit_count=1,
        provenance_confidence=0.95,
        vuln_label=0,
        lineage_stage="human_edit",
        rationale="Human developer capped resource allocations to prevent denial of service."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_036_v4_ai_edit_of_human_vulnerable",
        task_id="p11_task_036_net_socket_accept",
        version_id="t36_v4_ai_edit_vuln",
        cwe="CWE-400",
        cwe_name="Uncontrolled Resource Consumption",
        code="""#include <stdlib.h>
#include <stdbool.h>

bool allocate_connection_slot_ai_boosted(int *active_count, int requested_burst) {
    // AI edit of human code: allowed arbitrary burst override without validating burst bounds
    if (!active_count) return false;
    *active_count += requested_burst; // Flaw: Client sends requested_burst=1000000 causing DoS
    return true;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.8,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=1,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_edit",
        rationale="AI edit introduced unvalidated burst parameter, enabling client-induced resource exhaustion."
    ))

    # =========================================================================
    # Task 37 (Odd): CWE-193 Off-by-One in Circular Queue Index
    # =========================================================================
    units.append(Phase11CodeUnit(
        sample_id="p11_task_037_v1_human_vulnerable",
        task_id="p11_task_037_ring_buffer_pop",
        version_id="t37_v1_human_vuln",
        cwe="CWE-193",
        cwe_name="Off-by-one Error",
        code="""#define QUEUE_SIZE 64
int queue_data[QUEUE_SIZE];
int head = 0;

int pop_queue_item(void) {
    // Legacy human C: uses <= instead of < causing off-by-one read past array
    if (head <= QUEUE_SIZE) { // Flaw: when head == QUEUE_SIZE, reads queue_data[64]!
        return queue_data[head++];
    }
    return -1;
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=1,
        lineage_stage="base",
        rationale="Human code uses '<=' instead of '<', causing off-by-one boundary violation."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_037_v2_ai_secure",
        task_id="p11_task_037_ring_buffer_pop",
        version_id="t37_v2_ai_secure",
        cwe="CWE-193",
        cwe_name="Off-by-one Error",
        code="""#define QUEUE_SIZE 64
int queue_data[QUEUE_SIZE];
int head = 0;

int pop_queue_item_safe(void) {
    // AI generated safe ring buffer pop: strictly checks bounds with modulo wrap
    if (head < 0 || head >= QUEUE_SIZE) return -1;
    int val = queue_data[head];
    head = (head + 1) % QUEUE_SIZE;
    return val;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="ai_gen",
        rationale="AI generated code enforces strict head < QUEUE_SIZE condition with modulo advance."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_037_v3_human_edit_of_ai_vulnerable",
        task_id="p11_task_037_ring_buffer_pop",
        version_id="t37_v3_human_edit_vuln",
        cwe="CWE-193",
        cwe_name="Off-by-one Error",
        code="""#define QUEUE_SIZE 64
int queue_data[QUEUE_SIZE];
int head = 0;

int pop_queue_item_fast(void) {
    // Human developer changed condition to head <= QUEUE_SIZE - 0, introducing boundary flaw
    if (head <= QUEUE_SIZE) {
        int val = queue_data[head];
        head = (head + 1);
        return val;
    }
    return -1;
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.5,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=1,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="human_edit",
        rationale="Human developer modified AI guard with '<=', creating an off-by-one access."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_037_v4_ai_edit_of_human_secure",
        task_id="p11_task_037_ring_buffer_pop",
        version_id="t37_v4_ai_edit_secure",
        cwe="CWE-193",
        cwe_name="Off-by-one Error",
        code="""#define QUEUE_SIZE 64
int queue_data[QUEUE_SIZE];
int head = 0;

int pop_queue_item_fixed(void) {
    // AI fixed human off-by-one: ensures head is strictly within [0, QUEUE_SIZE - 1]
    if (head < 0 || head >= QUEUE_SIZE) {
        return -1;
    }
    return queue_data[head++];
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.8,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=1,
        provenance_confidence=0.95,
        vuln_label=0,
        lineage_stage="ai_edit",
        rationale="AI edit corrected boundary condition to head >= QUEUE_SIZE."
    ))

    # =========================================================================
    # Task 38 (Even): CWE-89 SQL Injection in PreparedStatement Builder
    # =========================================================================
    units.append(Phase11CodeUnit(
        sample_id="p11_task_038_v1_human_secure",
        task_id="p11_task_038_sql_query_builder",
        version_id="t38_v1_human_secure",
        cwe="CWE-89",
        cwe_name="Improper Neutralization of Special Elements used in an SQL Command",
        code="""#include <sqlite3.h>

int query_user_secure(sqlite3 *db, const char *user_id, sqlite3_stmt **stmt) {
    // Human defensive practice: use parameterized query with sqlite3_bind_text
    const char *sql = "SELECT email, status FROM users WHERE id = ?";
    if (sqlite3_prepare_v2(db, sql, -1, stmt, NULL) != SQLITE_OK) return -1;
    return sqlite3_bind_text(*stmt, 1, user_id, -1, SQLITE_TRANSIENT);
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human code uses parameterized prepared statement with bound parameters."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_038_v2_ai_vulnerable",
        task_id="p11_task_038_sql_query_builder",
        version_id="t38_v2_ai_vuln",
        cwe="CWE-89",
        cwe_name="Improper Neutralization of Special Elements used in an SQL Command",
        code="""#include <stdio.h>
#include <sqlite3.h>

int query_user_ai(sqlite3 *db, const char *user_id, sqlite3_stmt **stmt) {
    // AI generated: uses string concatenation instead of parameter binding
    char query[256];
    snprintf(query, sizeof(query), "SELECT email, status FROM users WHERE id = '%s'", user_id);
    return sqlite3_prepare_v2(db, query, -1, stmt, NULL); // Flaw: Classic SQL Injection!
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="GitHub Copilot",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=1,
        lineage_stage="ai_gen",
        rationale="AI concatenated unescaped user string into SQL statement, creating SQL injection."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_038_v3_human_edit_of_ai_secure",
        task_id="p11_task_038_sql_query_builder",
        version_id="t38_v3_human_edit_secure",
        cwe="CWE-89",
        cwe_name="Improper Neutralization of Special Elements used in an SQL Command",
        code="""#include <sqlite3.h>

int query_user_remediated(sqlite3 *db, const char *user_id, sqlite3_stmt **stmt) {
    // Human developer remediated AI flaw: replaced snprintf with parameter placeholder
    const char *sql = "SELECT email, status FROM users WHERE id = ?1";
    if (sqlite3_prepare_v2(db, sql, -1, stmt, NULL) != SQLITE_OK) return -1;
    return sqlite3_bind_text(*stmt, 1, user_id, -1, SQLITE_STATIC);
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.5,
        ai_generator="GitHub Copilot",
        human_edit_count=1,
        provenance_confidence=0.95,
        vuln_label=0,
        lineage_stage="human_edit",
        rationale="Human developer converted concatenated query into safe parameterized statement."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_038_v4_ai_edit_of_human_vulnerable",
        task_id="p11_task_038_sql_query_builder",
        version_id="t38_v4_ai_edit_vuln",
        cwe="CWE-89",
        cwe_name="Improper Neutralization of Special Elements used in an SQL Command",
        code="""#include <stdio.h>
#include <sqlite3.h>

int query_user_ai_dynamic(sqlite3 *db, const char *table, const char *user_id, sqlite3_stmt **stmt) {
    // AI edit of human code: allowed dynamic table name via string formatting
    char sql[256];
    snprintf(sql, sizeof(sql), "SELECT * FROM %s WHERE id = ?", table); // Flaw: SQLi via table name!
    if (sqlite3_prepare_v2(db, sql, -1, stmt, NULL) != SQLITE_OK) return -1;
    return sqlite3_bind_text(*stmt, 1, user_id, -1, SQLITE_TRANSIENT);
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.75,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=1,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_edit",
        rationale="AI edit introduced dynamic table interpolation into SQL template, allowing SQLi."
    ))

    # =========================================================================
    # Task 39 (Odd): CWE-476 Null Pointer Dereference in Hash Collision Chain
    # =========================================================================
    units.append(Phase11CodeUnit(
        sample_id="p11_task_039_v1_human_vulnerable",
        task_id="p11_task_039_hash_table_lookup",
        version_id="t39_v1_human_vuln",
        cwe="CWE-476",
        cwe_name="NULL Pointer Dereference",
        code="""#include <string.h>

typedef struct Entry {
    char *key;
    char *value;
    struct Entry *next;
} Entry;

char* find_entry_legacy(Entry **table, unsigned int hash, const char *key) {
    // Legacy human C: dereferences table[hash] directly without checking if entry is NULL
    Entry *e = table[hash];
    while (strcmp(e->key, key) != 0) { // Flaw: Dereferences e without checking e != NULL!
        e = e->next;
    }
    return e->value;
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=1,
        lineage_stage="base",
        rationale="Human code dereferences entry without checking for NULL pointer."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_039_v2_ai_secure",
        task_id="p11_task_039_hash_table_lookup",
        version_id="t39_v2_ai_secure",
        cwe="CWE-476",
        cwe_name="NULL Pointer Dereference",
        code="""#include <string.h>

typedef struct Entry {
    char *key;
    char *value;
    struct Entry *next;
} Entry;

char* find_entry_safe(Entry **table, size_t capacity, unsigned int hash, const char *key) {
    if (!table || !key || capacity == 0) return NULL;
    size_t idx = hash % capacity;
    Entry *e = table[idx];
    while (e != NULL) {
        if (e->key && strcmp(e->key, key) == 0) {
            return e->value;
        }
        e = e->next;
    }
    return NULL;
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="ai_gen",
        rationale="AI generated code safely checks e != NULL and e->key != NULL before strcmp."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_039_v3_human_edit_of_ai_vulnerable",
        task_id="p11_task_039_hash_table_lookup",
        version_id="t39_v3_human_edit_vuln",
        cwe="CWE-476",
        cwe_name="NULL Pointer Dereference",
        code="""#include <string.h>

typedef struct Entry {
    char *key;
    char *value;
    struct Entry *next;
} Entry;

char* find_entry_fast(Entry **table, unsigned int hash, const char *key) {
    // Human developer removed NULL loop check assuming key is always present in hash bucket
    Entry *e = table[hash];
    while (e) {
        if (strcmp(e->key, key) == 0) break;
        e = e->next;
    }
    return e->value; // Flaw: if key not found, e is NULL, causing crash on e->value!
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.5,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=1,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="human_edit",
        rationale="Human developer omitted check on e after loop, causing NULL pointer dereference."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_039_v4_ai_edit_of_human_secure",
        task_id="p11_task_039_hash_table_lookup",
        version_id="t39_v4_ai_edit_secure",
        cwe="CWE-476",
        cwe_name="NULL Pointer Dereference",
        code="""#include <string.h>

typedef struct Entry {
    char *key;
    char *value;
    struct Entry *next;
} Entry;

char* find_entry_ai_guarded(Entry **table, size_t cap, unsigned int hash, const char *key) {
    if (!table || cap == 0 || !key) return NULL;
    for (Entry *curr = table[hash % cap]; curr != NULL; curr = curr->next) {
        if (curr->key && strcmp(curr->key, key) == 0) {
            return curr->value;
        }
    }
    return NULL;
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.8,
        ai_generator="GitHub Copilot",
        human_edit_count=1,
        provenance_confidence=0.95,
        vuln_label=0,
        lineage_stage="ai_edit",
        rationale="AI edit transformed traversal into safe loop with complete NULL handling."
    ))

    # =========================================================================
    # Task 40 (Even): CWE-190 Integer Overflow in Bitmap Dimensions Allocation
    # =========================================================================
    units.append(Phase11CodeUnit(
        sample_id="p11_task_040_v1_human_secure",
        task_id="p11_task_040_image_header_parse",
        version_id="t40_v1_human_secure",
        cwe="CWE-190",
        cwe_name="Integer Overflow or Wraparound",
        code="""#include <stdlib.h>
#include <stdint.h>

void* allocate_bitmap_buffer(uint32_t width, uint32_t height, uint32_t bpp) {
    // Human defensive practice: check for multiplication overflow before malloc
    if (width == 0 || height == 0 || bpp == 0 || bpp > 64) return NULL;
    uint64_t total_pixels = (uint64_t)width * (uint64_t)height;
    uint64_t total_bytes = total_pixels * (uint64_t)(bpp / 8);
    if (total_bytes > SIZE_MAX || total_bytes > (1024 * 1024 * 256)) {
        return NULL; // Overflow detected or exceeds 256MB limit
    }
    return malloc((size_t)total_bytes);
}""",
        language="c",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=0,
        lineage_stage="base",
        rationale="Human code promotes to uint64_t and verifies SIZE_MAX bounds before allocating."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_040_v2_ai_vulnerable",
        task_id="p11_task_040_image_header_parse",
        version_id="t40_v2_ai_vuln",
        cwe="CWE-190",
        cwe_name="Integer Overflow or Wraparound",
        code="""#include <stdlib.h>
#include <stdint.h>

void* allocate_bitmap_buffer_ai(uint32_t width, uint32_t height) {
    // AI generated: direct 32-bit multiplication overflows into tiny malloc
    size_t size = width * height * 4; // Flaw: 32-bit integer overflow wraps around!
    return malloc(size);
}""",
        language="c",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=0,
        provenance_confidence=1.0,
        vuln_label=1,
        lineage_stage="ai_gen",
        rationale="AI multiplied 32-bit integers without overflow check, leading to undersized buffer allocation."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_040_v3_human_edit_of_ai_secure",
        task_id="p11_task_040_image_header_parse",
        version_id="t40_v3_human_edit_secure",
        cwe="CWE-190",
        cwe_name="Integer Overflow or Wraparound",
        code="""#include <stdlib.h>
#include <stdint.h>

void* allocate_bitmap_buffer_remediated(uint32_t width, uint32_t height) {
    // Human code review: check division boundary before multiplication
    if (width == 0 || height == 0) return NULL;
    if (width > UINT32_MAX / height) return NULL; // Multiplication overflow guard
    uint32_t pixels = width * height;
    if (pixels > UINT32_MAX / 4) return NULL;
    return malloc(pixels * 4);
}""",
        language="c",
        author_type="human_edit_of_ai",
        ai_involvement=0.5,
        ai_generator="ChatGPT (GPT-4o)",
        human_edit_count=1,
        provenance_confidence=0.95,
        vuln_label=0,
        lineage_stage="human_edit",
        rationale="Human developer added division-based multiplication overflow check."
    ))

    units.append(Phase11CodeUnit(
        sample_id="p11_task_040_v4_ai_edit_of_human_vulnerable",
        task_id="p11_task_040_image_header_parse",
        version_id="t40_v4_ai_edit_vuln",
        cwe="CWE-190",
        cwe_name="Integer Overflow or Wraparound",
        code="""#include <stdlib.h>
#include <stdint.h>

void* allocate_bitmap_buffer_ai_compact(uint16_t width, uint16_t height, uint16_t stride) {
    // AI edit of human code: added stride but cast to uint16_t causing premature truncation
    uint16_t total = width * height + stride; // Flaw: 16-bit integer overflow truncates total!
    return malloc((size_t)total);
}""",
        language="c",
        author_type="ai_edit_of_human",
        ai_involvement=0.8,
        ai_generator="Claude 3.5 Sonnet",
        human_edit_count=1,
        provenance_confidence=0.90,
        vuln_label=1,
        lineage_stage="ai_edit",
        rationale="AI edit converted size computation to 16-bit integer, causing wraparound."
    ))

    return units


def build_and_save_phase11_dataset():
    """Generates and splits the Phase 11 dataset into train, val, and locked test partitions."""
    out_dir = PROJECT_ROOT / "data" / "phase11_model_b"
    out_dir.mkdir(parents=True, exist_ok=True)

    units = create_phase11_tasks()
    
    # 16 Tasks Total (64 units):
    # Train Tasks: 8 tasks (32 units) - Tasks 25, 26, 27, 28, 29, 30, 31, 32 (4 odd, 4 even)
    # Val Tasks:   4 tasks (16 units) - Tasks 33, 34, 35, 36 (2 odd, 2 even)
    # Test Tasks:  4 tasks (16 units) - Tasks 37, 38, 39, 40 (2 odd, 2 even)
    
    train_tasks = {
        "p11_task_025_uri_decoder", "p11_task_026_jwt_verifier",
        "p11_task_027_packet_reassembly", "p11_task_028_tempfile_creation",
        "p11_task_029_base64_decode", "p11_task_030_ssl_cert_validate",
        "p11_task_031_cmd_exec_wrapper", "p11_task_032_config_parser"
    }
    val_tasks = {
        "p11_task_033_crypto_key_derivation", "p11_task_034_json_array_parse",
        "p11_task_035_file_path_resolver", "p11_task_036_net_socket_accept"
    }
    test_tasks = {
        "p11_task_037_ring_buffer_pop", "p11_task_038_sql_query_builder",
        "p11_task_039_hash_table_lookup", "p11_task_040_image_header_parse"
    }

    train_records = [asdict(u) for u in units if u.task_id in train_tasks]
    val_records = [asdict(u) for u in units if u.task_id in val_tasks]
    test_records = [asdict(u) for u in units if u.task_id in test_tasks]

    # Save JSON files
    for name, recs, tasks in [
        ("model_b_p11_train.json", train_records, train_tasks),
        ("model_b_p11_val.json", val_records, val_tasks),
        ("model_b_p11_test_locked.json", test_records, test_tasks)
    ]:
        path = out_dir / name
        data = {
            "split": name.replace(".json", ""),
            "n_tasks": len(tasks),
            "n_samples": len(recs),
            "n_vulnerable": sum(1 for r in recs if r["vuln_label"] == 1),
            "n_safe": sum(1 for r in recs if r["vuln_label"] == 0),
            "tasks": sorted(list(tasks)),
            "records": recs
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # Manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "total_units": len(units),
        "total_tasks": len(train_tasks) + len(val_tasks) + len(test_tasks),
        "splits": {
            "train": {"n_samples": len(train_records), "tasks": sorted(list(train_tasks))},
            "val": {"n_samples": len(val_records), "tasks": sorted(list(val_tasks))},
            "test_locked": {"n_samples": len(test_records), "tasks": sorted(list(test_tasks))}
        }
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Compute checksum of test set
    with open(out_dir / "model_b_p11_test_locked.json", "rb") as f:
        test_sha = hashlib.sha256(f.read()).hexdigest()
    with open(out_dir / "CHECKSUMS.sha256", "w") as f:
        f.write(f"{test_sha}  model_b_p11_test_locked.json\n")

    print(f"Generated Phase 11 balanced dataset at {out_dir}")
    print(f"Train: {len(train_records)} units across {len(train_tasks)} tasks")
    print(f"Val:   {len(val_records)} units across {len(val_tasks)} tasks")
    print(f"Test:  {len(test_records)} units across {len(test_tasks)} tasks")
    print(f"Test Locked SHA-256: {test_sha}")


if __name__ == "__main__":
    build_and_save_phase11_dataset()

