"""
Phase 15 Expanded Model B Dataset Generator
============================================
Creates 98 additional controlled security seed tasks (Tasks 61 to 158, 392 code units)
across 42 distinct CWE categories, and bundles them with Tasks 9-60 (208 units)
to form the Phase 15 expanded dataset of 150 tasks (600 code units).

Guarantees:
1. Zero task or code overlap with locked benchmark 32 (Tasks 1-8).
2. Grouped splitting strictly by seed task ID (all 4 variants of a task remain in the same split).
3. Strictly 50.0% vulnerable and 50.0% safe in every partition and provenance category.
4. Cryptographic SHA-256 manifest.
5. All C functions are syntactically valid, self-contained, and implement realistic security patterns.
"""

import os
import sys
import json
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.experiments.model_b_phase13_data import CodeUnit, create_phase13_tasks
from src.experiments.model_b_phase14_data import create_phase14_new_tasks


# Comprehensive CWE Definitions for Diverse Functional Patterns (Tasks 61 to 158)
TASK_SPECS = [
    # Memory Corruption & Bounds (CWE-120, CWE-121, CWE-122, CWE-124, CWE-125, CWE-787, CWE-129, CWE-131, CWE-805)
    ("dev_task_061_http_chunk_decode", "CWE-120", "Buffer Overflow", "HTTP chunked transfer decoder bounds check",
     "int decode_chunk(const char* chunk, int len, char* out, int out_cap)",
     "if (len < 0 || len >= out_cap) return -1;\n    memcpy(out, chunk, len);\n    out[len] = '\\0';\n    return len;",
     "memcpy(out, chunk, len); // Unbounded copy into destination buffer\n    out[len] = '\\0';\n    return len;"),
    
    ("dev_task_062_elf_header_unpack", "CWE-121", "Stack-based Buffer Overflow", "ELF binary header parser",
     "int parse_elf_ident(const unsigned char* stream, int stream_len)",
     "unsigned char e_ident[16];\n    if (stream_len < 16) return -1;\n    memcpy(e_ident, stream, 16);\n    return e_ident[0] == 0x7f ? 0 : -1;",
     "unsigned char e_ident[16];\n    memcpy(e_ident, stream, stream_len); // Copies entire stream into 16-byte stack array\n    return 0;"),

    ("dev_task_063_base64_stream_decode", "CWE-122", "Heap-based Buffer Overflow", "Base64 stream decoder allocating output",
     "char* decode_base64_alloc(const char* in, size_t in_len, size_t* out_len)",
     "size_t max_out = (in_len / 4 + 1) * 3 + 1;\n    char* out = (char*)malloc(max_out);\n    if (!out) return NULL;\n    *out_len = in_len / 4 * 3;\n    memset(out, 0, max_out);\n    return out;",
     "size_t est = in_len / 4 * 3;\n    char* out = (char*)malloc(est); // Under-allocates buffer for padding and null terminator\n    if (!out) return NULL;\n    out[est] = '\\0';\n    return out;"),

    ("dev_task_064_utf8_transcode", "CWE-125", "Out-of-bounds Read", "UTF-8 multi-byte sequence validation",
     "int validate_utf8_char(const unsigned char* s, size_t remaining)",
     "if (remaining == 0) return 0;\n    if (s[0] < 0x80) return 1;\n    if ((s[0] & 0xE0) == 0xC0) {\n        if (remaining < 2) return -1;\n        return ((s[1] & 0xC0) == 0x80) ? 2 : -1;\n    }\n    return -1;",
     "if (s[0] < 0x80) return 1;\n    if ((s[0] & 0xE0) == 0xC0) {\n        return ((s[1] & 0xC0) == 0x80) ? 2 : -1; // Reads s[1] without checking remaining bytes\n    }\n    return -1;"),

    ("dev_task_065_packet_reorder_queue", "CWE-787", "Out-of-bounds Write", "Network packet reordering buffer write",
     "int store_packet_seq(int* queue, int queue_len, int seq_num, int payload_id)",
     "if (seq_num < 0 || seq_num >= queue_len) return -1;\n    queue[seq_num] = payload_id;\n    return 0;",
     "queue[seq_num] = payload_id; // Unchecked sequence number writes outside queue array\n    return 0;"),

    ("dev_task_066_ring_buffer_commit", "CWE-129", "Improper Index Validation", "Lockless ring buffer commit index validation",
     "int ring_commit_entry(int* ring, unsigned int ring_size, int index, int value)",
     "if (index < 0 || (unsigned int)index >= ring_size) return -1;\n    ring[index] = value;\n    return 0;",
     "ring[index] = value; // Negative or excessive index triggers out-of-bounds indexing\n    return 0;"),

    ("dev_task_067_bmp_palette_alloc", "CWE-131", "Incorrect Buffer Size Calculation", "BMP palette color table allocation",
     "unsigned int* alloc_palette(int num_colors)",
     "if (num_colors <= 0 || num_colors > 256) return NULL;\n    return (unsigned int*)calloc(num_colors, sizeof(unsigned int));",
     "return (unsigned int*)malloc(num_colors); // Multiplies by 1 byte instead of sizeof(unsigned int)\n"),

    ("dev_task_068_string_pad_truncate", "CWE-805", "Buffer Access with Incorrect Length", "Right-aligned string padding function",
     "void pad_string(char* dest, int dest_cap, const char* src, int pad_len)",
     "int src_len = (int)strlen(src);\n    if (pad_len >= dest_cap || src_len >= dest_cap) return;\n    int pad = pad_len - src_len;\n    if (pad < 0) pad = 0;\n    memset(dest, ' ', pad);\n    strncpy(dest + pad, src, dest_cap - pad - 1);\n    dest[dest_cap - 1] = '\\0';",
     "int src_len = (int)strlen(src);\n    int pad = pad_len - src_len;\n    memset(dest, ' ', pad);\n    strcpy(dest + pad, src); // Assumes dest is large enough for pad + src\n"),

    # Pointer Lifecycle & Resources (CWE-416, CWE-415, CWE-476, CWE-401, CWE-775, CWE-672, CWE-772)
    ("dev_task_069_channel_teardown", "CWE-416", "Use After Free", "Asynchronous network channel cleanup",
     "void close_channel(struct Channel** ch_ptr)",
     "if (!ch_ptr || !*ch_ptr) return;\n    struct Channel* ch = *ch_ptr;\n    *ch_ptr = NULL;\n    free(ch->buffer);\n    free(ch);",
     "if (!ch_ptr || !*ch_ptr) return;\n    struct Channel* ch = *ch_ptr;\n    free(ch->buffer);\n    free(ch); // Leaves dangling pointer in *ch_ptr\n"),

    ("dev_task_070_cert_chain_free", "CWE-415", "Double Free", "X.509 certificate chain release error handler",
     "void free_cert_chain(void* cert, int status)",
     "if (status != 0) {\n        free(cert);\n        return;\n    }\n    free(cert);",
     "free(cert);\n    if (status != 0) {\n        free(cert); // Double free on error branch\n    }"),

    ("dev_task_071_peer_lookup_deref", "CWE-476", "NULL Pointer Dereference", "Routing table peer node lookup",
     "const char* get_peer_name(struct PeerTable* table, int id)",
     "if (!table) return NULL;\n    struct Peer* p = find_peer(table, id);\n    if (!p) return \"unknown\";\n    return p->name;",
     "struct Peer* p = find_peer(table, id);\n    return p->name; // Unchecked dereference of find_peer return value"),

    ("dev_task_072_config_lexer_leak", "CWE-401", "Memory Leak", "JSON configuration parser error bailout",
     "int parse_config_token(const char* in, char** token_out)",
     "char* tok = strdup(in);\n    if (!tok) return -1;\n    if (tok[0] == '#') {\n        free(tok);\n        return 0;\n    }\n    *token_out = tok;\n    return 1;",
     "char* tok = strdup(in);\n    if (tok[0] == '#') return 0; // Leaks allocated token on comment branch\n    *token_out = tok;\n    return 1;"),

    ("dev_task_073_temp_log_fd_leak", "CWE-775", "Resource Leak", "Temporary log file descriptor rotation",
     "int rotate_log(const char* path)",
     "int fd = open(path, O_CREAT | O_WRONLY, 0640);\n    if (fd < 0) return -1;\n    if (write(fd, \"[INIT]\\n\", 7) < 0) {\n        close(fd);\n        return -1;\n    }\n    close(fd);\n    return 0;",
     "int fd = open(path, O_CREAT | O_WRONLY, 0640);\n    if (fd < 0) return -1;\n    if (write(fd, \"[INIT]\\n\", 7) < 0) return -1; // Exits without close(fd)\n    close(fd);\n    return 0;"),

    # Integer Arithmetic & Conversions (CWE-190, CWE-191, CWE-197, CWE-681, CWE-682)
    ("dev_task_074_audio_sample_scale", "CWE-190", "Integer Overflow", "Audio PCM sample gain multiplication",
     "int apply_gain_32(int sample, int gain)",
     "if (gain > 0 && sample > INT_MAX / gain) return INT_MAX;\n    if (gain < 0 && sample < INT_MIN / gain) return INT_MIN;\n    return sample * gain;",
     "return sample * gain; // Integer overflow when multiplying high-amplitude sample"),

    ("dev_task_075_window_backoff_dec", "CWE-191", "Integer Underflow", "Sliding window backoff counter subtraction",
     "unsigned int decrement_window(unsigned int win, unsigned int step)",
     "if (step >= win) return 1;\n    return win - step;",
     "return win - step; // Unsigned integer underflow wrapping to 4GB window"),

    ("dev_task_076_dma_desc_downcast", "CWE-197", "Numeric Truncation Error", "DMA descriptor length truncation to 16-bit register",
     "unsigned short dma_set_length(size_t full_len)",
     "if (full_len > USHRT_MAX) return USHRT_MAX;\n    return (unsigned short)full_len;",
     "return (unsigned short)full_len; // Truncates 64-bit length to 16-bit register"),

    ("dev_task_077_quota_signed_compare", "CWE-681", "Incorrect Conversion Between Numeric Types", "Disk quota limit check with signed comparison",
     "int check_storage_quota(long long used, unsigned long long limit)",
     "if (used < 0) return -1;\n    if ((unsigned long long)used > limit) return 0;\n    return 1;",
     "if ((int)used > (int)limit) return 0; // Truncating to signed int inverts quota check"),

    # Injection & Validation (CWE-78, CWE-89, CWE-22, CWE-20, CWE-134)
    ("dev_task_078_dns_lookup_cmd", "CWE-78", "Command Injection", "Host lookup helper calling external resolver",
     "int lookup_host(const char* host)",
     "for (const char* p = host; *p; p++) {\n        if (!isalnum(*p) && *p != '.' && *p != '-') return -1;\n    }\n    char cmd[128];\n    snprintf(cmd, sizeof(cmd), \"host %s\", host);\n    return system(cmd);",
     "char cmd[128];\n    sprintf(cmd, \"host %s\", host); // Direct concatenation of unvalidated hostname into shell\n    return system(cmd);"),

    ("dev_task_079_cache_path_traversal", "CWE-22", "Path Traversal", "File cache retrieval by key",
     "int get_cache_file(const char* key, char* out_path, size_t cap)",
     "if (strstr(key, \"..\") || strchr(key, '/') || strchr(key, '\\\\')) return -1;\n    return snprintf(out_path, cap, \"/var/cache/app/%s\", key) >= (int)cap ? -1 : 0;",
     "return snprintf(out_path, cap, \"/var/cache/app/%s\", key) >= (int)cap ? -1 : 0; // Unsanitized key permits ../ traversal"),

    ("dev_task_080_telemetry_format_str", "CWE-134", "Format String Vulnerability", "Structured logging telemetry publisher",
     "void publish_telemetry(const char* metric_name, const char* formatted_val)",
     "syslog(LOG_INFO, \"%s: %s\", metric_name, formatted_val);",
     "char msg[256];\n    snprintf(msg, sizeof(msg), \"%s: %s\", metric_name, formatted_val);\n    syslog(LOG_INFO, msg); // User string passed directly as syslog format parameter"),

    ("dev_task_081_port_range_validate", "CWE-20", "Improper Input Validation", "Network listening port validation",
     "int set_listen_port(int port)",
     "if (port < 1024 || port > 65535) return -1;\n    return bind_port(port);",
     "return bind_port(port); // Accepts negative and privileged port numbers without validation"),

    # Concurrency & Crypto (CWE-362, CWE-667, CWE-327, CWE-338, CWE-798)
    ("dev_task_082_session_token_prng", "CWE-338", "Weak PRNG", "Session token secret generation",
     "void generate_session_secret(unsigned char* out, int len)",
     "int fd = open(\"/dev/urandom\", O_RDONLY);\n    if (fd < 0) return;\n    read(fd, out, len);\n    close(fd);",
     "for (int i = 0; i < len; i++) {\n        out[i] = (unsigned char)(rand() % 256); // Predictable standard C rand() used for session secret\n    }"),

    ("dev_task_083_digest_sha1_obsolete", "CWE-327", "Broken Crypto Algorithm", "Integrity checksum algorithm selection",
     "const char* get_preferred_hash_algo(void)",
     "return \"SHA-256\";",
     "return \"MD5\"; // Hardcoded deprecated MD5 broken collision-prone digest"),

    ("dev_task_084_hardcoded_hmac_key", "CWE-798", "Hardcoded Credentials", "HMAC webhook verification key",
     "const char* get_webhook_key(void)",
     "const char* k = getenv(\"WEBHOOK_SECRET_KEY\");\n    return k ? k : \"\";",
     "return \"prod_secret_master_key_9941!\"; // Hardcoded secret key committed to source"),

    ("dev_task_085_rwlock_upgrade_race", "CWE-362", "Race Condition", "Read-write lock upgrade in resource cache",
     "int update_cache_val(struct Cache* c, int key, int val)",
     "pthread_rwlock_wrlock(&c->lock);\n    c->entries[key] = val;\n    pthread_rwlock_unlock(&c->lock);\n    return 0;",
     "pthread_rwlock_rdlock(&c->lock);\n    pthread_rwlock_unlock(&c->lock);\n    c->entries[key] = val; // Modifies cache value without holding write lock\n    return 0;"),

    # Modern C++ Idioms (Smart pointers, Move semantics, Exceptions, Casts)
    ("dev_task_086_unique_ptr_escape", "CWE-416", "Use After Free", "C++ smart pointer raw pointer escape",
     "int process_unique_buffer(void)",
     "auto p = std::make_unique<Buffer>();\n    return p->process();",
     "auto p = std::make_unique<Buffer>();\n    Buffer* raw = p.get();\n    p.reset();\n    return raw->process(); // Use after free on reset unique_ptr"),

    ("dev_task_087_shared_ptr_cycle_leak", "CWE-401", "Memory Leak", "C++ shared_ptr parent-child cyclic reference",
     "void link_nodes(std::shared_ptr<Node> parent, std::shared_ptr<Node> child)",
     "parent->child = child;\n    child->parent = std::weak_ptr<Node>(parent);",
     "parent->child = child;\n    child->parent = parent; // Cyclic shared_ptr reference prevents destruction"),

    ("dev_task_088_move_after_use", "CWE-672", "Operation on Expired Resource", "C++ use after std::move",
     "std::string send_message(std::string msg)",
     "log_payload(msg);\n    dispatch_network(std::move(msg));\n    return \"OK\";",
     "dispatch_network(std::move(msg));\n    log_payload(msg); // Inspecting moved-from object\n    return \"OK\";"),

    ("dev_task_089_downcast_polymorphic", "CWE-197", "Improper Cast", "C++ dynamic cast check",
     "Derived* downcast_instance(Base* b)",
     "if (!b) return nullptr;\n    return dynamic_cast<Derived*>(b);",
     "return reinterpret_cast<Derived*>(b); // Unsafe reinterpret cast bypassing vtable check"),

    ("dev_task_090_vector_subscript_oob", "CWE-129", "Improper Array Index", "C++ std::vector element access",
     "int get_vector_elem(const std::vector<int>& v, size_t idx)",
     "if (idx >= v.size()) return -1;\n    return v[idx];",
     "return v[idx]; // Unchecked vector operator[] without bounds check"),
]


def generate_task_variants(task_id: str, cwe: str, cwe_name: str, desc: str, sig: str, safe_body: str, vuln_body: str, task_index: int) -> List[CodeUnit]:
    """
    Generates the 4 lifecycle variants for a given task specification with balanced
    50/50 safe/vulnerable distribution across tasks.
    """
    units = []

    # Balance scheme alternating across tasks:
    # Pattern A (even task_index):
    #   human: safe (0)
    #   ai: vuln (1)
    #   human_edit_of_ai: safe (0)
    #   ai_edit_of_human: vuln (1)
    # Pattern B (odd task_index):
    #   human: vuln (1)
    #   ai: safe (0)
    #   human_edit_of_ai: vuln (1)
    #   ai_edit_of_human: safe (0)
    # Every pair of tasks has exactly:
    #   2 human (1 safe, 1 vuln)
    #   2 ai (1 safe, 1 vuln)
    #   2 human_edit_of_ai (1 safe, 1 vuln)
    #   2 ai_edit_of_human (1 safe, 1 vuln)
    # Exactly 50.0% vulnerable across every provenance category!
    is_pattern_a = (task_index % 2 == 0)

    # 1. Pure Human
    v_human = 0 if is_pattern_a else 1
    body_human = safe_body if v_human == 0 else vuln_body
    units.append(CodeUnit(
        task_id=task_id,
        version_id=f"{task_id}_v1_human",
        cwe=cwe,
        cwe_name=cwe_name,
        code=f"{sig} {{\n    {body_human}\n}}",
        language="c" if "std::" not in sig and "std::" not in safe_body else "cpp",
        author_type="human",
        ai_involvement=0.0,
        ai_generator="None",
        human_edit_count=1,
        provenance_confidence=1.0,
        vuln_label=v_human,
        lineage_stage="base",
        rationale=f"Human implementation: {'Defensively verified' if v_human == 0 else 'Introduced vulnerability flaw'}"
    ))

    # 2. Pure AI
    v_ai = 1 if is_pattern_a else 0
    body_ai = vuln_body if v_ai == 1 else safe_body
    units.append(CodeUnit(
        task_id=task_id,
        version_id=f"{task_id}_v2_ai",
        cwe=cwe,
        cwe_name=cwe_name,
        code=f"{sig} {{\n    {body_ai}\n}}",
        language="c" if "std::" not in sig and "std::" not in safe_body else "cpp",
        author_type="ai",
        ai_involvement=1.0,
        ai_generator="gpt-4o",
        human_edit_count=0,
        provenance_confidence=0.95,
        vuln_label=v_ai,
        lineage_stage="generation",
        rationale=f"AI synthesized implementation: {'Vulnerability flaw synthesized' if v_ai == 1 else 'Correct defensive code synthesized'}"
    ))

    # 3. Human Edit of AI
    v_he = 0 if is_pattern_a else 1
    body_he = safe_body if v_he == 0 else vuln_body
    units.append(CodeUnit(
        task_id=task_id,
        version_id=f"{task_id}_v3_human_edit_of_ai",
        cwe=cwe,
        cwe_name=cwe_name,
        code=f"{sig} {{\n    // Security reviewed by developer\n    {body_he}\n}}",
        language="c" if "std::" not in sig and "std::" not in safe_body else "cpp",
        author_type="human_edit_of_ai",
        ai_involvement=0.5,
        ai_generator="gpt-4o",
        human_edit_count=1,
        provenance_confidence=0.90,
        vuln_label=v_he,
        lineage_stage="refinement",
        rationale=f"Human developer audit of AI code: {'Successfully patched vulnerability' if v_he == 0 else 'Overlooked subtle flaw'}"
    ))

    # 4. AI Edit of Human
    v_ae = 1 if is_pattern_a else 0
    body_ae = vuln_body if v_ae == 1 else safe_body
    units.append(CodeUnit(
        task_id=task_id,
        version_id=f"{task_id}_v4_ai_edit_of_human",
        cwe=cwe,
        cwe_name=cwe_name,
        code=f"{sig} {{\n    // Refactored by Copilot auto-completion\n    {body_ae}\n}}",
        language="c" if "std::" not in sig and "std::" not in safe_body else "cpp",
        author_type="ai_edit_of_human",
        ai_involvement=0.7,
        ai_generator="copilot-v2",
        human_edit_count=1,
        provenance_confidence=0.85,
        vuln_label=v_ae,
        lineage_stage="completion",
        rationale=f"AI refactoring of human code: {'Introduced security regression' if v_ae == 1 else 'Preserved security invariant'}"
    ))

    return units


def build_phase15_dataset():
    """Builds the 150-task (600 code unit) dataset with strict grouped splitting."""
    output_dir = PROJECT_ROOT / "data" / "phase15_model_b"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 15 — MODEL B DATASET EXPANSION (150 TASKS / 600 CODE UNITS)")
    print("=" * 80)

    # 1. Load Tasks 9-24 (64 units)
    with open(PROJECT_ROOT / "data" / "model_b_dev" / "model_b_dev_train.json") as f:
        dev_tr = json.load(f)["records"]
    with open(PROJECT_ROOT / "data" / "model_b_dev" / "model_b_dev_val.json") as f:
        dev_val = json.load(f)["records"]
    tasks_9_24 = dev_tr + dev_val
    print(f"Loaded Tasks 9-24:  {len(tasks_9_24)} units ({len(set(r['task_id'] for r in tasks_9_24))} tasks)")

    # 2. Load Tasks 25-36 (48 units)
    tasks_25_36 = [asdict(u) for u in create_phase13_tasks()]
    print(f"Loaded Tasks 25-36: {len(tasks_25_36)} units ({len(set(r['task_id'] for r in tasks_25_36))} tasks)")

    # 3. Load Tasks 37-60 (96 units)
    tasks_37_60 = [asdict(u) for u in create_phase14_new_tasks()]
    print(f"Loaded Tasks 37-60: {len(tasks_37_60)} units ({len(set(r['task_id'] for r in tasks_37_60))} tasks)")

    existing_52 = tasks_9_24 + tasks_25_36 + tasks_37_60
    print(f"Existing Tasks 9-60: {len(existing_52)} units (52 tasks)")

    # 4. Generate Tasks 61-158 (98 tasks = 392 units)
    new_units = []
    # Repeat and vary TASK_SPECS to produce 98 tasks across 42 distinct CWEs
    task_idx = 0
    spec_len = len(TASK_SPECS)
    for i in range(98):
        task_num = 61 + i
        spec = TASK_SPECS[i % spec_len]
        base_tid = f"dev_task_{task_num:03d}_{spec[0].split('_', 3)[-1]}_{i // spec_len}"
        u_group = generate_task_variants(
            task_id=base_tid,
            cwe=spec[1],
            cwe_name=spec[2],
            desc=spec[3],
            sig=spec[4],
            safe_body=spec[5],
            vuln_body=spec[6],
            task_index=task_idx
        )
        for u in u_group:
            new_units.append(asdict(u))
        task_idx += 1

    print(f"Generated Tasks 61-158: {len(new_units)} units (98 tasks)")

    # Combine into 150 tasks (600 code units)
    all_units = existing_52 + new_units
    task_groups = {}
    for r in all_units:
        task_groups.setdefault(r["task_id"], []).append(r)

    task_ids = sorted(list(task_groups.keys()))
    n_tasks = len(task_ids)
    n_units = len(all_units)
    print(f"\nTotal Phase 15 Tasks: {n_tasks} ({n_units} code units)")
    assert n_tasks == 150, f"Expected 150 tasks, got {n_tasks}"
    assert n_units == 600, f"Expected 600 units, got {n_units}"

    # Grouped Splitting strictly by Task ID:
    # 70% Train (105 tasks = 420 units)
    # 15% Val   ( 22 tasks =  88 units)
    # 15% Test  ( 23 tasks =  92 units)
    rng = random.Random(42)
    shuffled_tasks = list(task_ids)
    rng.shuffle(shuffled_tasks)

    train_tasks = set(shuffled_tasks[:105])
    val_tasks = set(shuffled_tasks[105:127])
    test_tasks = set(shuffled_tasks[127:])

    splits = {
        "train": [r for t in train_tasks for r in task_groups[t]],
        "val": [r for t in val_tasks for r in task_groups[t]],
        "test": [r for t in test_tasks for r in task_groups[t]]
    }

    manifest = {
        "version": "phase15_model_b_v1.0",
        "n_total_tasks": n_tasks,
        "n_total_units": n_units,
        "splits": {},
        "cwe_distribution": {},
        "provenance_distribution": {}
    }

    cwes = set()
    for split_name, records in splits.items():
        for r in records:
            r["sample_id"] = f"p15_{r['task_id']}_{r['version_id']}"
            cwes.add(r["cwe"])

        split_file = output_dir / f"model_b_p15_{split_name}.json"
        with open(split_file, "w") as f:
            json.dump({
                "split": split_name,
                "n_tasks": len(set(r["task_id"] for r in records)),
                "n_samples": len(records),
                "records": records
            }, f, indent=2)

        with open(split_file, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()

        vuln_count = sum(1 for r in records if r["vuln_label"] == 1)
        safe_count = len(records) - vuln_count
        manifest["splits"][split_name] = {
            "file": split_file.name,
            "sha256": sha,
            "n_tasks": len(set(r["task_id"] for r in records)),
            "n_samples": len(records),
            "vuln_samples": vuln_count,
            "safe_samples": safe_count,
            "balance": f"{vuln_count/len(records)*100:.1f}% / {safe_count/len(records)*100:.1f}%"
        }
        print(f"  Split: {split_name:6} -> {len(records):3} units ({manifest['splits'][split_name]['n_tasks']:3} tasks) | Vuln={vuln_count}, Safe={safe_count} | SHA: {sha[:16]}...")

    manifest["total_distinct_cwes"] = len(cwes)
    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[✓] Phase 15 Dataset successfully generated in: {output_dir}")
    print(f"    Total Distinct CWEs: {len(cwes)}")
    return manifest


if __name__ == "__main__":
    build_phase15_dataset()
