# Execution Oracle Audit Report
**Date:** 2026-11-22 01:28:28 UTC
**Total Units Evaluated:** 900
**Total Crashing (exec_label=1):** 72 (8.0%)
**Total Survived (exec_label=0):** 199 (22.1%)

## Part 1: Sample of 20 Crashing Units (exec_label = 1)
Below are 20 randomly sampled crashing units with their code and exact fuzzer sanitizer output for manual review.

### Crash Sample 1: `dev_task_126_ring_buffer_commit_2_v1_human`
- **Task:** `dev_task_126_ring_buffer_commit_2`
- **Role:** `v1_human`
- **Crash Classification:** `Crash (returncode -6)`

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

int ring_commit_entry(int* ring, unsigned int ring_size, int index, int value) {
    ring[index] = value; // Negative or excessive index triggers out-of-bounds indexing
    return 0;
}
```

**Fuzzer / Sanitizer Log:**
```text
INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 1447214085
INFO: Loaded 1 modules   (25 inline 8-bit counters): 25 [0x100550168, 0x100550181), 
INFO: Loaded 1 PC tables (25 PCs): 25 [0x100550188,0x100550318), 
INFO:       26 files found in /tmp/fuzzer_seed_corpus
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: seed corpus: files: 26 min: 1b max: 4096b total: 4724b rss: 34Mb
AddressSanitizer:DEADLYSIGNAL
=================================================================
==14100==ERROR: AddressSanitizer: BUS on unknown address (p
```

### Crash Sample 2: `dev_task_023_dangerous_function_pure_ai_real_t02`
- **Task:** `dev_task_023_dangerous_function`
- **Role:** `pure_ai_real_t02`
- **Crash Classification:** `Timeout: hang (>70s)`

```c
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include <stdio.h>

char *read_line_safely(char *buf, int max_size) {
    if (buf == NULL || max_size <= 0) {
        return NULL;
    }

    if (fgets(buf, max_size, stdin) == NULL) {
        return NULL;
    }

    int i = 0;
    while (buf[i] != '\0') {
        if (buf[i] == '\n') {
            buf[i] = '\0';
    
```

**Fuzzer / Sanitizer Log:**
```text
Subprocess timeout after 70s
```

### Crash Sample 3: `dev_task_100_cert_chain_free_1_v4_ai_edit_of_human`
- **Task:** `dev_task_100_cert_chain_free_1`
- **Role:** `v4_ai_edit_of_human`
- **Crash Classification:** `UBSan: runtime-error`

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

void free_cert_chain(void* cert, int status) {
    // Refactored by Copilot auto-completion
    if (status != 0) {
        free(cert);
        return;
    }
    free(cert);
}
```

**Fuzzer / Sanitizer Log:**
```text
INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 1758478485
INFO: Loaded 1 modules   (17 inline 8-bit counters): 17 [0x1000f0158, 0x1000f0169), 
INFO: Loaded 1 PC tables (17 PCs): 17 [0x1000f0170,0x1000f0280), 
INFO:       69 files found in /tmp/fuzzer_seed_corpus
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: seed corpus: files: 69 min: 1b max: 4096b total: 18750b rss: 34Mb
/var/folders/5p/_b7gs3q51472fq2l6wykzppr0000gn/T/fuzz_run_p10_1co0d48v/fuzz_dev_task_100_cert_chain_free_1_v4_ai_edit_of_human_13993.c:40:14: runtime erro
```

### Crash Sample 4: `dev_task_156_ring_buffer_commit_3_v1_human`
- **Task:** `dev_task_156_ring_buffer_commit_3`
- **Role:** `v1_human`
- **Crash Classification:** `Crash (returncode -6)`

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

int ring_commit_entry(int* ring, unsigned int ring_size, int index, int value) {
    ring[index] = value; // Negative or excessive index triggers out-of-bounds indexing
    return 0;
}
```

**Fuzzer / Sanitizer Log:**
```text
INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 1385790693
INFO: Loaded 1 modules   (25 inline 8-bit counters): 25 [0x100f50168, 0x100f50181), 
INFO: Loaded 1 PC tables (25 PCs): 25 [0x100f50188,0x100f50318), 
INFO:       20 files found in /tmp/fuzzer_seed_corpus
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: seed corpus: files: 20 min: 1b max: 4096b total: 4647b rss: 34Mb
#21	INITED cov: 5 ft: 5 corp: 2/14b exec/s: 0 rss: 35Mb
AddressSanitizer:DEADLYSIGNAL
=================================================================
==1
```

### Crash Sample 5: `dev_task_144_hardcoded_hmac_key_2_v1_human`
- **Task:** `dev_task_144_hardcoded_hmac_key_2`
- **Role:** `v1_human`
- **Crash Classification:** `ASan: global-buffer-overflow`

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

const char* get_webhook_key(void) {
    return "prod_secret_master_key_9941!"; // Hardcoded secret key committed to source
}
```

**Fuzzer / Sanitizer Log:**
```text
INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 2135126845
INFO: Loaded 1 modules   (7 inline 8-bit counters): 7 [0x102d80118, 0x102d8011f), 
INFO: Loaded 1 PC tables (7 PCs): 7 [0x102d80120,0x102d80190), 
INFO:       93 files found in /tmp/fuzzer_seed_corpus
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: seed corpus: files: 93 min: 1b max: 4096b total: 21146b rss: 35Mb
=================================================================
==14932==ERROR: AddressSanitizer: global-buffer-overflow on address 0x000102d726e0 at pc 0x0
```

### Crash Sample 6: `dev_task_067_bmp_palette_alloc_0_pure_ai_real_t02`
- **Task:** `dev_task_067_bmp_palette_alloc_0`
- **Role:** `pure_ai_real_t02`
- **Crash Classification:** `Crash (returncode 71)`

```c
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include <stdlib.h>

unsigned int* alloc_palette(int num_colors) {
    if (num_colors <= 0) {
        return NULL;
    }
    
    unsigned int* palette = (unsigned int*)malloc(num_colors * sizeof(unsigned int));
    return palette;
}
```

**Fuzzer / Sanitizer Log:**
```text
INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 2563726256
INFO: Loaded 1 modules   (11 inline 8-bit counters): 11 [0x10011c108, 0x10011c113), 
INFO: Loaded 1 PC tables (11 PCs): 11 [0x10011c118,0x10011c1c8), 
INFO:      143 files found in /tmp/fuzzer_seed_corpus
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: seed corpus: files: 143 min: 1b max: 4096b total: 21592b rss: 35Mb
==15500== ERROR: libFuzzer: out-of-memory (malloc(3638088916))
   To change the out-of-memory limit use -rss_limit_mb=<N>

    #0 0x0001009113a8 in __san
```

### Crash Sample 7: `t33_v2_ai_naive`
- **Task:** `dev_task_033_reachable_assertion`
- **Role:** `t33_v2_ai_naive`
- **Crash Classification:** `Crash (returncode 77)`

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include <assert.h>

int process_client_command(int cmd_id, int session_state) {
    // AI naive: uses assert() on untrusted user parameters, allowing remote DoS crash!
    assert(cmd_id >= 0 && cmd_id <= 100);
    assert(session_state == 1);
    return cmd_id * 2;
}
```

**Fuzzer / Sanitizer Log:**
```text
INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 1632430241
INFO: Loaded 1 modules   (11 inline 8-bit counters): 11 [0x1026c8130, 0x1026c813b), 
INFO: Loaded 1 PC tables (11 PCs): 11 [0x1026c8140,0x1026c81f0), 
INFO:       44 files found in /tmp/fuzzer_seed_corpus
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: seed corpus: files: 44 min: 1b max: 4096b total: 4867b rss: 34Mb
Assertion failed: (session_state == 1), function process_client_command, file fuzz_t33_v2_ai_naive_13989.c, line 12.
==14317== ERROR: libFuzzer: deadly sig
```

### Crash Sample 8: `dev_task_104_audio_sample_scale_1_v1_human`
- **Task:** `dev_task_104_audio_sample_scale_1`
- **Role:** `v1_human`
- **Crash Classification:** `UBSan: runtime-error`

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

int apply_gain_32(int sample, int gain) {
    return sample * gain; // Integer overflow when multiplying high-amplitude sample
}
```

**Fuzzer / Sanitizer Log:**
```text
INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 2134119597
INFO: Loaded 1 modules   (9 inline 8-bit counters): 9 [0x101038110, 0x101038119), 
INFO: Loaded 1 PC tables (9 PCs): 9 [0x101038120,0x1010381b0), 
INFO:       93 files found in /tmp/fuzzer_seed_corpus
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: seed corpus: files: 93 min: 1b max: 4096b total: 21146b rss: 34Mb
/var/folders/5p/_b7gs3q51472fq2l6wykzppr0000gn/T/fuzz_run_p10_1co0d48v/fuzz_dev_task_104_audio_sample_scale_1_v1_human_13996.c:8:19: runtime error: signed int
```

### Crash Sample 9: `dev_task_100_cert_chain_free_1_v1_human`
- **Task:** `dev_task_100_cert_chain_free_1`
- **Role:** `v1_human`
- **Crash Classification:** `UBSan: runtime-error`

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

void free_cert_chain(void* cert, int status) {
    free(cert);
    if (status != 0) {
        free(cert); // Double free on error branch
    }
}
```

**Fuzzer / Sanitizer Log:**
```text
INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 1755343429
INFO: Loaded 1 modules   (21 inline 8-bit counters): 21 [0x102aac158, 0x102aac16d), 
INFO: Loaded 1 PC tables (21 PCs): 21 [0x102aac170,0x102aac2c0), 
INFO:       62 files found in /tmp/fuzzer_seed_corpus
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: seed corpus: files: 62 min: 1b max: 4096b total: 18202b rss: 34Mb
/var/folders/5p/_b7gs3q51472fq2l6wykzppr0000gn/T/fuzz_run_p10_1co0d48v/fuzz_dev_task_100_cert_chain_free_1_v1_human_13990.c:38:14: runtime error: load of 
```

### Crash Sample 10: `dev_task_033_reachable_assertion_pure_ai_real_t08`
- **Task:** `dev_task_033_reachable_assertion`
- **Role:** `pure_ai_real_t08`
- **Crash Classification:** `Crash (returncode 77)`

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include <assert.h>

int process_client_command(int cmd_id, int session_state) {
    // Ensure that the session state is within a valid range
    // and that the command ID is non-negative, making this assertion reachable
    // under valid test conditions.
    assert(session_state >= 0 && session_
```

**Fuzzer / Sanitizer Log:**
```text
INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 2603132029
INFO: Loaded 1 modules   (13 inline 8-bit counters): 13 [0x100a4c118, 0x100a4c125), 
INFO: Loaded 1 PC tables (13 PCs): 13 [0x100a4c128,0x100a4c1f8), 
INFO:      143 files found in /tmp/fuzzer_seed_corpus
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: seed corpus: files: 143 min: 1b max: 4096b total: 21592b rss: 35Mb
Assertion failed: (session_state >= 0 && session_state <= 3), function process_client_command, file fuzz_dev_task_033_reachable_assertion_pure_ai_real_t0
```

### Crash Sample 11: `t35_v4_ai_regressive`
- **Task:** `dev_task_035_path_manipulation`
- **Role:** `t35_v4_ai_regressive`
- **Crash Classification:** `Crash (returncode -6)`

```c
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

#include <stdlib.h>
#include <string.h>

char *resolve_canonical_path(const char *rel_path) {
    char buf[128];
    // AI edit: increased buffer from 64 to 128, but still far below PATH_MAX (4096)!
    if (realpath(rel_path, buf) == NULL) return NULL;
    return strdup(buf);
}
```

**Fuzzer / Sanitizer Log:**
```text
INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 2473247616
INFO: Loaded 1 modules   (12 inline 8-bit counters): 12 [0x104cb00f8, 0x104cb0104), 
INFO: Loaded 1 PC tables (12 PCs): 12 [0x104cb0108,0x104cb01c8), 
INFO:      114 files found in /tmp/fuzzer_seed_corpus
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: seed corpus: files: 114 min: 1b max: 4096b total: 21482b rss: 35Mb

```

### Crash Sample 12: `dev_task_144_hardcoded_hmac_key_2_v4_ai_edit_of_human`
- **Task:** `dev_task_144_hardcoded_hmac_key_2`
- **Role:** `v4_ai_edit_of_human`
- **Crash Classification:** `ASan: global-buffer-overflow`

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

const char* get_webhook_key(void) {
    // Refactored by Copilot auto-completion
    const char* k = getenv("WEBHOOK_SECRET_KEY");
    return k ? k : "";
}
```

**Fuzzer / Sanitizer Log:**
```text
INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 2138204788
INFO: Loaded 1 modules   (11 inline 8-bit counters): 11 [0x100edc118, 0x100edc123), 
INFO: Loaded 1 PC tables (11 PCs): 11 [0x100edc128,0x100edc1d8), 
INFO:       93 files found in /tmp/fuzzer_seed_corpus
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: seed corpus: files: 93 min: 1b max: 4096b total: 21146b rss: 35Mb
=================================================================
==14955==ERROR: AddressSanitizer: global-buffer-overflow on address 0x000100ece781 at pc
```

### Crash Sample 13: `dev_task_070_cert_chain_free_0_v3_human_edit_of_ai`
- **Task:** `dev_task_070_cert_chain_free_0`
- **Role:** `v3_human_edit_of_ai`
- **Crash Classification:** `Crash (returncode -6)`

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

void free_cert_chain(void* cert, int status) {
    // Security reviewed by developer
    free(cert);
    if (status != 0) {
        free(cert); // Double free on error branch
    }
}
```

**Fuzzer / Sanitizer Log:**
```text
INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 1570094140
INFO: Loaded 1 modules   (15 inline 8-bit counters): 15 [0x1021b4118, 0x1021b4127), 
INFO: Loaded 1 PC tables (15 PCs): 15 [0x1021b4128,0x1021b4218), 
INFO:       44 files found in /tmp/fuzzer_seed_corpus
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: seed corpus: files: 44 min: 1b max: 4096b total: 4867b rss: 34Mb
=================================================================
==14214==ERROR: AddressSanitizer: attempting double-free on 0x602000000630 in thread T0:

```

### Crash Sample 14: `dev_task_130_cert_chain_free_2_v4_ai_edit_of_human`
- **Task:** `dev_task_130_cert_chain_free_2`
- **Role:** `v4_ai_edit_of_human`
- **Crash Classification:** `ASan: heap-use-after-free`

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

void free_cert_chain(void* cert, int status) {
    // Refactored by Copilot auto-completion
    if (status != 0) {
        free(cert);
        return;
    }
    free(cert);
}
```

**Fuzzer / Sanitizer Log:**
```text
INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 2448912069
INFO: Loaded 1 modules   (5 inline 8-bit counters): 5 [0x100340108, 0x10034010d), 
INFO: Loaded 1 PC tables (5 PCs): 5 [0x100340110,0x100340160), 
INFO:      113 files found in /tmp/fuzzer_seed_corpus
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: seed corpus: files: 113 min: 1b max: 4096b total: 21481b rss: 35Mb
=================================================================
==15377==ERROR: AddressSanitizer: heap-use-after-free on address 0x6020000006d0 at pc 0x000
```

### Crash Sample 15: `dev_task_126_ring_buffer_commit_2_v3_human_edit_of_ai`
- **Task:** `dev_task_126_ring_buffer_commit_2`
- **Role:** `v3_human_edit_of_ai`
- **Crash Classification:** `Crash (returncode -6)`

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

int ring_commit_entry(int* ring, unsigned int ring_size, int index, int value) {
    // Security reviewed by developer
    ring[index] = value; // Negative or excessive index triggers out-of-bounds indexing
    return 0;
}
```

**Fuzzer / Sanitizer Log:**
```text
INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 1448255354
INFO: Loaded 1 modules   (25 inline 8-bit counters): 25 [0x1022d8168, 0x1022d8181), 
INFO: Loaded 1 PC tables (25 PCs): 25 [0x1022d8188,0x1022d8318), 
INFO:       26 files found in /tmp/fuzzer_seed_corpus
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: seed corpus: files: 26 min: 1b max: 4096b total: 4724b rss: 34Mb
AddressSanitizer:DEADLYSIGNAL
=================================================================
==14115==ERROR: AddressSanitizer: BUS on unknown address (p
```

### Crash Sample 16: `dev_task_114_hardcoded_hmac_key_1_v4_ai_edit_of_human`
- **Task:** `dev_task_114_hardcoded_hmac_key_1`
- **Role:** `v4_ai_edit_of_human`
- **Crash Classification:** `ASan: global-buffer-overflow`

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

const char* get_webhook_key(void) {
    // Refactored by Copilot auto-completion
    const char* k = getenv("WEBHOOK_SECRET_KEY");
    return k ? k : "";
}
```

**Fuzzer / Sanitizer Log:**
```text
INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 1821781190
INFO: Loaded 1 modules   (10 inline 8-bit counters): 10 [0x1003f40f8, 0x1003f4102), 
INFO: Loaded 1 PC tables (10 PCs): 10 [0x1003f4108,0x1003f41a8), 
INFO:       69 files found in /tmp/fuzzer_seed_corpus
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: seed corpus: files: 69 min: 1b max: 4096b total: 18750b rss: 35Mb
=================================================================
==14585==ERROR: AddressSanitizer: global-buffer-overflow on address 0x0001003e6761 at pc
```

### Crash Sample 17: `dev_task_100_cert_chain_free_1_pure_ai_real_t08`
- **Task:** `dev_task_100_cert_chain_free_1`
- **Role:** `pure_ai_real_t08`
- **Crash Classification:** `ASan: heap-buffer-overflow`

```c
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 
 * Assuming a hypothetical certificate structure or API 
 * based on common C-based crypto libraries (like OpenSSL or similar).
 * Since the exact internal structure of void* cert is not provided,
 * we handle the generic deallocation safely. 
 * If 'cer
```

**Fuzzer / Sanitizer Log:**
```text
INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 2698119501
INFO: Loaded 1 modules   (29 inline 8-bit counters): 29 [0x104d481d8, 0x104d481f5), 
INFO: Loaded 1 PC tables (29 PCs): 29 [0x104d481f8,0x104d483c8), 
INFO:      187 files found in /tmp/fuzzer_seed_corpus
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: seed corpus: files: 187 min: 1b max: 4096b total: 23570b rss: 35Mb
/var/folders/5p/_b7gs3q51472fq2l6wykzppr0000gn/T/fuzz_run_p10_1co0d48v/fuzz_dev_task_100_cert_chain_free_1_pure_ai_real_t08_13993.c:78:14: runtime error:
```

### Crash Sample 18: `dev_task_065_packet_reorder_queue_0_v4_ai_edit_of_human`
- **Task:** `dev_task_065_packet_reorder_queue_0`
- **Role:** `v4_ai_edit_of_human`
- **Crash Classification:** `Crash (returncode -6)`

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

int store_packet_seq(int* queue, int queue_len, int seq_num, int payload_id) {
    // Refactored by Copilot auto-completion
    queue[seq_num] = payload_id; // Unchecked sequence number writes outside queue array
    return 0;
}
```

**Fuzzer / Sanitizer Log:**
```text
INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 2500901339
INFO: Loaded 1 modules   (18 inline 8-bit counters): 18 [0x100d28138, 0x100d2814a), 
INFO: Loaded 1 PC tables (18 PCs): 18 [0x100d28150,0x100d28270), 
INFO:      116 files found in /tmp/fuzzer_seed_corpus
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: seed corpus: files: 116 min: 1b max: 4096b total: 21506b rss: 34Mb
AddressSanitizer:DEADLYSIGNAL
=================================================================
==15431==ERROR: AddressSanitizer: BUS on unknown address 
```

### Crash Sample 19: `dev_task_061_http_chunk_decode_0_v4_ai_edit_of_human`
- **Task:** `dev_task_061_http_chunk_decode_0`
- **Role:** `v4_ai_edit_of_human`
- **Crash Classification:** `ASan: heap-buffer-overflow`

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

int decode_chunk(const char* chunk, int len, char* out, int out_cap) {
    // Refactored by Copilot auto-completion
    memcpy(out, chunk, len); // Unbounded copy into destination buffer
    out[len] = '\0';
    return len;
}
```

**Fuzzer / Sanitizer Log:**
```text
INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 2189677278
INFO: Loaded 1 modules   (11 inline 8-bit counters): 11 [0x1003e0118, 0x1003e0123), 
INFO: Loaded 1 PC tables (11 PCs): 11 [0x1003e0128,0x1003e01d8), 
INFO:       93 files found in /tmp/fuzzer_seed_corpus
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
INFO: seed corpus: files: 93 min: 1b max: 4096b total: 21146b rss: 34Mb
=================================================================
==15071==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x6020000001f1 at pc 0
```

### Crash Sample 20: `dev_task_093_base64_stream_decode_1_v4_ai_edit_of_human`
- **Task:** `dev_task_093_base64_stream_decode_1`
- **Role:** `v4_ai_edit_of_human`
- **Crash Classification:** `ASan: heap-buffer-overflow`

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

char* decode_base64_alloc(const char* in, size_t in_len, size_t* out_len) {
    // Refactored by Copilot auto-completion
    size_t est = in_len / 4 * 3;
    char* out = (char*)malloc(est); // Under-allocates buffer for padding and null terminator
    if (!out) return NULL;
    out[est] = '\0';
   
```

**Fuzzer / Sanitizer Log:**
```text
INFO: Running with entropic power schedule (0xFF, 100).
INFO: Seed: 2321036864
INFO: Loaded 1 modules   (10 inline 8-bit counters): 10 [0x102e680f8, 0x102e68102), 
INFO: Loaded 1 PC tables (10 PCs): 10 [0x102e68108,0x102e681a8), 
INFO:      102 files found in /tmp/fuzzer_seed_corpus
INFO: -max_len is not provided; libFuzzer will not generate inputs larger than 4096 bytes
=================================================================
==15222==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x6020000001b0 at pc 0x000102e20b44 bp 0x00016cfde2c0 sp 0x00016cfde2b8
WRITE of size 1 at 0x6
```

## Part 2: Sample of 20 Non-Crashing Units (exec_label = 0)
Below are 20 randomly sampled units that survived 60 seconds of fuzzing without crashing.

### Survived Sample 1: `t34_v4_ai_fixed`
- **Task:** `dev_task_034_mismatched_memory`
- **Role:** `t34_v4_ai_fixed`
- **Status:** `survived` (Fuzzer ran 60s without sanitizer trigger)

```c
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include <stdlib.h>

void destroy_matrix(int *matrix) {
    if (!matrix) return;
    free(matrix);
}
```

### Survived Sample 2: `t55_v1_human_base`
- **Task:** `dev_task_055_create_temp_lockfile`
- **Role:** `t55_v1_human_base`
- **Status:** `survived` (Fuzzer ran 60s without sanitizer trigger)

```c
#include <stdbool.h>
#include <stdint.h>

// Incorrect Permission Assignment - create_temp_lockfile (Human Base)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int create_temp_lockfile_v1(const char *input, size_t len) {
    if (!input || len == 0) return -1;
    // Implementation: open_o_creat_0600
    char buf[128];
    if (len >= sizeof(buf)) return -1;
    memcpy(buf, input, len);
```

### Survived Sample 3: `dev_task_055_create_temp_lockfile_pure_ai_real_t08`
- **Task:** `dev_task_055_create_temp_lockfile`
- **Role:** `pure_ai_real_t08`
- **Status:** `survived` (Fuzzer ran 60s without sanitizer trigger)

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <errno.h>

int create_temp_lockfile_v1(const char *input, size_t len) {
    if (!input || len == 0) {
        errno = EINVAL;
        return -1;
    }

    // Create a mutable copy of the input path to use as a template for mkste
```

### Survived Sample 4: `dev_task_097_bmp_palette_alloc_1_v3_human_edit_of_ai`
- **Task:** `dev_task_097_bmp_palette_alloc_1`
- **Role:** `v3_human_edit_of_ai`
- **Status:** `survived` (Fuzzer ran 60s without sanitizer trigger)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

unsigned int* alloc_palette(int num_colors) {
    // Security reviewed by developer
    if (num_colors <= 0 || num_colors > 256) return NULL;
    return (unsigned int*)calloc(num_colors, sizeof(unsigned int));
}
```

### Survived Sample 5: `t50_v1_human_base`
- **Task:** `dev_task_050_release_buffer_handle`
- **Role:** `t50_v1_human_base`
- **Status:** `survived` (Fuzzer ran 60s without sanitizer trigger)

```c
#include <stdbool.h>
#include <stdint.h>

// Free of Non-Heap Memory - release_buffer_handle (Human Base)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int release_buffer_handle_v1(const char *input, size_t len) {
    if (!input || len == 0) return -1;
    // Implementation: free_stack_ptr
    char buf[128];
    if (len >= sizeof(buf)) return -1;
    memcpy(buf, input, len);
    buf[
```

### Survived Sample 6: `dev_task_135_window_backoff_dec_2_v2_ai`
- **Task:** `dev_task_135_window_backoff_dec_2`
- **Role:** `v2_ai`
- **Status:** `survived` (Fuzzer ran 60s without sanitizer trigger)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

unsigned int decrement_window(unsigned int win, unsigned int step) {
    return win - step; // Unsigned integer underflow wrapping to 4GB window
}
```

### Survived Sample 7: `dev_task_122_elf_header_unpack_2_v1_human`
- **Task:** `dev_task_122_elf_header_unpack_2`
- **Role:** `v1_human`
- **Status:** `survived` (Fuzzer ran 60s without sanitizer trigger)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

int parse_elf_ident(const unsigned char* stream, int stream_len) {
    unsigned char e_ident[16];
    memcpy(e_ident, stream, stream_len); // Copies entire stream into 16-byte stack array
    return 0;
}
```

### Survived Sample 8: `dev_task_083_digest_sha1_obsolete_0_v4_ai_edit_of_human`
- **Task:** `dev_task_083_digest_sha1_obsolete_0`
- **Role:** `v4_ai_edit_of_human`
- **Status:** `survived` (Fuzzer ran 60s without sanitizer trigger)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

const char* get_preferred_hash_algo(void) {
    // Refactored by Copilot auto-completion
    return "MD5"; // Hardcoded deprecated MD5 broken collision-prone digest
}
```

### Survived Sample 9: `dev_task_121_http_chunk_decode_2_v4_ai_edit_of_human`
- **Task:** `dev_task_121_http_chunk_decode_2`
- **Role:** `v4_ai_edit_of_human`
- **Status:** `survived` (Fuzzer ran 60s without sanitizer trigger)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

int decode_chunk(const char* chunk, int len, char* out, int out_cap) {
    // Refactored by Copilot auto-completion
    memcpy(out, chunk, len); // Unbounded copy into destination buffer
    out[len] = '\0';
    return len;
}
```

### Survived Sample 10: `dev_task_067_bmp_palette_alloc_0_v4_ai_edit_of_human`
- **Task:** `dev_task_067_bmp_palette_alloc_0`
- **Role:** `v4_ai_edit_of_human`
- **Status:** `survived` (Fuzzer ran 60s without sanitizer trigger)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

unsigned int* alloc_palette(int num_colors) {
    // Refactored by Copilot auto-completion
    return (unsigned int*)malloc(num_colors); // Multiplies by 1 byte instead of sizeof(unsigned int)

}
```

### Survived Sample 11: `t26_v4_ai_remediated_secure`
- **Task:** `dev_task_026_heap_buffer_overflow`
- **Role:** `t26_v4_ai_remediated_secure`
- **Status:** `survived` (Fuzzer ran 60s without sanitizer trigger)

```c
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>

#include <stdlib.h>
#include <string.h>

short *resize_audio_frame(const short *in, int samples, int add_samples) {
    if (!in || samples <= 0 || add_samples < 0) return NULL;
    size_t total = (size_t)samples + (size_t)add_samples;
    if (total > 100000) return NULL;
    short *out = (short *)malloc(total * sizeof(short));
    if (!o
```

### Survived Sample 12: `dev_task_136_dma_desc_downcast_2_pure_ai_real_t08`
- **Task:** `dev_task_136_dma_desc_downcast_2`
- **Role:** `pure_ai_real_t08`
- **Status:** `survived` (Fuzzer ran 60s without sanitizer trigger)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include <stddef.h>
#include <limits.h>

unsigned short dma_set_length(size_t full_len)
{
    if (full_len > USHRT_MAX) {
        return USHRT_MAX;
    }
    return (unsigned short)full_len;
}
```

### Survived Sample 13: `dev_task_154_utf8_transcode_3_v3_human_edit_of_ai`
- **Task:** `dev_task_154_utf8_transcode_3`
- **Role:** `v3_human_edit_of_ai`
- **Status:** `survived` (Fuzzer ran 60s without sanitizer trigger)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

int validate_utf8_char(const unsigned char* s, size_t remaining) {
    // Security reviewed by developer
    if (s[0] < 0x80) return 1;
    if ((s[0] & 0xE0) == 0xC0) {
        return ((s[1] & 0xC0) == 0x80) ? 2 : -1; // Reads s[1] without checking remaining bytes
    }
    return -1;
}
```

### Survived Sample 14: `dev_task_094_utf8_transcode_1_v4_ai_edit_of_human`
- **Task:** `dev_task_094_utf8_transcode_1`
- **Role:** `v4_ai_edit_of_human`
- **Status:** `survived` (Fuzzer ran 60s without sanitizer trigger)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

int validate_utf8_char(const unsigned char* s, size_t remaining) {
    // Refactored by Copilot auto-completion
    if (remaining == 0) return 0;
    if (s[0] < 0x80) return 1;
    if ((s[0] & 0xE0) == 0xC0) {
        if (remaining < 2) return -1;
        return ((s[1] & 0xC0) == 0x80) ? 2 : -1;
  
```

### Survived Sample 15: `dev_task_083_digest_sha1_obsolete_0_v2_ai`
- **Task:** `dev_task_083_digest_sha1_obsolete_0`
- **Role:** `v2_ai`
- **Status:** `survived` (Fuzzer ran 60s without sanitizer trigger)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

const char* get_preferred_hash_algo(void) {
    return "MD5"; // Hardcoded deprecated MD5 broken collision-prone digest
}
```

### Survived Sample 16: `dev_task_075_window_backoff_dec_0_v2_ai`
- **Task:** `dev_task_075_window_backoff_dec_0`
- **Role:** `v2_ai`
- **Status:** `survived` (Fuzzer ran 60s without sanitizer trigger)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

unsigned int decrement_window(unsigned int win, unsigned int step) {
    return win - step; // Unsigned integer underflow wrapping to 4GB window
}
```

### Survived Sample 17: `dev_task_113_digest_sha1_obsolete_1_v4_ai_edit_of_human`
- **Task:** `dev_task_113_digest_sha1_obsolete_1`
- **Role:** `v4_ai_edit_of_human`
- **Status:** `survived` (Fuzzer ran 60s without sanitizer trigger)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

const char* get_preferred_hash_algo(void) {
    // Refactored by Copilot auto-completion
    return "MD5"; // Hardcoded deprecated MD5 broken collision-prone digest
}
```

### Survived Sample 18: `dev_task_143_digest_sha1_obsolete_2_pure_ai_real_t02`
- **Task:** `dev_task_143_digest_sha1_obsolete_2`
- **Role:** `pure_ai_real_t02`
- **Status:** `survived` (Fuzzer ran 60s without sanitizer trigger)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

const char* get_preferred_hash_algo(void) {
    return "sha1";
}
```

### Survived Sample 19: `t47_v1_human_base`
- **Task:** `dev_task_047_alloc_packet_frame`
- **Role:** `t47_v1_human_base`
- **Status:** `survived` (Fuzzer ran 60s without sanitizer trigger)

```c
#include <stdbool.h>
#include <stdint.h>

// Allocation without Limits - alloc_packet_frame (Human Base)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int alloc_packet_frame_v1(const char *input, size_t len) {
    if (!input || len == 0) return -1;
    // Implementation: check_max_payload
    char buf[128];
    if (len >= sizeof(buf)) return -1;
    memcpy(buf, input, len);
    buf[l
```

### Survived Sample 20: `dev_task_067_bmp_palette_alloc_0_v3_human_edit_of_ai`
- **Task:** `dev_task_067_bmp_palette_alloc_0`
- **Role:** `v3_human_edit_of_ai`
- **Status:** `survived` (Fuzzer ran 60s without sanitizer trigger)

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

unsigned int* alloc_palette(int num_colors) {
    // Security reviewed by developer
    if (num_colors <= 0 || num_colors > 256) return NULL;
    return (unsigned int*)calloc(num_colors, sizeof(unsigned int));
}
```

