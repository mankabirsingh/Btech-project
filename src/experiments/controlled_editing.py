"""
Controlled Editing Experiment Module
=====================================
Framework for creating and tracking code versions with
different levels of AI involvement.

This module provides:
- Structures for tracking code evolution
- Tools for recording edit stages
- Analysis of security changes across versions
"""

import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field

logger = logging.getLogger(__name__)


@dataclass
class CodeVersion:
    """Represents one version of a code sample in the editing experiment."""
    version_id: str
    code: str
    language: str
    task_id: str
    author_type: str  # 'human', 'ai', 'human_edit_of_ai', 'ai_edit_of_human'
    ai_generator: Optional[str] = None
    ai_involvement: float = 0.0  # 0.0 = fully human, 1.0 = fully AI
    parent_version_id: Optional[str] = None
    edit_description: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    security_findings: Optional[Dict] = None
    vuln_label: Optional[int] = None  # If known ground truth
    code_hash: str = ''
    
    def __post_init__(self):
        self.code_hash = hashlib.sha256(self.code.encode()).hexdigest()[:16]


@dataclass
class EditingExperiment:
    """Tracks a complete editing experiment for one programming task."""
    task_id: str
    task_description: str
    language: str
    versions: List[CodeVersion] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    
    def add_version(self, version: CodeVersion):
        """Add a code version to the experiment."""
        self.versions.append(version)
        logger.info(f"Added version {version.version_id} "
                    f"(author={version.author_type}, "
                    f"ai_involvement={version.ai_involvement})")
    
    def get_version(self, version_id: str) -> Optional[CodeVersion]:
        for v in self.versions:
            if v.version_id == version_id:
                return v
        return None
    
    def get_lineage(self, version_id: str) -> List[CodeVersion]:
        """Get the full edit history chain for a version."""
        chain = []
        current = self.get_version(version_id)
        while current:
            chain.append(current)
            if current.parent_version_id:
                current = self.get_version(current.parent_version_id)
            else:
                break
        return list(reversed(chain))
    
    def to_dict(self) -> Dict:
        return {
            'task_id': self.task_id,
            'task_description': self.task_description,
            'language': self.language,
            'n_versions': len(self.versions),
            'versions': [asdict(v) for v in self.versions],
            'metadata': self.metadata,
        }
    
    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Experiment saved to {path}")


def create_pilot_experiment() -> EditingExperiment:
    """
    Create a small pilot controlled editing experiment.
    
    Task: Implement a function to read user input and process it.
    
    Versions:
    1. Human-written (0% AI): Known secure implementation
    2. AI-generated (100% AI): Potentially insecure implementation
    3. Human-edited AI (50% AI): AI code with human security fixes
    4. Human-written insecure (0% AI): Known insecure for comparison
    
    All versions are clearly labeled as CONTROLLED EXAMPLES.
    """
    experiment = EditingExperiment(
        task_id="pilot_001_buffer_handling",
        task_description="Implement a function that reads a string from user input, "
                        "processes it, and copies it to a fixed-size buffer.",
        language="C",
        metadata={
            'experiment_type': 'controlled_pilot',
            'purpose': 'Demonstrate provenance-tracking framework and '
                      'security analysis across code versions',
            'note': 'These are CONTROLLED EXAMPLES, not from a public benchmark',
            'created_at': datetime.now().isoformat(),
        }
    )
    
    # Version 1: Human-written secure implementation (0% AI)
    v1 = CodeVersion(
        version_id="pilot_001_v1_human_secure",
        code='''#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define BUFFER_SIZE 256

int process_input(const char *input, char *output, size_t output_size) {
    if (input == NULL || output == NULL || output_size == 0) {
        return -1;
    }
    
    size_t input_len = strlen(input);
    if (input_len >= output_size) {
        fprintf(stderr, "Input too long: %zu >= %zu\\n", input_len, output_size);
        return -1;
    }
    
    strncpy(output, input, output_size - 1);
    output[output_size - 1] = '\\0';
    
    return 0;
}

int main(void) {
    char buffer[BUFFER_SIZE];
    char input[BUFFER_SIZE];
    
    printf("Enter input: ");
    if (fgets(input, sizeof(input), stdin) == NULL) {
        fprintf(stderr, "Failed to read input\\n");
        return 1;
    }
    
    // Remove trailing newline
    size_t len = strlen(input);
    if (len > 0 && input[len - 1] == '\\n') {
        input[len - 1] = '\\0';
    }
    
    if (process_input(input, buffer, sizeof(buffer)) != 0) {
        fprintf(stderr, "Processing failed\\n");
        return 1;
    }
    
    printf("Processed: %s\\n", buffer);
    return 0;
}
''',
        language="C",
        task_id="pilot_001_buffer_handling",
        author_type="human",
        ai_generator=None,
        ai_involvement=0.0,
        parent_version_id=None,
        edit_description="Human-written secure implementation with bounds checking",
        vuln_label=0,  # Known: not vulnerable
    )
    experiment.add_version(v1)
    
    # Version 2: AI-generated potentially insecure (100% AI)
    v2 = CodeVersion(
        version_id="pilot_001_v2_ai_insecure",
        code='''#include <stdio.h>
#include <string.h>

void process_input(char *input) {
    char buffer[64];
    strcpy(buffer, input);  // No bounds checking!
    printf("Processed: %s\\n", buffer);
}

int main() {
    char input[1024];
    printf("Enter input: ");
    gets(input);  // Deprecated and unsafe!
    process_input(input);
    return 0;
}
''',
        language="C",
        task_id="pilot_001_buffer_handling",
        author_type="ai",
        ai_generator="controlled_example_simulated",
        ai_involvement=1.0,
        parent_version_id=None,
        edit_description="AI-generated code with common insecure patterns "
                        "(CONTROLLED EXAMPLE - simulated for experiment)",
        vuln_label=1,  # Known: vulnerable
    )
    experiment.add_version(v2)
    
    # Version 3: Human-edited AI code (50% AI involvement)
    v3 = CodeVersion(
        version_id="pilot_001_v3_human_edited_ai",
        code='''#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define BUFFER_SIZE 64

void process_input(const char *input) {
    char buffer[BUFFER_SIZE];
    
    if (input == NULL) {
        fprintf(stderr, "Null input\\n");
        return;
    }
    
    // Human fix: replaced strcpy with strncpy + null termination
    strncpy(buffer, input, BUFFER_SIZE - 1);
    buffer[BUFFER_SIZE - 1] = '\\0';
    
    printf("Processed: %s\\n", buffer);
}

int main() {
    char input[1024];
    printf("Enter input: ");
    
    // Human fix: replaced gets() with fgets()
    if (fgets(input, sizeof(input), stdin) == NULL) {
        return 1;
    }
    
    // Remove newline
    input[strcspn(input, "\\n")] = '\\0';
    
    process_input(input);
    return 0;
}
''',
        language="C",
        task_id="pilot_001_buffer_handling",
        author_type="human_edit_of_ai",
        ai_generator="controlled_example_simulated",
        ai_involvement=0.5,
        parent_version_id="pilot_001_v2_ai_insecure",
        edit_description="Human security review of AI code: replaced strcpy→strncpy, gets→fgets, "
                        "added null checks (CONTROLLED EXAMPLE)",
        vuln_label=0,  # Known: fixed
    )
    experiment.add_version(v3)
    
    # Version 4: Human-written insecure (0% AI, but still vulnerable)
    v4 = CodeVersion(
        version_id="pilot_001_v4_human_insecure",
        code='''#include <stdio.h>
#include <string.h>

int main() {
    char buffer[32];
    char name[256];
    
    printf("Enter your name: ");
    scanf("%s", name);  // No length limit
    
    sprintf(buffer, "Hello, %s!", name);  // Buffer overflow possible
    printf("%s\\n", buffer);
    
    return 0;
}
''',
        language="C",
        task_id="pilot_001_buffer_handling",
        author_type="human",
        ai_generator=None,
        ai_involvement=0.0,
        parent_version_id=None,
        edit_description="Human-written code with vulnerabilities "
                        "(demonstrates that human code can also be insecure) "
                        "(CONTROLLED EXAMPLE)",
        vuln_label=1,  # Known: vulnerable
    )
    experiment.add_version(v4)
    
    return experiment


def analyze_experiment_security(experiment: EditingExperiment) -> Dict:
    """
    Run security analysis on all versions and compare results.
    
    Returns a summary of how security properties change across versions.
    """
    from src.security.static_analysis import analyze_code
    
    results = []
    
    for version in experiment.versions:
        analysis = analyze_code(version.code, version.language)
        version.security_findings = analysis
        
        results.append({
            'version_id': version.version_id,
            'author_type': version.author_type,
            'ai_involvement': version.ai_involvement,
            'n_findings': analysis['n_findings'],
            'severity_counts': analysis['severity_counts'],
            'cwes': analysis['cwes'],
            'risk_level': analysis['risk_level'],
            'risk_score': analysis['risk_score'],
            'known_vuln_label': version.vuln_label,
        })
    
    # Analyze transitions (parent -> child)
    transitions = []
    for version in experiment.versions:
        if version.parent_version_id:
            parent = experiment.get_version(version.parent_version_id)
            if parent and parent.security_findings and version.security_findings:
                parent_cwes = set(parent.security_findings.get('cwes', []))
                child_cwes = set(version.security_findings.get('cwes', []))
                
                transition = {
                    'from_version': parent.version_id,
                    'to_version': version.version_id,
                    'from_author': parent.author_type,
                    'to_author': version.author_type,
                    'from_ai_involvement': parent.ai_involvement,
                    'to_ai_involvement': version.ai_involvement,
                    'findings_removed': parent.security_findings['n_findings'] - version.security_findings['n_findings'],
                    'risk_change': version.security_findings['risk_score'] - parent.security_findings['risk_score'],
                    'cwes_fixed': sorted(list(parent_cwes - child_cwes)),
                    'cwes_introduced': sorted(list(child_cwes - parent_cwes)),
                    'cwes_persistent': sorted(list(parent_cwes & child_cwes)),
                    'vuln_persisted': version.vuln_label == 1 if version.vuln_label is not None else None,
                    'vuln_fixed': parent.vuln_label == 1 and version.vuln_label == 0 if (parent.vuln_label is not None and version.vuln_label is not None) else None,
                }
                transitions.append(transition)
    
    return {
        'task_id': experiment.task_id,
        'n_versions': len(experiment.versions),
        'version_results': results,
        'transitions': transitions,
        'summary': {
            'versions_analyzed': len(results),
            'versions_with_findings': sum(1 for r in results if r['n_findings'] > 0),
            'total_unique_cwes': sorted(list(set(
                cwe for r in results for cwe in r['cwes']
            ))),
        },
    }


def create_multi_cwe_benchmark() -> List[EditingExperiment]:
    """
    Creates a rigorous 8-task, 32-version controlled benchmark covering 8 major CWEs.
    
    Each task defines a 4-version lineage:
      v1: Human Secure Baseline (0% AI, vuln=0)
      v2: AI-Generated Naive Implementation (100% AI, vuln=1)
      v3: Human-Remediated Hybrid (50% AI, vuln=0)
      v4: AI-Assisted Regressive/Incomplete Fix (75% AI, vuln=1)
    
    Total: 32 code units (16 vulnerable, 16 secure, balanced).
    """
    benchmarks = []
    
    # Task 1: CWE-120 Buffer Overflow
    t1 = create_pilot_experiment()
    benchmarks.append(t1)
    
    # Task 2: CWE-134 Format String Vulnerability
    t2 = EditingExperiment(
        task_id="task_002_format_string",
        task_description="Log user activity message to console or file with timestamps.",
        language="C",
        metadata={'cwe': 'CWE-134', 'category': 'Format String'}
    )
    t2.add_version(CodeVersion(
        version_id="t2_v1_human_secure",
        code='''#include <stdio.h>
#include <time.h>
void log_activity(const char *username, const char *msg) {
    if (!username || !msg) return;
    time_t now = time(NULL);
    printf("[%ld] user=%s message=%s\\n", (long)now, username, msg);
}
int main(void) {
    log_activity("alice", "User logged in successfully");
    return 0;
}''',
        language="C", task_id="task_002_format_string",
        author_type="human", ai_involvement=0.0, vuln_label=0,
        edit_description="Human secure: uses static format string literal with %s specifiers"
    ))
    t2.add_version(CodeVersion(
        version_id="t2_v2_ai_insecure",
        code='''#include <stdio.h>
void log_activity(const char *msg) {
    printf(msg); // Format string vulnerability!
}
int main(int argc, char **argv) {
    if (argc > 1) log_activity(argv[1]);
    return 0;
}''',
        language="C", task_id="task_002_format_string",
        author_type="ai", ai_generator="simulated_copilot", ai_involvement=1.0, vuln_label=1,
        edit_description="AI naive: passes user-controlled string directly as format argument"
    ))
    t2.add_version(CodeVersion(
        version_id="t2_v3_human_hybrid_fixed",
        code='''#include <stdio.h>
void log_activity(const char *msg) {
    if (msg == NULL) return;
    printf("%s\\n", msg); // Human fix: static format string
}
int main(int argc, char **argv) {
    if (argc > 1) log_activity(argv[1]);
    return 0;
}''',
        language="C", task_id="task_002_format_string", parent_version_id="t2_v2_ai_insecure",
        author_type="human_edit_of_ai", ai_generator="simulated_copilot", ai_involvement=0.5, vuln_label=0,
        edit_description="Human remediation: added NULL check and constant format string %s\\n"
    ))
    t2.add_version(CodeVersion(
        version_id="t2_v4_ai_regressive",
        code='''#include <stdio.h>
#include <syslog.h>
void log_activity(const char *msg) {
    syslog(LOG_INFO, msg); // AI refactor: still vulnerable format string in syslog
}
int main(int argc, char **argv) {
    if (argc > 1) log_activity(argv[1]);
    return 0;
}''',
        language="C", task_id="task_002_format_string", parent_version_id="t2_v3_human_hybrid_fixed",
        author_type="ai_edit_of_human", ai_generator="simulated_copilot", ai_involvement=0.75, vuln_label=1,
        edit_description="AI quick fix: moved from printf to syslog but forgot format specifier"
    ))
    benchmarks.append(t2)
    
    # Task 3: CWE-190 Integer Overflow to Buffer Allocation
    t3 = EditingExperiment(
        task_id="task_003_integer_overflow",
        task_description="Allocate memory buffer for an array of elements specified by external count.",
        language="C",
        metadata={'cwe': 'CWE-190', 'category': 'Integer Overflow'}
    )
    t3.add_version(CodeVersion(
        version_id="t3_v1_human_secure",
        code='''#include <stdlib.h>
#include <stdint.h>
int *allocate_buffer(size_t count) {
    if (count == 0 || count > SIZE_MAX / sizeof(int)) return NULL;
    return (int *)malloc(count * sizeof(int));
}''',
        language="C", task_id="task_003_integer_overflow",
        author_type="human", ai_involvement=0.0, vuln_label=0,
        edit_description="Human secure: checks for multiplication overflow against SIZE_MAX"
    ))
    t3.add_version(CodeVersion(
        version_id="t3_v2_ai_insecure",
        code='''#include <stdlib.h>
int *allocate_buffer(int count) {
    // No integer overflow check!
    int *arr = (int *)malloc(count * sizeof(int));
    return arr;
}''',
        language="C", task_id="task_003_integer_overflow",
        author_type="ai", ai_generator="simulated_copilot", ai_involvement=1.0, vuln_label=1,
        edit_description="AI naive: unchecked multiplication count * sizeof(int)"
    ))
    t3.add_version(CodeVersion(
        version_id="t3_v3_human_hybrid_fixed",
        code='''#include <stdlib.h>
#include <stddef.h>
int *allocate_buffer(int count) {
    if (count <= 0) return NULL;
    size_t bytes;
    if (__builtin_mul_overflow((size_t)count, sizeof(int), &bytes)) return NULL;
    int *arr = (int *)malloc(bytes);
    return arr;
}''',
        language="C", task_id="task_003_integer_overflow", parent_version_id="t3_v2_ai_insecure",
        author_type="human_edit_of_ai", ai_generator="simulated_copilot", ai_involvement=0.5, vuln_label=0,
        edit_description="Human fix: __builtin_mul_overflow check on bytes"
    ))
    t3.add_version(CodeVersion(
        version_id="t3_v4_ai_regressive",
        code='''#include <stdlib.h>
int *allocate_buffer(int count) {
    // AI incomplete fix: checks count < 100000000 but count can be negative!
    if (count > 100000000) return NULL;
    return (int *)malloc(count * sizeof(int));
}''',
        language="C", task_id="task_003_integer_overflow", parent_version_id="t3_v3_human_hybrid_fixed",
        author_type="ai_edit_of_human", ai_generator="simulated_copilot", ai_involvement=0.75, vuln_label=1,
        edit_description="AI regressive fix: checks upper bound but ignores negative signed integers"
    ))
    benchmarks.append(t3)
    
    # Task 4: CWE-78 OS Command Injection
    t4 = EditingExperiment(
        task_id="task_004_command_injection",
        task_description="Execute network diagnostic utility with host argument.",
        language="C",
        metadata={'cwe': 'CWE-78', 'category': 'Command Injection'}
    )
    t4.add_version(CodeVersion(
        version_id="t4_v1_human_secure",
        code='''#include <unistd.h>
#include <sys/wait.h>
#include <regex.h>
int ping_host(const char *host) {
    if (!host) return -1;
    // Strict IP or hostname validation
    for (const char *p = host; *p; p++) {
        if (!((*p >= 'a' && *p <= 'z') || (*p >= '0' && *p <= '9') || *p == '.' || *p == '-')) return -1;
    }
    pid_t pid = fork();
    if (pid == 0) {
        execl("/bin/ping", "ping", "-c", "1", host, (char *)NULL);
        _exit(127);
    }
    int status;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}''',
        language="C", task_id="task_004_command_injection",
        author_type="human", ai_involvement=0.0, vuln_label=0,
        edit_description="Human secure: character whitelist + execve without shell invocation"
    ))
    t4.add_version(CodeVersion(
        version_id="t4_v2_ai_insecure",
        code='''#include <stdio.h>
#include <stdlib.h>
int ping_host(const char *host) {
    char cmd[256];
    sprintf(cmd, "ping -c 1 %s", host);
    return system(cmd); // Command injection via shell!
}''',
        language="C", task_id="task_004_command_injection",
        author_type="ai", ai_generator="simulated_copilot", ai_involvement=1.0, vuln_label=1,
        edit_description="AI naive: passes unvalidated string to sprintf and system()"
    ))
    t4.add_version(CodeVersion(
        version_id="t4_v3_human_hybrid_fixed",
        code='''#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int ping_host(const char *host) {
    if (!host || strlen(host) > 64) return -1;
    if (strpbrk(host, ";|&`$><\\n\\r")) return -1; // Block shell metacharacters
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "ping -c 1 %s", host);
    return system(cmd);
}''',
        language="C", task_id="task_004_command_injection", parent_version_id="t4_v2_ai_insecure",
        author_type="human_edit_of_ai", ai_generator="simulated_copilot", ai_involvement=0.5, vuln_label=0,
        edit_description="Human fix: blacklist shell metacharacters and bounded snprintf"
    ))
    t4.add_version(CodeVersion(
        version_id="t4_v4_ai_regressive",
        code='''#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int ping_host(const char *host) {
    // AI incomplete fix: blocks semicolons only, forgetting | and & and backticks
    if (strchr(host, ';') != NULL) return -1;
    char cmd[256];
    sprintf(cmd, "ping -c 1 %s", host);
    return system(cmd);
}''',
        language="C", task_id="task_004_command_injection", parent_version_id="t4_v3_human_hybrid_fixed",
        author_type="ai_edit_of_human", ai_generator="simulated_copilot", ai_involvement=0.75, vuln_label=1,
        edit_description="AI incomplete fix: only filters semicolon, still vulnerable to pipe or ampersand"
    ))
    benchmarks.append(t4)
    
    # Task 5: CWE-416 Use After Free
    t5 = EditingExperiment(
        task_id="task_005_use_after_free",
        task_description="Manage session node and retrieve cached status upon deallocation.",
        language="C",
        metadata={'cwe': 'CWE-416', 'category': 'Use After Free'}
    )
    t5.add_version(CodeVersion(
        version_id="t5_v1_human_secure",
        code='''#include <stdlib.h>
typedef struct { int status; char data[32]; } session_t;
int close_session(session_t **sess_ptr) {
    if (!sess_ptr || !*sess_ptr) return -1;
    int st = (*sess_ptr)->status;
    free(*sess_ptr);
    *sess_ptr = NULL; // Zero pointer prevents dangling reference
    return st;
}''',
        language="C", task_id="task_005_use_after_free",
        author_type="human", ai_involvement=0.0, vuln_label=0,
        edit_description="Human secure: captures status before free and sets double pointer to NULL"
    ))
    t5.add_version(CodeVersion(
        version_id="t5_v2_ai_insecure",
        code='''#include <stdlib.h>
typedef struct { int status; char data[32]; } session_t;
int close_session(session_t *sess) {
    free(sess);
    return sess->status; // Use After Free!
}''',
        language="C", task_id="task_005_use_after_free",
        author_type="ai", ai_generator="simulated_copilot", ai_involvement=1.0, vuln_label=1,
        edit_description="AI naive: accesses struct member after calling free()"
    ))
    t5.add_version(CodeVersion(
        version_id="t5_v3_human_hybrid_fixed",
        code='''#include <stdlib.h>
typedef struct { int status; char data[32]; } session_t;
int close_session(session_t *sess) {
    if (!sess) return -1;
    int st = sess->status; // Human fix: read before free
    free(sess);
    return st;
}''',
        language="C", task_id="task_005_use_after_free", parent_version_id="t5_v2_ai_insecure",
        author_type="human_edit_of_ai", ai_generator="simulated_copilot", ai_involvement=0.5, vuln_label=0,
        edit_description="Human remediation: stores status in local variable prior to free()"
    ))
    t5.add_version(CodeVersion(
        version_id="t5_v4_ai_regressive",
        code='''#include <stdlib.h>
typedef struct { int status; char *token; } session_t;
void free_session(session_t *sess) { free(sess); }
int close_session(session_t *sess) {
    free_session(sess);
    return sess != NULL ? sess->status : -1; // AI helper refactor: still accesses freed memory
}''',
        language="C", task_id="task_005_use_after_free", parent_version_id="t5_v3_human_hybrid_fixed",
        author_type="ai_edit_of_human", ai_generator="simulated_copilot", ai_involvement=0.75, vuln_label=1,
        edit_description="AI refactoring: wrapped free in helper but checks sess != NULL on stale pointer"
    ))
    benchmarks.append(t5)
    
    # Task 6: CWE-22 Path Traversal
    t6 = EditingExperiment(
        task_id="task_006_path_traversal",
        task_description="Read file from user-specified name within restricted directory.",
        language="C",
        metadata={'cwe': 'CWE-22', 'category': 'Path Traversal'}
    )
    t6.add_version(CodeVersion(
        version_id="t6_v1_human_secure",
        code='''#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
FILE *open_user_file(const char *filename) {
    if (!filename || strstr(filename, "..") || strchr(filename, '/')) return NULL;
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "/var/safe_dir/%s", filename);
    return fopen(path, "r");
}''',
        language="C", task_id="task_006_path_traversal",
        author_type="human", ai_involvement=0.0, vuln_label=0,
        edit_description="Human secure: rejects slashes and .. components"
    ))
    t6.add_version(CodeVersion(
        version_id="t6_v2_ai_insecure",
        code='''#include <stdio.h>
FILE *open_user_file(const char *filename) {
    char path[256];
    sprintf(path, "/var/uploads/%s", filename); // Path traversal vulnerable!
    return fopen(path, "r");
}''',
        language="C", task_id="task_006_path_traversal",
        author_type="ai", ai_generator="simulated_copilot", ai_involvement=1.0, vuln_label=1,
        edit_description="AI naive: concatenates filename directly into filepath"
    ))
    t6.add_version(CodeVersion(
        version_id="t6_v3_human_hybrid_fixed",
        code='''#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
FILE *open_user_file(const char *filename) {
    char candidate[PATH_MAX];
    char resolved[PATH_MAX];
    snprintf(candidate, sizeof(candidate), "/var/uploads/%s", filename);
    if (realpath(candidate, resolved) == NULL) return NULL;
    if (strncmp(resolved, "/var/uploads/", 13) != 0) return NULL; // Human fix: realpath prefix
    return fopen(resolved, "r");
}''',
        language="C", task_id="task_006_path_traversal", parent_version_id="t6_v2_ai_insecure",
        author_type="human_edit_of_ai", ai_generator="simulated_copilot", ai_involvement=0.5, vuln_label=0,
        edit_description="Human remediation: canonicalize with realpath and enforce directory root"
    ))
    t6.add_version(CodeVersion(
        version_id="t6_v4_ai_regressive",
        code='''#include <stdio.h>
#include <string.h>
FILE *open_user_file(const char *filename) {
    // AI incomplete fix: naive single-pass replacement of ../ bypassable via ....//
    char clean[256] = {0};
    const char *p = strstr(filename, "../");
    if (p) strcpy(clean, p + 3);
    else strcpy(clean, filename);
    char path[256];
    sprintf(path, "/var/uploads/%s", clean);
    return fopen(path, "r");
}''',
        language="C", task_id="task_006_path_traversal", parent_version_id="t6_v3_human_hybrid_fixed",
        author_type="ai_edit_of_human", ai_generator="simulated_copilot", ai_involvement=0.75, vuln_label=1,
        edit_description="AI regressive fix: naive substring strip bypassable by nested sequences"
    ))
    benchmarks.append(t6)
    
    # Task 7: CWE-327 Broken Cryptography
    t7 = EditingExperiment(
        task_id="task_007_broken_crypto",
        task_description="Encrypt sensitive user authentication token.",
        language="C",
        metadata={'cwe': 'CWE-327', 'category': 'Broken Cryptography'}
    )
    t7.add_version(CodeVersion(
        version_id="t7_v1_human_secure",
        code='''#include <openssl/evp.h>
#include <openssl/rand.h>
int encrypt_token(const unsigned char *pt, int pt_len, const unsigned char *key, unsigned char *ct) {
    unsigned char iv[12];
    if (RAND_bytes(iv, sizeof(iv)) != 1) return -1;
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return -1;
    // Modern AES-256-GCM authenticated encryption
    EVP_EncryptInit_ex(ctx, EVP_aes_256_gcm(), NULL, key, iv);
    EVP_CIPHER_CTX_free(ctx);
    return 0;
}''',
        language="C", task_id="task_007_broken_crypto",
        author_type="human", ai_involvement=0.0, vuln_label=0,
        edit_description="Human secure: AES-256-GCM with cryptographically random IV"
    ))
    t7.add_version(CodeVersion(
        version_id="t7_v2_ai_insecure",
        code='''#include <openssl/des.h>
void encrypt_token(const unsigned char *in, unsigned char *out, DES_key_schedule *ks) {
    // Insecure obsolete DES in ECB mode!
    DES_ecb_encrypt((const_DES_cblock *)in, (DES_cblock *)out, ks, DES_ENCRYPT);
}''',
        language="C", task_id="task_007_broken_crypto",
        author_type="ai", ai_generator="simulated_copilot", ai_involvement=1.0, vuln_label=1,
        edit_description="AI naive: legacy DES_ecb_encrypt (CWE-327 broken crypto)"
    ))
    t7.add_version(CodeVersion(
        version_id="t7_v3_human_hybrid_fixed",
        code='''#include <openssl/evp.h>
int encrypt_token(const unsigned char *in, int in_len, const unsigned char *key, const unsigned char *iv, unsigned char *out) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    if (!ctx) return -1;
    EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv); // Human fix: AES-256-CBC
    EVP_CIPHER_CTX_free(ctx);
    return 0;
}''',
        language="C", task_id="task_007_broken_crypto", parent_version_id="t7_v2_ai_insecure",
        author_type="human_edit_of_ai", ai_generator="simulated_copilot", ai_involvement=0.5, vuln_label=0,
        edit_description="Human remediation: replaced DES with AES-256"
    ))
    t7.add_version(CodeVersion(
        version_id="t7_v4_ai_regressive",
        code='''#include <openssl/aes.h>
void encrypt_token(const unsigned char *in, unsigned char *out, AES_KEY *key) {
    // AI incomplete fix: upgraded to AES but used ECB mode without IV!
    AES_ecb_encrypt(in, out, key, AES_ENCRYPT);
}''',
        language="C", task_id="task_007_broken_crypto", parent_version_id="t7_v3_human_hybrid_fixed",
        author_type="ai_edit_of_human", ai_generator="simulated_copilot", ai_involvement=0.75, vuln_label=1,
        edit_description="AI quick fix: migrated to AES but selected ECB mode without IV"
    ))
    benchmarks.append(t7)
    
    # Task 8: CWE-476 NULL Pointer Dereference
    t8 = EditingExperiment(
        task_id="task_008_null_dereference",
        task_description="Lookup cache entry and update access statistics.",
        language="C",
        metadata={'cwe': 'CWE-476', 'category': 'NULL Pointer Dereference'}
    )
    t8.add_version(CodeVersion(
        version_id="t8_v1_human_secure",
        code='''#include <stdlib.h>
typedef struct { int hits; char val[64]; } entry_t;
extern entry_t *cache_lookup(const char *k);
int record_hit(const char *key) {
    if (!key) return -1;
    entry_t *e = cache_lookup(key);
    if (e == NULL) return -1; // Rigorous null check
    e->hits++;
    return e->hits;
}''',
        language="C", task_id="task_008_null_dereference",
        author_type="human", ai_involvement=0.0, vuln_label=0,
        edit_description="Human secure: checks return of cache_lookup before dereferencing"
    ))
    t8.add_version(CodeVersion(
        version_id="t8_v2_ai_insecure",
        code='''#include <stdlib.h>
typedef struct { int hits; char val[64]; } entry_t;
extern entry_t *cache_lookup(const char *k);
int record_hit(const char *key) {
    entry_t *e = cache_lookup(key);
    e->hits++; // NULL pointer dereference if key missing!
    return e->hits;
}''',
        language="C", task_id="task_008_null_dereference",
        author_type="ai", ai_generator="simulated_copilot", ai_involvement=1.0, vuln_label=1,
        edit_description="AI naive: dereferences cache_lookup result directly"
    ))
    t8.add_version(CodeVersion(
        version_id="t8_v3_human_hybrid_fixed",
        code='''#include <stdlib.h>
typedef struct { int hits; char val[64]; } entry_t;
extern entry_t *cache_lookup(const char *k);
int record_hit(const char *key) {
    if (!key) return -1;
    entry_t *e = cache_lookup(key);
    if (!e) return 0; // Human fix: return 0 if miss
    e->hits++;
    return e->hits;
}''',
        language="C", task_id="task_008_null_dereference", parent_version_id="t8_v2_ai_insecure",
        author_type="human_edit_of_ai", ai_generator="simulated_copilot", ai_involvement=0.5, vuln_label=0,
        edit_description="Human remediation: added NULL guard on lookup result"
    ))
    t8.add_version(CodeVersion(
        version_id="t8_v4_ai_regressive",
        code='''#include <stdlib.h>
#include <stdio.h>
typedef struct { int hits; char val[64]; } entry_t;
extern entry_t *cache_lookup(const char *k);
int record_hit(const char *key) {
    entry_t *e = cache_lookup(key);
    if (e != NULL) {
        e->hits++;
    }
    // AI incomplete fix: still logs value outside guard!
    printf("Cache state: %s\\n", e->val); // NULL dereference on cache miss!
    return e ? e->hits : -1;
}''',
        language="C", task_id="task_008_null_dereference", parent_version_id="t8_v3_human_hybrid_fixed",
        author_type="ai_edit_of_human", ai_generator="simulated_copilot", ai_involvement=0.75, vuln_label=1,
        edit_description="AI refactoring: guards hits increment but dereferences e->val in log statement"
    ))
    benchmarks.append(t8)
    
    return benchmarks
