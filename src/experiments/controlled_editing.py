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
