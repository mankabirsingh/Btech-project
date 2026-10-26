"""
Static Security Analysis Module
=================================
Provides code security analysis using pattern-matching heuristics
and optional Semgrep integration.
"""

import re
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class SecurityFinding:
    """Represents a single security finding."""
    rule_id: str
    cwe: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    line: Optional[int] = None
    end_line: Optional[int] = None
    code_snippet: Optional[str] = None
    tool: str = 'pattern'  # 'pattern', 'semgrep', 'model'
    confidence: str = 'medium'  # 'low', 'medium', 'high'


# Common vulnerability patterns for C/C++
C_VULNERABILITY_PATTERNS = [
    {
        'pattern': r'\bstrcpy\s*\(',
        'rule_id': 'unsafe-strcpy',
        'cwe': 'CWE-120',
        'severity': 'high',
        'message': 'Use of strcpy() without bounds checking — potential buffer overflow',
    },
    {
        'pattern': r'\bstrcat\s*\(',
        'rule_id': 'unsafe-strcat',
        'cwe': 'CWE-120',
        'severity': 'high',
        'message': 'Use of strcat() without bounds checking — potential buffer overflow',
    },
    {
        'pattern': r'\bsprintf\s*\(',
        'rule_id': 'unsafe-sprintf',
        'cwe': 'CWE-120',
        'severity': 'high',
        'message': 'Use of sprintf() — potential buffer overflow, prefer snprintf()',
    },
    {
        'pattern': r'\bgets\s*\(',
        'rule_id': 'unsafe-gets',
        'cwe': 'CWE-120',
        'severity': 'critical',
        'message': 'Use of gets() — always vulnerable to buffer overflow',
    },
    {
        'pattern': r'\bmalloc\s*\([^)]*\)\s*;(?!\s*if)',
        'rule_id': 'unchecked-malloc',
        'cwe': 'CWE-252',
        'severity': 'medium',
        'message': 'malloc() return value may not be checked for NULL',
    },
    {
        'pattern': r'\bfree\s*\(\s*\w+\s*\)',
        'rule_id': 'pointer-free',
        'cwe': 'CWE-416',
        'severity': 'low',
        'message': 'Pointer deallocation via free() — ensure pointer is zeroed or not reused',
        'confidence': 'low',
    },
    {
        'pattern': r'\bsystem\s*\(',
        'rule_id': 'command-injection',
        'cwe': 'CWE-78',
        'severity': 'high',
        'message': 'Use of system() — potential command injection',
    },
    {
        'pattern': r'\b(?:printf|fprintf|sprintf|snprintf)\s*\(\s*[^",\)]+\s*\)',
        'rule_id': 'format-string',
        'cwe': 'CWE-134',
        'severity': 'high',
        'message': 'Potential format string vulnerability: non-literal format argument',
    },
    {
        'pattern': r'\bscanf\s*\(\s*"[^"]*%s',
        'rule_id': 'unsafe-scanf',
        'cwe': 'CWE-120',
        'severity': 'high',
        'message': 'scanf with %s — no bounds on input, potential buffer overflow',
    },
    {
        'pattern': r'(?:int|long|short|size_t)\s+\w+\s*=\s*[^;]*(?:\+|\*|-)[^;]*;',
        'rule_id': 'integer-overflow',
        'cwe': 'CWE-190',
        'severity': 'medium',
        'message': 'Arithmetic operation may cause integer overflow',
        'confidence': 'low',
    },
]

# Common vulnerability patterns for Python
PYTHON_VULNERABILITY_PATTERNS = [
    {
        'pattern': r'\beval\s*\(',
        'rule_id': 'unsafe-eval',
        'cwe': 'CWE-94',
        'severity': 'critical',
        'message': 'Use of eval() — potential code injection',
    },
    {
        'pattern': r'\bexec\s*\(',
        'rule_id': 'unsafe-exec',
        'cwe': 'CWE-94',
        'severity': 'critical',
        'message': 'Use of exec() — potential code injection',
    },
    {
        'pattern': r'subprocess\.\w+\(.*shell\s*=\s*True',
        'rule_id': 'shell-injection',
        'cwe': 'CWE-78',
        'severity': 'high',
        'message': 'subprocess with shell=True — potential command injection',
    },
    {
        'pattern': r'pickle\.load[s]?\s*\(',
        'rule_id': 'unsafe-pickle',
        'cwe': 'CWE-502',
        'severity': 'high',
        'message': 'Deserializing untrusted data with pickle — potential code execution',
    },
    {
        'pattern': r'yaml\.load\s*\([^)]*\)(?!\s*,\s*Loader)',
        'rule_id': 'unsafe-yaml',
        'cwe': 'CWE-502',
        'severity': 'high',
        'message': 'yaml.load without Loader argument — potential code execution',
    },
    {
        'pattern': r'\.format\s*\(.*input|\.format\s*\(.*request',
        'rule_id': 'format-injection',
        'cwe': 'CWE-134',
        'severity': 'medium',
        'message': 'String format with user input — potential format string vulnerability',
    },
    {
        'pattern': r'execute\s*\(\s*["\'].*%|execute\s*\(\s*f["\']',
        'rule_id': 'sql-injection',
        'cwe': 'CWE-89',
        'severity': 'critical',
        'message': 'SQL query with string interpolation — potential SQL injection',
    },
]


def detect_language(code: str) -> str:
    """Simple heuristic language detection."""
    if re.search(r'#include\s*<|void\s+\w+\s*\(|int\s+main\s*\(', code):
        return 'c/c++'
    elif re.search(r'\bdef\s+\w+\s*\(|import\s+\w+|from\s+\w+\s+import', code):
        return 'python'
    elif re.search(r'\bfunction\s+\w+|const\s+\w+\s*=|let\s+\w+\s*=', code):
        return 'javascript'
    elif re.search(r'public\s+(?:static\s+)?(?:void|int|String)|class\s+\w+\s*\{', code):
        return 'java'
    return 'unknown'


def pattern_analysis(code: str, language: Optional[str] = None) -> List[SecurityFinding]:
    """
    Run pattern-based vulnerability detection.
    
    This is NOT a replacement for proper static analysis.
    Findings are heuristic and may contain false positives.
    """
    if language is None:
        language = detect_language(code)
    
    patterns = []
    if language.lower() in ('c', 'c++', 'c/c++', 'cpp'):
        patterns = C_VULNERABILITY_PATTERNS
    elif language.lower() == 'python':
        patterns = PYTHON_VULNERABILITY_PATTERNS
    else:
        patterns = C_VULNERABILITY_PATTERNS + PYTHON_VULNERABILITY_PATTERNS
    
    findings = []
    lines = code.split('\n')
    
    for pat_info in patterns:
        for i, line in enumerate(lines):
            # Strip single-line comments for pattern matching to avoid false positives
            code_part = re.sub(r'//.*$|#.*$', '', line)
            if re.search(pat_info['pattern'], code_part):
                finding = SecurityFinding(
                    rule_id=pat_info['rule_id'],
                    cwe=pat_info['cwe'],
                    severity=pat_info['severity'],
                    message=pat_info['message'],
                    line=i + 1,
                    code_snippet=line.strip(),
                    tool='pattern',
                    confidence=pat_info.get('confidence', 'medium'),
                )
                findings.append(finding)
    
    return findings


def try_semgrep_analysis(code: str, language: str = 'c') -> List[SecurityFinding]:
    """
    Attempt Semgrep analysis if installed.
    Returns empty list if Semgrep is not available.
    """
    try:
        # Check if semgrep is available
        result = subprocess.run(['semgrep', '--version'], 
                                capture_output=True, timeout=5)
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("Semgrep not available")
        return []
    
    # Write code to temp file
    ext_map = {'c': '.c', 'c++': '.cpp', 'cpp': '.cpp', 'python': '.py', 
               'java': '.java', 'javascript': '.js'}
    ext = ext_map.get(language.lower(), '.c')
    
    findings = []
    with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=True) as f:
        f.write(code)
        f.flush()
        
        try:
            result = subprocess.run(
                ['semgrep', '--config', 'auto', '--json', f.name],
                capture_output=True, text=True, timeout=30
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                for r in data.get('results', []):
                    finding = SecurityFinding(
                        rule_id=r.get('check_id', 'unknown'),
                        cwe=r.get('extra', {}).get('metadata', {}).get('cwe', ['unknown'])[0] if isinstance(r.get('extra', {}).get('metadata', {}).get('cwe'), list) else r.get('extra', {}).get('metadata', {}).get('cwe', 'unknown'),
                        severity=r.get('extra', {}).get('severity', 'medium').lower(),
                        message=r.get('extra', {}).get('message', 'Semgrep finding'),
                        line=r.get('start', {}).get('line'),
                        end_line=r.get('end', {}).get('line'),
                        code_snippet=r.get('extra', {}).get('lines', ''),
                        tool='semgrep',
                        confidence='high',
                    )
                    findings.append(finding)
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            logger.warning(f"Semgrep analysis failed: {e}")
    
    return findings


def analyze_code(code: str, language: Optional[str] = None) -> Dict:
    """
    Full security analysis of a code snippet.
    
    Combines pattern-based and optional Semgrep analysis.
    Clearly labels the source of each finding.
    
    Returns:
        Dict with findings, summary, and risk assessment.
    """
    if language is None:
        language = detect_language(code)
    
    # Pattern-based analysis
    pattern_findings = pattern_analysis(code, language)
    
    # Try Semgrep
    semgrep_findings = try_semgrep_analysis(code, language)
    
    all_findings = pattern_findings + semgrep_findings
    
    # Risk assessment
    severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
    cwe_set = set()
    
    for f in all_findings:
        severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1
        if f.cwe and f.cwe != 'unknown':
            cwe_set.add(f.cwe)
    
    # Overall risk score (simplified)
    risk_score = (
        severity_counts.get('critical', 0) * 4 +
        severity_counts.get('high', 0) * 3 +
        severity_counts.get('medium', 0) * 2 +
        severity_counts.get('low', 0) * 1
    )
    
    if risk_score == 0:
        risk_level = 'Low'
    elif risk_score <= 3:
        risk_level = 'Medium'
    elif risk_score <= 8:
        risk_level = 'High'
    else:
        risk_level = 'Critical'
    
    return {
        'language': language,
        'findings': [asdict(f) for f in all_findings],
        'n_findings': len(all_findings),
        'severity_counts': severity_counts,
        'cwes': sorted(list(cwe_set)),
        'risk_score': risk_score,
        'risk_level': risk_level,
        'tools_used': list(set(f.tool for f in all_findings)) if all_findings else ['pattern'],
        'note': 'Pattern-based findings are heuristic and may contain false positives. '
                'They are not confirmed vulnerabilities.',
    }
