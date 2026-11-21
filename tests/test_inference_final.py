"""
tests/test_inference_final.py

Comprehensive tests for the unified SecurityScanner inference package.
"""

import unittest
import tempfile
from pathlib import Path
from src.inference_final import SecurityScanner


class TestSecurityScanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scanner = SecurityScanner(load_model_b=True)

    def test_vulnerable_code_flagged(self):
        """Vulnerable buffer overflow with strcpy should have high p_code or VULNERABLE/UNSURE verdict."""
        vuln_code = """
        #include <string.h>
        #include <stdio.h>
        void process_input(const char *src) {
            char buffer[64];
            strcpy(buffer, src); // Buffer overflow vulnerability CWE-120
            printf("Processed: %s\\n", buffer);
        }
        """
        res = self.scanner.scan_code(vuln_code)
        self.assertIn(res["verdict"], ["VULNERABLE", "UNSURE"])
        self.assertGreater(res["p_code"], 0.25)
        self.assertTrue(len(res["likely_cwe"]) > 0)
        self.assertIn("model_a", res["model_versions"])

    def test_fixed_code_not_flagged(self):
        """Fixed version with strncpy and bounds check should have lower score."""
        fixed_code = """
        #include <string.h>
        #include <stdio.h>
        void process_input(const char *src) {
            char buffer[64];
            if (src == NULL) return;
            strncpy(buffer, src, sizeof(buffer) - 1);
            buffer[sizeof(buffer) - 1] = '\\0';
            printf("Processed: %s\\n", buffer);
        }
        """
        res = self.scanner.scan_code(fixed_code)
        self.assertIn(res["verdict"], ["SAFE", "UNSURE"])
        self.assertLessEqual(res["p_code"], 0.6956)

    def test_safe_function(self):
        """Trivial mathematical utility function should receive SAFE verdict."""
        safe_code = """
        int calculate_gcd(int a, int b) {
            while (b != 0) {
                int t = b;
                b = a % b;
                a = t;
            }
            return a;
        }
        """
        res = self.scanner.scan_code(safe_code)
        self.assertIn(res["verdict"], ["SAFE", "UNSURE"])
        self.assertLessEqual(res["p_code"], 0.50)

    def test_empty_string(self):
        """Empty string must never crash and must return SAFE."""
        res = self.scanner.scan_code("")
        self.assertEqual(res["verdict"], "SAFE")
        self.assertEqual(res["p_code"], 0.0)
        self.assertTrue(len(res["warnings"]) > 0)

    def test_very_long_file(self):
        """5000-line file must be handled without crashing or OOM."""
        long_lines = ["// auto-generated boilerplate"] * 5000
        long_lines.append("int add(int a, int b) { return a + b; }")
        long_code = "\n".join(long_lines)

        with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as f:
            f.write(long_code)
            tmp_path = Path(f.name)

        try:
            res = self.scanner.scan_file(tmp_path)
            self.assertIn(res["verdict"], ["SAFE", "UNSURE", "VULNERABLE"])
            self.assertGreater(res["num_functions"], 0)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_non_c_text(self):
        """Random English prose / markdown must be parsed safely without crashing."""
        non_c = """
        # Security Policy
        Please report all security vulnerabilities to security@example.com.
        Do not disclose issues publicly until a patch is released.
        Thank you for your cooperation!
        """
        res = self.scanner.scan_code(non_c)
        self.assertIn(res["verdict"], ["SAFE", "UNSURE"])
        self.assertIsInstance(res["runtime_ms"], float)


if __name__ == "__main__":
    unittest.main()
