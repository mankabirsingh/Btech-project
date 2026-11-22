#!/usr/bin/env python3
"""
scripts/p12/capture_screenshots.py

Automates launching app_final.py on 127.0.0.1:8501 and capturing 3 PNG screenshots:
1. Scan of examples/vulnerable.c -> report_bundle/screenshots/scan_vulnerable.png
2. Scan of examples/fixed.c -> report_bundle/screenshots/scan_fixed.png
3. Model Performance tab -> report_bundle/screenshots/model_performance.png

Terminates the Streamlit app process cleanly upon completion.
"""

import os
import sys
import time
import subprocess
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCREENSHOTS_DIR = BASE_DIR / "report_bundle" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 8501
URL = f"http://127.0.0.1:{PORT}"


def wait_for_server(url: str, timeout_s: float = 30.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    print("=" * 80)
    print("P12: CAPTURING AUTOMATED APP SCREENSHOTS VIA PLAYWRIGHT")
    print("=" * 80)

    # 1. Start Streamlit app in background process
    print(f"Starting Streamlit app on port {PORT}...")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(BASE_DIR / "app_final.py"),
        "--server.port",
        str(PORT),
        "--server.headless",
        "true",
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        print("Waiting for Streamlit server to become responsive...")
        if not wait_for_server(URL, timeout_s=40.0):
            print("ERROR: Streamlit server failed to start within timeout.")
            return

        print("Streamlit server is ready. Launching Playwright browser...")
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=CHROME_PATH,
                headless=True,
                args=["--no-sandbox", "--disable-gpu"],
            )
            context = browser.new_context(viewport={"width": 1400, "height": 950})
            page = context.new_page()

            # -------------------------------------------------------------
            # Screenshot 1: Scan of examples/vulnerable.c
            # -------------------------------------------------------------
            print("\n[Step 1] Loading and scanning examples/vulnerable.c...")
            page.goto(URL)
            page.wait_for_load_state("networkidle")
            time.sleep(2.0)

            code_vuln = (BASE_DIR / "examples" / "vulnerable.c").read_text()
            textarea = page.locator('textarea[aria-label="Source Code for Security Inspection:"]')
            textarea.fill(code_vuln)
            time.sleep(1.0)

            # Click Run Security Audit
            print("  Clicking 'Run Security Audit'...")
            page.locator('button:has-text("Run Security Audit")').click()
            page.wait_for_selector('h3:has-text("Audit Findings")', timeout=30000)
            time.sleep(2.0)

            # Scroll slightly to capture verdict and findings well
            page.evaluate("window.scrollTo(0, 200);")
            time.sleep(1.0)

            out_vulnerable = SCREENSHOTS_DIR / "scan_vulnerable.png"
            page.screenshot(path=str(out_vulnerable), full_page=False)
            print(f"  Successfully captured: {out_vulnerable}")

            # -------------------------------------------------------------
            # Screenshot 2: Scan of examples/fixed.c
            # -------------------------------------------------------------
            print("\n[Step 2] Loading and scanning examples/fixed.c...")
            page.evaluate("window.scrollTo(0, 0);")
            time.sleep(1.0)

            code_fixed = (BASE_DIR / "examples" / "fixed.c").read_text()
            textarea.fill(code_fixed)
            time.sleep(1.0)

            print("  Clicking 'Run Security Audit'...")
            page.locator('button:has-text("Run Security Audit")').click()
            time.sleep(3.0)

            page.evaluate("window.scrollTo(0, 200);")
            time.sleep(1.0)

            out_fixed = SCREENSHOTS_DIR / "scan_fixed.png"
            page.screenshot(path=str(out_fixed), full_page=False)
            print(f"  Successfully captured: {out_fixed}")

            # -------------------------------------------------------------
            # Screenshot 3: Model Performance tab
            # -------------------------------------------------------------
            print("\n[Step 3] Navigating to 'Model Performance' tab...")
            page.evaluate("window.scrollTo(0, 0);")
            time.sleep(1.0)

            # Click Tab 2: "Model Performance"
            page.locator('button[data-baseweb="tab"]:has-text("Model Performance")').click()
            time.sleep(2.0)

            out_perf = SCREENSHOTS_DIR / "model_performance.png"
            page.screenshot(path=str(out_perf), full_page=False)
            print(f"  Successfully captured: {out_perf}")

            browser.close()
            print("\nPlaywright browser session completed cleanly.")

    finally:
        print("Terminating Streamlit server process...")
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Streamlit process terminated.")


if __name__ == "__main__":
    main()
