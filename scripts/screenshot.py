"""Start the API, score a sample message, run the obfuscation probe and save docs/ui.png (needs playwright + Edge).

    python scripts/screenshot.py docs/ui.png
"""
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SPAM = "WINNER!! As a valued network customer you have been selected to receive a 900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only."


def main(out, port=8013):
    server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'spam_api:app', '--port', str(port), '--log-level', 'warning'], cwd=ROOT)
    try:
        for _ in range(90):
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2)
                break
            except OSError:
                time.sleep(1)
        # A throwaway browser profile, so the script runs on any machine (it used a fixed D:/work path).
        with sync_playwright() as p, tempfile.TemporaryDirectory(prefix='spamshield-shot-') as profile:
            ctx = p.chromium.launch_persistent_context(profile, channel='msedge', headless=True,
                                                       viewport={'width': 1100, 'height': 900})
            page = ctx.new_page()
            page.goto(f'http://127.0.0.1:{port}/')
            page.fill('#msg', SPAM)
            page.click('#go')
            page.locator('#out .label').first.wait_for()
            page.click('#probe')
            page.locator('#probeout table').wait_for()
            page.screenshot(path=str(out), full_page=True)
            ctx.close()
    finally:
        server.terminate()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / 'docs' / 'ui.png')
