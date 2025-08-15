from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import requests


"""
CAISO Today's Outlook (exact source, cited)
- JSON: http://content.caiso.com/outlook/SystemLoad.json
"""


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download CAISO Today's Outlook system load JSON")
    parser.add_argument("--out-dir", type=str, default=os.path.join(os.getcwd(), "power_pred", "data", "caiso_outlook"))
    args = parser.parse_args()

    url = "https://content.caiso.com/outlook/SystemLoad.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    payload = r.json()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"caiso_outlook_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"Saved CAISO Today's Outlook to: {path}")
    print("Cite: CAISO – https://content.caiso.com/outlook/SystemLoad.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


