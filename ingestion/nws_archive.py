from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Any

import requests
from dotenv import load_dotenv


"""
NWS forecast archiver (exact source, cited)
- API docs: https://www.weather.gov/documentation/services-web-api
- Flow: GET /points/{lat},{lon} -> properties.forecastHourly -> fetch JSON
- Policy: provide User-Agent with contact info per NWS requirements
"""


def get_user_agent() -> str:
    load_dotenv()
    contact = os.getenv("NWS_CONTACT", "contact@example.com")
    # Recommended format per NWS API guidance
    return f"cpau-grid-archive/1.0 (youremail={contact})"


def resolve_hourly_url(lat: float, lon: float, session: requests.Session) -> str:
    url = f"https://api.weather.gov/points/{lat},{lon}"
    r = session.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["properties"]["forecastHourly"]


def fetch_hourly(hourly_url: str, session: requests.Session) -> Dict[str, Any]:
    r = session.get(hourly_url, timeout=30)
    r.raise_for_status()
    return r.json()


def write_snapshot(out_dir: str, lat: float, lon: float, payload: Dict[str, Any]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts_issue = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fname = f"nws_hourly_{lat:.4f}_{lon:.4f}_{ts_issue}.json"
    path = os.path.join(out_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Archive NWS hourly forecast issuance for Palo Alto")
    parser.add_argument("--lat", type=float, default=37.4419)
    parser.add_argument("--lon", type=float, default=-122.1430)
    parser.add_argument("--out-dir", type=str, default=os.path.join(os.getcwd(), "power_pred", "data", "nws_archive"))
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({"User-Agent": get_user_agent(), "Accept": "application/geo+json"})

    hourly_url = resolve_hourly_url(args.lat, args.lon, session)
    payload = fetch_hourly(hourly_url, session)
    path = write_snapshot(args.out_dir, args.lat, args.lon, payload)
    print(f"Saved NWS hourly forecast issuance to: {path}")
    print("Cite: National Weather Service API – https://www.weather.gov/documentation/services-web-api")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


