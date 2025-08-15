from __future__ import annotations

import os
from typing import Dict, Any, List

import pandas as pd
import requests
from dotenv import load_dotenv


"""
AFDC EV stations (exact source, cited)
- Docs: https://developer.nrel.gov/docs/transportation/alt-fuel-stations-v1/
"""


def get_api_key() -> str:
    load_dotenv()
    # AFDC uses the NREL Developer Network key. Accept either env var.
    key = os.getenv("AFDC_API_KEY") or os.getenv("NREL_API_KEY")
    if not key:
        raise RuntimeError("AFDC_API_KEY or NREL_API_KEY not set in environment")
    return key


def fetch_ev_stations_paginated(
    lat: float,
    lon: float,
    radius_miles: float = 50.0,
    status: str = "E",
    limit: int = 200,
    max_pages: int = 50,
) -> pd.DataFrame:
    """Paginate AFDC nearest endpoint across offsets to collect up to limit*max_pages records.

    Docs: https://developer.nrel.gov/docs/transportation/alt-fuel-stations-v1/
    """
    url = "https://developer.nrel.gov/api/alt-fuel-stations/v1/nearest.json"
    all_rows: List[Dict[str, Any]] = []
    api_key = get_api_key()
    for page in range(max_pages):
        offset = page * limit
        params = {
            "api_key": api_key,
            "latitude": lat,
            "longitude": lon,
            "radius": radius_miles,
            "fuel_type": "ELEC",
            "status": status,
            "limit": min(limit, 200),
            "offset": offset,
        }
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data: Dict[str, Any] = r.json()
        stations = data.get("fuel_stations", [])
        if not stations:
            break
        all_rows.extend(stations)
        if len(stations) < limit:
            break
    return pd.json_normalize(all_rows) if all_rows else pd.DataFrame()


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Download AFDC EV stations in Bay Area vicinity (paginated)")
    parser.add_argument("--lat", type=float, default=37.7749, help="Center latitude (default: SF)")
    parser.add_argument("--lon", type=float, default=-122.4194, help="Center longitude (default: SF)")
    parser.add_argument("--radius", type=float, default=50.0, help="Search radius in miles (default: 50 for Bay Area)")
    parser.add_argument("--status", type=str, default="E", help="Station status filter (E operational, P planned, T temporarily unavailable)")
    parser.add_argument("--limit", type=int, default=200, help="Page size (max 200)")
    parser.add_argument("--max-pages", type=int, default=50, help="Maximum pages to fetch")
    parser.add_argument("--out-path", type=str, default=os.path.join(os.getcwd(), "power_pred", "data", "afdc_ev_stations.parquet"))
    args = parser.parse_args()

    df = fetch_ev_stations_paginated(
        args.lat,
        args.lon,
        args.radius,
        status=args.status,
        limit=args.limit,
        max_pages=args.max_pages,
    )
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    df.to_parquet(args.out_path, index=False)
    print(f"Saved AFDC EV stations to: {args.out_path}")
    print("Cite: DOE/NREL AFDC – https://developer.nrel.gov/docs/transportation/alt-fuel-stations-v1/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


