from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv


"""
NSRDB irradiance downloader (exact source, cited)
- Portal: https://nsrdb.nrel.gov/
- API docs: https://developer.nrel.gov/docs/solar/nsrdb/
Note: Requires API key and registration; subject to terms.
"""


def get_api_key() -> str:
    load_dotenv()
    key = os.getenv("NREL_API_KEY")
    if not key:
        raise RuntimeError("NREL_API_KEY not set in environment")
    return key


def fetch_nsrdb(lat: float, lon: float, year: int) -> pd.DataFrame:
    url = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv"
    params = {
        "api_key": get_api_key(),
        "wkt": f"POINT({lon} {lat})",
        "names": year,
        "leap_day": "false",
        "interval": "60",
        "utc": "true",
        "email": os.getenv("NWS_CONTACT", "contact@example.com"),
        "attributes": "ghi,dni,dhi,cloud_type",
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    # NSRDB returns header lines starting with '#'
    lines = [ln for ln in r.text.splitlines() if not ln.startswith("#")]
    df = pd.read_csv(pd.compat.StringIO("\n".join(lines)))
    # Standardize names
    rename = {
        "Year": "year",
        "Month": "month",
        "Day": "day",
        "Hour": "hour",
        "Minute": "minute",
        "GHI": "ghi",
        "DNI": "dni",
        "DHI": "dhi",
        "Cloud Type": "cloud_type",
    }
    df = df.rename(columns=rename)
    df["ts"] = pd.to_datetime(df[["year", "month", "day", "hour", "minute"]], utc=True)
    return df[["ts", "ghi", "dni", "dhi", "cloud_type"]]


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Download NSRDB hourly irradiance for given year")
    parser.add_argument("--lat", type=float, default=37.4419)
    parser.add_argument("--lon", type=float, default=-122.1430)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=int(datetime.utcnow().strftime("%Y")))
    parser.add_argument("--out-path", type=str, default=os.path.join(os.getcwd(), "power_pred", "data", "nsrdb_irradiance.parquet"))
    args = parser.parse_args()

    frames = []
    for year in range(args.start_year, args.end_year + 1):
        frames.append(fetch_nsrdb(args.lat, args.lon, year))
    df = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    df.to_parquet(args.out_path, index=False)
    print(f"Saved NSRDB irradiance to: {args.out_path}")
    print("Cite: NREL NSRDB – https://developer.nrel.gov/docs/solar/nsrdb/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


