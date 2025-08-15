from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from meteostat import Hourly, Point


"""
Meteostat hourly history (exact source, cited)
- Docs: https://dev.meteostat.net/python/hourly.html
"""


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download Meteostat hourly weather history for Palo Alto vicinity")
    parser.add_argument("--lat", type=float, default=37.4419)
    parser.add_argument("--lon", type=float, default=-122.1430)
    parser.add_argument("--start", type=str, default="2019-01-01")
    parser.add_argument("--end", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--out-path", type=str, default=os.path.join(os.getcwd(), "power_pred", "data", "meteostat_hourly.parquet"))
    args = parser.parse_args()

    location = Point(args.lat, args.lon)
    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)
    data = Hourly(location, start=start_dt, end=end_dt).fetch()
    # Standardize column names subset
    cols = {
        "temp": "temp",
        "dwpt": "dew_point",
        "wspd": "wind_speed",
        "wpgt": "wind_gust",
        "prcp": "precip",
    }
    df = data.rename(columns=cols)[list(cols.values())]
    df = df.reset_index().rename(columns={"time": "ts"})
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    df.to_parquet(args.out_path, index=False)
    print(f"Saved Meteostat hourly history to: {args.out_path}")
    print("Cite: Meteostat – https://dev.meteostat.net/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


