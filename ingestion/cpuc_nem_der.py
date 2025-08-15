from __future__ import annotations

import os
import re
from typing import Optional

import pandas as pd


"""
CPUC NEM DER interconnections (exact source, cited)
- Portal: https://www.cpuc.ca.gov/industries-and-topics/electrical-energy/demand-side-management/net-energy-metering
- Note: Exact file URL may change; this script expects a downloaded CSV/XLSX path and parses to a tidy table.
"""


def parse_nem(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    # Attempt to standardize minimal columns
    colmap = {}
    for c in df.columns:
        cl = c.strip().lower()
        if re.search(r"(zip|postal)", cl):
            colmap[c] = "zip_or_feeder"
        elif re.search(r"size|kw|capacity", cl):
            colmap[c] = "installed_kw"
        elif re.search(r"interconnect|c.o.d|permission to operate|pto|date", cl):
            colmap[c] = "interconnection_date"
    out = df.rename(columns=colmap)
    keep = [c for c in ("zip_or_feeder", "installed_kw", "interconnection_date") if c in out.columns]
    return out[keep]


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Parse CPUC NEM interconnections CSV/XLSX to tidy DER table")
    parser.add_argument("--in-path", type=str, required=True)
    parser.add_argument("--out-path", type=str, default=os.path.join(os.getcwd(), "power_pred", "data", "cpuc_nem_der.parquet"))
    args = parser.parse_args()

    df = parse_nem(args.in_path)
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    df.to_parquet(args.out_path, index=False)
    print(f"Saved CPUC NEM DER table to: {args.out_path}")
    print("Cite: CPUC – Net Energy Metering: https://www.cpuc.ca.gov/industries-and-topics/electrical-energy/demand-side-management/net-energy-metering")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


