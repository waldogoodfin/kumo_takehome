from __future__ import annotations

import os
import io
from typing import Dict, Any, Optional

import pandas as pd
import requests


"""
California DG Stats (real data, cited)
- Portal: https://www.californiadgstats.ca.gov/
- Real interconnection data for distributed generation
"""


def fetch_ca_dg_data() -> Optional[pd.DataFrame]:
    """Attempt to fetch California DG Stats data via known endpoints."""
    
    # Try common API endpoints and data URLs
    endpoints = [
        "https://www.californiadgstats.ca.gov/charts/nem/",
        "https://www.californiadgstats.ca.gov/downloads/",
        "https://www.californiadgstats.ca.gov/api/",
    ]
    
    headers = {"User-Agent": "sf-grid-research/1.0 (contact@example.com)"}
    
    for url in endpoints:
        try:
            print(f"Trying: {url}")
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            
            # Check if response looks like CSV data
            content = r.text
            if 'zip' in content.lower() or 'interconnection' in content.lower():
                print(f"Found potential data at: {url}")
                # Try to parse as CSV
                try:
                    df = pd.read_csv(io.StringIO(content))
                    print(f"Successfully parsed CSV with {len(df)} rows, {len(df.columns)} columns")
                    return df
                except Exception as e:
                    print(f"Failed to parse as CSV: {e}")
            
        except Exception as e:
            print(f"Failed to access {url}: {e}")
    
    return None


def filter_sf_interconnections(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for San Francisco interconnections."""
    
    # SF ZIP codes
    sf_zips = [
        '94102', '94103', '94104', '94105', '94107', '94108', '94109', '94110',
        '94111', '94112', '94114', '94115', '94116', '94117', '94118', '94119',
        '94120', '94121', '94122', '94123', '94124', '94125', '94126', '94127',
        '94128', '94129', '94130', '94131', '94132', '94133', '94134', '94137',
        '94158', '94164', '94172', '94177', '94188'
    ]
    
    print(f"Original data columns: {list(df.columns)}")
    
    # Try to find ZIP code column
    zip_col = None
    for col in df.columns:
        if any(term in col.lower() for term in ['zip', 'postal']):
            zip_col = col
            break
    
    # Try to find city column as alternative
    city_col = None
    for col in df.columns:
        if 'city' in col.lower():
            city_col = col
            break
    
    if zip_col:
        print(f"Using ZIP column: {zip_col}")
        df[zip_col] = df[zip_col].astype(str).str.zfill(5)
        sf_df = df[df[zip_col].isin(sf_zips)]
        print(f"Filtered by ZIP: {len(df)} → {len(sf_df)} records")
        return sf_df
    elif city_col:
        print(f"Using city column: {city_col}")
        sf_df = df[df[city_col].str.contains('San Francisco|SF', case=False, na=False)]
        print(f"Filtered by city: {len(df)} → {len(sf_df)} records")
        return sf_df
    else:
        print("Warning: No ZIP or city column found, returning all data")
        return df


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Download California DG Stats for SF")
    parser.add_argument("--manual-csv", type=str, help="Path to manually downloaded CSV")
    parser.add_argument("--out-path", type=str, default=os.path.join(os.getcwd(), "power_pred", "data", "ca_dg_sf.parquet"))
    args = parser.parse_args()

    if args.manual_csv:
        # Process manually downloaded file
        if not os.path.exists(args.manual_csv):
            raise FileNotFoundError(f"Manual CSV not found: {args.manual_csv}")
        
        df = pd.read_csv(args.manual_csv)
        sf_df = filter_sf_interconnections(df)
        
        os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
        sf_df.to_parquet(args.out_path, index=False)
        print(f"Saved SF DG data to: {args.out_path}")
        print("Cite: California DG Stats – https://www.californiadgstats.ca.gov/")
        
    else:
        # Try to fetch data automatically
        print("Attempting to fetch California DG Stats data...")
        df = fetch_ca_dg_data()
        
        if df is not None:
            sf_df = filter_sf_interconnections(df)
            os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
            sf_df.to_parquet(args.out_path, index=False)
            print(f"Saved SF DG data to: {args.out_path}")
            print("Cite: California DG Stats – https://www.californiadgstats.ca.gov/")
        else:
            print("Could not automatically fetch data.")
            print("Visit https://www.californiadgstats.ca.gov/ to download manually,")
            print("then use --manual-csv flag.")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
