from __future__ import annotations

import os
from typing import Dict, Any

import pandas as pd
import requests
from bs4 import BeautifulSoup


"""
PG&E Energy Data Request Portal (real data, cited)
- Portal: https://www.pge-energydatarequest.com/
- Quarterly aggregated usage by ZIP code and customer segment
"""


def scrape_available_datasets() -> Dict[str, Any]:
    """Scrape PG&E portal to find available datasets."""
    url = "https://www.pge-energydatarequest.com/"
    headers = {"User-Agent": "sf-grid-research/1.0 (contact@example.com)"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, 'html.parser')
    
    # Look for download links or dataset references
    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if any(ext in href.lower() for ext in ['.csv', '.xlsx', '.zip']):
            links.append({
                'url': href,
                'text': link.get_text(strip=True),
                'full_url': href if href.startswith('http') else f"https://www.pge-energydatarequest.com{href}"
            })
    
    return {
        'portal_url': url,
        'scraped_links': links,
        'note': 'Manual download may be required from portal'
    }


def filter_sf_zip_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataset for San Francisco ZIP codes."""
    # SF ZIP codes (comprehensive list)
    sf_zips = [
        '94102', '94103', '94104', '94105', '94107', '94108', '94109', '94110',
        '94111', '94112', '94114', '94115', '94116', '94117', '94118', '94119',
        '94120', '94121', '94122', '94123', '94124', '94125', '94126', '94127',
        '94128', '94129', '94130', '94131', '94132', '94133', '94134', '94137',
        '94158', '94164', '94172', '94177', '94188'
    ]
    
    # Try common ZIP column names
    zip_col = None
    for col in df.columns:
        if any(term in col.lower() for term in ['zip', 'postal', 'code']):
            zip_col = col
            break
    
    if zip_col is None:
        print("Warning: No ZIP code column found")
        return df
    
    # Convert to string and filter
    df[zip_col] = df[zip_col].astype(str).str.zfill(5)
    sf_df = df[df[zip_col].isin(sf_zips)]
    print(f"Filtered from {len(df)} to {len(sf_df)} records for SF ZIP codes")
    return sf_df


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Access PG&E Energy Data Request Portal for SF data")
    parser.add_argument("--manual-csv", type=str, help="Path to manually downloaded CSV from PG&E portal")
    parser.add_argument("--out-path", type=str, default=os.path.join(os.getcwd(), "power_pred", "data", "pge_sf_energy.parquet"))
    args = parser.parse_args()

    if args.manual_csv:
        # Process manually downloaded file
        if not os.path.exists(args.manual_csv):
            raise FileNotFoundError(f"Manual CSV not found: {args.manual_csv}")
        
        df = pd.read_csv(args.manual_csv)
        sf_df = filter_sf_zip_codes(df)
        
        os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
        sf_df.to_parquet(args.out_path, index=False)
        print(f"Saved SF-filtered PG&E data to: {args.out_path}")
        print("Cite: PG&E Energy Data Request Portal – https://www.pge-energydatarequest.com/")
        
    else:
        # Scrape portal for available datasets
        info = scrape_available_datasets()
        print("PG&E Energy Data Portal Info:")
        print(f"Portal: {info['portal_url']}")
        print(f"Found {len(info['scraped_links'])} potential download links:")
        for link in info['scraped_links'][:10]:
            print(f"  - {link['text']}: {link['full_url']}")
        print("\nNote: Manual download may be required from the portal.")
        print("Use --manual-csv flag after downloading data.")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
