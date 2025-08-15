from __future__ import annotations

import io
import os
import zipfile
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from time import sleep
from io import StringIO


"""
CAISO OASIS LMP downloader (exact source, cited)
- Docs: http://oasis.caiso.com/mrioasis/
- PRC_LMP with MARKET_RUN_ID=DAM for DLAP_PGAE-APND (preferred) or TH_NP15 (fallback)
"""


def build_url(start: str, end: str, node: str, market_run_id: str = "DAM") -> str:
    params = {
        "queryname": "PRC_LMP",
        "startdatetime": f"{start}T00:00-0000",
        "enddatetime": f"{end}T23:59-0000",
        "version": 1,
        "market_run_id": market_run_id,
        "node": node,
        "resultformat": 6,  # CSV zipped
    }
    q = "&".join(f"{k}={v}" for k, v in params.items())
    return f"http://oasis.caiso.com/oasisapi/SingleZip?{q}"


def fetch_zip_to_df(url: str, session: requests.Session | None = None) -> pd.DataFrame:
    sess = session or requests.Session()
    sess.headers.update({"User-Agent": "cpau-grid-ingest/1.0 (contact@cpau.local)"})
    r = sess.get(url, timeout=60)
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    # Expect single CSV
    names = zf.namelist()
    if not names:
        raise RuntimeError("Empty ZIP from CAISO OASIS")
    with zf.open(names[0]) as f:
        text = f.read().decode("utf-8", errors="ignore")
    # Find the header line that contains INTERVALSTARTTIME_GMT
    lines = text.splitlines()
    try:
        header_idx = next(i for i, ln in enumerate(lines) if "INTERVALSTARTTIME_GMT" in ln)
    except StopIteration:
        # Fallback: attempt direct read
        return pd.read_csv(StringIO(text))
    csv_text = "\n".join(lines[header_idx:])
    return pd.read_csv(StringIO(csv_text))


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    # Common OASIS PRC_LMP columns
    # INTERVALSTARTTIME_GMT, LMP_TYPE, LMP, NODE, MARKET_RUN_ID, etc.
    keep = {
        "INTERVALSTARTTIME_GMT": "ts",
        "LMP": "lmp",
        "LMP_TYPE": "lmp_type",
        "NODE": "node_id",
        "MARKET_RUN_ID": "market_run_id",
    }
    out = df.rename(columns=keep)[list(keep.values())]
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out


def daterange_chunks(start_date: datetime, end_date: datetime, days: int = 7):
    cur = start_date
    while cur <= end_date:
        chunk_end = min(cur + timedelta(days=days - 1), end_date)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download CAISO OASIS LMP (DAM) for DLAP_PGAE-APND (preferred) or NP15 fallback")
    parser.add_argument("--start", type=str, default=(datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d"))
    parser.add_argument("--end", type=str, default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--node", type=str, default="DLAP_PGAE-APND")
    parser.add_argument("--fallback-node", type=str, default="TH_NP15")
    parser.add_argument("--out-path", type=str, default=os.path.join(os.getcwd(), "power_pred", "data", "caiso_lmp.parquet"))
    args = parser.parse_args()

    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)
    session = requests.Session()
    frames = []
    for s, e in daterange_chunks(start_dt, end_dt, days=7):
        url = build_url(s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"), args.node)
        try:
            df = fetch_zip_to_df(url, session)
        except Exception as ex:
            # Fallback to NP15
            url_fb = build_url(s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d"), args.fallback_node)
            # Simple backoff if 429
            for attempt in range(3):
                try:
                    df = fetch_zip_to_df(url_fb, session)
                    break
                except requests.HTTPError as he:
                    if he.response is not None and he.response.status_code == 429 and attempt < 2:
                        sleep(2 ** attempt)
                        continue
                    raise
        frames.append(df)
    if not frames:
        raise SystemExit("No data downloaded from CAISO OASIS")
    out = normalize(pd.concat(frames, ignore_index=True))
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    out.to_parquet(args.out_path, index=False)
    print(f"Saved CAISO OASIS LMP to: {args.out_path}")
    print("Cite: CAISO OASIS – http://oasis.caiso.com/mrioasis/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


