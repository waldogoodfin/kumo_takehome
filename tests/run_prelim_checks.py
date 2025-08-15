from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List

import pandas as pd


DATA_DIR = os.path.join(os.getcwd(), "power_pred", "data")
RUNS_DIR = os.path.join(os.getcwd(), "power_pred", "runs")


@dataclass
class RunArtifacts:
    run_id: str
    md_path: str
    json_path: str


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def load_parquet_if_exists(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def summarize_meteostat() -> Dict[str, Any]:
    path = os.path.join(DATA_DIR, "meteostat_hourly.parquet")
    df = load_parquet_if_exists(path)
    out: Dict[str, Any] = {"path": path, "exists": df is not None}
    if df is not None and not df.empty:
        out["shape"] = list(df.shape)
        out["columns"] = list(df.columns)
        # last 30 days summary
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=30)
        recent = df[df["ts"] >= cutoff]
        if not recent.empty:
            out["recent_30d_mean_temp"] = float(recent["temp"].mean()) if "temp" in recent else None
            out["recent_30d_max_gust"] = float(recent.get("wind_gust", pd.Series(dtype=float)).max()) if "wind_gust" in recent else None
    return out


def load_latest_nws_json() -> Optional[str]:
    folder = os.path.join(DATA_DIR, "nws_archive")
    if not os.path.isdir(folder):
        return None
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json")]
    if not files:
        return None
    files.sort()
    return files[-1]


def summarize_nws() -> Dict[str, Any]:
    path = load_latest_nws_json()
    out: Dict[str, Any] = {"latest_file": path, "exists": path is not None}
    if path is None:
        return out
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    props = data.get("properties", {}) if isinstance(data, dict) else {}
    periods: List[Dict[str, Any]] = props.get("periods", [])
    out["top_level_keys"] = list(data.keys()) if isinstance(data, dict) else []
    out["properties_keys"] = list(props.keys())
    out["num_periods"] = len(periods)
    # derive next 24h peak temperature and CDD65
    if periods:
        # Convert to DataFrame for convenience
        recs = []
        for p in periods:
            try:
                recs.append({
                    "start": pd.to_datetime(p.get("startTime"), utc=True),
                    "temp": float(p.get("temperature")),
                    "unit": p.get("temperatureUnit"),
                    "pop": p.get("probabilityOfPrecipitation", {}).get("value"),
                    "dewpoint": p.get("dewpoint", {}).get("value"),
                })
            except Exception:
                continue
        if recs:
            ndf = pd.DataFrame.from_records(recs).dropna(subset=["start", "temp"])
            now = pd.Timestamp.utcnow()
            horizon = ndf[(ndf["start"] >= now) & (ndf["start"] < now + pd.Timedelta(hours=24))]
            if not horizon.empty:
                idx = horizon["temp"].idxmax()
                peak_row = horizon.loc[idx]
                out["next_24h_peak_temp_F"] = float(peak_row["temp"]) if pd.notna(peak_row["temp"]) else None
                out["next_24h_peak_time_utc"] = peak_row["start"].isoformat()
                # CDD65 across horizon (assumes Fahrenheit)
                cdd = (horizon["temp"] - 65.0).clip(lower=0).sum()
                out["next_24h_cdd65_sum"] = float(cdd)
    return out


def summarize_afdc() -> Dict[str, Any]:
    path = os.path.join(DATA_DIR, "afdc_ev_stations.parquet")
    df = load_parquet_if_exists(path)
    out: Dict[str, Any] = {"path": path, "exists": df is not None}
    if df is not None and not df.empty:
        out["shape"] = list(df.shape)
        out["columns"] = list(df.columns)[:15]
        if "ev_network" in df.columns:
            vc = df["ev_network"].value_counts().head(10)
            out["top_networks"] = {k: int(v) for k, v in vc.to_dict().items()}
    return out


def write_run_artifacts(results: Dict[str, Any]) -> RunArtifacts:
    ensure_dir(RUNS_DIR)
    rid = now_ts()
    md_path = os.path.join(RUNS_DIR, f"prelim_{rid}.md")
    json_path = os.path.join(RUNS_DIR, f"prelim_{rid}.json")
    # JSON
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2, default=str)
    # Markdown report
    lines = []
    lines.append(f"### Preliminary data checks – run {rid}")
    lines.append("")
    lines.append("- Summary of what was run:")
    lines.append("  - Loaded Meteostat hourly history and computed last-30d summaries")
    lines.append("  - Loaded latest NWS hourly forecast snapshot; computed next-24h peak temperature and CDD65")
    lines.append("  - Loaded AFDC EV stations; summarized dataset and top networks")
    lines.append("")
    # Meteostat
    m = results.get("meteostat", {})
    lines.append("#### Meteostat hourly history")
    lines.append(f"- Path: `{m.get('path')}`")
    lines.append(f"- Exists: {m.get('exists')}")
    if m.get("exists"):
        lines.append(f"- Shape: {m.get('shape')}")
        lines.append(f"- Columns: {m.get('columns')}")
        if m.get("recent_30d_mean_temp") is not None:
            lines.append(f"- Recent 30d mean temp: {m.get('recent_30d_mean_temp'):.2f}")
        if m.get("recent_30d_max_gust") is not None:
            lines.append(f"- Recent 30d max gust: {m.get('recent_30d_max_gust'):.2f}")
    lines.append("")
    # NWS
    n = results.get("nws", {})
    lines.append("#### NWS forecast snapshot (latest)")
    lines.append(f"- File: `{n.get('latest_file')}`")
    lines.append(f"- Exists: {n.get('exists')}")
    if n.get("exists"):
        lines.append(f"- Num periods: {n.get('num_periods')}")
        if n.get("next_24h_peak_temp_F") is not None:
            lines.append(f"- Next 24h peak temp (F): {n.get('next_24h_peak_temp_F'):.1f}")
            lines.append(f"- Peak time (UTC): {n.get('next_24h_peak_time_utc')}")
            lines.append(f"- Next 24h CDD65 sum: {n.get('next_24h_cdd65_sum'):.1f}")
    lines.append("")
    # AFDC
    a = results.get("afdc", {})
    lines.append("#### AFDC EV stations (Bay Area fetch)")
    lines.append(f"- Path: `{a.get('path')}`")
    lines.append(f"- Exists: {a.get('exists')}")
    if a.get("exists"):
        lines.append(f"- Shape: {a.get('shape')}")
        if a.get("top_networks"):
            lines.append(f"- Top networks: {a.get('top_networks')}")
    lines.append("")
    lines.append("Sources: Meteostat; NWS API; DOE/NREL AFDC")
    with open(md_path, "w", encoding="utf-8") as mf:
        mf.write("\n".join(lines))
    return RunArtifacts(run_id=rid, md_path=md_path, json_path=json_path)


def main() -> int:
    ensure_dir(RUNS_DIR)
    results = {
        "meteostat": summarize_meteostat(),
        "nws": summarize_nws(),
        "afdc": summarize_afdc(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    artifacts = write_run_artifacts(results)
    print(f"Wrote run report: {artifacts.md_path}")
    print(f"Wrote run json:   {artifacts.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


