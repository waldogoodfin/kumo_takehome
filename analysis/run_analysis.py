from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import pandas as pd


DATA_DIR = os.path.join(os.getcwd(), "power_pred", "data")
RUNS_DIR = os.path.join(os.getcwd(), "power_pred", "runs")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def load_parquet(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


def summarize_weather_history() -> Dict[str, Any]:
    path = os.path.join(DATA_DIR, "meteostat_hourly.parquet")
    df = load_parquet(path)
    out: Dict[str, Any] = {"path": path, "exists": df is not None}
    if df is None or df.empty:
        return out
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["date"] = df["ts"].dt.date
    daily = (
        df.groupby("date")
        .agg(temp_mean=("temp", "mean"), temp_max=("temp", "max"), gust_max=("wind_gust", "max"))
        .reset_index()
    )
    # Last 365 days (compare using plain dates to avoid tz issues)
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    cutoff_date = (_dt.now(_tz.utc) - _td(days=365)).date()
    last_year = daily[daily["date"] >= cutoff_date]
    out["days_covered"] = int(daily.shape[0])
    out["last_year_days"] = int(last_year.shape[0])
    if not last_year.empty:
        out["last_year_mean_temp"] = float(last_year["temp_mean"].mean())
        out["top10_hottest"] = (
            last_year.sort_values("temp_max", ascending=False)
            .head(10)[["date", "temp_max"]]
            .assign(date=lambda x: pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d"))
            .to_dict(orient="records")
        )
        out["max_gust"] = float(last_year["gust_max"].max()) if "gust_max" in last_year else None
    # Monthly climatology
    df["month"] = df["ts"].dt.to_period("M").astype(str)
    monthly = df.groupby("month").agg(temp_mean=("temp", "mean")).reset_index()
    out["monthly_temp_mean"] = (
        monthly.sort_values("month").head(12).to_dict(orient="records")
    )
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


def summarize_latest_forecast() -> Dict[str, Any]:
    path = load_latest_nws_json()
    out: Dict[str, Any] = {"path": path, "exists": path is not None}
    if path is None:
        return out
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    props = data.get("properties", {}) if isinstance(data, dict) else {}
    periods: List[Dict[str, Any]] = props.get("periods", [])
    out["num_periods"] = len(periods)
    if not periods:
        return out
    recs = []
    for p in periods:
        try:
            recs.append({
                "start": pd.to_datetime(p.get("startTime"), utc=True),
                "temp_F": float(p.get("temperature")),
                "pop": (p.get("probabilityOfPrecipitation", {}) or {}).get("value"),
            })
        except Exception:
            continue
    if not recs:
        return out
    fdf = pd.DataFrame.from_records(recs).dropna(subset=["start", "temp_F"]).sort_values("start")
    now = pd.Timestamp.utcnow()
    horizon = fdf[(fdf["start"] >= now) & (fdf["start"] < now + pd.Timedelta(hours=24))]
    if not horizon.empty:
        peak_idx = horizon["temp_F"].idxmax()
        peak_row = horizon.loc[peak_idx]
        out["next_24h_peak_temp_F"] = float(peak_row["temp_F"]) if pd.notna(peak_row["temp_F"]) else None
        out["next_24h_peak_time_utc"] = peak_row["start"].isoformat()
        out["next_24h_cdd65_sum"] = float((horizon["temp_F"] - 65.0).clip(lower=0).sum())
        # Provide a preview table
        out["next_24h_table"] = horizon.head(8).assign(
            start=lambda x: x["start"].dt.strftime("%Y-%m-%d %H:%MZ")
        ).to_dict(orient="records")
    return out


def summarize_ev_infrastructure() -> Dict[str, Any]:
    path = os.path.join(DATA_DIR, "afdc_ev_stations.parquet")
    df = load_parquet(path)
    out: Dict[str, Any] = {"path": path, "exists": df is not None}
    if df is None or df.empty:
        return out
    out["shape"] = list(df.shape)
    # Estimate total ports if fields available
    for col in ["ev_dc_fast_num", "ev_level2_evse_num", "ev_level1_evse_num"]:
        if col not in df.columns:
            df[col] = 0
    df["ports_est"] = df["ev_dc_fast_num"].fillna(0) + df["ev_level2_evse_num"].fillna(0) + df["ev_level1_evse_num"].fillna(0)
    # Top networks by station count and by estimated ports
    if "ev_network" in df.columns:
        out["networks_by_stations_top10"] = df["ev_network"].value_counts().head(10).to_dict()
        out["networks_by_ports_top10"] = (
            df.groupby("ev_network")["ports_est"].sum().sort_values(ascending=False).head(10).astype(int).to_dict()
        )
    # Top cities if present
    city_col = "city" if "city" in df.columns else None
    if city_col:
        out["cities_by_stations_top10"] = df[city_col].value_counts().head(10).to_dict()
    return out


def write_report(results: Dict[str, Any]) -> str:
    ensure_dir(RUNS_DIR)
    rid = now_ts()
    md_path = os.path.join(RUNS_DIR, f"analysis_{rid}.md")
    ln: List[str] = []
    ln.append(f"### Daily analysis – run {rid}")
    ln.append("")
    ln.append("- This run summarizes weather history, the latest forecast, and EV infrastructure in the Bay Area.")
    ln.append("")
    # Weather history
    w = results.get("weather_history", {})
    ln.append("#### Meteostat hourly weather (history)")
    ln.append(f"- Path: `{w.get('path')}`")
    ln.append(f"- Exists: {w.get('exists')}")
    if w.get("exists"):
        ln.append(f"- Days covered: {w.get('days_covered')}")
        ln.append(f"- Last year days: {w.get('last_year_days')}")
        if w.get("last_year_mean_temp") is not None:
            ln.append(f"- Last-year mean temp: {w.get('last_year_mean_temp'):.2f}")
        if w.get("max_gust") is not None:
            ln.append(f"- Max gust last year: {w.get('max_gust'):.2f}")
        if w.get("top10_hottest"):
            ln.append("- Top 10 hottest days last year (date, temp_max):")
            for rec in w["top10_hottest"]:
                ln.append(f"  - {rec['date']}: {rec['temp_max']:.1f}")
    ln.append("")
    # Forecast
    f = results.get("forecast", {})
    ln.append("#### NWS hourly forecast (latest)")
    ln.append(f"- File: `{f.get('path')}`")
    ln.append(f"- Periods: {f.get('num_periods')}")
    if f.get("next_24h_peak_temp_F") is not None:
        ln.append(f"- Next 24h peak temp (F): {f.get('next_24h_peak_temp_F'):.1f}")
        ln.append(f"- Peak time (UTC): {f.get('next_24h_peak_time_utc')}")
        ln.append(f"- Next 24h CDD65 sum: {f.get('next_24h_cdd65_sum'):.1f}")
        ln.append("- Preview next 8 hours:")
        for rec in f.get("next_24h_table", [])[:8]:
            ln.append(f"  - {rec['start']}: {rec['temp_F']:.1f}F, PoP={rec['pop']}")
    ln.append("")
    # EV infra
    e = results.get("ev", {})
    ln.append("#### AFDC EV infrastructure (Bay Area)")
    ln.append(f"- Path: `{e.get('path')}`")
    if e.get("exists"):
        ln.append(f"- Shape: {e.get('shape')}")
        if e.get("networks_by_stations_top10"):
            ln.append("- Top networks by stations:")
            for k, v in list(e["networks_by_stations_top10"].items())[:10]:
                ln.append(f"  - {k}: {v}")
        if e.get("networks_by_ports_top10"):
            ln.append("- Top networks by estimated ports:")
            for k, v in list(e["networks_by_ports_top10"].items())[:10]:
                ln.append(f"  - {k}: {v}")
        if e.get("cities_by_stations_top10"):
            ln.append("- Top cities by stations:")
            for k, v in list(e["cities_by_stations_top10"].items())[:10]:
                ln.append(f"  - {k}: {v}")
    ln.append("")
    ln.append("Sources: Meteostat; NWS API; DOE/NREL AFDC")
    with open(md_path, "w", encoding="utf-8") as fmd:
        fmd.write("\n".join(ln))
    return md_path


def main() -> int:
    ensure_dir(RUNS_DIR)
    results = {
        "weather_history": summarize_weather_history(),
        "forecast": summarize_latest_forecast(),
        "ev": summarize_ev_infrastructure(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = write_report(results)
    print(f"Wrote analysis report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


