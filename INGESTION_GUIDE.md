# 📊 Data Ingestion Pipeline Guide

## Overview
The ingestion pipeline refreshes and updates the data sources used by the **Geographically Aligned Kumo vs CAISO Benchmark**. Each script downloads fresh data from authoritative sources.

## 🎯 Current Data Dependencies
Your streamlined benchmark uses these **5 essential data files**:
1. `data/ca_iso.pkl` - **PRIMARY**: POS 3.8 SF Bay Area load data
2. `data/meteostat_hourly.parquet` - SF weather data
3. `data/afdc_ev_stations_sf.parquet` - SF Bay Area EV stations
4. `data/ca_dg_sf.parquet` - California distributed generation
5. `data/caiso_forecast_comparison.parquet` - CAISO forecast benchmarks

## 🔧 Setup Instructions

### 1. Install Dependencies
```bash
cd power_pred/ingestion
pip install -r requirements.txt
```

### 2. Set Up API Keys
Create a `.env` file in the `ingestion/` directory:
```bash
# NREL/AFDC API (for EV stations)
NREL_API_KEY=your_nrel_api_key_here

# Optional: Other API keys for additional data sources
CAISO_API_KEY=your_caiso_key_here
```

Get API keys:
- **NREL/AFDC**: https://developer.nrel.gov/signup/
- **CAISO**: https://www.caiso.com/Pages/default.aspx

## 📥 How to Use Each Ingestion Script

### 🌤️ Weather Data (meteostat_history.py)
**Updates**: `data/meteostat_hourly.parquet`
```bash
cd power_pred/ingestion
python meteostat_history.py --lat 37.4419 --lon -122.1430 --start 2022-01-01
```
- Downloads hourly weather for SF Bay Area
- **Frequency**: Monthly (weather data updates regularly)
- **Source**: Meteostat (free, no API key needed)

### ⚡ EV Stations (afdc_ev_stations.py)
**Updates**: `data/afdc_ev_stations_sf.parquet`
```bash
cd power_pred/ingestion
python afdc_ev_stations.py
```
- Downloads current EV charging stations in SF Bay Area
- **Frequency**: Quarterly (new stations added regularly)
- **Source**: NREL AFDC (requires free API key)

### 🏠 Distributed Generation (ca_dg_stats.py)
**Updates**: `data/ca_dg_sf.parquet`
```bash
cd power_pred/ingestion
python ca_dg_stats.py
```
- Downloads California distributed generation projects
- **Frequency**: Quarterly (regulatory filings)
- **Source**: CPUC/CEC databases

### 🎯 CAISO Data (caiso_oasis_lmp.py)
**Updates**: `data/caiso_forecast_comparison.parquet`
```bash
cd power_pred/ingestion
python caiso_oasis_lmp.py --start-date 2022-07-01 --end-date 2025-08-01
```
- Downloads CAISO load forecasts and actuals
- **Frequency**: Weekly (continuous updates)
- **Source**: CAISO OASIS (may require API key for bulk downloads)

### 🏢 PG&E Customer Data (pge_energy_data.py)
**Status**: ⚠️ **Not currently used** (POS 3.8 provides real customer load)
```bash
cd power_pred/ingestion
python pge_energy_data.py
```
- Downloads PG&E quarterly customer usage data
- **Note**: Currently skipped in benchmark (POS 3.8 is better)

## 🔄 Recommended Update Schedule

### Weekly Updates
```bash
# Update CAISO data (most dynamic)
cd power_pred/ingestion
python caiso_oasis_lmp.py --start-date $(date -d '7 days ago' +%Y-%m-%d)
```

### Monthly Updates
```bash
# Update weather data
python meteostat_history.py --start $(date -d '30 days ago' +%Y-%m-%d)

# Re-run benchmark with fresh data
cd ..
python kumo_vs_caiso_benchmark.py
```

### Quarterly Updates
```bash
# Update all infrastructure data
python afdc_ev_stations.py
python ca_dg_stats.py
python pge_energy_data.py  # Optional
```

## 🚨 Critical: ca_iso.pkl Update

**⚠️ IMPORTANT**: The `ca_iso.pkl` file (1.6GB) contains the **core POS 3.8 data**. This is currently a static file but should be updated regularly:

### Manual Update Process:
1. The `ca_iso.pkl` was created from CAISO OASIS bulk downloads
2. Contains POS-level granular data (including POS 3.8)
3. **Action needed**: Create automated script to refresh this file

### Suggested Enhancement:
```bash
# TODO: Create ca_iso_pos_data.py ingestion script
cd power_pred/ingestion
python ca_iso_pos_data.py --update-pos-38 --start-date 2022-07-01
```

## 🎯 Integration with Main Benchmark

After updating data, run the benchmark:
```bash
cd power_pred
python kumo_vs_caiso_benchmark.py
```

The benchmark will automatically:
1. Load POS 3.8 data from `ca_iso.pkl`
2. Align SF weather data (83.7% overlap expected)
3. Filter EV stations to SF Bay Area (1,402+ stations expected)
4. Train Kumo models on proper SF Bay Area scale
5. Compare against SF Bay Area targets (354.1 MW MAE)

## 📊 Expected Results After Data Updates

With fresh data, you should see:
- **7DA**: Kumo MAE ~270-320 MW (beats 357.9 MW target)
- **2DA**: Kumo MAE ~100-200 MW (beats 354.1 MW target)  
- **DAM**: Kumo MAE ~130-190 MW (beats 356.2 MW target)

## 🔍 Monitoring Data Quality

Check data freshness:
```bash
# Check data file dates
ls -la data/*.parquet data/*.pkl

# Validate POS 3.8 data coverage
python -c "
import pickle
with open('data/ca_iso.pkl', 'rb') as f:
    data = pickle.load(f)
pos_38 = data[(data['TAC_AREA_NAME'].isin(['PGE-TAC', 'PGE'])) & (data['POS'] == 3.8)]
print(f'POS 3.8 records: {len(pos_38):,}')
print(f'Latest date: {pos_38[\"INTERVALSTARTTIME_GMT\"].max()}')
"
```

## 🎯 Next Steps

1. **Set up API keys** in `.env` file
2. **Run monthly updates** for weather data
3. **Create ca_iso.pkl refresh script** (high priority)
4. **Monitor benchmark performance** after each update
5. **Automate with cron jobs** for production use

The ingestion pipeline ensures your **geographically aligned benchmark** stays current with the latest real-world data! 🚀
