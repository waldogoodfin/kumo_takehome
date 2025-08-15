# 📊 Data Setup Instructions

## 🚨 **Missing Large Data File**

The main data file `data/ca_iso.pkl` (1.5GB) is **not included** in this repository due to GitHub size limits.

## 🔧 **How to Obtain the Data**

### **Option 1: Use the Ingestion Scripts**
```bash
# This will recreate ca_iso.pkl from CAISO's public API
python ingestion/caiso_oasis_lmp.py
```

### **Option 2: Contact Repository Owner**
The `ca_iso.pkl` file contains processed CAISO data from 2022-2025. Contact the repository maintainer for access to the preprocessed file.

## 📁 **Included Data Files**

The following smaller data files **are included** and ready to use:

- ✅ `data/afdc_ev_stations_sf.parquet` (204KB) - SF Bay Area EV charging stations
- ✅ `data/ca_dg_sf.parquet` (68KB) - SF distributed generation projects  
- ✅ `data/meteostat_hourly.parquet` (284KB) - SF weather data
- ✅ `data/caiso_forecast_comparison.parquet` (1.8MB) - CAISO forecast benchmarks

## 🏃‍♂️ **Quick Test Without Full Data**

You can test the configuration and basic functionality even without `ca_iso.pkl`:

```bash
# Test configuration loading
python -c "from config_loader import ConfigLoader; print(ConfigLoader('config.yaml').get_config_summary())"

# Test ingestion scripts
python ingestion/afdc_ev_stations.py
python ingestion/meteostat_history.py
```

## 📊 **Expected Data Structure**

When you have the full `ca_iso.pkl` file, it should contain:

```python
import pickle
with open('data/ca_iso.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Records: {len(data):,}")
print(f"Columns: {list(data.columns)}")
# Expected: ~18M records with columns like:
# ['TAC_AREA_NAME', 'POS', 'MW', 'EXECUTION_TYPE', 'INTERVALSTARTTIME_GMT']
```

## 🎯 **POS 3.8 Data**

The benchmark specifically uses **POS 3.8** data from `ca_iso.pkl`:

```python
# This is what the benchmark extracts:
pos_38_data = data[
    (data['TAC_AREA_NAME'].isin(['PGE-TAC', 'PGE'])) &
    (data['POS'] == 3.8) &
    (data['EXECUTION_TYPE'] == 'ACTUAL')
]
# Expected: ~27,047 records (SF Bay Area hourly load 2022-2025)
```

## 🔄 **Data Updates**

To keep data fresh, run the ingestion scripts regularly:

```bash
# Update all data sources
python ingestion/afdc_ev_stations.py      # EV stations (monthly)
python ingestion/meteostat_history.py     # Weather (daily)
python ingestion/ca_dg_stats.py          # DG projects (quarterly)
python ingestion/caiso_oasis_lmp.py      # CAISO data (daily)
```

## ⚠️ **Important Notes**

1. **File Size**: `ca_iso.pkl` is 1.5GB - ensure you have sufficient disk space
2. **Processing Time**: Initial data ingestion can take 30+ minutes
3. **API Limits**: Some ingestion scripts may hit rate limits - run during off-peak hours
4. **Data Quality**: Always validate data ranges after ingestion

## 🆘 **Troubleshooting**

### **"FileNotFoundError: ca_iso.pkl"**
- Run the ingestion script: `python ingestion/caiso_oasis_lmp.py`
- Or contact the repository maintainer for the preprocessed file

### **"Empty POS 3.8 data"**
- Check that `ca_iso.pkl` contains PGE territory data with POS values
- Verify the TAC_AREA_NAME filtering logic

### **"Kumo training failed"**
- Ensure you have set `KUMO_TOKEN` environment variable
- Check that you have sufficient data (>1000 records for training)

---

📧 **Need help?** Open an issue or contact the repository maintainer!
