# Configurable Kumo vs CAISO Benchmark

This document explains the new configuration system that replaces all hardcoded values in the benchmark script.

## Overview

The benchmark script has been completely refactored to remove all hardcoded values and make everything configurable through a YAML configuration file. This makes the system:

- **Flexible**: Easy to modify parameters without changing code
- **Maintainable**: All settings in one place
- **Testable**: Different configurations for different scenarios
- **Secure**: API tokens loaded from environment variables

## Files

- `kumo_vs_caiso_benchmark.py` - Main benchmark script (now fully configurable)
- `config_loader.py` - Configuration loading and validation system
- `config.yaml` - Main configuration file with all settings
- `test_config.py` - Test script to verify configuration system

## Configuration Structure

### Data Sources
```yaml
data_sources:
  weather:
    file_path: "data/meteostat_hourly.parquet"
    timestamp_columns: ["ts", "time"]
    feature_keywords: ["temp", "humid", "wind", "pressure", "precip", "snow"]
  
  ev_stations:
    file_path: "data/afdc_ev_stations_sf.parquet"
    # ... column mappings and settings
```

### EV Infrastructure Settings
```yaml
ev_infrastructure:
  base_load_per_station_mw: 0.05
  dc_fast_multiplier: 3.0
  hourly_load_profiles:
    0: 0.3   # Midnight
    1: 0.2
    # ... 24-hour profile
  peak_charging_hours: [16, 17, 18, 19]
```

### Kumo Configuration
```yaml
kumo:
  api_token_env_var: "KUMO_API_TOKEN"  # Environment variable name
  training:
    sample_sizes: [10, 25, 50]
    table_name_prefix: "temporal_"
  forecast_horizons:
    "7DA": 7   # 7-day ahead
    "2DA": 2   # 2-day ahead
    "DAM": 1   # Day-ahead market
```

### CAISO Benchmarks
```yaml
caiso_benchmarks:
  "7DA":
    mae_mw: 676.8
    rmse_mw: 927.2
    mape_percent: 4.9
  # ... other horizons
```

## Usage

### Basic Usage
```bash
# Use default config.yaml
python kumo_vs_caiso_benchmark.py

# Use custom configuration file
python kumo_vs_caiso_benchmark.py my_custom_config.yaml
```

### Environment Setup
```bash
# Set Kumo API token (required)
export KUMO_API_TOKEN="your_api_token_here"

# Run the benchmark
python kumo_vs_caiso_benchmark.py
```

### Testing Configuration
```bash
# Test that configuration system works
python test_config.py
```

## Key Improvements

### 🔧 **No More Hardcoded Values**
- ✅ API tokens from environment variables
- ✅ File paths configurable
- ✅ All thresholds and constants in config
- ✅ Model parameters adjustable
- ✅ Feature selection customizable

### 📊 **Data Source Flexibility**
- Configure column names for different datasets
- Set minimum dataset size thresholds
- Customize feature extraction keywords
- Handle missing data gracefully

### ⚡ **EV Infrastructure Modeling**
- Configurable load profiles by hour
- Adjustable charging technology weights
- Customizable peak/off-peak periods
- Flexible infrastructure scoring

### 🌤️ **Weather Features**
- Configurable lag and rolling windows
- Adjustable feature keywords
- Customizable derived feature calculations

### 🏠 **Distributed Generation**
- California scaling factors configurable
- Solar generation profiles by hour/season
- Storage discharge patterns adjustable
- Default values when data unavailable

### 🧠 **Kumo Training**
- Sample sizes configurable
- Table naming conventions adjustable
- Forecast horizons customizable
- API configuration secure

### 📈 **Benchmarking**
- CAISO target metrics configurable
- File patterns customizable
- Feature selection rules adjustable

## Configuration Validation

The system includes comprehensive validation:

```python
from config_loader import ConfigLoader

config = ConfigLoader("config.yaml")
print(config.get_config_summary())

# Check if data files exist
existence = config.validate_data_files_exist()
```

## Example Customizations

### Change Sample Sizes
```yaml
kumo:
  training:
    sample_sizes: [5, 15, 30, 100]  # Test different sizes
```

### Adjust EV Load Modeling
```yaml
ev_infrastructure:
  base_load_per_station_mw: 0.08  # Higher base load
  dc_fast_multiplier: 4.0         # Higher DC fast multiplier
  peak_charging_hours: [15, 16, 17, 18, 19, 20]  # Longer peak period
```

### Modify Weather Features
```yaml
weather_features:
  lag_hours: 48              # 48-hour lag instead of 24
  rolling_window_hours: 72   # 72-hour rolling window
```

### Update CAISO Benchmarks
```yaml
caiso_benchmarks:
  "7DA":
    mae_mw: 650.0    # Updated target
    rmse_mw: 900.0   # Updated target
    mape_percent: 4.5  # Updated target
```

## Error Handling

The system includes robust error handling:

- **Missing config file**: Clear error message
- **Invalid YAML**: Parsing error details
- **Missing data files**: Warnings with file paths
- **Missing environment variables**: Clear instructions
- **Invalid configuration**: Validation errors

## Migration from Hardcoded Version

If you have the old hardcoded version:

1. **Backup your changes**: Save any custom modifications
2. **Update imports**: Add `from config_loader import ConfigLoader`
3. **Initialize with config**: Pass config to class constructors
4. **Set environment variables**: Export `KUMO_API_TOKEN`
5. **Run tests**: Use `python test_config.py` to verify

## Benefits

### 🎯 **Flexibility**
- Easy parameter tuning without code changes
- Different configurations for different experiments
- Quick A/B testing of approaches

### 🔒 **Security**
- API tokens in environment variables
- No sensitive data in code
- Secure configuration management

### 🧪 **Testing**
- Separate test configurations
- Reproducible experiments
- Configuration validation

### 🚀 **Maintainability**
- All settings in one place
- Clear documentation of parameters
- Easy to understand and modify

## Troubleshooting

### Common Issues

1. **"Configuration file not found"**
   - Check file path: `config.yaml` should be in current directory
   - Use absolute path if needed

2. **"Kumo API token not found"**
   - Set environment variable: `export KUMO_API_TOKEN="your_token"`
   - Check variable name in config matches

3. **"Data file not found"**
   - Update file paths in `config.yaml`
   - Check that data files exist

4. **"Missing required configuration section"**
   - Ensure all required sections exist in config.yaml
   - Use the provided config.yaml as template

### Getting Help

Run the test script to diagnose issues:
```bash
python test_config.py
```

This will show:
- ✅ Configuration loading status
- ✅ Data file existence
- ✅ Hardcoded value removal verification
- ✅ Configuration usage statistics
