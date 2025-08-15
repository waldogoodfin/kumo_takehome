#!/usr/bin/env python3
"""
Configuration Loader for Kumo vs CAISO Benchmark
================================================

Loads and validates configuration from YAML files, replacing all hardcoded values.
"""

import os
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path


class ConfigLoader:
    """
    Loads and provides access to configuration settings.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration file path."""
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"✅ Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"❌ Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            print(f"❌ Error parsing configuration file: {e}")
            raise
    
    def _validate_config(self):
        """Validate required configuration sections exist."""
        required_sections = [
            'data_sources', 'directories', 'file_patterns',
            'kumo', 'caiso_benchmarks', 'feature_selection'
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        print("✅ Configuration validation passed")
    
    # Data Sources
    def get_data_source_path(self, source_name: str) -> str:
        """Get file path for a data source."""
        return self.config['data_sources'][source_name]['file_path']
    
    def get_weather_config(self) -> Dict[str, Any]:
        """Get weather data configuration."""
        return self.config['data_sources']['weather']
    
    def get_ev_config(self) -> Dict[str, Any]:
        """Get EV infrastructure configuration."""
        return {
            **self.config['data_sources']['ev_stations'],
            **self.config['ev_infrastructure']
        }
    
    def get_dg_config(self) -> Dict[str, Any]:
        """Get distributed generation configuration."""
        return {
            **self.config['data_sources']['distributed_generation'],
            **self.config['distributed_generation']
        }
    
    def get_pge_config(self) -> Dict[str, Any]:
        """Get PG&E customer data configuration."""
        return self.config['data_sources']['pge_customers']
    
    def get_caiso_config(self) -> Dict[str, Any]:
        """Get CAISO data configuration."""
        return self.config['data_sources']['caiso_data']
    
    # Directories and File Patterns
    def get_runs_dir(self) -> str:
        """Get runs directory path."""
        return self.config['directories']['runs_dir']
    
    def get_data_dir(self) -> str:
        """Get data directory path."""
        return self.config['directories']['data_dir']
    
    def get_file_pattern(self, pattern_name: str) -> str:
        """Get file pattern for generated files."""
        return self.config['file_patterns'][pattern_name]
    
    # Weather Features
    def get_weather_features_config(self) -> Dict[str, Any]:
        """Get weather features configuration."""
        return self.config['weather_features']
    
    # Temporal Features
    def get_temporal_config(self) -> Dict[str, Any]:
        """Get temporal features configuration."""
        return self.config['temporal_features']
    
    # Kumo Configuration
    def get_kumo_api_token(self) -> Optional[str]:
        """Get Kumo API token from environment variable."""
        env_var = self.config['kumo']['api_token_env_var']
        return os.environ.get(env_var)
    
    def get_kumo_training_config(self) -> Dict[str, Any]:
        """Get Kumo training configuration."""
        return self.config['kumo']['training']
    
    def get_forecast_horizons(self) -> Dict[str, int]:
        """Get forecast horizons configuration."""
        return self.config['kumo']['forecast_horizons']
    
    # CAISO Benchmarks
    def get_caiso_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Get CAISO benchmark metrics."""
        return self.config['caiso_benchmarks']
    
    def get_caiso_benchmark(self, horizon: str) -> Dict[str, float]:
        """Get benchmark metrics for specific forecast horizon."""
        return self.config['caiso_benchmarks'][horizon]
    
    # Feature Selection
    def get_real_features(self) -> List[str]:
        """Get list of real features to keep."""
        features = []
        for category, feature_list in self.config['feature_selection']['real_features'].items():
            features.extend(feature_list)
        return features
    
    def get_exclude_columns(self) -> List[str]:
        """Get columns to exclude."""
        return self.config['feature_selection']['exclude_columns']
    
    def get_nan_handling_config(self) -> Dict[str, Any]:
        """Get NaN handling configuration."""
        return self.config['feature_selection']['nan_handling']
    
    # Data Quality
    def get_min_dataset_size(self, dataset_name: str) -> int:
        """Get minimum dataset size threshold."""
        return self.config['data_quality']['min_dataset_sizes'].get(dataset_name, 0)
    
    def get_value_limits(self) -> Dict[str, Any]:
        """Get value limits configuration."""
        return self.config['data_quality']['value_limits']
    
    def get_geographic_bounds(self) -> Dict[str, Any]:
        """Get geographic bounds configuration."""
        return self.config['data_quality']['geographic_bounds']
    
    # Logging
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config['logging']
    
    # Utility methods
    def should_skip_dataset(self, dataset_name: str, dataset_size: int) -> bool:
        """Check if dataset should be skipped due to size."""
        min_size = self.get_min_dataset_size(dataset_name)
        return dataset_size < min_size
    
    def get_all_data_source_paths(self) -> Dict[str, str]:
        """Get all data source file paths."""
        paths = {}
        for source_name, config in self.config['data_sources'].items():
            if 'file_path' in config:
                paths[source_name] = config['file_path']
        return paths
    
    def validate_data_files_exist(self) -> Dict[str, bool]:
        """Check which data files actually exist."""
        paths = self.get_all_data_source_paths()
        existence = {}
        
        for source_name, file_path in paths.items():
            exists = Path(file_path).exists()
            existence[source_name] = exists
            if not exists:
                print(f"⚠️ Data file not found: {source_name} -> {file_path}")
        
        return existence
    
    def get_config_summary(self) -> str:
        """Get a summary of the loaded configuration."""
        summary = [
            "📋 Configuration Summary:",
            f"  Data Sources: {len(self.config['data_sources'])}",
            f"  Forecast Horizons: {list(self.get_forecast_horizons().keys())}",
            f"  Real Features: {len(self.get_real_features())}",
            f"  Runs Directory: {self.get_runs_dir()}",
        ]
        
        # Check data file existence
        existence = self.validate_data_files_exist()
        available_files = sum(existence.values())
        total_files = len(existence)
        summary.append(f"  Available Data Files: {available_files}/{total_files}")
        
        return "\n".join(summary)


def load_config(config_path: str = "config.yaml") -> ConfigLoader:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        ConfigLoader instance
    """
    return ConfigLoader(config_path)


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = load_config("config.yaml")
        print(config.get_config_summary())
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
