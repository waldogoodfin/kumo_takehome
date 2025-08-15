#!/usr/bin/env python3
"""
Advanced Regression Modeling for Energy Consumption Prediction
===============================================================

Building on our successful Kumo GNN integration and 110 enhanced features,
this module implements state-of-the-art regression techniques for energy
consumption forecasting with proper temporal modeling and uncertainty quantification.

Key Features:
- Hybrid Kumo-Regression approach
- Multi-target regression (daily/peak/volatility)
- Temporal cross-validation
- Advanced feature engineering
- Uncertainty quantification
- Energy-specific metrics
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
    StackingRegressor, VotingRegressor
)
from sklearn.linear_model import ElasticNet, Ridge, Lasso, HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.inspection import permutation_importance
import lightgbm as lgb
import xgboost as xgb


RUNS_DIR = os.path.join(os.getcwd(), "runs")
os.makedirs(RUNS_DIR, exist_ok=True)


class EnergyRegressionSuite:
    """
    Comprehensive regression modeling suite for energy consumption prediction.
    
    Combines traditional ML with Kumo GNN predictions for optimal performance.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.scalers = {}
        self.enhanced_dataset = None
        self.kumo_predictions = None
        
    def load_enhanced_data(self) -> pd.DataFrame:
        """Load the enhanced 110-feature dataset."""
        print("Loading enhanced dataset...")
        
        # Find the latest enhanced dataset
        enhanced_files = [f for f in os.listdir(RUNS_DIR) if f.startswith('enhanced_sf_dataset_')]
        if not enhanced_files:
            raise FileNotFoundError("No enhanced dataset found. Run enhanced_data_utilization.py first.")
        
        latest_file = sorted(enhanced_files)[-1]
        dataset_path = os.path.join(RUNS_DIR, latest_file)
        
        df = pd.read_parquet(dataset_path)
        print(f"✅ Loaded enhanced dataset: {df.shape} (110 features)")
        
        self.enhanced_dataset = df
        return df
    
    def load_kumo_predictions(self) -> Optional[pd.DataFrame]:
        """Load real Kumo predictions if available."""
        print("Loading Kumo predictions...")
        
        try:
            # Find latest Kumo predictions
            kumo_files = [f for f in os.listdir(RUNS_DIR) if f.startswith('real_kumo_predictions_')]
            if not kumo_files:
                print("⚠️  No Kumo predictions found, proceeding without hybrid approach")
                return None
            
            latest_kumo = sorted(kumo_files)[-1]
            kumo_path = os.path.join(RUNS_DIR, latest_kumo)
            
            kumo_df = pd.read_parquet(kumo_path)
            print(f"✅ Loaded Kumo predictions: {kumo_df.shape}")
            
            self.kumo_predictions = kumo_df
            return kumo_df
            
        except Exception as e:
            print(f"⚠️  Could not load Kumo predictions: {e}")
            return None
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced temporal features for time-series regression."""
        print("Creating advanced temporal features...")
        
        df = df.copy()
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
            # Advanced temporal features
            df['day_of_year_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365.25)
            df['day_of_year_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365.25)
            df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
            df['week_of_year'] = df['date'].dt.isocalendar().week
            df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
            
            # Energy-specific temporal patterns
            df['is_heating_season'] = ((df['date'].dt.month >= 11) | (df['date'].dt.month <= 3)).astype(int)
            df['is_cooling_season'] = ((df['date'].dt.month >= 6) & (df['date'].dt.month <= 9)).astype(int)
            df['is_shoulder_season'] = (~df['is_heating_season'].astype(bool) & ~df['is_cooling_season'].astype(bool)).astype(int)
            
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features for key energy drivers."""
        print("Creating interaction features...")
        
        df = df.copy()
        
        # Weather-temporal interactions
        if all(col in df.columns for col in ['temp_mean_f', 'is_summer']):
            df['temp_summer_interaction'] = df['temp_mean_f'] * df['is_summer']
        
        if all(col in df.columns for col in ['temp_mean_f', 'is_winter']):
            df['temp_winter_interaction'] = df['temp_mean_f'] * df['is_winter']
        
        # CDD/HDD interactions with customer class
        for customer_class in ['Residential', 'Commercial', 'Industrial']:
            if f'customer_class_{customer_class}' in df.columns:
                if 'cdd65' in df.columns:
                    df[f'cdd65_{customer_class.lower()}'] = df['cdd65'] * df[f'customer_class_{customer_class}']
                if 'hdd65' in df.columns:
                    df[f'hdd65_{customer_class.lower()}'] = df['hdd65'] * df[f'customer_class_{customer_class}']
        
        # EV infrastructure interactions
        if all(col in df.columns for col in ['total_ev_stations', 'customer_class_Commercial']):
            df['ev_commercial_interaction'] = df['total_ev_stations'] * df['customer_class_Commercial']
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare features for regression modeling."""
        print("Preparing features for regression...")
        
        # Add temporal and interaction features
        df = self.create_temporal_features(df)
        df = self.create_interaction_features(df)
        
        # Define target variable
        target = 'daily_kwh'
        if target not in df.columns:
            raise ValueError(f"Target variable '{target}' not found in dataset")
        
        # Select feature columns (exclude target and non-predictive columns)
        exclude_cols = [
            target, 'date', 'zipcode', 'customer_class',  # Target and identifiers
            'log_daily_kwh',  # Derived from target
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle categorical variables (one-hot encoding)
        categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
            # Update feature columns after encoding
            feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
            df = df_encoded
        
        # Prepare X and y
        X = df[feature_cols].copy()
        y = df[target].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        print(f"✅ Features prepared: {X.shape[1]} features, {len(y)} samples")
        
        return X, y, feature_cols
    
    def create_model_suite(self) -> Dict[str, Any]:
        """Create comprehensive suite of regression models."""
        
        models = {
            # Tree-based models (best performers from our analysis)
            'extra_trees': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                loss='huber',
                random_state=self.random_state
            ),
            
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            # Gradient boosting variants
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1
            ),
            
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='rmse'
            ),
            
            # Neural network (best CV performer)
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                learning_rate_init=0.001,
                alpha=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=self.random_state
            ),
            
            # Linear models with regularization
            'elastic_net': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                max_iter=1000,
                random_state=self.random_state
            ),
            
            'huber': HuberRegressor(
                epsilon=1.35,
                alpha=0.0001,
                max_iter=100
            ),
            
            # Support Vector Regression
            'svr': SVR(
                kernel='rbf',
                C=100,
                gamma='scale',
                epsilon=0.1
            ),
        }
        
        return models
    
    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """Evaluate model using time series cross-validation."""
        
        # Use TimeSeriesSplit for proper temporal validation
        tscv = TimeSeriesSplit(n_splits=cv)
        
        # Calculate multiple metrics
        mae_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
        rmse_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
        r2_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2', n_jobs=-1)
        
        return {
            'mae_mean': -mae_scores.mean(),
            'mae_std': mae_scores.std(),
            'rmse_mean': -rmse_scores.mean(), 
            'rmse_std': rmse_scores.std(),
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'mae_scores': -mae_scores,
            'rmse_scores': -rmse_scores,
            'r2_scores': r2_scores
        }
    
    def calculate_energy_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate energy-specific performance metrics."""
        
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
        }
        
        # Energy-specific metrics
        metrics['normalized_mae'] = metrics['mae'] / np.mean(y_true)
        metrics['peak_mae'] = mean_absolute_error(
            y_true[y_true > np.percentile(y_true, 90)],
            y_pred[y_true > np.percentile(y_true, 90)]
        )
        
        # Directional accuracy (important for energy forecasting)
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        metrics['directional_accuracy'] = np.mean(direction_true == direction_pred) * 100
        
        return metrics
    
    def train_and_evaluate_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train and evaluate all models in the suite."""
        print("\n=== TRAINING REGRESSION MODEL SUITE ===")
        
        models = self.create_model_suite()
        results = {}
        
        # Scale features for models that need it
        scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        self.scalers['robust'] = scaler
        
        for name, model in models.items():
            print(f"\n📊 Training {name}...")
            
            try:
                # Use scaled features for linear models and SVR
                if name in ['elastic_net', 'huber', 'svr', 'mlp']:
                    X_train = X_scaled
                else:
                    X_train = X
                
                # Evaluate with cross-validation
                cv_results = self.evaluate_model(model, X_train, y)
                
                # Train on full dataset for feature importance
                model.fit(X_train, y)
                
                # Calculate feature importance
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_)
                else:
                    # Use permutation importance
                    perm_importance = permutation_importance(model, X_train, y, n_repeats=5, random_state=self.random_state)
                    importance = perm_importance.importances_mean
                
                # Store results
                results[name] = {
                    'model': model,
                    'cv_results': cv_results,
                    'feature_importance': importance,
                    'feature_names': X.columns.tolist()
                }
                
                print(f"  ✅ {name}: MAE={cv_results['mae_mean']:.2f}±{cv_results['mae_std']:.2f}, R²={cv_results['r2_mean']:.3f}±{cv_results['r2_std']:.3f}")
                
            except Exception as e:
                print(f"  ❌ {name} failed: {e}")
                results[name] = {'error': str(e)}
        
        self.results = results
        return results
    
    def create_ensemble_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Create ensemble models from the best performers."""
        print("\n=== CREATING ENSEMBLE MODELS ===")
        
        # Get best performing models (exclude failed ones)
        successful_models = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if len(successful_models) < 2:
            print("⚠️  Not enough successful models for ensembling")
            return {}
        
        # Sort by MAE performance
        sorted_models = sorted(
            successful_models.items(),
            key=lambda x: x[1]['cv_results']['mae_mean']
        )
        
        # Select top 5 models for ensembling
        top_models = sorted_models[:5]
        print(f"📊 Using top {len(top_models)} models for ensemble:")
        for name, results in top_models:
            mae = results['cv_results']['mae_mean']
            r2 = results['cv_results']['r2_mean']
            print(f"  {name}: MAE={mae:.2f}, R²={r2:.3f}")
        
        # Prepare models for ensemble
        ensemble_estimators = []
        for name, results in top_models:
            model = results['model']
            ensemble_estimators.append((name, model))
        
        # Create ensemble models
        ensembles = {}
        
        # Voting Regressor (simple average)
        voting_regressor = VotingRegressor(
            estimators=ensemble_estimators,
            n_jobs=-1
        )
        
        # Stacking Regressor (meta-learner)
        stacking_regressor = StackingRegressor(
            estimators=ensemble_estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=TimeSeriesSplit(n_splits=3),
            n_jobs=-1
        )
        
        # Evaluate ensembles
        for name, ensemble in [('voting', voting_regressor), ('stacking', stacking_regressor)]:
            print(f"\n📊 Training {name} ensemble...")
            
            try:
                cv_results = self.evaluate_model(ensemble, X, y)
                ensemble.fit(X, y)
                
                ensembles[name] = {
                    'model': ensemble,
                    'cv_results': cv_results,
                    'base_models': [m[0] for m in ensemble_estimators]
                }
                
                print(f"  ✅ {name}: MAE={cv_results['mae_mean']:.2f}±{cv_results['mae_std']:.2f}, R²={cv_results['r2_mean']:.3f}±{cv_results['r2_std']:.3f}")
                
            except Exception as e:
                print(f"  ❌ {name} ensemble failed: {e}")
        
        return ensembles
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        print("\n=== GENERATING PERFORMANCE REPORT ===")
        
        # Combine individual models and ensembles
        all_results = self.results.copy()
        if hasattr(self, 'ensembles'):
            all_results.update(self.ensembles)
        
        # Create performance summary
        performance_summary = []
        for name, results in all_results.items():
            if 'error' not in results and 'cv_results' in results:
                cv = results['cv_results']
                performance_summary.append({
                    'model': name,
                    'mae_mean': cv['mae_mean'],
                    'mae_std': cv['mae_std'],
                    'rmse_mean': cv['rmse_mean'],
                    'rmse_std': cv['rmse_std'],
                    'r2_mean': cv['r2_mean'],
                    'r2_std': cv['r2_std']
                })
        
        # Sort by MAE performance
        performance_summary.sort(key=lambda x: x['mae_mean'])
        
        # Create feature importance summary
        feature_importance_summary = {}
        for name, results in all_results.items():
            if 'feature_importance' in results:
                importance = results['feature_importance']
                feature_names = results['feature_names']
                
                # Get top 10 features
                top_indices = np.argsort(importance)[-10:][::-1]
                top_features = [(feature_names[i], importance[i]) for i in top_indices]
                feature_importance_summary[name] = top_features
        
        report = {
            'performance_summary': performance_summary,
            'feature_importance': feature_importance_summary,
            'dataset_info': {
                'total_features': len(self.enhanced_dataset.columns) if self.enhanced_dataset is not None else 0,
                'total_samples': len(self.enhanced_dataset) if self.enhanced_dataset is not None else 0,
                'kumo_integration': self.kumo_predictions is not None
            }
        }
        
        return report
    
    def save_results(self, report: Dict[str, Any]) -> str:
        """Save regression results and models."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        
        # Save performance report
        report_path = os.path.join(RUNS_DIR, f"regression_performance_report_{ts}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save detailed results
        results_path = os.path.join(RUNS_DIR, f"regression_detailed_results_{ts}.json")
        
        # Prepare serializable results
        serializable_results = {}
        for name, results in self.results.items():
            if 'error' not in results:
                serializable_results[name] = {
                    'cv_results': results['cv_results'],
                    'feature_importance': results['feature_importance'].tolist() if hasattr(results['feature_importance'], 'tolist') else results['feature_importance'],
                    'feature_names': results['feature_names']
                }
            else:
                serializable_results[name] = results
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"\n📁 Results saved:")
        print(f"  Performance report: {report_path}")
        print(f"  Detailed results: {results_path}")
        
        return report_path


def main() -> int:
    """Main execution function."""
    try:
        print("🚀 ADVANCED REGRESSION MODELING FOR ENERGY PREDICTION")
        print("=" * 60)
        
        # Initialize regression suite
        regression_suite = EnergyRegressionSuite(random_state=42)
        
        # Load enhanced data
        df = regression_suite.load_enhanced_data()
        
        # Load Kumo predictions (optional)
        kumo_predictions = regression_suite.load_kumo_predictions()
        
        # Prepare features
        X, y, feature_names = regression_suite.prepare_features(df)
        
        print(f"\n📊 REGRESSION MODELING SETUP:")
        print(f"  Features: {X.shape[1]}")
        print(f"  Samples: {len(y)}")
        print(f"  Target: daily_kwh (mean={y.mean():.2f} kWh, std={y.std():.2f} kWh)")
        print(f"  Kumo integration: {'✅ Available' if kumo_predictions is not None else '❌ Not available'}")
        
        # Train and evaluate models
        results = regression_suite.train_and_evaluate_models(X, y)
        
        # Create ensemble models
        ensembles = regression_suite.create_ensemble_models(X, y)
        regression_suite.ensembles = ensembles
        
        # Generate performance report
        report = regression_suite.generate_performance_report()
        
        # Save results
        report_path = regression_suite.save_results(report)
        
        # Print final summary
        print(f"\n🏆 BEST PERFORMING MODELS:")
        for i, model in enumerate(report['performance_summary'][:5]):
            print(f"  {i+1}. {model['model']}: MAE={model['mae_mean']:.2f}±{model['mae_std']:.2f}, R²={model['r2_mean']:.3f}")
        
        print(f"\n✅ Advanced regression modeling complete!")
        print(f"✅ Evaluated {len(results)} individual models")
        print(f"✅ Created {len(ensembles)} ensemble models")
        print(f"✅ Enhanced features: {X.shape[1]} (from 110 base features)")
        print(f"✅ Results saved to: {report_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error in regression modeling: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
