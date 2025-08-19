#!/usr/bin/env python3
"""
Feature Importance Analysis for Kumo Electricity Prediction
===========================================================

Analyzes which features are actually contributing to the 2.3% improvement over CAISO.
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FeatureImportanceAnalyzer:
    """Analyze feature importance for electricity demand prediction."""
    
    def __init__(self):
        self.enhanced_data = None
        self.feature_importance_results = {}
        
    def load_latest_enhanced_data(self) -> pd.DataFrame:
        """Load the latest enhanced dataset."""
        print("📊 Loading latest enhanced dataset...")
        
        runs_dir = "runs"
        dataset_files = [f for f in os.listdir(runs_dir) if f.startswith("enhanced_caiso_dataset_")]
        if not dataset_files:
            raise FileNotFoundError("No enhanced dataset found")
        
        latest_file = sorted(dataset_files)[-1]
        data_path = os.path.join(runs_dir, latest_file)
        
        self.enhanced_data = pd.read_parquet(data_path)
        print(f"✅ Loaded {latest_file}: {self.enhanced_data.shape[0]:,} records, {self.enhanced_data.shape[1]} features")
        
        return self.enhanced_data
    
    def analyze_feature_categories(self) -> Dict[str, List[str]]:
        """Categorize features by type."""
        print("\n🔍 Categorizing features...")
        
        categories = {
            'Weather': [col for col in self.enhanced_data.columns if col.startswith('weather_')],
            'EV Infrastructure': [col for col in self.enhanced_data.columns if col.startswith('ev_')],
            'Distributed Generation': [col for col in self.enhanced_data.columns if col.startswith('dg_')],
            'Load Patterns': [col for col in self.enhanced_data.columns if any(x in col.lower() for x in ['load_', 'lag_', 'rolling_', 'daily_'])],
            'Temporal': [col for col in self.enhanced_data.columns if any(x in col.lower() for x in ['hour', 'day_', 'month', 'quarter', 'year', 'season', 'holiday'])],
            'CAISO Forecasts': [col for col in self.enhanced_data.columns if col.startswith('forecast_')],
            'Target': [col for col in self.enhanced_data.columns if col == 'actual_mw']
        }
        
        # Print category summary
        total_features = sum(len(features) for features in categories.values())
        print(f"📈 Feature Categories ({total_features} total features):")
        for category, features in categories.items():
            if features:
                print(f"  {category}: {len(features)} features")
                if len(features) <= 5:
                    print(f"    {features}")
        
        return categories
    
    def calculate_correlation_with_target(self) -> pd.Series:
        """Calculate correlation between features and actual_mw."""
        print("\n📊 Calculating feature correlations with target...")
        
        # Select only numeric columns
        numeric_data = self.enhanced_data.select_dtypes(include=[np.number])
        
        if 'actual_mw' not in numeric_data.columns:
            raise ValueError("Target column 'actual_mw' not found")
        
        correlations = numeric_data.corr()['actual_mw'].abs().sort_values(ascending=False)
        
        print(f"🔝 Top 10 features correlated with actual_mw:")
        for i, (feature, corr) in enumerate(correlations.head(10).items()):
            if feature != 'actual_mw':
                print(f"  {i+1}. {feature}: {corr:.3f}")
        
        return correlations
    
    def random_forest_importance(self) -> Dict[str, float]:
        """Use Random Forest to determine feature importance."""
        print("\n🌳 Random Forest feature importance analysis...")
        
        # Prepare data
        numeric_data = self.enhanced_data.select_dtypes(include=[np.number])
        
        if 'actual_mw' not in numeric_data.columns:
            raise ValueError("Target column 'actual_mw' not found")
        
        # Remove rows with NaN values
        clean_data = numeric_data.dropna()
        print(f"  📊 Clean data: {len(clean_data):,} records")
        
        # Separate features and target
        X = clean_data.drop(columns=['actual_mw'])
        y = clean_data['actual_mw']
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importance
        importance_dict = dict(zip(X.columns, rf.feature_importances_))
        
        # Sort by importance
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        print(f"🔝 Top 10 Random Forest important features:")
        for i, (feature, importance) in enumerate(sorted_importance[:10]):
            print(f"  {i+1}. {feature}: {importance:.4f}")
        
        # Calculate MAE to see model performance
        y_pred = rf.predict(X)
        mae = mean_absolute_error(y, y_pred)
        print(f"  📊 Random Forest MAE: {mae:.1f} MW")
        
        return importance_dict
    
    def analyze_caiso_forecast_value(self) -> Dict[str, float]:
        """Analyze how much value CAISO forecasts add."""
        print("\n🎯 Analyzing CAISO forecast feature value...")
        
        # Prepare data
        numeric_data = self.enhanced_data.select_dtypes(include=[np.number])
        clean_data = numeric_data.dropna()
        
        X = clean_data.drop(columns=['actual_mw'])
        y = clean_data['actual_mw']
        
        # Find CAISO forecast columns
        caiso_cols = [col for col in X.columns if col.startswith('forecast_')]
        non_caiso_cols = [col for col in X.columns if not col.startswith('forecast_')]
        
        print(f"  📊 CAISO forecast features: {len(caiso_cols)}")
        print(f"  📊 Non-CAISO features: {len(non_caiso_cols)}")
        
        results = {}
        
        if caiso_cols:
            # Model 1: Only CAISO forecasts
            rf_caiso = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_caiso.fit(X[caiso_cols], y)
            y_pred_caiso = rf_caiso.predict(X[caiso_cols])
            mae_caiso = mean_absolute_error(y, y_pred_caiso)
            results['caiso_only'] = mae_caiso
            print(f"  🔵 CAISO forecasts only MAE: {mae_caiso:.1f} MW")
            
            # Model 2: Without CAISO forecasts
            rf_no_caiso = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_no_caiso.fit(X[non_caiso_cols], y)
            y_pred_no_caiso = rf_no_caiso.predict(X[non_caiso_cols])
            mae_no_caiso = mean_absolute_error(y, y_pred_no_caiso)
            results['no_caiso'] = mae_no_caiso
            print(f"  🟡 Without CAISO forecasts MAE: {mae_no_caiso:.1f} MW")
            
            # Model 3: All features
            rf_all = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_all.fit(X, y)
            y_pred_all = rf_all.predict(X)
            mae_all = mean_absolute_error(y, y_pred_all)
            results['all_features'] = mae_all
            print(f"  🟢 All features MAE: {mae_all:.1f} MW")
            
            # Calculate improvements
            caiso_improvement = ((mae_no_caiso - mae_caiso) / mae_no_caiso) * 100
            enhanced_improvement = ((mae_caiso - mae_all) / mae_caiso) * 100
            
            print(f"  📈 CAISO forecasts improve by: {caiso_improvement:.1f}%")
            print(f"  📈 Enhanced features improve CAISO by: {enhanced_improvement:.1f}%")
            
            results['caiso_improvement_pct'] = caiso_improvement
            results['enhanced_improvement_pct'] = enhanced_improvement
        
        return results
    
    def identify_redundant_features(self, importance_threshold: float = 0.001) -> List[str]:
        """Identify features with very low importance."""
        print(f"\n🗑️ Identifying redundant features (importance < {importance_threshold})...")
        
        # Get Random Forest importance
        rf_importance = self.random_forest_importance()
        
        # Find low-importance features
        redundant_features = [feature for feature, importance in rf_importance.items() 
                            if importance < importance_threshold]
        
        print(f"  📊 Found {len(redundant_features)} potentially redundant features:")
        for feature in redundant_features[:10]:  # Show first 10
            print(f"    {feature}: {rf_importance[feature]:.6f}")
        
        if len(redundant_features) > 10:
            print(f"    ... and {len(redundant_features) - 10} more")
        
        return redundant_features
    
    def generate_feature_report(self) -> Dict:
        """Generate comprehensive feature analysis report."""
        print("\n📋 Generating comprehensive feature analysis report...")
        
        # Run all analyses
        categories = self.analyze_feature_categories()
        correlations = self.calculate_correlation_with_target()
        rf_importance = self.random_forest_importance()
        caiso_analysis = self.analyze_caiso_forecast_value()
        redundant_features = self.identify_redundant_features()
        
        # Compile report
        report = {
            'dataset_info': {
                'total_records': len(self.enhanced_data),
                'total_features': len(self.enhanced_data.columns),
                'feature_categories': {cat: len(features) for cat, features in categories.items()}
            },
            'top_correlations': correlations.head(15).to_dict(),
            'top_rf_importance': dict(sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:15]),
            'caiso_forecast_analysis': caiso_analysis,
            'redundant_features': redundant_features,
            'recommendations': self._generate_recommendations(correlations, rf_importance, caiso_analysis, redundant_features)
        }
        
        return report
    
    def _generate_recommendations(self, correlations, rf_importance, caiso_analysis, redundant_features) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Top performing features
        top_corr_features = list(correlations.head(5).index)
        top_rf_features = list(dict(sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)[:5]).keys())
        
        recommendations.append(f"🏆 Focus on top correlating features: {', '.join(top_corr_features[:3])}")
        recommendations.append(f"🌳 Random Forest identifies these as most important: {', '.join(top_rf_features[:3])}")
        
        # CAISO analysis
        if 'enhanced_improvement_pct' in caiso_analysis:
            improvement = caiso_analysis['enhanced_improvement_pct']
            if improvement > 5:
                recommendations.append(f"✅ Enhanced features add significant value ({improvement:.1f}% improvement)")
            elif improvement > 0:
                recommendations.append(f"⚠️ Enhanced features add modest value ({improvement:.1f}% improvement)")
            else:
                recommendations.append("❌ Enhanced features may not be adding value - investigate further")
        
        # Feature reduction
        if len(redundant_features) > 10:
            recommendations.append(f"🗑️ Consider removing {len(redundant_features)} low-importance features")
            recommendations.append("🎯 Focus on feature quality over quantity")
        
        # Specific improvements
        weather_features = [f for f in top_rf_features if f.startswith('weather_')]
        load_features = [f for f in top_rf_features if any(x in f for x in ['load_', 'lag_', 'rolling_'])]
        
        if weather_features:
            recommendations.append(f"🌤️ Weather features are valuable: {', '.join(weather_features[:2])}")
        if load_features:
            recommendations.append(f"📊 Load pattern features are key: {', '.join(load_features[:2])}")
        
        return recommendations
    
    def save_report(self, report: Dict) -> str:
        """Save feature analysis report."""
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%SZ")
        report_path = f"runs/feature_analysis_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Feature analysis report saved: {report_path}")
        return report_path

def main():
    """Run feature importance analysis."""
    print("🔍 FEATURE IMPORTANCE ANALYSIS FOR KUMO ELECTRICITY PREDICTION")
    print("=" * 70)
    
    try:
        # Initialize analyzer
        analyzer = FeatureImportanceAnalyzer()
        
        # Load data
        analyzer.load_latest_enhanced_data()
        
        # Generate comprehensive report
        report = analyzer.generate_feature_report()
        
        # Save report
        report_path = analyzer.save_report(report)
        
        # Print key findings
        print(f"\n🏆 KEY FINDINGS:")
        for rec in report['recommendations']:
            print(f"  {rec}")
        
        print(f"\n✅ Feature analysis complete!")
        print(f"📊 Analyzed {report['dataset_info']['total_features']} features")
        print(f"📋 Report saved to: {report_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Feature analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
