#!/usr/bin/env python3
"""
ML-Based Settlement Predictor
==============================
Uses machine learning to predict:
1. Time to bottom (ms)
2. Sell wave volume (USD)
3. Price at T+100ms, T+500ms, T+1s, T+5s

Trains on historical settlement data with orderbook features.

Usage:
    # Train models
    python3 ml_settlement_predictor.py --train settlement_predictability_analysis.csv
    
    # Predict in real-time
    from ml_settlement_predictor import MLSettlementPredictor
    
    predictor = MLSettlementPredictor()
    predictor.load_models('models/')
    
    features = extract_features_from_orderbook(...)
    predictions = predictor.predict(features)
    # Returns: {
    #   'time_to_bottom_ms': 450,
    #   'sell_volume_usd': 15000,
    #   'price_100ms_bps': -45,
    #   'price_500ms_bps': -85,
    #   'price_1s_bps': -95,
    #   'price_5s_bps': -60,
    #   'confidence': 0.85
    # }
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb


class MLSettlementPredictor:
    """ML-based predictor for settlement outcomes."""
    
    # Feature columns to use
    FEATURE_COLS = [
        # Funding rate (MOST IMPORTANT!)
        'fr_bps',
        'fr_abs_bps',
        
        # Spread features (orderbook.1)
        'spread_mean_bps',
        'spread_std_bps',
        'spread_max_bps',
        'qty_imb_mean',
        
        # Depth features (orderbook.50)
        'depth_imb_mean',
        'bid10_mean_usd',
        'ask10_mean_usd',
        
        # Total depth (orderbook.200)
        'total_depth_imb_mean',
        'total_bid_mean_usd',
        'total_ask_mean_usd',
        
        # Trade flow
        'trade_flow_imb',
        'pre_trade_count',
        'pre_total_vol_usd',
        'pre_avg_trade_size_usd',
        'pre_price_vol_bps',
    ]
    
    # Target columns
    TARGET_COLS = {
        'time_to_bottom_ms': 'time_to_bottom_ms',
        'sell_volume_usd': 'post_sell_vol_usd',
        'price_100ms_bps': 'drop_100ms_bps',
        'price_500ms_bps': 'drop_500ms_bps',
        'price_1s_bps': 'drop_min_bps',  # Using min as proxy for 1s
        'price_5s_bps': 'drop_final_bps',
    }
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.training_stats = {}
    
    def _create_model(self, target_name: str):
        """Create model for specific target."""
        # Use LightGBM for best performance
        return lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
    
    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
        """Prepare features and targets from raw data."""
        
        # Filter to rows with required features
        feature_mask = df[self.FEATURE_COLS].notna().all(axis=1)
        df_clean = df[feature_mask].copy()
        
        print(f"Data preparation:")
        print(f"  Total rows: {len(df)}")
        print(f"  Rows with all features: {len(df_clean)} ({len(df_clean)/len(df)*100:.1f}%)")
        
        # Extract features
        X = df_clean[self.FEATURE_COLS].copy()
        
        # Extract targets
        targets = {}
        for target_name, col_name in self.TARGET_COLS.items():
            if col_name in df_clean.columns:
                targets[target_name] = df_clean[col_name].copy()
                
                # Handle missing values
                n_missing = targets[target_name].isna().sum()
                if n_missing > 0:
                    print(f"  Warning: {target_name} has {n_missing} missing values")
                    targets[target_name] = targets[target_name].fillna(targets[target_name].median())
        
        return X, targets
    
    def train(self, df: pd.DataFrame, cv_folds: int = 5) -> Dict:
        """Train models for all targets.
        
        Args:
            df: DataFrame with features and targets
            cv_folds: Number of cross-validation folds
        
        Returns:
            Dict with training statistics
        """
        print("\n" + "="*80)
        print("TRAINING ML SETTLEMENT PREDICTOR")
        print("="*80 + "\n")
        
        X, targets = self._prepare_data(df)
        
        if len(X) < 10:
            raise ValueError(f"Insufficient data: only {len(X)} samples with complete features")
        
        results = {}
        
        for target_name, y in targets.items():
            print(f"\n{'='*80}")
            print(f"Training: {target_name}")
            print(f"{'='*80}")
            
            # Create scaler for features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Create and train model
            model = self._create_model(target_name)
            
            # Cross-validation
            kf = KFold(n_splits=min(cv_folds, len(X)), shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                model, X_scaled, y,
                cv=kf,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            
            cv_mae = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Train on full data
            model.fit(X_scaled, y)
            
            # Get predictions for R²
            y_pred = model.predict(X_scaled)
            r2 = r2_score(y, y_pred)
            train_mae = mean_absolute_error(y, y_pred)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': self.FEATURE_COLS,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\nTop 5 features:")
                for _, row in importance.head(5).iterrows():
                    print(f"  {row['feature']:30s}: {row['importance']:.3f}")
                
                self.feature_importance[target_name] = importance
            
            # Store model and scaler
            self.models[target_name] = model
            self.scalers[target_name] = scaler
            
            # Store stats
            stats = {
                'cv_mae': cv_mae,
                'cv_std': cv_std,
                'train_mae': train_mae,
                'r2': r2,
                'n_samples': len(X),
                'target_mean': y.mean(),
                'target_std': y.std(),
            }
            self.training_stats[target_name] = stats
            results[target_name] = stats
            
            print(f"\nPerformance:")
            print(f"  CV MAE:     {cv_mae:.1f} ± {cv_std:.1f}")
            print(f"  Train MAE:  {train_mae:.1f}")
            print(f"  R²:         {r2:.3f}")
            print(f"  Target mean: {y.mean():.1f} ± {y.std():.1f}")
        
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print(f"{'='*80}\n")
        
        return results
    
    def predict(self, features: Dict[str, float]) -> Dict:
        """Make predictions for all targets.
        
        Args:
            features: Dict of feature values
        
        Returns:
            Dict of predictions with confidence scores
        """
        if not self.models:
            raise ValueError("Models not trained. Call train() or load_models() first.")
        
        # Convert features to DataFrame
        X = pd.DataFrame([features])[self.FEATURE_COLS]
        
        # Handle missing features
        X = X.fillna(X.mean())
        
        predictions = {}
        confidences = {}
        
        for target_name, model in self.models.items():
            scaler = self.scalers[target_name]
            X_scaled = scaler.transform(X)
            
            pred = model.predict(X_scaled)[0]
            predictions[target_name] = pred
            
            # Estimate confidence based on CV performance
            stats = self.training_stats[target_name]
            cv_mae = stats['cv_mae']
            target_std = stats['target_std']
            
            # Confidence = 1 - (MAE / std)
            # Higher when MAE is low relative to target variance
            confidence = max(0, min(1, 1 - (cv_mae / target_std)))
            confidences[target_name] = confidence
        
        # Add overall confidence (average)
        predictions['confidence'] = np.mean(list(confidences.values()))
        predictions['confidences'] = confidences
        
        return predictions
    
    def save_models(self, output_dir: Path):
        """Save trained models to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for target_name, model in self.models.items():
            model_path = output_dir / f"{target_name}_model.pkl"
            scaler_path = output_dir / f"{target_name}_scaler.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[target_name], f)
        
        # Save training stats
        stats_path = output_dir / "training_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        # Save feature importance
        for target_name, importance in self.feature_importance.items():
            importance_path = output_dir / f"{target_name}_importance.csv"
            importance.to_csv(importance_path, index=False)
        
        print(f"\n✓ Models saved to {output_dir}/")
    
    def load_models(self, model_dir: Path):
        """Load trained models from disk."""
        model_dir = Path(model_dir)
        
        if not model_dir.exists():
            raise ValueError(f"Model directory not found: {model_dir}")
        
        for target_name in self.TARGET_COLS.keys():
            model_path = model_dir / f"{target_name}_model.pkl"
            scaler_path = model_dir / f"{target_name}_scaler.pkl"
            
            if model_path.exists() and scaler_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[target_name] = pickle.load(f)
                
                with open(scaler_path, 'rb') as f:
                    self.scalers[target_name] = pickle.load(f)
        
        # Load training stats
        stats_path = model_dir / "training_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                self.training_stats = json.load(f)
        
        print(f"✓ Loaded {len(self.models)} models from {model_dir}/")
    
    def format_prediction(self, predictions: Dict) -> str:
        """Format predictions for human-readable output."""
        lines = []
        lines.append("="*80)
        lines.append("ML SETTLEMENT PREDICTIONS")
        lines.append("="*80)
        
        lines.append(f"\nOverall Confidence: {predictions['confidence']:.1%}")
        lines.append("")
        
        lines.append("Timing:")
        if 'time_to_bottom_ms' in predictions:
            lines.append(f"  Time to bottom:   {predictions['time_to_bottom_ms']:.0f} ms")
            conf = predictions['confidences'].get('time_to_bottom_ms', 0)
            lines.append(f"                    (confidence: {conf:.1%})")
        
        lines.append("\nSell Wave:")
        if 'sell_volume_usd' in predictions:
            lines.append(f"  Sell volume:      ${predictions['sell_volume_usd']:.0f}")
            conf = predictions['confidences'].get('sell_volume_usd', 0)
            lines.append(f"                    (confidence: {conf:.1%})")
        
        lines.append("\nPrice Predictions:")
        for horizon, key in [
            ('T+100ms', 'price_100ms_bps'),
            ('T+500ms', 'price_500ms_bps'),
            ('T+1s', 'price_1s_bps'),
            ('T+5s', 'price_5s_bps'),
        ]:
            if key in predictions:
                conf = predictions['confidences'].get(key, 0)
                lines.append(f"  {horizon:10s}    {predictions[key]:+7.1f} bps  (conf: {conf:.1%})")
        
        lines.append("="*80)
        
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="ML Settlement Predictor")
    parser.add_argument('--train', type=str, help='CSV file with training data')
    parser.add_argument('--output', type=str, default='models/', help='Output directory for models')
    parser.add_argument('--cv-folds', type=int, default=5, help='Cross-validation folds')
    
    args = parser.parse_args()
    
    if args.train:
        # Load training data
        print(f"Loading training data from {args.train}...")
        df = pd.read_csv(args.train)
        print(f"Loaded {len(df)} settlements\n")
        
        # Train models
        predictor = MLSettlementPredictor()
        results = predictor.train(df, cv_folds=args.cv_folds)
        
        # Save models
        predictor.save_models(Path(args.output))
        
        # Print summary
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        
        for target_name, stats in results.items():
            print(f"\n{target_name}:")
            print(f"  CV MAE:  {stats['cv_mae']:.1f} ± {stats['cv_std']:.1f}")
            print(f"  R²:      {stats['r2']:.3f}")
            print(f"  Samples: {stats['n_samples']}")
        
        print(f"\n✓ Models saved to {args.output}/")
        print("\nTo use in production:")
        print(f"  predictor = MLSettlementPredictor()")
        print(f"  predictor.load_models('{args.output}')")
        print(f"  predictions = predictor.predict(features)")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
