#!/usr/bin/env python3
"""
Feature Ablation Study for TEM Target Locator
==============================================
Systematically test different feature combinations to understand
which features contribute most to model performance.

This helps answer:
- Which features are critical vs redundant?
- Is the model learning physics or memorizing patterns?
- Can we simplify the model for better interpretability?

Usage:
    python feature_ablation_study.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from scipy.signal import savgol_filter

# Import from main script
from CNN_target_locator import parse_tem_file, load_all_data, MAX_STATIONS, STATION_SPACING

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def create_feature_profile_custom(df, metadata, feature_set='full'):
    """
    Create feature profile with different feature combinations.

    Parameters:
    -----------
    df : DataFrame
        TEM survey data
    metadata : dict
        File metadata
    feature_set : str
        Which feature set to use:
        - 'full': All features (baseline)
        - 'minimal': Only late_sum_Z_residual
        - 'physics_core': Decay ratios + residuals (no gradients/anomaly metrics)
        - 'no_residuals': Raw features only (no background removal)
        - 'no_gradients': All except spatial gradients
        - 'late_only': Only late-time channels
        - 'ratios_only': Only decay ratios
        - 'z_component_only': Only Z-component features
    """
    # Define time windows
    early_channels = [f'CH{i}' for i in range(1, 8)]
    mid_channels = [f'CH{i}' for i in range(8, 15)]
    late_channels = [f'CH{i}' for i in range(15, 21)]
    all_channel_cols = [col for col in df.columns if col.startswith('CH')]

    # Normalize
    max_val = df[all_channel_cols].abs().max().max()
    if max_val > 0:
        for col in all_channel_cols:
            df[col] /= max_val

    # Log transform
    for col in all_channel_cols:
        if col in df.columns:
            df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

    # Extract features per station and component
    feature_dfs = []
    for (st, c), g in df.groupby(['STATION', 'COMPONENT']):
        early_sum = g[early_channels].sum(axis=1).iloc[0]
        mid_sum = g[mid_channels].sum(axis=1).iloc[0]
        late_sum = g[late_channels].sum(axis=1).iloc[0]

        feature_dfs.append({
            'STATION': st,
            'COMPONENT': c,
            'early_sum': early_sum,
            'mid_sum': mid_sum,
            'late_sum': late_sum
        })

    if not feature_dfs:
        return None

    # Pivot
    feature_df = pd.DataFrame(feature_dfs)
    pivoted_df = feature_df.pivot(index='STATION', columns='COMPONENT',
                                   values=['early_sum', 'mid_sum', 'late_sum'])
    pivoted_df.columns = ['_'.join(col).strip() for col in pivoted_df.columns.values]
    pivoted_df = pivoted_df.reset_index()

    # TFA
    for time in ['early', 'mid', 'late']:
        sum_cols = [f'{time}_sum_{c}' for c in ['X','Y','Z']
                   if f'{time}_sum_{c}' in pivoted_df.columns]
        if sum_cols:
            pivoted_df[f'tfa_{time}'] = np.sqrt(
                np.sum([pivoted_df[col]**2 for col in sum_cols], axis=0))
        else:
            pivoted_df[f'tfa_{time}'] = 0.0

    # Decay ratios
    for comp in ['X','Y','Z']:
        early_col = f'early_sum_{comp}'
        late_col = f'late_sum_{comp}'
        if early_col in pivoted_df.columns and late_col in pivoted_df.columns:
            pivoted_df[f'ratio_{comp}'] = pivoted_df[late_col] / (
                pivoted_df[early_col] + 1e-6)
        else:
            pivoted_df[f'ratio_{comp}'] = 0.0

    if 'tfa_early' in pivoted_df.columns and 'tfa_late' in pivoted_df.columns:
        pivoted_df['tfa_ratio'] = pivoted_df['tfa_late'] / (
            pivoted_df['tfa_early'] + 1e-6)
    else:
        pivoted_df['tfa_ratio'] = 0.0

    pivoted_df = pivoted_df.sort_values(by='STATION').reset_index(drop=True)

    # ========================================================================
    # Feature Set Selection
    # ========================================================================

    if feature_set == 'minimal':
        # MINIMAL: Only late_sum_Z with background removal
        base_features = ['late_sum_Z']

    elif feature_set == 'physics_core':
        # PHYSICS CORE: Decay ratios + key residuals
        base_features = ['ratio_X', 'ratio_Y', 'ratio_Z', 'tfa_ratio',
                        'late_sum_X', 'late_sum_Y', 'late_sum_Z', 'tfa_late']

    elif feature_set == 'no_residuals':
        # NO RESIDUALS: Raw features only
        base_features = ([f'{f}_{c}' for f in ['early_sum', 'mid_sum', 'late_sum', 'ratio']
                         for c in ['X', 'Y', 'Z']] +
                        ['tfa_early', 'tfa_mid', 'tfa_late', 'tfa_ratio'])

    elif feature_set == 'no_gradients':
        # NO GRADIENTS: All features except spatial gradients
        base_features = ([f'{f}_{c}' for f in ['early_sum', 'mid_sum', 'late_sum', 'ratio']
                         for c in ['X', 'Y', 'Z']] +
                        ['tfa_early', 'tfa_mid', 'tfa_late', 'tfa_ratio'])

    elif feature_set == 'late_only':
        # LATE ONLY: Only late-time features
        base_features = (['late_sum_X', 'late_sum_Y', 'late_sum_Z', 'tfa_late'] +
                        ['ratio_X', 'ratio_Y', 'ratio_Z', 'tfa_ratio'])

    elif feature_set == 'ratios_only':
        # RATIOS ONLY: Only decay ratios
        base_features = ['ratio_X', 'ratio_Y', 'ratio_Z', 'tfa_ratio']

    elif feature_set == 'z_component_only':
        # Z ONLY: Only Z-component features
        base_features = ['early_sum_Z', 'mid_sum_Z', 'late_sum_Z', 'ratio_Z']

    else:  # 'full'
        # FULL: All features
        base_features = ([f'{f}_{c}' for f in ['early_sum', 'mid_sum', 'late_sum', 'ratio']
                         for c in ['X', 'Y', 'Z']] +
                        ['tfa_early', 'tfa_mid', 'tfa_late', 'tfa_ratio'])

    # Ensure features exist
    for col in base_features:
        if col not in pivoted_df.columns:
            pivoted_df[col] = 0.0

    # ========================================================================
    # Residuals (if not excluded)
    # ========================================================================
    if feature_set not in ['no_residuals', 'ratios_only']:
        for col in base_features:
            if len(pivoted_df[col]) >= 51:
                try:
                    background = savgol_filter(pivoted_df[col], window_length=51, polyorder=3)
                    pivoted_df[f'{col}_residual'] = pivoted_df[col] - background
                except:
                    pivoted_df[f'{col}_residual'] = 0
            else:
                pivoted_df[f'{col}_residual'] = 0

    # ========================================================================
    # Gradients (if not excluded)
    # ========================================================================
    grad_df = None
    if feature_set not in ['no_gradients', 'minimal', 'ratios_only']:
        grad_df = pivoted_df[base_features].diff().fillna(0)
        grad_df.columns = [f'{c}_grad' for c in base_features]

    # ========================================================================
    # Anomaly metrics (if applicable)
    # ========================================================================
    anomaly_df = None
    if feature_set == 'full':
        residual_cols = [c for c in pivoted_df.columns if 'residual' in c]
        if residual_cols:
            pivoted_df['anomaly_peak'] = pivoted_df[residual_cols].abs().max(axis=1)
            pivoted_df['anomaly_energy'] = np.sqrt(
                (pivoted_df[residual_cols]**2).sum(axis=1))
            anomaly_df = pivoted_df[['anomaly_peak', 'anomaly_energy']]

    # ========================================================================
    # Combine features
    # ========================================================================
    residual_cols = [c for c in pivoted_df.columns if 'residual' in c]

    combine_list = [pivoted_df[['STATION'] + base_features]]

    if residual_cols and feature_set not in ['no_residuals', 'ratios_only']:
        combine_list.append(pivoted_df[residual_cols])

    if grad_df is not None:
        combine_list.append(grad_df)

    if anomaly_df is not None:
        combine_list.append(anomaly_df)

    final_features_df = pd.concat(combine_list, axis=1)

    # Add config metadata
    final_features_df['offset'] = metadata['offset']
    final_features_df['config_type_encoded'] = 1 if metadata['config_type'] == 'trailing' else 0

    # Create spatial profile
    full_profile = np.zeros((MAX_STATIONS, len(final_features_df.columns) - 1))
    feature_cols = [col for col in final_features_df.columns if col != 'STATION']

    for _, row in final_features_df.iterrows():
        station_idx = int(row['STATION'] / STATION_SPACING)
        if 0 <= station_idx < MAX_STATIONS:
            full_profile[station_idx, :] = row[feature_cols].values

    return full_profile


def load_data_custom(base_dir, feature_set='full'):
    """Load data with custom feature set."""
    all_profiles, all_labels, all_metadata = [], [], []

    folders = [d for d in os.listdir(base_dir)
              if os.path.isdir(os.path.join(base_dir, d))]

    for loc_str in sorted(folders):
        try:
            label = float(loc_str)
            folder_path = os.path.join(base_dir, loc_str)
            files = [f for f in os.listdir(folder_path) if f.endswith('.tem')]

            for fname in files:
                df, metadata = parse_tem_file(os.path.join(folder_path, fname))
                if df is not None and metadata is not None:
                    profile = create_feature_profile_custom(df, metadata, feature_set)
                    if profile is not None:
                        all_profiles.append(profile)
                        all_labels.append(label)
                        all_metadata.append(metadata)
        except ValueError:
            continue

    return (np.array(all_profiles, dtype=np.float32),
            np.array(all_labels, dtype=np.float32),
            all_metadata)


def build_simple_model(input_shape):
    """Simpler model for faster ablation testing."""
    input_layer = Input(shape=input_shape, name='input')

    # Single CNN layer
    x = Conv1D(64, 7, padding='same', activation='relu')(input_layer)
    x = Dropout(0.3)(x)

    # LSTM
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.4)(x)

    # Dense
    x = Dense(32, activation='relu')(x)
    output_layer = Dense(1, activation='linear', name='output')(x)

    model = Model(inputs=input_layer, outputs=output_layer, name='Ablation_Model')

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error',
                 metrics=['mean_absolute_error'])

    return model


def test_feature_set(feature_set_name, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate model with specific feature set."""
    print(f"\n{'='*70}")
    print(f"Testing Feature Set: {feature_set_name.upper()}")
    print(f"{'='*70}")
    print(f"  Feature shape: {X_train.shape}")
    print(f"  Number of features: {X_train.shape[2]}")

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_simple_model(input_shape)

    # Train (fewer epochs for ablation study)
    print(f"  Training...")
    history = model.fit(
        X_train, y_train,
        epochs=50,  # Reduced for speed
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=0
    )

    # Evaluate
    test_preds = model.predict(X_test, verbose=0).flatten()
    test_mae = np.mean(np.abs(test_preds - y_test))

    print(f"  ✓ Test MAE: {test_mae:.2f} m")

    return test_mae, X_train.shape[2]


def main():
    """Main ablation study."""
    print("="*70)
    print("FEATURE ABLATION STUDY")
    print("="*70)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, ".")

    # Define feature sets to test
    feature_sets = {
        'full': 'All features (baseline)',
        'minimal': 'Only late_sum_Z_residual',
        'physics_core': 'Decay ratios + key residuals',
        'no_residuals': 'Raw features (no background removal)',
        'no_gradients': 'All features except gradients',
        'late_only': 'Only late-time features',
        'ratios_only': 'Only decay ratios',
        'z_component_only': 'Only Z-component features'
    }

    results = {}

    for feature_set, description in feature_sets.items():
        print(f"\n[{list(feature_sets.keys()).index(feature_set)+1}/{len(feature_sets)}] "
              f"Loading data with feature set: {feature_set}")
        print(f"  Description: {description}")

        # Load data with this feature set
        X, y, metadata = load_data_custom(data_path, feature_set=feature_set)

        if len(X) == 0:
            print(f"  ✗ No data loaded, skipping...")
            continue

        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=RANDOM_SEED, stratify=y_temp
        )

        # Scale
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        scaler.fit(X_train_reshaped)

        X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
        X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        # Train and evaluate
        mae, n_features = test_feature_set(
            feature_set, X_train_scaled, y_train,
            X_val_scaled, y_val, X_test_scaled, y_test
        )

        results[feature_set] = {
            'mae': mae,
            'n_features': n_features,
            'description': description
        }

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)

    # Sort by MAE
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mae'])

    print(f"\n{'Feature Set':<20} {'N Features':<12} {'MAE (m)':<10} {'vs Full':<10} Description")
    print("-"*100)

    baseline_mae = results.get('full', {}).get('mae', 0)

    for feature_set, data in sorted_results:
        vs_full = ((data['mae'] - baseline_mae) / baseline_mae * 100) if baseline_mae > 0 else 0
        vs_full_str = f"+{vs_full:.1f}%" if vs_full > 0 else f"{vs_full:.1f}%"

        print(f"{feature_set:<20} {data['n_features']:<12} {data['mae']:<10.2f} "
              f"{vs_full_str:<10} {data['description']}")

    # Analysis
    print(f"\n{'='*70}")
    print("KEY FINDINGS:")
    print(f"{'='*70}")

    # Find best simple model
    simple_models = {k: v for k, v in results.items()
                    if k not in ['full'] and v['n_features'] < 20}
    if simple_models:
        best_simple = min(simple_models.items(), key=lambda x: x[1]['mae'])
        print(f"\nBest Simple Model: {best_simple[0]}")
        print(f"  MAE: {best_simple[1]['mae']:.2f} m ({best_simple[1]['n_features']} features)")
        print(f"  Performance: {(baseline_mae - best_simple[1]['mae'])/baseline_mae*100:.1f}% of full model")

    # Check if residuals matter
    if 'no_residuals' in results:
        residual_impact = results['no_residuals']['mae'] - baseline_mae
        print(f"\nResiduals (background removal) impact: +{residual_impact:.2f}m error if removed")
        if residual_impact > 5:
            print("  → CRITICAL: Residuals are essential!")
        else:
            print("  → Residuals have modest impact")

    # Check if gradients matter
    if 'no_gradients' in results:
        gradient_impact = results['no_gradients']['mae'] - baseline_mae
        print(f"\nSpatial gradients impact: +{gradient_impact:.2f}m error if removed")
        if gradient_impact > 5:
            print("  → IMPORTANT: Gradients help localization")
        else:
            print("  → Gradients have modest impact")

    # Save results
    results_df = pd.DataFrame([
        {
            'feature_set': k,
            'n_features': v['n_features'],
            'mae': v['mae'],
            'vs_full_percent': ((v['mae'] - baseline_mae) / baseline_mae * 100) if baseline_mae > 0 else 0,
            'description': v['description']
        }
        for k, v in results.items()
    ])
    results_df.to_csv('feature_ablation_results.csv', index=False)
    print(f"\n✓ Results saved to 'feature_ablation_results.csv'")

    print("="*70)


if __name__ == '__main__':
    main()
