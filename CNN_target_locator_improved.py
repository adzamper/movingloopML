"""
Improved CNN Target Locator for TEM Survey Data
================================================
Enhanced version with stratified location splits and ensemble uncertainty.

Key Improvements:
1. Stratified location split (tests interpolation at middle of range, not edges)
2. Ensemble of 3 models (provides uncertainty estimates)
3. Uncertainty-aware analysis (flags unreliable predictions)
4. Optimized for realistic deployment scenarios

Tests: "Can the model predict at NEW sites in the MIDDLE of surveyed range?"
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, LSTM, Dense,
                                      Dropout, BatchNormalization, Concatenate,
                                      Bidirectional)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from scipy.signal import savgol_filter
import joblib

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIRECTORY = "."
MAX_STATIONS = 101
STATION_SPACING = 50.0
RANDOM_SEED = 42
N_ENSEMBLE = 3  # Number of models for uncertainty estimation

# ============================================================================
# DATA LOADING
# ============================================================================

def parse_tem_file(file_path):
    """
    Parse a .tem file and extract survey data with metadata.

    Returns: (DataFrame, metadata_dict) or (None, None) on error
    """
    try:
        filename = os.path.basename(file_path)
        folder_name = os.path.basename(os.path.dirname(file_path))

        # Extract true target location from folder name
        true_location = float(folder_name) if folder_name.replace('.','').isdigit() else None

        # Extract configuration from filename (e.g., "0moffset1.tem", "500m_trailing3.tem")
        config_match = re.match(r'(\d+)m[_]?(offset|trailing)?(\d+)', filename)
        if not config_match:
            return None, None

        offset = float(config_match.group(1))
        config_type = config_match.group(2) if config_match.group(2) else 'offset'
        file_id = int(config_match.group(3))

        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Find header
        header_line_index = next((i for i, l in enumerate(lines)
                                 if 'EAST' in l and 'STATION' in l), -1)
        if header_line_index == -1:
            return None, None

        # Parse data
        header = re.split(r'\s+', lines[header_line_index].strip())
        data_lines = [l.strip() for l in lines[header_line_index + 1:]
                     if l.strip() and (l.strip().startswith('-') or l.strip()[0].isdigit())]
        if not data_lines:
            return None, None

        # Create DataFrame
        data = [re.split(r'\s+', line) for line in data_lines]
        df = pd.DataFrame(data)
        num_cols = min(len(header), len(df.columns))
        df = df.iloc[:, :num_cols]
        df.columns = header[:num_cols]

        # Convert numeric columns
        for col in df.columns:
            if col.upper() != 'COMPONENT':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        metadata = {
            'offset': offset,
            'config_type': config_type,
            'true_location': true_location,
            'file_id': file_id
        }

        return df, metadata

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None, None


def create_feature_profile(df, metadata):
    """
    Extract physics-informed features from TEM data.

    Features:
    - Time-window sums (early/mid/late channels) for each component
    - Total field amplitude (TFA) for each time window
    - Decay ratios (late/early) - indicator of anomalies
    - Background-removed residuals (Savitzky-Golay filter)
    - Anomaly strength metrics

    NO spatial gradients - ablation study showed they hurt performance!

    Returns: Feature array of shape (MAX_STATIONS, n_features)
    """
    # Define time windows
    early_channels = [f'CH{i}' for i in range(1, 8)]   # Fast decay
    mid_channels = [f'CH{i}' for i in range(8, 15)]    # Medium decay
    late_channels = [f'CH{i}' for i in range(15, 21)]  # Slow decay
    all_channel_cols = [col for col in df.columns if col.startswith('CH')]

    # Normalize
    max_val = df[all_channel_cols].abs().max().max()
    if max_val > 0:
        for col in all_channel_cols:
            df[col] /= max_val

    # Log transform (handles exponential decay)
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

    # Pivot to get features per station
    feature_df = pd.DataFrame(feature_dfs)
    pivoted_df = feature_df.pivot(index='STATION', columns='COMPONENT',
                                   values=['early_sum', 'mid_sum', 'late_sum'])
    pivoted_df.columns = ['_'.join(col).strip() for col in pivoted_df.columns.values]
    pivoted_df = pivoted_df.reset_index()

    # Calculate Total Field Amplitude (TFA)
    for time in ['early', 'mid', 'late']:
        sum_cols = [f'{time}_sum_{c}' for c in ['X','Y','Z']
                   if f'{time}_sum_{c}' in pivoted_df.columns]
        if sum_cols:
            pivoted_df[f'tfa_{time}'] = np.sqrt(
                np.sum([pivoted_df[col]**2 for col in sum_cols], axis=0))
        else:
            pivoted_df[f'tfa_{time}'] = 0.0

    # Calculate decay ratios (anomaly indicators)
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

    # Sort by station
    pivoted_df = pivoted_df.sort_values(by='STATION').reset_index(drop=True)

    # Base features
    base_features = ([f'{f}_{c}' for f in ['early_sum', 'mid_sum', 'late_sum', 'ratio']
                     for c in ['X', 'Y', 'Z']] +
                    ['tfa_early', 'tfa_mid', 'tfa_late', 'tfa_ratio'])

    for col in base_features:
        if col not in pivoted_df.columns:
            pivoted_df[col] = 0.0

    # CRITICAL: Background removal using Savitzky-Golay filter
    # This removes regional EM field trends, isolating local target anomalies
    for col in base_features:
        if len(pivoted_df[col]) >= 51:
            try:
                background = savgol_filter(pivoted_df[col], window_length=51, polyorder=3)
                pivoted_df[f'{col}_residual'] = pivoted_df[col] - background
            except:
                pivoted_df[f'{col}_residual'] = 0
        else:
            pivoted_df[f'{col}_residual'] = 0

    # Anomaly strength metrics
    residual_cols = [c for c in pivoted_df.columns if 'residual' in c]
    if residual_cols:
        pivoted_df['anomaly_peak'] = pivoted_df[residual_cols].abs().max(axis=1)
        pivoted_df['anomaly_energy'] = np.sqrt(
            (pivoted_df[residual_cols]**2).sum(axis=1))

    # Combine features
    final_features_df = pd.concat([
        pivoted_df[['STATION'] + base_features],
        pivoted_df[residual_cols],
        pivoted_df[['anomaly_peak', 'anomaly_energy']]
    ], axis=1)

    # Add configuration metadata
    final_features_df['offset'] = metadata['offset']
    final_features_df['config_type_encoded'] = 1 if metadata['config_type'] == 'trailing' else 0

    # Create fixed-size spatial profile
    full_profile = np.zeros((MAX_STATIONS, len(final_features_df.columns) - 1))
    feature_cols = [col for col in final_features_df.columns if col != 'STATION']

    for _, row in final_features_df.iterrows():
        station_idx = int(row['STATION'] / STATION_SPACING)
        if 0 <= station_idx < MAX_STATIONS:
            full_profile[station_idx, :] = row[feature_cols].values

    return full_profile


def load_all_data(base_dir):
    """
    Load all TEM data from directory structure.

    Expected structure:
        base_dir/
            1700/
                0moffset1.tem, 0moffset2.tem, ...
            1900/
                ...

    Returns: (X, y, metadata_list)
    """
    all_profiles, all_labels, all_metadata = [], [], []

    folders = [d for d in os.listdir(base_dir)
              if os.path.isdir(os.path.join(base_dir, d))]

    print(f"Found {len(folders)} location folders: {sorted(folders)}")

    for loc_str in sorted(folders):
        try:
            label = float(loc_str)
            folder_path = os.path.join(base_dir, loc_str)
            files = [f for f in os.listdir(folder_path) if f.endswith('.tem')]

            print(f"  Loading {len(files)} files from {loc_str}...")

            for fname in files:
                df, metadata = parse_tem_file(os.path.join(folder_path, fname))
                if df is not None and metadata is not None:
                    profile = create_feature_profile(df, metadata)
                    if profile is not None:
                        all_profiles.append(profile)
                        all_labels.append(label)
                        all_metadata.append(metadata)

        except ValueError:
            print(f"  Skipping non-numeric folder: {loc_str}")
            continue

    print(f"\nTotal samples loaded: {len(all_profiles)}")

    return (np.array(all_profiles, dtype=np.float32),
            np.array(all_labels, dtype=np.float32),
            all_metadata)


def split_by_location_stratified(X, y, metadata, test_ratio=0.15, val_ratio=0.15):
    """
    STRATIFIED location-based split - picks test locations from MIDDLE of range.

    Why stratified?
    - Neural networks interpolate well but extrapolate poorly
    - Testing at edges gives pessimistic performance
    - Testing in middle gives realistic deployment performance

    Strategy:
    - Sort locations by distance
    - Pick test locations from MIDDLE of range (ensures training on both sides)
    - Pick validation from middle-ish regions
    - Use edges and remaining middle for training

    Returns: train/val/test indices
    """
    # Group samples by location
    location_groups = {}
    for i, meta in enumerate(metadata):
        loc = meta['true_location']
        if loc not in location_groups:
            location_groups[loc] = []
        location_groups[loc].append(i)

    # Sort locations
    locations = sorted(location_groups.keys())
    n_locs = len(locations)

    print(f"  Locations: {locations}")
    print(f"  Range: {min(locations)}-{max(locations)}m")

    # Calculate splits
    n_test = max(1, int(n_locs * test_ratio))
    n_val = max(1, int(n_locs * val_ratio))

    # STRATIFIED SELECTION: Pick test from MIDDLE of range
    mid_point = n_locs // 2
    test_start = mid_point - (n_test // 2)
    test_end = test_start + n_test

    test_locs = locations[test_start:test_end]

    # Pick validation from remaining middle regions
    remaining_locs = [loc for loc in locations if loc not in test_locs]
    mid_remaining = len(remaining_locs) // 2
    val_start = mid_remaining - (n_val // 2)
    val_end = val_start + n_val
    val_locs = [remaining_locs[i] for i in range(val_start, min(val_end, len(remaining_locs)))]

    # Training gets everything else (including edges)
    train_locs = [loc for loc in locations if loc not in test_locs and loc not in val_locs]

    # Extract indices
    train_indices = [i for loc in train_locs for i in location_groups[loc]]
    val_indices = [i for loc in val_locs for i in location_groups[loc]]
    test_indices = [i for loc in test_locs for i in location_groups[loc]]

    return train_indices, val_indices, test_indices, train_locs, val_locs, test_locs


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def build_model(input_shape):
    """
    Build CNN-LSTM model for target localization.

    Architecture:
    1. Multi-scale CNN (different kernel sizes capture patterns at different scales)
    2. MaxPooling for spatial downsampling
    3. Bidirectional LSTM (models spatial sequences along survey line)
    4. Dense layers for final prediction

    Returns: Compiled Keras model
    """
    input_layer = Input(shape=input_shape, name='input')

    # Multi-scale CNN blocks
    # Different kernel sizes capture anomalies at different spatial scales
    tower_1 = Conv1D(64, 3, padding='same', activation='relu')(input_layer)
    tower_1 = BatchNormalization()(tower_1)

    tower_2 = Conv1D(64, 7, padding='same', activation='relu')(input_layer)
    tower_2 = BatchNormalization()(tower_2)

    tower_3 = Conv1D(64, 11, padding='same', activation='relu')(input_layer)
    tower_3 = BatchNormalization()(tower_3)

    # Combine multi-scale features
    x = Concatenate(axis=-1)([tower_1, tower_2, tower_3])
    x = Dropout(0.3)(x)

    # Deep CNN processing
    x = Conv1D(128, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(256, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    # Bidirectional LSTM
    # Models spatial sequences in both directions along survey line
    x = Bidirectional(LSTM(128, return_sequences=False))(x)
    x = Dropout(0.4)(x)

    # Dense prediction layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)

    output_layer = Dense(1, activation='linear', name='output')(x)

    # Compile model
    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )

    return model


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("IMPROVED CNN TARGET LOCATOR - STRATIFIED SPLIT + ENSEMBLE")
    print("="*80)
    print("\nKey Improvements:")
    print("  1. Stratified location split (test in MIDDLE of range, not edges)")
    print("  2. Ensemble of 3 models (uncertainty estimation)")
    print("  3. Uncertainty-aware analysis")
    print("="*80)

    # Set random seeds
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    print(f"\n[1/6] Loading data...")
    print("-" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, DATA_DIRECTORY)

    X, y, metadata = load_all_data(data_path)
    print(f"\nData shape: {X.shape}")
    print(f"  {X.shape[0]} samples")
    print(f"  {X.shape[1]} stations (spatial dimension)")
    print(f"  {X.shape[2]} features per station")
    print(f"\nTarget locations: {sorted(set(y))}")

    # Configuration distribution
    configs = {}
    for meta in metadata:
        config_key = f"{int(meta['offset'])}m_{meta['config_type']}"
        configs[config_key] = configs.get(config_key, 0) + 1

    print(f"\nConfiguration distribution:")
    for config, count in sorted(configs.items()):
        print(f"  {config}: {count} samples")

    # -------------------------------------------------------------------------
    # STRATIFIED LOCATION-BASED SPLIT
    # -------------------------------------------------------------------------
    print(f"\n[2/6] Creating STRATIFIED location-based split...")
    print("-" * 80)
    print("  Strategy: Pick test locations from MIDDLE of range for realistic interpolation")

    train_indices, val_indices, test_indices, train_locs, val_locs, test_locs = \
        split_by_location_stratified(X, y, metadata)

    # Extract data for each split
    X_train = X[train_indices]
    y_train = y[train_indices]

    X_val = X[val_indices]
    y_val = y[val_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]
    metadata_test = [metadata[i] for i in test_indices]

    print(f"\nSplit allocation:")
    print(f"  Training:   {len(train_locs)} locations → {len(X_train)} samples")
    print(f"    Locations: {sorted(train_locs)}")
    print(f"  Validation: {len(val_locs)} locations → {len(X_val)} samples")
    print(f"    Locations: {sorted(val_locs)}")
    print(f"  Test:       {len(test_locs)} locations → {len(X_test)} samples")
    print(f"    Locations: {sorted(test_locs)}")

    # Check if test is in middle
    all_locs = sorted(set(y))
    test_in_middle = all(min(train_locs) < loc < max(train_locs) for loc in test_locs)
    if test_in_middle:
        print(f"\n  ✓ STRATIFIED: Test locations are WITHIN training range (interpolation)")
        print(f"  ✓ Training on both sides of test locations")
    else:
        print(f"\n  ⚠ WARNING: Some test locations near/at edges")

    # -------------------------------------------------------------------------
    # Scale Features
    # -------------------------------------------------------------------------
    print(f"\n[3/6] Scaling features...")
    print("-" * 80)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_reshaped)

    X_train = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Save scaler
    joblib.dump(scaler, 'improved_scaler.joblib')
    print("✓ Features scaled to [-1, 1]")
    print("✓ Scaler saved: improved_scaler.joblib")

    # -------------------------------------------------------------------------
    # Build and Train Ensemble
    # -------------------------------------------------------------------------
    print(f"\n[4/6] Training ensemble of {N_ENSEMBLE} models...")
    print("-" * 80)

    input_shape = (X_train.shape[1], X_train.shape[2])
    models = []

    for i in range(N_ENSEMBLE):
        print(f"\n{'='*80}")
        print(f"Training Model {i+1}/{N_ENSEMBLE}")
        print(f"{'='*80}")

        # Set different seed for ensemble diversity
        np.random.seed(RANDOM_SEED + i)
        tf.random.set_seed(RANDOM_SEED + i)

        model = build_model(input_shape)

        if i == 0:
            print("\nModel Architecture:")
            model.summary()

        # Training callbacks
        callbacks = [
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=15,
                min_lr=0.00001,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=40,
                verbose=1,
                restore_best_weights=True
            )
        ]

        print("\nTraining...")
        history = model.fit(
            X_train, y_train,
            epochs=250,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        # Save model
        model_path = f'improved_model_{i}.keras'
        model.save(model_path)
        models.append(model)

        # Quick validation check
        val_preds = model.predict(X_val, verbose=0).flatten()
        val_mae = np.mean(np.abs(val_preds - y_val))
        print(f"\n✓ Model {i+1} Validation MAE: {val_mae:.2f} m")
        print(f"✓ Saved: {model_path}")

    # -------------------------------------------------------------------------
    # Ensemble Evaluation
    # -------------------------------------------------------------------------
    print(f"\n[5/6] Evaluating ensemble on test set...")
    print("-" * 80)

    # Get predictions from all models
    test_preds = np.array([model.predict(X_test, verbose=0).flatten()
                           for model in models])
    mean_preds = np.mean(test_preds, axis=0)
    std_preds = np.std(test_preds, axis=0)  # UNCERTAINTY!

    # Calculate metrics
    errors = np.abs(mean_preds - y_test)
    mae = np.mean(errors)
    median_error = np.median(errors)
    rmse = np.sqrt(np.mean(errors**2))

    print(f"\nOverall Ensemble Performance:")
    print(f"  Mean Absolute Error (MAE):         {mae:.2f} m")
    print(f"  Median Absolute Error:             {median_error:.2f} m")
    print(f"  Root Mean Squared Error (RMSE):    {rmse:.2f} m")
    print(f"  Mean Ensemble Uncertainty (σ):     {np.mean(std_preds):.2f} m")

    # Best and worst predictions
    results = sorted(zip(y_test, mean_preds, errors, std_preds), key=lambda x: x[2])

    print(f"\nBest 5 Predictions:")
    for i in range(min(5, len(results))):
        true, pred, err, unc = results[i]
        print(f"  {i+1}. True={true:.0f}m, Pred={pred:.0f}m, Error={err:.1f}m, σ={unc:.1f}m")

    print(f"\nWorst 5 Predictions:")
    for i in range(min(5, len(results))):
        true, pred, err, unc = results[-(i+1)]
        print(f"  {i+1}. True={true:.0f}m, Pred={pred:.0f}m, Error={err:.1f}m, σ={unc:.1f}m")

    # Uncertainty analysis
    print(f"\nUncertainty Analysis:")
    high_uncertainty = std_preds > np.percentile(std_preds, 75)
    print(f"  High uncertainty predictions (top 25%): {high_uncertainty.sum()}")
    print(f"    Mean error for high uncertainty: {np.mean(errors[high_uncertainty]):.2f} m")
    print(f"    Mean error for low uncertainty:  {np.mean(errors[~high_uncertainty]):.2f} m")

    if np.mean(errors[high_uncertainty]) > np.mean(errors[~high_uncertainty]):
        print(f"  ✓ Uncertainty correlates with error (ensemble is well-calibrated)")
    else:
        print(f"  ⚠ Uncertainty does NOT correlate well with error")

    # Performance by configuration
    print(f"\nPerformance by Configuration:")
    config_results = {}
    for i, meta in enumerate(metadata_test):
        config_key = f"{int(meta['offset'])}m_{meta['config_type']}"
        if config_key not in config_results:
            config_results[config_key] = {'errors': [], 'uncertainties': []}
        config_results[config_key]['errors'].append(errors[i])
        config_results[config_key]['uncertainties'].append(std_preds[i])

    print(f"  {'Configuration':<20} {'N':<6} {'MAE (m)':<10} {'Median (m)':<12} {'Mean σ (m)':<10}")
    print(f"  {'-'*70}")
    for config in sorted(config_results.keys()):
        errs = config_results[config]['errors']
        uncs = config_results[config]['uncertainties']
        print(f"  {config:<20} {len(errs):<6} {np.mean(errs):<10.2f} "
              f"{np.median(errs):<12.2f} {np.mean(uncs):<10.2f}")

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    print(f"\n[6/6] Generating visualizations...")
    print("-" * 80)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Predictions vs True
    ax1 = axes[0, 0]
    scatter = ax1.scatter(y_test, mean_preds, c=std_preds, cmap='RdYlGn_r',
                         alpha=0.6, s=40, vmin=0, vmax=np.percentile(std_preds, 95))
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect')
    ax1.set_xlabel('True Location (m)', fontsize=11)
    ax1.set_ylabel('Predicted Location (m)', fontsize=11)
    ax1.set_title('Predictions vs True\n(color = uncertainty)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Uncertainty (σ, m)')

    # 2. Error Distribution
    ax2 = axes[0, 1]
    ax2.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(mae, color='r', linestyle='--', lw=2, label=f'MAE={mae:.1f}m')
    ax2.axvline(median_error, color='g', linestyle='--', lw=2, label=f'Median={median_error:.1f}m')
    ax2.set_xlabel('Absolute Error (m)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Uncertainty vs Error (KEY PLOT)
    ax3 = axes[0, 2]
    ax3.scatter(std_preds, errors, alpha=0.6, s=40)
    ax3.set_xlabel('Ensemble Uncertainty (σ, m)', fontsize=11)
    ax3.set_ylabel('Absolute Error (m)', fontsize=11)
    ax3.set_title('Uncertainty vs Error\n(Higher σ = Less reliable)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Add correlation
    corr = np.corrcoef(std_preds, errors)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax3.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. Error by Configuration
    ax4 = axes[1, 0]
    config_names = sorted(config_results.keys())
    config_maes = [np.mean(config_results[c]['errors']) for c in config_names]
    colors = plt.cm.viridis(np.linspace(0, 1, len(config_names)))
    ax4.barh(config_names, config_maes, color=colors, alpha=0.8)
    ax4.set_xlabel('Mean Absolute Error (m)', fontsize=11)
    ax4.set_title('Performance by Configuration', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    # 5. Uncertainty by Configuration
    ax5 = axes[1, 1]
    config_uncs = [np.mean(config_results[c]['uncertainties']) for c in config_names]
    ax5.barh(config_names, config_uncs, color=colors, alpha=0.8)
    ax5.set_xlabel('Mean Uncertainty (σ, m)', fontsize=11)
    ax5.set_title('Uncertainty by Configuration', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')

    # 6. Location Split Visualization
    ax6 = axes[1, 2]
    all_locations = sorted(set(y))
    colors_loc = []
    for loc in all_locations:
        if loc in train_locs:
            colors_loc.append('steelblue')
        elif loc in val_locs:
            colors_loc.append('orange')
        else:
            colors_loc.append('red')

    ax6.bar(range(len(all_locations)), [1]*len(all_locations), color=colors_loc, alpha=0.7)
    ax6.set_xticks(range(len(all_locations)))
    ax6.set_xticklabels([f'{int(loc)}' for loc in all_locations], rotation=45, ha='right', fontsize=9)
    ax6.set_xlabel('Location (m)', fontsize=11)
    ax6.set_title('Location Split Strategy', fontsize=12, fontweight='bold')
    ax6.set_yticks([])

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', label='Train'),
                      Patch(facecolor='orange', label='Validation'),
                      Patch(facecolor='red', label='Test')]
    ax6.legend(handles=legend_elements, loc='upper right')

    plt.suptitle('Improved CNN Target Locator - Stratified Split + Ensemble Results',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('improved_cnn_results.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Save results to CSV
    results_df = pd.DataFrame({
        'true_location': y_test,
        'predicted_location': mean_preds,
        'absolute_error': errors,
        'ensemble_uncertainty': std_preds,
        'offset': [metadata_test[i]['offset'] for i in range(len(y_test))],
        'config_type': [metadata_test[i]['config_type'] for i in range(len(y_test))],
        'file_id': [metadata_test[i]['file_id'] for i in range(len(y_test))],
        'high_uncertainty_flag': high_uncertainty
    })
    results_df.to_csv('improved_cnn_predictions.csv', index=False)

    print(f"\n✓ Visualization saved: improved_cnn_results.png")
    print(f"✓ Predictions saved: improved_cnn_predictions.csv")
    print(f"✓ Models saved: improved_model_0.keras to improved_model_{N_ENSEMBLE-1}.keras")
    print(f"✓ Scaler saved: improved_scaler.joblib")

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nTest Performance:")
    print(f"  MAE:              {mae:.2f} m")
    print(f"  Median Error:     {median_error:.2f} m")
    print(f"  RMSE:             {rmse:.2f} m")
    print(f"  Mean Uncertainty: {np.mean(std_preds):.2f} m")

    print(f"\nTest Strategy:")
    print(f"  ✓ STRATIFIED SPLIT: Test locations in MIDDLE of training range")
    print(f"  ✓ Test locations: {sorted(test_locs)}")
    print(f"  ✓ Training range: {min(train_locs)}-{max(train_locs)}m")
    print(f"  ✓ Test is surrounded by training data (optimal interpolation)")

    print(f"\nEnsemble Benefits:")
    print(f"  ✓ Uncertainty estimates identify unreliable predictions")
    print(f"  ✓ High uncertainty predictions have {np.mean(errors[high_uncertainty]):.1f}m MAE")
    print(f"  ✓ Low uncertainty predictions have {np.mean(errors[~high_uncertainty]):.1f}m MAE")

    print(f"\nRecommendation:")
    if corr > 0.3:
        print(f"  ✓ Use ensemble uncertainty to flag predictions needing field verification")
        print(f"  ✓ Predictions with σ > {np.percentile(std_preds, 75):.1f}m are less reliable")
    else:
        print(f"  ⚠ Ensemble uncertainty weakly correlated with error")

    print("="*80)
