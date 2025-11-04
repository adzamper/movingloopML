"""
Simple CNN Target Locator for TEM Survey Data
==============================================
Trains a CNN to predict conductor locations from TEM survey data.

Key Focus:
- Location-based train/test split (tests generalization to NEW survey sites)
- Physics-informed features (time-window sums, decay ratios, background removal)
- Clean, straightforward implementation

What this tests: "Can the model predict conductor location at a NEW survey site?"
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

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIRECTORY = "."
MAX_STATIONS = 101
STATION_SPACING = 50.0
RANDOM_SEED = 42

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
    print("SIMPLE CNN TARGET LOCATOR - LOCATION-BASED SPLIT")
    print("="*80)
    print("\nWhat this tests: Can the model predict conductor location at NEW survey sites?")
    print("="*80)

    # Set random seeds
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    print(f"\n[1/5] Loading data...")
    print("-" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, DATA_DIRECTORY)

    X, y, metadata = load_all_data(data_path)

    # Check if data was loaded
    if len(X) == 0:
        print("\n" + "!"*80)
        print("ERROR: No data found!")
        print("!"*80)
        print(f"\nSearched in: {data_path}")
        print("\nExpected directory structure:")
        print("  your_directory/")
        print("    1700/")
        print("      0moffset1.tem, 0moffset2.tem, ...")
        print("    1900/")
        print("      0moffset1.tem, ...")
        print("    ...")
        print("\nPlease ensure:")
        print("  1. Run script from directory containing numbered location folders")
        print("  2. Or set DATA_DIRECTORY to path containing location folders")
        print("  3. Location folders should be numeric (1700, 1900, 2100, etc.)")
        print("  4. Each folder contains .tem files")
        print("="*80)
        exit(1)

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
    # LOCATION-BASED SPLIT
    # -------------------------------------------------------------------------
    # CRITICAL: Group by LOCATION only (not by configuration!)
    #
    # Why?
    # - Different configs (0m vs 500m) have different physics
    # - Can't train on one config and test on another (proven to fail)
    # - Real deployment: Train at sites A,B,C → Deploy at NEW site D
    # - At each site, we measure ALL configs
    #
    # This split tests: "Can model work at a completely NEW survey site?"
    # -------------------------------------------------------------------------
    print(f"\n[2/5] Creating LOCATION-BASED split...")
    print("-" * 80)

    # Group samples by location (all configs at each location stay together)
    location_groups = {}
    for i, meta in enumerate(metadata):
        loc = meta['true_location']
        if loc not in location_groups:
            location_groups[loc] = []
        location_groups[loc].append(i)

    print(f"Found {len(location_groups)} unique locations:")
    for loc in sorted(location_groups.keys()):
        print(f"  {loc}m: {len(location_groups[loc])} samples (all configs × noise variations)")

    # Shuffle and split locations (70% train, 15% val, 15% test)
    locations = sorted(location_groups.keys())
    np.random.shuffle(locations)

    n_locs = len(locations)
    n_test = max(1, int(n_locs * 0.15))
    n_val = max(1, int(n_locs * 0.15))

    test_locs = locations[:n_test]
    val_locs = locations[n_test:n_test+n_val]
    train_locs = locations[n_test+n_val:]

    # Extract samples for each split
    train_indices = [i for loc in train_locs for i in location_groups[loc]]
    val_indices = [i for loc in val_locs for i in location_groups[loc]]
    test_indices = [i for loc in test_locs for i in location_groups[loc]]

    X_train = X[train_indices]
    y_train = y[train_indices]
    metadata_train = [metadata[i] for i in train_indices]

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

    print(f"\n✓ Location-based split complete")
    print(f"✓ Tests: Can model generalize to NEW survey sites?")

    # -------------------------------------------------------------------------
    # Scale Features
    # -------------------------------------------------------------------------
    print(f"\n[3/5] Scaling features...")
    print("-" * 80)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_reshaped)

    X_train = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    print("✓ Features scaled to [-1, 1]")

    # -------------------------------------------------------------------------
    # Build and Train Model
    # -------------------------------------------------------------------------
    print(f"\n[4/5] Building and training model...")
    print("-" * 80)

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape)

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

    # -------------------------------------------------------------------------
    # Evaluate Model
    # -------------------------------------------------------------------------
    print(f"\n[5/5] Evaluating model on test set...")
    print("-" * 80)

    # Get predictions
    y_pred = model.predict(X_test, verbose=0).flatten()

    # Calculate metrics
    errors = np.abs(y_pred - y_test)
    mae = np.mean(errors)
    median_error = np.median(errors)
    rmse = np.sqrt(np.mean(errors**2))

    print(f"\nOverall Performance:")
    print(f"  Mean Absolute Error (MAE):    {mae:.2f} m")
    print(f"  Median Absolute Error:        {median_error:.2f} m")
    print(f"  Root Mean Squared Error:      {rmse:.2f} m")

    # Best and worst predictions
    results = sorted(zip(y_test, y_pred, errors), key=lambda x: x[2])

    print(f"\nBest 5 Predictions:")
    for i in range(min(5, len(results))):
        true, pred, err = results[i]
        print(f"  {i+1}. True={true:.0f}m, Pred={pred:.0f}m, Error={err:.1f}m")

    print(f"\nWorst 5 Predictions:")
    for i in range(min(5, len(results))):
        true, pred, err = results[-(i+1)]
        print(f"  {i+1}. True={true:.0f}m, Pred={pred:.0f}m, Error={err:.1f}m")

    # Performance by configuration
    print(f"\nPerformance by Configuration:")
    config_results = {}
    for i, meta in enumerate(metadata_test):
        config_key = f"{int(meta['offset'])}m_{meta['config_type']}"
        if config_key not in config_results:
            config_results[config_key] = []
        config_results[config_key].append(errors[i])

    print(f"  {'Configuration':<20} {'N':<6} {'MAE (m)':<10} {'Median (m)':<10}")
    print(f"  {'-'*50}")
    for config in sorted(config_results.keys()):
        config_errors = config_results[config]
        print(f"  {config:<20} {len(config_errors):<6} "
              f"{np.mean(config_errors):<10.2f} {np.median(config_errors):<10.2f}")

    # -------------------------------------------------------------------------
    # Visualize Results
    # -------------------------------------------------------------------------
    print(f"\nGenerating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Predictions vs True
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_pred, alpha=0.6, s=40)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect')
    ax1.set_xlabel('True Location (m)', fontsize=11)
    ax1.set_ylabel('Predicted Location (m)', fontsize=11)
    ax1.set_title('Predictions vs True Values', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

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

    # 3. Training History
    ax3 = axes[1, 0]
    ax3.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax3.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Loss (MSE)', fontsize=11)
    ax3.set_title('Training History', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')

    # 4. Error by Configuration
    ax4 = axes[1, 1]
    config_names = sorted(config_results.keys())
    config_maes = [np.mean(config_results[c]) for c in config_names]
    colors = plt.cm.viridis(np.linspace(0, 1, len(config_names)))
    ax4.barh(config_names, config_maes, color=colors, alpha=0.8)
    ax4.set_xlabel('Mean Absolute Error (m)', fontsize=11)
    ax4.set_title('Performance by Configuration', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Simple CNN Target Locator - Location-Based Split Results',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('simple_cnn_results.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Save results to CSV
    results_df = pd.DataFrame({
        'true_location': y_test,
        'predicted_location': y_pred,
        'absolute_error': errors,
        'offset': [metadata_test[i]['offset'] for i in range(len(y_test))],
        'config_type': [metadata_test[i]['config_type'] for i in range(len(y_test))],
        'file_id': [metadata_test[i]['file_id'] for i in range(len(y_test))]
    })
    results_df.to_csv('simple_cnn_predictions.csv', index=False)

    # Save model
    model.save('simple_cnn_model.keras')

    print(f"\n✓ Visualization saved: simple_cnn_results.png")
    print(f"✓ Predictions saved: simple_cnn_predictions.csv")
    print(f"✓ Model saved: simple_cnn_model.keras")

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nTest Performance:")
    print(f"  MAE:    {mae:.2f} m")
    print(f"  Median: {median_error:.2f} m")
    print(f"  RMSE:   {rmse:.2f} m")
    print(f"\nTest Set:")
    print(f"  {len(test_locs)} unseen locations: {sorted(test_locs)}")
    print(f"  {len(X_test)} samples total")
    print(f"\nInterpretation:")
    print(f"  This model was tested on {len(test_locs)} locations it has NEVER seen.")
    print(f"  Test locations: {sorted(test_locs)}")
    print(f"  Training range: {min(train_locs)}-{max(train_locs)}m")

    # Check for extrapolation
    if any(loc < min(train_locs) or loc > max(train_locs) for loc in test_locs):
        print(f"\n⚠ WARNING: Test includes locations OUTSIDE training range!")
        print(f"  Neural networks cannot extrapolate well.")
        print(f"  For best results, ensure test locations are WITHIN training range.")
    else:
        print(f"\n✓ All test locations are within training range (interpolation)")

    print("="*80)
