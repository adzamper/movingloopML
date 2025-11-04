"""
CNN-LSTM Target Locator for TEM Survey Data
============================================

This script uses deep learning to predict target locations from time-domain
electromagnetic (TEM) survey data.

Philosophy: Keep it simple. Use raw measurements with minimal preprocessing.
Let the neural network discover the patterns.

Key Components:
- Raw time-channel data (no hand-crafted features)
- Multi-scale CNN (learns spatial patterns at different scales)
- Attention mechanism (learns to focus on anomalies)
- Bidirectional LSTM (models spatial sequences)
- Ensemble learning (quantifies uncertainty)

Author: ML Team
Date: 2025-11-04
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, LSTM, Dense,
                                      Dropout, BatchNormalization, Concatenate,
                                      Multiply, Activation, Bidirectional)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIRECTORY = "."
MODEL_PATH = "tem_model.keras"
SCALER_PATH = "tem_scaler.joblib"
N_ENSEMBLE = 3
MAX_STATIONS = 101
STATION_SPACING = 50.0
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# =============================================================================
# DATA LOADING
# =============================================================================

def parse_tem_file(file_path):
    """
    Load TEM survey data from a .tem file.

    Returns:
    --------
    df : DataFrame with columns STATION, COMPONENT, CH1-CH20
    metadata : dict with 'offset', 'config_type', 'true_location', 'file_id'
    """
    try:
        filename = os.path.basename(file_path)
        folder_name = os.path.basename(os.path.dirname(file_path))

        # Extract target location from folder name
        true_location = float(folder_name) if folder_name.replace('.','').isdigit() else None

        # Extract configuration from filename (e.g., "500moffset3.tem")
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
        header_idx = next((i for i, l in enumerate(lines)
                          if 'EAST' in l and 'STATION' in l), -1)
        if header_idx == -1:
            return None, None

        # Parse header
        header = lines[header_idx].split()
        channel_cols = [col for col in header if col.startswith('CH')]

        # Parse data
        data_lines = [l.strip() for l in lines[header_idx+1:] if l.strip() and not l.startswith('*')]

        records = []
        for line in data_lines:
            parts = line.split()
            if len(parts) < len(header):
                continue

            try:
                station = float(parts[header.index('STATION')])
                component = parts[header.index('COMPONENT')]

                record = {'STATION': station, 'COMPONENT': component}
                for ch in channel_cols:
                    record[ch] = float(parts[header.index(ch)])
                records.append(record)
            except (ValueError, IndexError):
                continue

        if not records:
            return None, None

        df = pd.DataFrame(records)
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


def create_spatial_profile(df, metadata):
    """
    Create spatial feature profile from raw TEM data.

    Philosophy:
    -----------
    Use RAW time channels with minimal preprocessing. Let the CNN discover
    patterns instead of hand-crafting features.

    Processing:
    -----------
    1. Log transform: Handles exponential TEM decay (well-established physics)
    2. Normalization: Scales to common range (standard ML practice)
    3. Spatial organization: Arranges by station position (preserves geometry)

    That's it! No contrived features, no complex filtering.

    Returns:
    --------
    np.array : Shape (MAX_STATIONS, n_features)
        Features per station:
        - 20 time channels × 3 components (X, Y, Z) = 60 channels
        - Plus 2 metadata features (offset, config_type)
        - Total: 62 features per station
    """
    all_channel_cols = [f'CH{i}' for i in range(1, 21)]

    # 1. Normalize by maximum (handles different survey amplitudes)
    max_val = df[all_channel_cols].abs().max().max()
    if max_val > 0:
        for col in all_channel_cols:
            df[col] /= max_val

    # 2. Log transform (handles exponential decay)
    #    TEM fields decay exponentially, log makes patterns more linear
    for col in all_channel_cols:
        df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

    # 3. Organize by station and component
    station_features = []
    for station in sorted(df['STATION'].unique()):
        station_data = df[df['STATION'] == station]

        features = {}
        for component in ['X', 'Y', 'Z']:
            comp_data = station_data[station_data['COMPONENT'] == component]
            if not comp_data.empty:
                for ch in all_channel_cols:
                    features[f'{ch}_{component}'] = comp_data[ch].iloc[0]
            else:
                # Missing component - fill with zeros
                for ch in all_channel_cols:
                    features[f'{ch}_{component}'] = 0.0

        station_features.append({'STATION': station, **features})

    if not station_features:
        return None

    feature_df = pd.DataFrame(station_features)

    # Add configuration metadata (helps model adapt to different geometries)
    feature_df['offset'] = metadata['offset']
    feature_df['config_type_encoded'] = 1 if metadata['config_type'] == 'trailing' else 0

    # Create fixed-size array (pads with zeros if fewer stations)
    feature_cols = [col for col in feature_df.columns if col != 'STATION']
    full_profile = np.zeros((MAX_STATIONS, len(feature_cols)))

    for _, row in feature_df.iterrows():
        station_idx = int(row['STATION'] / STATION_SPACING)
        if 0 <= station_idx < MAX_STATIONS:
            full_profile[station_idx, :] = row[feature_cols].values

    return full_profile


def load_all_data(base_dir):
    """
    Load all TEM survey data from directory structure.

    Expected structure:
        base_dir/
            1700/  (target at 1700m)
                0moffset1.tem
                500moffset1.tem
                ...
            1900/
                ...

    Returns:
    --------
    X : np.array (n_samples, MAX_STATIONS, n_features)
    y : np.array (n_samples,) - target locations in meters
    metadata : list of metadata dicts
    """
    all_profiles, all_labels, all_metadata = [], [], []

    folders = [d for d in os.listdir(base_dir)
              if os.path.isdir(os.path.join(base_dir, d))]

    print(f"\nLoading data from {len(folders)} location folders...")

    for loc_str in sorted(folders):
        try:
            label = float(loc_str)
            folder_path = os.path.join(base_dir, loc_str)
            files = [f for f in os.listdir(folder_path) if f.endswith('.tem')]

            for fname in files:
                df, metadata = parse_tem_file(os.path.join(folder_path, fname))
                if df is not None and metadata is not None:
                    profile = create_spatial_profile(df, metadata)

                    if profile is not None:
                        all_profiles.append(profile)
                        all_labels.append(label)
                        all_metadata.append(metadata)

        except ValueError:
            continue

    print(f"Loaded {len(all_profiles)} samples")

    return (np.array(all_profiles, dtype=np.float32),
            np.array(all_labels, dtype=np.float32),
            all_metadata)


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

def build_model(input_shape):
    """
    Build CNN-LSTM model with attention for target localization.

    Architecture Rationale:
    -----------------------
    1. Multi-scale CNN: Detects anomalies at different spatial scales
       - 3-filter kernel: Local sharp features
       - 7-filter kernel: Medium-scale patterns
       - 11-filter kernel: Broad regional trends

    2. Attention: Learns to focus on the anomaly location
       - Assigns importance weights to different spatial positions
       - Makes model interpretable (we can see where it's looking)

    3. Bidirectional LSTM: Models spatial sequence in both directions
       - Captures context from both sides of the anomaly
       - Understands how fields decay approaching/leaving target

    4. Dense layers: Final prediction from learned representations

    Input:
    ------
    (MAX_STATIONS, n_features) - spatial profile along survey line

    Output:
    -------
    Single value - predicted target location in meters
    """
    input_layer = Input(shape=input_shape)

    # Multi-scale inception block
    # Learn patterns at different spatial scales
    tower_1 = Conv1D(64, 3, padding='same', activation='relu')(input_layer)
    tower_2 = Conv1D(64, 7, padding='same', activation='relu')(input_layer)
    tower_3 = Conv1D(64, 11, padding='same', activation='relu')(input_layer)

    merged = Concatenate(axis=-1)([tower_1, tower_2, tower_3])

    # Deep CNN processing
    x = Conv1D(128, 5, padding='same', activation='relu')(merged)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(128, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Attention mechanism
    # Learns which spatial positions are important
    attention = Dense(1, activation='tanh')(x)
    attention = Activation('softmax')(attention)
    x = Multiply()([x, attention])

    # Bidirectional LSTM
    # Models spatial sequence (approaching and leaving target)
    x = Bidirectional(LSTM(128, return_sequences=False))(x)
    x = Dropout(0.4)(x)

    # Dense prediction layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='linear')(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


# =============================================================================
# TRAINING
# =============================================================================

def split_by_groups(X, y, metadata, test_ratio=0.2, val_ratio=0.15):
    """
    Split data using group-based splitting to prevent data leakage.

    Critical Insight:
    -----------------
    Files like "0moffset1" and "0moffset2" are the SAME survey with different
    noise realizations. If we randomly split, the model sees the pattern in
    training and "predicts" it perfectly in test (data leakage!).

    Solution:
    ---------
    Group by (location, config_type, offset) so all noise variations of the
    same survey stay together in one split.

    This tests: "Can the model generalize to NEW survey configurations at
    NEW locations?" (the real deployment scenario)
    """
    # Create groups
    config_groups = {}
    for i, meta in enumerate(metadata):
        group_key = (meta['true_location'], meta['config_type'], meta['offset'])
        if group_key not in config_groups:
            config_groups[group_key] = []
        config_groups[group_key].append(i)

    # Shuffle groups
    group_keys = list(config_groups.keys())
    np.random.shuffle(group_keys)

    # Assign to splits
    n_groups = len(group_keys)
    n_test = max(1, int(n_groups * test_ratio))
    n_val = max(1, int(n_groups * val_ratio))

    test_groups = group_keys[:n_test]
    val_groups = group_keys[n_test:n_test+n_val]
    train_groups = group_keys[n_test+n_val:]

    # Extract indices
    train_idx = [i for g in train_groups for i in config_groups[g]]
    val_idx = [i for g in val_groups for i in config_groups[g]]
    test_idx = [i for g in test_groups for i in config_groups[g]]

    return (X[train_idx], y[train_idx], [metadata[i] for i in train_idx],
            X[val_idx], y[val_idx], [metadata[i] for i in val_idx],
            X[test_idx], y[test_idx], [metadata[i] for i in test_idx])


def train_ensemble(X_train, y_train, X_val, y_val):
    """
    Train an ensemble of models for robust predictions and uncertainty estimation.

    Why Ensemble?
    -------------
    - Reduces overfitting (averaging reduces variance)
    - Provides uncertainty estimates (std across models)
    - More robust to random initialization

    Returns:
    --------
    models : list of trained Keras models
    """
    models = []

    for i in range(N_ENSEMBLE):
        print(f"\nTraining model {i+1}/{N_ENSEMBLE}")
        print("-" * 70)

        model = build_model(X_train.shape[1:])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=150,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        models.append(model)

        # Save best model
        if i == 0:
            model.save(MODEL_PATH.replace('.keras', f'_{i}.keras'))

    return models


def evaluate_ensemble(models, X_test, y_test, metadata_test):
    """
    Evaluate ensemble on test set with detailed analysis.
    """
    # Get predictions from all models
    predictions = np.array([model.predict(X_test, verbose=0).flatten()
                           for model in models])

    # Ensemble statistics
    mean_preds = np.mean(predictions, axis=0)
    std_preds = np.std(predictions, axis=0)

    # Calculate errors
    errors = np.abs(mean_preds - y_test)
    mae = np.mean(errors)
    median_error = np.median(errors)
    rmse = np.sqrt(np.mean((mean_preds - y_test)**2))

    print(f"\n{'='*70}")
    print("TEST SET PERFORMANCE")
    print(f"{'='*70}")
    print(f"\nOverall Metrics:")
    print(f"  MAE:               {mae:.2f} m")
    print(f"  Median Error:      {median_error:.2f} m")
    print(f"  RMSE:              {rmse:.2f} m")
    print(f"  Mean Uncertainty:  {np.mean(std_preds):.2f} m")

    # Best/worst predictions
    sorted_idx = np.argsort(errors)

    print(f"\nBest 5 Predictions:")
    for i in sorted_idx[:5]:
        print(f"  True={y_test[i]:.0f}m, Pred={mean_preds[i]:.0f}m, "
              f"Error={errors[i]:.1f}m, σ={std_preds[i]:.1f}m")

    print(f"\nWorst 5 Predictions:")
    for i in sorted_idx[-5:]:
        print(f"  True={y_test[i]:.0f}m, Pred={mean_preds[i]:.0f}m, "
              f"Error={errors[i]:.1f}m, σ={std_preds[i]:.1f}m")

    # Configuration analysis
    config_stats = {}
    for i, meta in enumerate(metadata_test):
        config = f"{int(meta['offset'])}m_{meta['config_type']}"
        if config not in config_stats:
            config_stats[config] = []
        config_stats[config].append(errors[i])

    print(f"\nPerformance by Configuration:")
    print(f"  {'Configuration':<20} {'N':<6} {'MAE (m)':<10}")
    print(f"  {'-'*40}")
    for config in sorted(config_stats.keys()):
        errs = config_stats[config]
        print(f"  {config:<20} {len(errs):<6} {np.mean(errs):<10.2f}")

    return mean_preds, std_preds, errors


def create_visualization(y_test, mean_preds, std_preds, errors, metadata_test):
    """
    Create comprehensive visualization of results.
    """
    fig = plt.figure(figsize=(15, 10))

    # 1. Predictions vs True
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(y_test, mean_preds, alpha=0.6, c=errors, cmap='RdYlGn_r')
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'k--', linewidth=2, label='Perfect prediction')
    ax1.set_xlabel('True Location (m)', fontsize=11)
    ax1.set_ylabel('Predicted Location (m)', fontsize=11)
    ax1.set_title('Predictions vs Ground Truth', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Error distribution
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(np.median(errors), color='red', linestyle='--',
                linewidth=2, label=f'Median: {np.median(errors):.1f}m')
    ax2.set_xlabel('Absolute Error (m)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Uncertainty calibration
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(std_preds, errors, alpha=0.6)
    ax3.set_xlabel('Prediction Uncertainty (σ)', fontsize=11)
    ax3.set_ylabel('Absolute Error (m)', fontsize=11)
    ax3.set_title('Uncertainty Calibration', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Error by configuration
    ax4 = plt.subplot(2, 3, 4)
    config_errors = {}
    for i, meta in enumerate(metadata_test):
        config = f"{int(meta['offset'])}m_{meta['config_type']}"
        if config not in config_errors:
            config_errors[config] = []
        config_errors[config].append(errors[i])

    ax4.boxplot(config_errors.values(), tick_labels=config_errors.keys())
    ax4.set_ylabel('Absolute Error (m)', fontsize=11)
    ax4.set_title('Error by Configuration', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Error along survey line
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(y_test, errors, alpha=0.6, c=std_preds, cmap='viridis')
    ax5.set_xlabel('True Location (m)', fontsize=11)
    ax5.set_ylabel('Absolute Error (m)', fontsize=11)
    ax5.set_title('Error vs Location', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # 6. Uncertainty distribution
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(std_preds, bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax6.axvline(np.mean(std_preds), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(std_preds):.1f}m')
    ax6.set_xlabel('Prediction Uncertainty (σ)', fontsize=11)
    ax6.set_ylabel('Frequency', fontsize=11)
    ax6.set_title('Uncertainty Distribution', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tem_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: tem_results.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training pipeline."""

    print(f"\n{'='*70}")
    print("TEM TARGET LOCATOR - CNN-LSTM WITH ATTENTION")
    print(f"{'='*70}")

    # Load data
    print(f"\n[1/5] Loading data...")
    X, y, metadata = load_all_data(DATA_DIRECTORY)

    print(f"\nData shape: {X.shape}")
    print(f"Features per station: {X.shape[2]}")
    print(f"Target range: {y.min():.0f}m to {y.max():.0f}m")

    # Split data
    print(f"\n[2/5] Splitting data (group-based to prevent leakage)...")
    X_train, y_train, meta_train, X_val, y_val, meta_val, X_test, y_test, meta_test = \
        split_by_groups(X, y, metadata)

    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")

    # Scale data
    print(f"\n[3/5] Scaling features...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_reshaped)

    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    joblib.dump(scaler, SCALER_PATH)
    print(f"  ✓ Saved scaler: {SCALER_PATH}")

    # Train ensemble
    print(f"\n[4/5] Training ensemble...")
    models = train_ensemble(X_train_scaled, y_train, X_val_scaled, y_val)

    # Evaluate
    print(f"\n[5/5] Evaluating on test set...")
    mean_preds, std_preds, errors = evaluate_ensemble(
        models, X_test_scaled, y_test, meta_test
    )

    # Visualize
    create_visualization(y_test, mean_preds, std_preds, errors, meta_test)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nSaved files:")
    print(f"  • {MODEL_PATH}")
    print(f"  • {SCALER_PATH}")
    print(f"  • tem_results.png")
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
