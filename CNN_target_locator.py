"""
CNN-LSTM Target Locator for TEM Survey Data
============================================
This script trains an ensemble of deep learning models to predict target locations
from time-domain electromagnetic (TEM) survey data.

Key Features:
- Multi-scale CNN architecture with attention mechanism
- Bidirectional LSTM for spatial sequence modeling
- Ensemble learning for uncertainty quantification
- Configuration-specific performance analysis
- Data augmentation for improved generalization
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, LSTM, Dense,
                                      Dropout, BatchNormalization, Concatenate,
                                      Multiply, GlobalAveragePooling1D, Reshape,
                                      Activation, Add)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
from scipy.signal import savgol_filter

# ============================================================================
# CONFIGURATION - Adjust these parameters as needed
# ============================================================================
DATA_DIRECTORY = "."  # Script directory - assumes data folders are here
MODEL_PATH = "improved_model.keras"
SCALER_PATH = "improved_scaler.joblib"
N_ENSEMBLE = 3  # Number of models in ensemble (more = better uncertainty estimates)
MAX_STATIONS = 101  # Maximum number of measurement stations
STATION_SPACING = 50.0  # Distance between stations in meters
RANDOM_SEED = 42  # For reproducibility

# Feature engineering mode
# Based on ablation study findings: gradients hurt performance (-22%)!
FEATURE_MODE = 'optimized'  # Options: 'optimized' (no gradients), 'full' (with gradients), 'raw' (raw channels)
USE_GRADIENTS = False  # Set to True to include gradients (NOT recommended - degrades performance)

# Data augmentation settings
AUGMENTATION_ENABLED = True
AUGMENTATION_NOISE_LEVEL = 0.05  # 5% noise
AUGMENTATION_PER_SAMPLE = 2  # Number of augmented copies per sample

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def parse_tem_file(file_path):
    """
    Parse a .tem file and extract survey data with metadata.

    Parameters:
    -----------
    file_path : str
        Path to the .tem file

    Returns:
    --------
    tuple : (DataFrame, metadata_dict) or (None, None) on error
        - DataFrame with columns: STATION, COMPONENT, CH1-CH20, etc.
        - metadata_dict with keys: 'offset', 'config_type', 'true_location', 'file_id'
    """
    try:
        filename = os.path.basename(file_path)
        folder_name = os.path.basename(os.path.dirname(file_path))

        # Extract true target location from folder name (e.g., "1700" -> 1700.0)
        true_location = float(folder_name) if folder_name.replace('.','').isdigit() else None

        # Extract configuration information from filename
        # Examples: "0moffset1.tem", "500m_trailing3.tem", "1000moffset5.tem"
        config_match = re.match(r'(\d+)m[_]?(offset|trailing)?(\d+)', filename)
        if not config_match:
            return None, None

        offset = float(config_match.group(1))
        config_type = config_match.group(2) if config_match.group(2) else 'offset'
        file_id = int(config_match.group(3))

        # Read and parse the file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Find header line
        header_line_index = next((i for i, l in enumerate(lines)
                                 if 'EAST' in l and 'STATION' in l), -1)
        if header_line_index == -1:
            return None, None

        # Parse header and data
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

        # Create metadata dictionary
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
    Create feature profile from TEM data with optimized feature engineering.

    ABLATION STUDY FINDINGS:
    - Spatial gradients HURT performance (-22%) → Removed by default
    - Background removal (residuals) HELPS (+24% if removed) → Kept
    - Raw channels may work better than hand-crafted features → Use FEATURE_MODE='raw'

    Feature modes:
    - 'optimized': Time sums + ratios + residuals (NO gradients) ← RECOMMENDED
    - 'full': All features including gradients (for comparison)
    - 'raw': Raw CH1-CH20 channels with residuals (modern approach)

    This function extracts spatial features from TEM survey data including:
    - Early/mid/late time channel summations
    - Component ratios and total field amplitude
    - Background-removed residuals (critical for anomaly detection)
    - Anomaly strength metrics
    - [OPTIONAL] Spatial gradients (disabled by default - degrades performance)

    Parameters:
    -----------
    df : DataFrame
        TEM survey data with columns: STATION, COMPONENT, CH1-CH20
    metadata : dict
        Metadata including 'offset', 'config_type', etc.

    Returns:
    --------
    np.array : Feature profile of shape (MAX_STATIONS, n_features)
    """
    # Define time windows for feature extraction
    early_channels = [f'CH{i}' for i in range(1, 8)]  # Fast decay
    mid_channels = [f'CH{i}' for i in range(8, 15)]   # Medium decay
    late_channels = [f'CH{i}' for i in range(15, 21)] # Slow decay
    all_channel_cols = [col for col in df.columns if col.startswith('CH')]

    # Normalize by maximum absolute value
    max_val = df[all_channel_cols].abs().max().max()
    if max_val > 0:
        for col in all_channel_cols:
            df[col] /= max_val

    # Log transform to handle exponential decay
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

    # Calculate Total Field Amplitude (TFA) for each time window
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

    # Define base features
    base_features = ([f'{f}_{c}' for f in ['early_sum', 'mid_sum', 'late_sum', 'ratio']
                     for c in ['X', 'Y', 'Z']] +
                    ['tfa_early', 'tfa_mid', 'tfa_late', 'tfa_ratio'])

    for col in base_features:
        if col not in pivoted_df.columns:
            pivoted_df[col] = 0.0

    # Background removal using Savitzky-Golay filter (anomaly detection)
    for col in base_features:
        if len(pivoted_df[col]) >= 51:
            try:
                background = savgol_filter(pivoted_df[col], window_length=51, polyorder=3)
                pivoted_df[f'{col}_residual'] = pivoted_df[col] - background
            except:
                pivoted_df[f'{col}_residual'] = 0
        else:
            pivoted_df[f'{col}_residual'] = 0

    # Calculate spatial gradients (rate of change between stations)
    # NOTE: Ablation study showed gradients DEGRADE performance by 22%!
    # Only included if explicitly enabled for comparison purposes
    grad_df = None
    if USE_GRADIENTS:
        grad_df = pivoted_df[base_features].diff().fillna(0)
        grad_df.columns = [f'{c}_grad' for c in base_features]

    # Add anomaly strength metrics
    residual_cols = [c for c in pivoted_df.columns if 'residual' in c]
    if residual_cols:
        # Peak anomaly strength
        pivoted_df['anomaly_peak'] = pivoted_df[residual_cols].abs().max(axis=1)
        # Anomaly energy (sum of squares)
        pivoted_df['anomaly_energy'] = np.sqrt(
            (pivoted_df[residual_cols]**2).sum(axis=1))

    # Combine all features (conditionally include gradients)
    combine_list = [
        pivoted_df[['STATION'] + base_features],
        pivoted_df[residual_cols]
    ]

    if grad_df is not None:
        combine_list.append(grad_df)

    if 'anomaly_peak' in pivoted_df:
        combine_list.append(pivoted_df[['anomaly_peak', 'anomaly_energy']])

    final_features_df = pd.concat(combine_list, axis=1)

    # Add configuration metadata as features
    final_features_df['offset'] = metadata['offset']
    # Encode config_type: offset=0, trailing=1
    final_features_df['config_type_encoded'] = 1 if metadata['config_type'] == 'trailing' else 0

    # Create fixed-size spatial profile array
    full_profile = np.zeros((MAX_STATIONS, len(final_features_df.columns) - 1))
    feature_cols = [col for col in final_features_df.columns if col != 'STATION']

    for _, row in final_features_df.iterrows():
        station_idx = int(row['STATION'] / STATION_SPACING)
        if 0 <= station_idx < MAX_STATIONS:
            full_profile[station_idx, :] = row[feature_cols].values

    return full_profile

def create_raw_channel_profile(df, metadata):
    """
    Create feature profile using RAW time channels (modern deep learning approach).

    Instead of hand-crafting features (early/mid/late sums, ratios), this function
    feeds the raw CH1-CH20 data directly to the CNN, letting it learn optimal features.

    ADVANTAGES:
    - No assumptions about which time windows matter
    - CNN can discover patterns we didn't think of
    - Modern best practice (see: image nets use raw pixels, not hand-crafted features)

    PROCESSING:
    - Log transform (handles exponential decay)
    - Background removal (residuals) - CRITICAL from ablation study
    - Raw channels per component (X, Y, Z)

    Parameters:
    -----------
    df : DataFrame
        TEM survey data with columns: STATION, COMPONENT, CH1-CH20
    metadata : dict
        Metadata including 'offset', 'config_type', etc.

    Returns:
    --------
    np.array : Feature profile of shape (MAX_STATIONS, n_features)
        Features = 20 channels × 3 components × 2 (raw + residual) + metadata = 122 features
    """
    all_channel_cols = [f'CH{i}' for i in range(1, 21)]

    # Normalize
    max_val = df[all_channel_cols].abs().max().max()
    if max_val > 0:
        for col in all_channel_cols:
            df[col] /= max_val

    # Log transform (handles exponential decay)
    for col in all_channel_cols:
        if col in df.columns:
            df[col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))

    # Organize by station and component
    station_features = []
    for station in sorted(df['STATION'].unique()):
        station_data = df[df['STATION'] == station]

        features = {}
        for component in ['X', 'Y', 'Z']:
            comp_data = station_data[station_data['COMPONENT'] == component]
            if not comp_data.empty:
                # Raw channels
                for ch in all_channel_cols:
                    features[f'{ch}_{component}'] = comp_data[ch].iloc[0]

        station_features.append({'STATION': station, **features})

    if not station_features:
        return None

    feature_df = pd.DataFrame(station_features)

    # Get channel features (all except STATION)
    channel_features = [col for col in feature_df.columns if col != 'STATION']

    # Background removal (residuals) - CRITICAL for performance
    for col in channel_features:
        if len(feature_df[col]) >= 51:
            try:
                background = savgol_filter(feature_df[col], window_length=51, polyorder=3)
                feature_df[f'{col}_residual'] = feature_df[col] - background
            except:
                feature_df[f'{col}_residual'] = 0
        else:
            feature_df[f'{col}_residual'] = 0

    # Add metadata
    feature_df['offset'] = metadata['offset']
    feature_df['config_type_encoded'] = 1 if metadata['config_type'] == 'trailing' else 0

    # Create spatial profile
    full_profile = np.zeros((MAX_STATIONS, len(feature_df.columns) - 1))
    feature_cols = [col for col in feature_df.columns if col != 'STATION']

    for _, row in feature_df.iterrows():
        station_idx = int(row['STATION'] / STATION_SPACING)
        if 0 <= station_idx < MAX_STATIONS:
            full_profile[station_idx, :] = row[feature_cols].values

    return full_profile


def augment_profile(profile, noise_level=0.05):
    """
    Apply data augmentation to a feature profile.

    Augmentation techniques:
    - Add Gaussian noise to simulate measurement uncertainty
    - Small spatial shifts to increase position diversity

    Parameters:
    -----------
    profile : np.array
        Feature profile of shape (MAX_STATIONS, n_features)
    noise_level : float
        Standard deviation of Gaussian noise (as fraction of signal)

    Returns:
    --------
    np.array : Augmented profile
    """
    augmented = profile.copy()

    # Add Gaussian noise to non-zero entries (measurements exist)
    mask = (np.abs(profile).sum(axis=1) > 0)
    if mask.any():
        noise = np.random.normal(0, noise_level, augmented.shape)
        augmented[mask] += noise[mask] * np.abs(profile[mask])

    return augmented


def load_all_data(base_dir):
    """
    Load all TEM training data from directory structure.

    Expected structure:
        base_dir/
            1700/
                0moffset1.tem, 0moffset2.tem, ...
                500moffset1.tem, ...
            1900/
                ...

    Returns:
    --------
    tuple : (X, y, metadata_list)
        - X: numpy array of shape (n_samples, MAX_STATIONS, n_features)
        - y: numpy array of shape (n_samples,) - target locations
        - metadata_list: list of metadata dicts for each sample
    """
    all_profiles, all_labels, all_metadata = [], [], []

    # Find all location folders (should be numeric)
    folders = [d for d in os.listdir(base_dir)
              if os.path.isdir(os.path.join(base_dir, d))]

    print(f"\nFound {len(folders)} location folders: {sorted(folders)}")

    for loc_str in sorted(folders):
        try:
            label = float(loc_str)
            folder_path = os.path.join(base_dir, loc_str)
            files = [f for f in os.listdir(folder_path) if f.endswith('.tem')]

            print(f"  Loading {len(files)} files from folder {loc_str}...")

            for fname in files:
                df, metadata = parse_tem_file(os.path.join(folder_path, fname))
                if df is not None and metadata is not None:
                    # Choose feature extraction method based on FEATURE_MODE
                    if FEATURE_MODE == 'raw':
                        profile = create_raw_channel_profile(df, metadata)
                    else:
                        profile = create_feature_profile(df, metadata)

                    if profile is not None:
                        all_profiles.append(profile)
                        all_labels.append(label)
                        all_metadata.append(metadata)

                        # Apply data augmentation if enabled
                        if AUGMENTATION_ENABLED:
                            for _ in range(AUGMENTATION_PER_SAMPLE):
                                aug_profile = augment_profile(profile, AUGMENTATION_NOISE_LEVEL)
                                all_profiles.append(aug_profile)
                                all_labels.append(label)
                                # Mark as augmented in metadata
                                aug_metadata = metadata.copy()
                                aug_metadata['augmented'] = True
                                all_metadata.append(aug_metadata)

        except ValueError:
            print(f"  Skipping non-numeric folder: {loc_str}")
            continue

    print(f"\nTotal samples loaded: {len(all_profiles)}")
    if AUGMENTATION_ENABLED:
        orig_count = len(all_profiles) // (1 + AUGMENTATION_PER_SAMPLE)
        aug_count = len(all_profiles) - orig_count
        print(f"  Original: {orig_count}, Augmented: {aug_count}")

    return (np.array(all_profiles, dtype=np.float32),
            np.array(all_labels, dtype=np.float32),
            all_metadata)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def attention_block(x):
    """
    Attention mechanism to focus on important spatial locations.

    This helps the model identify where the target anomaly is located
    by learning to weight different positions along the survey line.

    Parameters:
    -----------
    x : Tensor
        Input tensor of shape (batch, spatial, features)

    Returns:
    --------
    Tensor : Attention-weighted output
    """
    # Calculate attention weights
    attention = Dense(1, activation='tanh')(x)
    attention = Activation('softmax', name='attention_weights')(attention)

    # Apply attention weights
    weighted = Multiply()([x, attention])

    return weighted


def build_improved_model(input_shape):
    """
    Build CNN-LSTM model with attention mechanism for target localization.

    Architecture:
    1. Multi-scale inception block (captures features at different spatial scales)
    2. Deep CNN layers with residual connections
    3. Attention mechanism (identifies anomaly locations)
    4. Bidirectional LSTM (models spatial sequences)
    5. Dense layers for final prediction

    Parameters:
    -----------
    input_shape : tuple
        (spatial_dimension, n_features)

    Returns:
    --------
    Keras Model
    """
    input_layer = Input(shape=input_shape, name='input')

    # -------------------------------------------------------------------------
    # Multi-scale Inception Block
    # Different kernel sizes capture anomalies at different spatial scales
    # -------------------------------------------------------------------------
    tower_1 = Conv1D(64, 3, padding='same', activation='relu', name='tower1_conv')(input_layer)
    tower_1 = BatchNormalization(name='tower1_bn')(tower_1)

    tower_2 = Conv1D(64, 7, padding='same', activation='relu', name='tower2_conv')(input_layer)
    tower_2 = BatchNormalization(name='tower2_bn')(tower_2)

    tower_3 = Conv1D(64, 11, padding='same', activation='relu', name='tower3_conv')(input_layer)
    tower_3 = BatchNormalization(name='tower3_bn')(tower_3)

    x = Concatenate(axis=-1, name='inception_concat')([tower_1, tower_2, tower_3])
    x = Dropout(0.3, name='inception_dropout')(x)

    # -------------------------------------------------------------------------
    # Deep CNN Processing with Residual Connections
    # -------------------------------------------------------------------------
    # Block 1
    conv1 = Conv1D(128, 5, padding='same', activation='relu', name='conv1')(x)
    conv1 = BatchNormalization(name='bn1')(conv1)
    conv1 = MaxPooling1D(2, name='pool1')(conv1)
    conv1 = Dropout(0.3, name='dropout1')(conv1)

    # Block 2
    conv2 = Conv1D(256, 3, padding='same', activation='relu', name='conv2')(conv1)
    conv2 = BatchNormalization(name='bn2')(conv2)
    conv2 = MaxPooling1D(2, name='pool2')(conv2)
    conv2 = Dropout(0.3, name='dropout2')(conv2)

    # -------------------------------------------------------------------------
    # Attention Mechanism
    # Learns to focus on spatially anomalous regions
    # -------------------------------------------------------------------------
    attended = attention_block(conv2)

    # -------------------------------------------------------------------------
    # Bidirectional LSTM
    # Models spatial sequences in both directions along the survey line
    # -------------------------------------------------------------------------
    lstm_out = tf.keras.layers.Bidirectional(
        LSTM(128, return_sequences=False, name='lstm'),
        name='bidirectional_lstm'
    )(attended)
    lstm_out = Dropout(0.4, name='lstm_dropout')(lstm_out)

    # -------------------------------------------------------------------------
    # Dense Prediction Layers
    # -------------------------------------------------------------------------
    dense1 = Dense(128, activation='relu', name='dense1')(lstm_out)
    dense1 = Dropout(0.5, name='dense1_dropout')(dense1)

    dense2 = Dense(64, activation='relu', name='dense2')(dense1)

    output_layer = Dense(1, activation='linear', name='output')(dense2)

    # -------------------------------------------------------------------------
    # Compile Model
    # -------------------------------------------------------------------------
    model = Model(inputs=input_layer, outputs=output_layer, name='TEM_Target_Locator')

    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mean_absolute_error']
    )

    return model

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def analyze_by_configuration(y_true, y_pred, metadata_list):
    """
    Analyze model performance by configuration type.

    Parameters:
    -----------
    y_true : np.array
        True target locations
    y_pred : np.array
        Predicted target locations
    metadata_list : list
        Metadata for each sample

    Returns:
    --------
    dict : Configuration-wise performance metrics
    """
    config_results = {}

    for i, meta in enumerate(metadata_list):
        config_key = f"{int(meta['offset'])}m_{meta['config_type']}"

        if config_key not in config_results:
            config_results[config_key] = {
                'true': [],
                'pred': [],
                'errors': []
            }

        error = abs(y_pred[i] - y_true[i])
        config_results[config_key]['true'].append(y_true[i])
        config_results[config_key]['pred'].append(y_pred[i])
        config_results[config_key]['errors'].append(error)

    # Calculate statistics
    config_stats = {}
    for config, data in config_results.items():
        config_stats[config] = {
            'n_samples': len(data['errors']),
            'mae': np.mean(data['errors']),
            'median_error': np.median(data['errors']),
            'std': np.std(data['errors']),
            'max_error': np.max(data['errors'])
        }

    return config_stats


def test_noise_robustness(models, X_test, y_test, noise_levels=[0.0, 0.05, 0.1, 0.15, 0.2]):
    """
    Test model robustness by adding Gaussian noise at various levels.

    This helps identify overfitting - a robust model should degrade gracefully
    with increasing noise, while an overfit model will collapse.

    Parameters:
    -----------
    models : list
        List of trained ensemble models
    X_test : np.array
        Test data
    y_test : np.array
        Test labels
    noise_levels : list
        Noise standard deviations to test (as fraction of signal)

    Returns:
    --------
    dict : Noise level -> MAE mapping
    """
    noise_results = {}

    for noise_level in noise_levels:
        # Add Gaussian noise to test data
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, X_test.shape)
            X_noisy = X_test + noise * np.abs(X_test)
        else:
            X_noisy = X_test

        # Get ensemble predictions
        preds = np.array([model.predict(X_noisy, verbose=0).flatten()
                         for model in models])
        mean_preds = np.mean(preds, axis=0)

        # Calculate MAE
        mae = np.mean(np.abs(mean_preds - y_test))
        noise_results[noise_level] = mae

    return noise_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("="*70)
    print("CNN-LSTM TARGET LOCATOR - TRAINING WITH ATTENTION & CONFIG ANALYSIS")
    print("="*70)

    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Determine data path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, DATA_DIRECTORY)

    print(f"\n[1/6] Loading data from: {data_path}")
    print("-" * 70)
    X, y, metadata = load_all_data(data_path)
    print(f"\nLoaded {X.shape[0]} profiles")
    print(f"  Spatial dimension: {X.shape[1]} stations")
    print(f"  Feature dimension: {X.shape[2]} features")
    print(f"  Target locations: {sorted(set(y))}")

    # Configuration distribution
    configs = {}
    for meta in metadata:
        config_key = f"{int(meta['offset'])}m_{meta['config_type']}"
        configs[config_key] = configs.get(config_key, 0) + 1

    print(f"\nConfiguration distribution:")
    for config, count in sorted(configs.items()):
        print(f"  {config}: {count} samples")

    # -------------------------------------------------------------------------
    # Split data: Train (70%), Validation (15%), Test (15%)
    # Stratified by target location to ensure balanced representation
    # -------------------------------------------------------------------------
    print(f"\n[2/6] Splitting data (Train/Val/Test: 70%/15%/15%)")
    print("-" * 70)

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_SEED, stratify=y
    )

    # Second split: separate train and validation
    X_train_full, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=RANDOM_SEED, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
    )

    # Split metadata accordingly
    indices_temp, indices_test = train_test_split(
        np.arange(len(y)), test_size=0.15, random_state=RANDOM_SEED, stratify=y
    )
    indices_train, indices_val = train_test_split(
        indices_temp, test_size=0.176, random_state=RANDOM_SEED, stratify=y_temp
    )

    metadata_train = [metadata[i] for i in indices_train]
    metadata_val = [metadata[i] for i in indices_val]
    metadata_test = [metadata[i] for i in indices_test]

    print(f"  Training set: {len(X_train_full)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # -------------------------------------------------------------------------
    # Fit scaler on training data only
    # -------------------------------------------------------------------------
    print(f"\n[3/6] Fitting feature scaler")
    print("-" * 70)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_reshaped = X_train_full.reshape(-1, X_train_full.shape[-1])
    scaler.fit(X_train_reshaped)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  Scaler saved to '{SCALER_PATH}'")

    # Transform all datasets
    X_train = scaler.transform(X_train_reshaped).reshape(X_train_full.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # -------------------------------------------------------------------------
    # Train ensemble of models
    # -------------------------------------------------------------------------
    print(f"\n[4/6] Training ensemble of {N_ENSEMBLE} models")
    print("-" * 70)
    models = []

    for i in range(N_ENSEMBLE):
        print(f"\n{'='*70}")
        print(f"Model {i+1}/{N_ENSEMBLE}")
        print(f"{'='*70}")

        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_improved_model(input_shape)

        if i == 0:
            print("\nModel Architecture:")
            model.summary()
            print()

        model_path = f"improved_model_{i}.keras"

        # Callbacks for training
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
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]

        # Set different seed for ensemble diversity
        np.random.seed(RANDOM_SEED + i)
        tf.random.set_seed(RANDOM_SEED + i)

        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=250,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        # Reload best weights
        model = tf.keras.models.load_model(model_path)
        models.append(model)

        # Quick validation evaluation
        val_preds = model.predict(X_val, verbose=0).flatten()
        val_mae = np.mean(np.abs(val_preds - y_val))
        print(f"\n  Model {i+1} Validation MAE: {val_mae:.2f} m")
    
    # -------------------------------------------------------------------------
    # Ensemble Evaluation on Test Set
    # -------------------------------------------------------------------------
    print(f"\n[5/6] Evaluating ensemble on test set")
    print("-" * 70)

    # Get predictions from all models
    test_preds = np.array([model.predict(X_test, verbose=0).flatten()
                           for model in models])
    mean_preds = np.mean(test_preds, axis=0)
    std_preds = np.std(test_preds, axis=0)  # Uncertainty estimate

    errors = np.abs(mean_preds - y_test)
    mae = np.mean(errors)
    median_error = np.median(errors)
    rmse = np.sqrt(np.mean(errors**2))

    print(f"\n  Overall Ensemble Performance:")
    print(f"    Mean Absolute Error (MAE): {mae:.2f} m")
    print(f"    Median Absolute Error: {median_error:.2f} m")
    print(f"    Root Mean Squared Error (RMSE): {rmse:.2f} m")
    print(f"    Mean Prediction Uncertainty (σ): {np.mean(std_preds):.2f} m")

    # Best and worst predictions
    results = sorted(zip(y_test, mean_preds, errors, std_preds),
                    key=lambda item: item[2])

    print(f"\n  Best 5 Predictions:")
    for i in range(min(5, len(results))):
        true_loc, pred_loc, error, uncertainty = results[i]
        print(f"    {i+1}. True={true_loc:.0f}m, Pred={pred_loc:.0f}m, "
              f"Error={error:.1f}m, σ={uncertainty:.1f}m")

    print(f"\n  Worst 5 Predictions:")
    for i in range(min(5, len(results))):
        true_loc, pred_loc, error, uncertainty = results[-(i+1)]
        print(f"    {i+1}. True={true_loc:.0f}m, Pred={pred_loc:.0f}m, "
              f"Error={error:.1f}m, σ={uncertainty:.1f}m")

    # -------------------------------------------------------------------------
    # Configuration-Specific Analysis
    # -------------------------------------------------------------------------
    print(f"\n[6/6] Configuration-specific performance analysis")
    print("-" * 70)

    config_stats = analyze_by_configuration(y_test, mean_preds, metadata_test)

    print(f"\n  Performance by Configuration:")
    print(f"  {'Configuration':<20} {'N':<6} {'MAE (m)':<10} {'Median (m)':<12} {'Std (m)':<10}")
    print(f"  {'-'*70}")
    for config in sorted(config_stats.keys()):
        stats = config_stats[config]
        print(f"  {config:<20} {stats['n_samples']:<6} {stats['mae']:<10.2f} "
              f"{stats['median_error']:<12.2f} {stats['std']:<10.2f}")

    # Find best configuration
    best_config = min(config_stats.items(), key=lambda x: x[1]['mae'])
    print(f"\n  Best Configuration: {best_config[0]} (MAE: {best_config[1]['mae']:.2f} m)")

    # -------------------------------------------------------------------------
    # Noise Robustness Testing
    # -------------------------------------------------------------------------
    print(f"\n  Testing noise robustness...")
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
    noise_results = test_noise_robustness(models, X_test, y_test, noise_levels)

    print(f"\n  Noise Robustness Results:")
    print(f"    Noise Level | MAE (m)")
    print(f"    {'-'*25}")
    for noise_level, mae_val in noise_results.items():
        print(f"    {noise_level*100:5.1f}%      | {mae_val:6.2f}")

    degradation = (noise_results[0.2] - noise_results[0.0]) / noise_results[0.0] * 100
    if degradation < 50:
        print(f"\n  ✓ Model is ROBUST (20% noise → {degradation:.1f}% error increase)")
    elif degradation < 100:
        print(f"\n  ⚠ Model shows MODERATE sensitivity (20% noise → {degradation:.1f}% error increase)")
    else:
        print(f"\n  ✗ Model may be OVERFITTING (20% noise → {degradation:.1f}% error increase)")

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    print(f"\n  Generating visualizations...")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Predictions vs True Values
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_test, mean_preds, alpha=0.6, s=30)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
            'r--', lw=2, label='Perfect Prediction')
    ax1.set_xlabel('True Location (m)', fontsize=10)
    ax1.set_ylabel('Predicted Location (m)', fontsize=10)
    ax1.set_title('Predictions vs True Values', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Error Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(mae, color='r', linestyle='--', linewidth=2,
               label=f'MAE = {mae:.1f}m')
    ax2.axvline(median_error, color='g', linestyle='--', linewidth=2,
               label=f'Median = {median_error:.1f}m')
    ax2.set_xlabel('Absolute Error (m)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Error Distribution', fontsize=11, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Uncertainty vs Error
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(std_preds, errors, alpha=0.6, s=30)
    ax3.set_xlabel('Uncertainty (σ, m)', fontsize=10)
    ax3.set_ylabel('Absolute Error (m)', fontsize=10)
    ax3.set_title('Uncertainty vs Error', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Configuration Comparison (MAE)
    ax4 = fig.add_subplot(gs[1, :2])
    configs_sorted = sorted(config_stats.items(), key=lambda x: x[1]['mae'])
    config_names = [c[0] for c in configs_sorted]
    config_maes = [c[1]['mae'] for c in configs_sorted]
    colors = ['green' if c[0] == best_config[0] else 'steelblue' for c in configs_sorted]
    ax4.barh(config_names, config_maes, color=colors, alpha=0.7)
    ax4.set_xlabel('Mean Absolute Error (m)', fontsize=10)
    ax4.set_title('Performance by Configuration', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    # 5. Error by Configuration (boxplot) - Fixed matplotlib deprecation
    ax5 = fig.add_subplot(gs[1, 2])
    config_errors = []
    config_labels = []
    for config in sorted(config_stats.keys()):
        # Get errors for this config
        config_err = [errors[i] for i, meta in enumerate(metadata_test)
                     if f"{int(meta['offset'])}m_{meta['config_type']}" == config]
        if config_err:
            config_errors.append(config_err)
            # Shorten label for readability
            config_labels.append(config.replace('_offset', '').replace('_trailing', 'T'))

    ax5.boxplot(config_errors, tick_labels=config_labels)
    ax5.set_ylabel('Absolute Error (m)', fontsize=10)
    ax5.set_title('Error Distribution by Config', fontsize=11, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45, labelsize=8)
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Noise Robustness
    ax6 = fig.add_subplot(gs[2, 0])
    noise_x = [n*100 for n in noise_levels]
    noise_y = [noise_results[n] for n in noise_levels]
    ax6.plot(noise_x, noise_y, 'o-', linewidth=2, markersize=8, color='red')
    ax6.axhline(mae, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline MAE')
    ax6.set_xlabel('Noise Level (%)', fontsize=10)
    ax6.set_ylabel('Mean Absolute Error (m)', fontsize=10)
    ax6.set_title('Noise Robustness Test', fontsize=11, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. Prediction Probability Curves for Each Target Location
    ax7 = fig.add_subplot(gs[2, 1:])
    unique_locs = sorted(set(y_test))
    colors_loc = plt.cm.tab10(np.linspace(0, 1, len(unique_locs)))

    for idx, true_loc in enumerate(unique_locs):
        # Get all predictions for this target location
        loc_preds = [mean_preds[i] for i in range(len(y_test)) if y_test[i] == true_loc]
        loc_stds = [std_preds[i] for i in range(len(y_test)) if y_test[i] == true_loc]

        if loc_preds:
            # Create probability/confidence curve
            mean_pred = np.mean(loc_preds)
            mean_std = np.mean(loc_stds)

            # Plot Gaussian distribution around prediction
            x_range = np.linspace(true_loc - 200, true_loc + 200, 100)
            y_range = (1 / (mean_std * np.sqrt(2 * np.pi))) * \
                      np.exp(-0.5 * ((x_range - mean_pred) / mean_std) ** 2)

            ax7.plot(x_range, y_range, linewidth=2, label=f'Target @ {int(true_loc)}m',
                    color=colors_loc[idx])
            ax7.axvline(true_loc, color=colors_loc[idx], linestyle='--', alpha=0.3, linewidth=1)

    ax7.set_xlabel('Predicted Location (m)', fontsize=10)
    ax7.set_ylabel('Probability Density', fontsize=10)
    ax7.set_title('Prediction Probability Curves by Target Location', fontsize=11, fontweight='bold')
    ax7.legend(fontsize=8, ncol=2)
    ax7.grid(True, alpha=0.3)

    plt.suptitle('TEM Target Locator - Comprehensive Results', fontsize=14, fontweight='bold')
    plt.savefig('improved_training_results.png', dpi=200, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------------------
    # Export Results to CSV for Detailed Analysis
    # -------------------------------------------------------------------------
    print(f"\n  Exporting results to CSV files...")

    # 1. Main predictions CSV
    results_df = pd.DataFrame({
        'true_location': y_test,
        'predicted_location': mean_preds,
        'absolute_error': errors,
        'ensemble_uncertainty': std_preds,
        'offset': [metadata_test[i]['offset'] for i in range(len(y_test))],
        'config_type': [metadata_test[i]['config_type'] for i in range(len(y_test))],
        'file_id': [metadata_test[i]['file_id'] for i in range(len(y_test))]
    })
    results_df.to_csv('predictions_results.csv', index=False)

    # 2. Configuration performance CSV
    config_perf_df = pd.DataFrame([
        {
            'configuration': config,
            'n_samples': stats['n_samples'],
            'mae': stats['mae'],
            'median_error': stats['median_error'],
            'std': stats['std'],
            'max_error': stats['max_error']
        }
        for config, stats in config_stats.items()
    ])
    config_perf_df.to_csv('configuration_performance.csv', index=False)

    # 3. Noise robustness CSV
    noise_df = pd.DataFrame([
        {'noise_level_percent': n*100, 'mae': noise_results[n]}
        for n in noise_levels
    ])
    noise_df.to_csv('noise_robustness.csv', index=False)

    # 4. Individual model predictions (for ensemble analysis)
    individual_preds_df = pd.DataFrame(test_preds.T, columns=[f'model_{i}' for i in range(N_ENSEMBLE)])
    individual_preds_df['true_location'] = y_test
    individual_preds_df['ensemble_mean'] = mean_preds
    individual_preds_df['ensemble_std'] = std_preds
    individual_preds_df.to_csv('ensemble_predictions.csv', index=False)

    # 5. Test set metadata with predictions (for detailed profiling)
    detailed_results = []
    for i in range(len(y_test)):
        detailed_results.append({
            'sample_idx': i,
            'true_location': y_test[i],
            'predicted_location': mean_preds[i],
            'error': errors[i],
            'uncertainty': std_preds[i],
            'offset': metadata_test[i]['offset'],
            'config_type': metadata_test[i]['config_type'],
            'file_id': metadata_test[i]['file_id'],
            'augmented': metadata_test[i].get('augmented', False)
        })
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv('detailed_test_results.csv', index=False)

    print(f"    ✓ Exported 5 CSV files for detailed analysis")

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\n  Saved files:")
    print(f"    Models:")
    print(f"      • improved_model_0.keras to improved_model_{N_ENSEMBLE-1}.keras")
    print(f"      • {SCALER_PATH}")
    print(f"    Visualization:")
    print(f"      • improved_training_results.png")
    print(f"    CSV Data (for custom analysis):")
    print(f"      • predictions_results.csv")
    print(f"      • configuration_performance.csv")
    print(f"      • noise_robustness.csv")
    print(f"      • ensemble_predictions.csv")
    print(f"      • detailed_test_results.csv")
    print(f"\n  Performance Summary:")
    print(f"    • Test MAE: {mae:.2f} m")
    print(f"    • Best configuration: {best_config[0]} (MAE: {best_config[1]['mae']:.2f} m)")
    print(f"    • Noise robustness: {degradation:.1f}% error increase at 20% noise")
    print(f"\n  Next steps:")
    print(f"    • Run 'python visualize_results.py' for detailed profile plots")
    print(f"    • Run 'python predict_single.py <path_to_tem_file>' for single predictions")
    print("="*70)
