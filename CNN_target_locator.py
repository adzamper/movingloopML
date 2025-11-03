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
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib
from scipy.signal import savgol_filter

# --- CONFIGURATION for tom change this to where-ever you put the training data on your drive---
DATA_DIRECTORY = r"C:\Users\anthony\Documents\Work\Nova\consulting\orano\movingloop\maxwell\training3"
MODEL_PATH = "improved_model.keras"
SCALER_PATH = "improved_scaler.joblib"
N_ENSEMBLE = 3  # Start small
MAX_STATIONS = 101
STATION_SPACING = 50.0

# --- DATA LOADING ---
def parse_tem_file(file_path):
    try:
        filename = os.path.basename(file_path)
        true_location_str = os.path.basename(os.path.dirname(file_path))
        true_location = float(true_location_str) if true_location_str.replace('.','').isdigit() else None
        offset_match = re.match(r'(\d+)m', filename)
        if not offset_match: return None, None, None
        offset = float(offset_match.group(1))
        
        with open(file_path, 'r', encoding='utf-8') as f: 
            lines = f.readlines()
        
        header_line_index = next((i for i, l in enumerate(lines) 
                                 if 'EAST' in l and 'STATION' in l), -1)
        if header_line_index == -1: return None, None, None
        
        header = re.split(r'\s+', lines[header_line_index].strip())
        data_lines = [l.strip() for l in lines[header_line_index + 1:] 
                     if l.strip() and (l.strip().startswith('-') or l.strip()[0].isdigit())]
        if not data_lines: return None, None, None
        
        data = [re.split(r'\s+', line) for line in data_lines]
        df = pd.DataFrame(data)
        num_cols = min(len(header), len(df.columns))
        df = df.iloc[:, :num_cols]
        df.columns = header[:num_cols]
        
        for col in df.columns:
            if col.upper() != 'COMPONENT': 
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        
        return df, offset, true_location
    except Exception as e: 
        return None, None, None

# Create Feature Profile
def create_feature_profile(df, offset):
    """features """
    
    early_channels = [f'CH{i}' for i in range(1, 8)]
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
    
    feature_dfs = []
    for (st, c), g in df.groupby(['STATION', 'COMPONENT']):
        early_sum = g[early_channels].sum(axis=1).iloc[0]
        late_sum = g[late_channels].sum(axis=1).iloc[0]
        
        feature_dfs.append({
            'STATION': st, 
            'COMPONENT': c,
            'early_sum': early_sum,
            'late_sum': late_sum
        })
    
    if not feature_dfs: return None
    
    feature_df = pd.DataFrame(feature_dfs)
    pivoted_df = feature_df.pivot(index='STATION', columns='COMPONENT', 
                                   values=['early_sum', 'late_sum'])
    pivoted_df.columns = ['_'.join(col).strip() for col in pivoted_df.columns.values]
    pivoted_df = pivoted_df.reset_index()
    
    # sum the x,y,z
    for time in ['early', 'late']:
        sum_cols = [f'{time}_sum_{c}' for c in ['X','Y','Z'] 
                   if f'{time}_sum_{c}' in pivoted_df.columns]
        if sum_cols:
            pivoted_df[f'tfa_{time}'] = np.sqrt(
                np.sum([pivoted_df[col]**2 for col in sum_cols], axis=0))
        else:
            pivoted_df[f'tfa_{time}'] = 0.0
    
    # Ratios of x,y,z
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
    
    base_features = ([f'{f}_{c}' for f in ['early_sum', 'late_sum', 'ratio'] 
                     for c in ['X', 'Y', 'Z']] + 
                    ['tfa_early', 'tfa_late', 'tfa_ratio'])
    
    for col in base_features:
        if col not in pivoted_df.columns: 
            pivoted_df[col] = 0.0
    
    # ADD ONLY RESIDUALS
    for col in base_features:
        if len(pivoted_df[col]) >= 51:
            try:
                background = savgol_filter(pivoted_df[col], window_length=51, polyorder=3)
                pivoted_df[f'{col}_residual'] = pivoted_df[col] - background
            except:
                pivoted_df[f'{col}_residual'] = 0
        else:
            pivoted_df[f'{col}_residual'] = 0
    
    # Gradients station to station
    grad_df = pivoted_df[base_features].diff().fillna(0)
    grad_df.columns = [f'{c}_grad' for c in base_features]
    
    # Combine features
    residual_cols = [c for c in pivoted_df.columns if 'residual' in c]
    final_features_df = pd.concat([
        pivoted_df[['STATION'] + base_features],
        pivoted_df[residual_cols],
        grad_df
    ], axis=1)
    
    final_features_df['offset'] = offset
    
    # Create spatial profile
    full_profile = np.zeros((MAX_STATIONS, len(final_features_df.columns) - 1))
    feature_cols = [col for col in final_features_df.columns if col != 'STATION']
    
    for _, row in final_features_df.iterrows():
        station_idx = int(row['STATION'] / STATION_SPACING)
        if 0 <= station_idx < MAX_STATIONS: 
            full_profile[station_idx, :] = row[feature_cols].values
    
    return full_profile

def load_all_data(base_dir):
    """Load all training data"""
    all_profiles, all_labels = [], []
    folders = [d for d in os.listdir(base_dir) 
              if os.path.isdir(os.path.join(base_dir, d))]
    
    for loc_str in folders:
        try:
            label = float(loc_str)
            folder_path = os.path.join(base_dir, loc_str)
            files = [f for f in os.listdir(folder_path) if f.endswith('.tem')]
            
            for fname in files:
                df, offset, _ = parse_tem_file(os.path.join(folder_path, fname))
                if df is not None:
                    profile = create_feature_profile(df, offset)
                    if profile is not None:
                        all_profiles.append(profile)
                        all_labels.append(label)
        except ValueError: 
            continue
    
    return np.array(all_profiles, dtype=np.float32), np.array(all_labels, dtype=np.float32)

# --- MODEL BUILDS HERE ---
def build_improved_model(input_shape):
    input_layer = Input(shape=input_shape)
    
    # Multi-scale inception (the diff kernel sizes)
    tower_1 = Conv1D(64, 3, padding='same', activation='relu')(input_layer)
    tower_1 = BatchNormalization()(tower_1)
    
    tower_2 = Conv1D(64, 7, padding='same', activation='relu')(input_layer)
    tower_2 = BatchNormalization()(tower_2)
    
    tower_3 = Conv1D(64, 11, padding='same', activation='relu')(input_layer)
    tower_3 = BatchNormalization()(tower_3)
    
    x = Concatenate(axis=-1)([tower_1, tower_2, tower_3])
    x = Dropout(0.3)(x)
    
    # Deeper processing
    x = Conv1D(128, 5, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    x = Conv1D(256, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    # Bidirectional LSTM
    x = tf.keras.layers.Bidirectional(LSTM(128, return_sequences=False))(x)
    x = Dropout(0.4)(x)
    
    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    
    output_layer = Dense(1, activation='linear')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    optimizer = Adam(learning_rate=0.0005)
    model.compile(
        optimizer=optimizer, 
        loss='mean_squared_error',  # Start with MSE, not Huber
        metrics=['mean_absolute_error']
    )
    
    return model

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    print("="*60)
    print("IMPROVED TARGET LOCATOR - TRAINING")
    print("="*60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, DATA_DIRECTORY)
    
    # Load data
    print("\nLoading data...")
    X, y = load_all_data(data_path)
    print(f"Loaded {X.shape[0]} profiles with {X.shape[2]} features each")
    
    # Split data
    X_train_full, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set: {len(X_train_full)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Fit scaler (same as baseline)
    print("\nFitting scaler...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_reshaped = X_train_full.reshape(-1, X_train_full.shape[-1])
    scaler.fit(X_train_reshaped)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to '{SCALER_PATH}'")
    
    # Transform data
    X_train = scaler.transform(X_train_reshaped).reshape(X_train_full.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    
    # Train ensemble
    print(f"\nTraining {N_ENSEMBLE} models...")
    models = []
    
    for i in range(N_ENSEMBLE):
        print(f"\n{'='*60}")
        print(f"Training Model {i+1}/{N_ENSEMBLE}")
        print(f"{'='*60}")
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_improved_model(input_shape)
        
        if i == 0:
            model.summary()
        
        model_path = f"improved_model_{i}.keras"
        
        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, 
                            min_lr=0.00001, verbose=1),
            EarlyStopping(monitor='val_loss', patience=40, verbose=1, 
                         restore_best_weights=True),
            ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, 
                          verbose=1)
        ]
        
        # Different seed for diversity
        np.random.seed(42 + i)
        tf.random.set_seed(42 + i)
        
        history = model.fit(
            X_train, y_train,
            epochs=250,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Reload best
        model = tf.keras.models.load_model(model_path)
        models.append(model)
        
        # Evaluate
        val_preds = model.predict(X_val, verbose=0).flatten()
        val_mae = np.mean(np.abs(val_preds - y_val))
        print(f"\nModel {i+1} Validation MAE: {val_mae:.2f} m")
    
    # Ensemble evaluation
    print("\n" + "="*60)
    print("ENSEMBLE EVALUATION")
    print("="*60)
    
    ensemble_preds = np.array([model.predict(X_val, verbose=0).flatten() 
                               for model in models])
    mean_preds = np.mean(ensemble_preds, axis=0)
    std_preds = np.std(ensemble_preds, axis=0)
    
    errors = np.abs(mean_preds - y_val)
    mae = np.mean(errors)
    median_error = np.median(errors)
    
    print(f"\nEnsemble Performance:")
    print(f"Mean Absolute Error: {mae:.2f} m")
    print(f"Median Error: {median_error:.2f} m")
    print(f"Mean Uncertainty (σ): {np.mean(std_preds):.2f} m")
    
    # Best/worst
    results = sorted(zip(y_val, mean_preds, errors, std_preds), 
                    key=lambda item: item[2])
    
    print("\n--- BEST 5 PREDICTIONS ---")
    for i in range(min(5, len(results))):
        true_loc, pred_loc, error, uncertainty = results[i]
        print(f"{i+1}. True={true_loc:.0f}m, Pred={pred_loc:.0f}m, "
              f"Error={error:.1f}m, σ={uncertainty:.1f}m")
    
    print("\n--- WORST 5 PREDICTIONS ---")
    for i in range(min(5, len(results))):
        true_loc, pred_loc, error, uncertainty = results[-(i+1)]
        print(f"{i+1}. True={true_loc:.0f}m, Pred={pred_loc:.0f}m, "
              f"Error={error:.1f}m, σ={uncertainty:.1f}m")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].scatter(y_val, mean_preds, alpha=0.5)
    axes[0, 0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 
                    'r--', lw=2)
    axes[0, 0].set_xlabel('True Location (m)')
    axes[0, 0].set_ylabel('Predicted Location (m)')
    axes[0, 0].set_title('Predictions vs True Values')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(mae, color='r', linestyle='--', linewidth=2, 
                      label=f'Mean = {mae:.1f}m')
    axes[0, 1].set_xlabel('Absolute Error (m)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Error Distribution')
    axes[0, 1].legend()
    
    axes[1, 0].scatter(std_preds, errors, alpha=0.5)
    axes[1, 0].set_xlabel('Uncertainty (σ, m)')
    axes[1, 0].set_ylabel('Absolute Error (m)')
    axes[1, 0].set_title('Uncertainty vs Error')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(std_preds, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Uncertainty (σ, m)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Uncertainty Distribution')
    
    plt.tight_layout()
    plt.savefig('improved_training_results.png', dpi=150)
    plt.close()
    
    print("\nTraining complete!")
    print(f"Models saved as 'improved_model_0.keras' through 'improved_model_{N_ENSEMBLE-1}.keras'")
    print(f"Results saved to 'improved_training_results.png'")
