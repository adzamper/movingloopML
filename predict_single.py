#!/usr/bin/env python3
"""
Single TEM File Prediction with Probability Curve
==================================================
This script loads trained models and predicts the target location for
a single unseen .tem file, showing prediction probability distribution.

Usage:
    python predict_single.py <path_to_tem_file>

Example:
    python predict_single.py 1700/0moffset1.tem
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from CNN_target_locator import parse_tem_file, create_feature_profile

# Configuration
N_ENSEMBLE = 3
SCALER_PATH = "improved_scaler.joblib"
MODEL_PREFIX = "improved_model"


def load_models_and_scaler():
    """Load all trained models and scaler."""
    print("\nLoading trained models and scaler...")

    # Check files exist
    if not os.path.exists(SCALER_PATH):
        print(f"  ✗ Error: Scaler not found at '{SCALER_PATH}'")
        print("    Please run CNN_target_locator.py first to train models.")
        sys.exit(1)

    # Load scaler
    scaler = joblib.load(SCALER_PATH)
    print(f"  ✓ Loaded scaler")

    # Load models
    models = []
    for i in range(N_ENSEMBLE):
        model_path = f"{MODEL_PREFIX}_{i}.keras"
        if not os.path.exists(model_path):
            print(f"  ✗ Error: Model not found at '{model_path}'")
            sys.exit(1)
        model = tf.keras.models.load_model(model_path)
        models.append(model)
        print(f"  ✓ Loaded model {i+1}/{N_ENSEMBLE}")

    return models, scaler


def predict_single_file(file_path, models, scaler):
    """
    Predict target location for a single .tem file.

    Returns:
    --------
    tuple : (mean_pred, std_pred, individual_preds, metadata, raw_profile)
    """
    print(f"\nProcessing file: {file_path}")

    # Parse file
    df, metadata = parse_tem_file(file_path)
    if df is None or metadata is None:
        print(f"  ✗ Error: Could not parse file")
        return None

    print(f"  ✓ Parsed successfully")
    print(f"    Configuration: {int(metadata['offset'])}m_{metadata['config_type']}")
    print(f"    True location: {metadata['true_location']}m" if metadata['true_location'] else "")

    # Create feature profile
    profile = create_feature_profile(df, metadata)
    if profile is None:
        print(f"  ✗ Error: Could not create feature profile")
        return None

    raw_profile = profile.copy()  # Save raw profile for visualization
    print(f"  ✓ Created feature profile: {profile.shape}")

    # Scale features
    profile_reshaped = profile.reshape(-1, profile.shape[-1])
    profile_scaled = scaler.transform(profile_reshaped).reshape(profile.shape)
    profile_scaled = np.expand_dims(profile_scaled, axis=0)  # Add batch dimension

    # Get predictions from all models
    print(f"\n  Getting ensemble predictions...")
    individual_preds = []
    for i, model in enumerate(models):
        pred = model.predict(profile_scaled, verbose=0)[0, 0]
        individual_preds.append(pred)
        print(f"    Model {i+1}: {pred:.1f}m")

    # Calculate ensemble statistics
    mean_pred = np.mean(individual_preds)
    std_pred = np.std(individual_preds)

    print(f"\n  Ensemble Prediction: {mean_pred:.1f} ± {std_pred:.1f} m")

    if metadata['true_location']:
        error = abs(mean_pred - metadata['true_location'])
        print(f"  True Location: {metadata['true_location']:.0f}m")
        print(f"  Absolute Error: {error:.1f}m")

    return mean_pred, std_pred, individual_preds, metadata, raw_profile, df


def visualize_prediction(mean_pred, std_pred, individual_preds, metadata, raw_profile, df, file_path):
    """Create comprehensive visualization for single prediction."""
    print(f"\n  Creating visualization...")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Probability Curve (main plot)
    ax1 = fig.add_subplot(gs[0, :])
    x_range = np.linspace(mean_pred - 4*std_pred, mean_pred + 4*std_pred, 200)
    y_range = (1 / (std_pred * np.sqrt(2 * np.pi))) * \
              np.exp(-0.5 * ((x_range - mean_pred) / std_pred) ** 2)

    ax1.fill_between(x_range, y_range, alpha=0.3, color='steelblue', label='Prediction PDF')
    ax1.plot(x_range, y_range, linewidth=3, color='steelblue')
    ax1.axvline(mean_pred, color='red', linestyle='--', linewidth=2, label=f'Prediction: {mean_pred:.1f}m')

    if metadata['true_location']:
        ax1.axvline(metadata['true_location'], color='green', linestyle='--',
                   linewidth=2, label=f'True: {metadata["true_location"]:.0f}m')
        error = abs(mean_pred - metadata['true_location'])
        ax1.set_title(f'Prediction Probability Distribution | Error: {error:.1f}m',
                     fontsize=14, fontweight='bold')
    else:
        ax1.set_title('Prediction Probability Distribution',
                     fontsize=14, fontweight='bold')

    # Shade confidence intervals
    x_1std = x_range[(x_range >= mean_pred - std_pred) & (x_range <= mean_pred + std_pred)]
    y_1std = (1 / (std_pred * np.sqrt(2 * np.pi))) * \
             np.exp(-0.5 * ((x_1std - mean_pred) / std_pred) ** 2)
    ax1.fill_between(x_1std, y_1std, alpha=0.5, color='orange', label='±1σ (68% confidence)')

    ax1.set_xlabel('Location (m)', fontsize=11)
    ax1.set_ylabel('Probability Density', fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Individual Model Predictions
    ax2 = fig.add_subplot(gs[1, 0])
    model_indices = [f'M{i+1}' for i in range(len(individual_preds))]
    bars = ax2.bar(model_indices, individual_preds, alpha=0.7, color='steelblue')
    ax2.axhline(mean_pred, color='red', linestyle='--', linewidth=2, label='Ensemble Mean')
    if metadata['true_location']:
        ax2.axhline(metadata['true_location'], color='green', linestyle='--',
                   linewidth=2, label='True Location')
    ax2.set_ylabel('Predicted Location (m)', fontsize=10)
    ax2.set_title('Individual Model Predictions', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, individual_preds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # 3. Raw TEM Data - X Component Profile
    ax3 = fig.add_subplot(gs[1, 1])
    # Extract X component data
    x_comp = df[df['COMPONENT'] == 'X']
    if not x_comp.empty:
        stations = x_comp['STATION'].values
        ch_cols = [col for col in x_comp.columns if col.startswith('CH')]
        # Plot early channels as proxy for anomaly
        for ch in ch_cols[:5]:  # First 5 channels
            ax3.plot(stations, x_comp[ch].values, alpha=0.7, linewidth=1)
        ax3.set_xlabel('Station (m)', fontsize=10)
        ax3.set_ylabel('Response (nV/Am²)', fontsize=10)
        ax3.set_title('Raw X-Component Profile (Early Channels)', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3)

    # 4. Raw TEM Data - Z Component Profile
    ax4 = fig.add_subplot(gs[1, 2])
    z_comp = df[df['COMPONENT'] == 'Z']
    if not z_comp.empty:
        stations = z_comp['STATION'].values
        for ch in ch_cols[:5]:  # First 5 channels
            ax4.plot(stations, z_comp[ch].values, alpha=0.7, linewidth=1)
        ax4.set_xlabel('Station (m)', fontsize=10)
        ax4.set_ylabel('Response (nV/Am²)', fontsize=10)
        ax4.set_title('Raw Z-Component Profile (Early Channels)', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)

    # 5. Feature Profile Summary (spatial features)
    ax5 = fig.add_subplot(gs[2, :])
    # Plot non-zero stations
    non_zero_mask = np.abs(raw_profile).sum(axis=1) > 0
    non_zero_indices = np.where(non_zero_mask)[0]

    if len(non_zero_indices) > 0:
        # Plot first few important features
        feature_indices = [0, 1, 2]  # First few features
        stations_m = non_zero_indices * 50  # Convert to meters (STATION_SPACING = 50)

        for feat_idx in feature_indices:
            if feat_idx < raw_profile.shape[1]:
                feature_vals = raw_profile[non_zero_mask, feat_idx]
                ax5.plot(stations_m, feature_vals, '-o', alpha=0.7, linewidth=2,
                        markersize=4, label=f'Feature {feat_idx+1}')

        ax5.axvline(mean_pred, color='red', linestyle='--', linewidth=2,
                   alpha=0.7, label='Predicted Location')
        if metadata['true_location']:
            ax5.axvline(metadata['true_location'], color='green', linestyle='--',
                       linewidth=2, alpha=0.7, label='True Location')

        ax5.set_xlabel('Station Location (m)', fontsize=11)
        ax5.set_ylabel('Feature Value', fontsize=11)
        ax5.set_title('Spatial Feature Profile (Processed)', fontsize=11, fontweight='bold')
        ax5.legend(fontsize=9, ncol=3)
        ax5.grid(True, alpha=0.3)

    # Add metadata text
    config_str = f"{int(metadata['offset'])}m_{metadata['config_type']}"
    filename = os.path.basename(file_path)
    folder = os.path.basename(os.path.dirname(file_path))

    fig.text(0.02, 0.98, f"File: {folder}/{filename} | Config: {config_str}",
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'Single File Prediction Analysis: {filename}',
                fontsize=16, fontweight='bold', y=0.99)

    output_file = f'prediction_{filename.replace(".tem", "")}.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved visualization: {output_file}")


def main():
    """Main execution."""
    print("="*70)
    print("SINGLE TEM FILE PREDICTION")
    print("="*70)

    # Check command line arguments
    if len(sys.argv) < 2:
        print("\n✗ Error: No file path provided")
        print("\nUsage:")
        print("  python predict_single.py <path_to_tem_file>")
        print("\nExample:")
        print("  python predict_single.py 1700/0moffset1.tem")
        sys.exit(1)

    file_path = sys.argv[1]

    # Check file exists
    if not os.path.exists(file_path):
        print(f"\n✗ Error: File not found: {file_path}")
        sys.exit(1)

    # Load models
    models, scaler = load_models_and_scaler()

    # Predict
    result = predict_single_file(file_path, models, scaler)
    if result is None:
        sys.exit(1)

    mean_pred, std_pred, individual_preds, metadata, raw_profile, df = result

    # Visualize
    visualize_prediction(mean_pred, std_pred, individual_preds, metadata,
                        raw_profile, df, file_path)

    print("\n" + "="*70)
    print("PREDICTION COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
