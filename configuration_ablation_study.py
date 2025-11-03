"""
Configuration Ablation Study for TEM Target Locator
====================================================

Answers the question: "Which survey configurations should I run in the field?"

Tests:
1. Which single configuration generalizes best to others?
2. Which pairs of configurations give best overall performance?
3. What's the ROI on running 3, 4, or all 5 configurations?

This helps optimize field operations:
- Running all 5 configs = expensive, slow
- Running 1-2 configs = cheap, fast, but lower accuracy
- This study finds the sweet spot!
"""

import os
import re
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, LSTM, Dense,
                                      Dropout, BatchNormalization, Concatenate,
                                      Multiply, GlobalAveragePooling1D, Reshape,
                                      Activation, Add, Bidirectional)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import functions from main script
import sys
sys.path.insert(0, '.')
from CNN_target_locator import (parse_tem_file, create_feature_profile,
                                 create_raw_channel_profile, FEATURE_MODE,
                                 MAX_STATIONS, STATION_SPACING, RANDOM_SEED)

# Configuration
DATA_DIRECTORY = "."
N_ENSEMBLE = 2  # Reduced for speed (this is exploratory)
EPOCHS = 50     # Reduced for speed

def build_model(input_shape):
    """Simplified model for faster training during ablation study."""
    input_layer = Input(shape=input_shape)

    # Multi-scale inception
    tower_1 = Conv1D(32, 3, padding='same', activation='relu')(input_layer)
    tower_2 = Conv1D(32, 7, padding='same', activation='relu')(input_layer)
    tower_3 = Conv1D(32, 11, padding='same', activation='relu')(input_layer)
    merged = Concatenate(axis=-1)([tower_1, tower_2, tower_3])

    # CNN processing
    x = Conv1D(64, 5, padding='same', activation='relu')(merged)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # Attention
    attention = Dense(1, activation='tanh')(x)
    attention = Activation('softmax')(attention)
    x = Multiply()([x, attention])

    # LSTM
    x = Bidirectional(LSTM(32, return_sequences=False))(x)
    x = Dropout(0.3)(x)

    # Output
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='linear')(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    return model


def load_data_by_configs(base_dir, included_configs):
    """
    Load data only from specified configurations.

    Parameters:
    -----------
    base_dir : str
        Base directory
    included_configs : list
        List of config names like ['0m_offset', '500m_offset']

    Returns:
    --------
    tuple : (X, y, metadata, config_labels)
    """
    all_profiles, all_labels, all_metadata, all_configs = [], [], [], []

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
                    # Check if this config is included
                    config_name = f"{int(metadata['offset'])}m_{metadata['config_type']}"

                    if config_name in included_configs:
                        if FEATURE_MODE == 'raw':
                            profile = create_raw_channel_profile(df, metadata)
                        else:
                            profile = create_feature_profile(df, metadata)

                        if profile is not None:
                            all_profiles.append(profile)
                            all_labels.append(label)
                            all_metadata.append(metadata)
                            all_configs.append(config_name)
        except ValueError:
            continue

    return (np.array(all_profiles, dtype=np.float32),
            np.array(all_labels, dtype=np.float32),
            all_metadata,
            all_configs)


def split_by_config_groups(X, y, metadata, test_configs):
    """
    Split data: train on some configs, test on others.

    Parameters:
    -----------
    X, y, metadata : arrays
        Full dataset
    test_configs : list
        Configs to use for testing (e.g., ['800m_offset'])

    Returns:
    --------
    tuple : (X_train, y_train, X_test, y_test, metadata_train, metadata_test)
    """
    train_indices = []
    test_indices = []

    for i, meta in enumerate(metadata):
        config_name = f"{int(meta['offset'])}m_{meta['config_type']}"

        if config_name in test_configs:
            test_indices.append(i)
        else:
            train_indices.append(i)

    return (X[train_indices], y[train_indices],
            X[test_indices], y[test_indices],
            [metadata[i] for i in train_indices],
            [metadata[i] for i in test_indices])


def train_and_evaluate(X_train, y_train, X_test, y_test, verbose=0):
    """Train ensemble and return test MAE."""
    # Scale
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_reshaped)

    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Train ensemble
    predictions = []
    for i in range(N_ENSEMBLE):
        model = build_model(X_train_scaled.shape[1:])

        early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

        model.fit(X_train_scaled, y_train,
                 epochs=EPOCHS, batch_size=16,
                 callbacks=[early_stop], verbose=verbose)

        preds = model.predict(X_test_scaled, verbose=0).flatten()
        predictions.append(preds)

    # Ensemble average
    mean_preds = np.mean(predictions, axis=0)
    mae = np.mean(np.abs(mean_preds - y_test))

    return mae


def run_configuration_ablation():
    """
    Main ablation study.

    Tests all possible combinations of configurations to find:
    1. Which single config generalizes best?
    2. Which pairs work best together?
    3. Optimal number of configs to run in field?
    """
    print("="*70)
    print("CONFIGURATION ABLATION STUDY")
    print("="*70)
    print("\nQuestion: Which configurations should you run in the field?")
    print("Trade-off: More configs = better accuracy but higher cost/time\n")

    # Load full dataset to get all available configs
    print("[1/4] Loading full dataset...")
    all_X, all_y, all_metadata, all_config_labels = load_data_by_configs(DATA_DIRECTORY, [
        '0m_offset', '500m_offset', '800m_offset', '1000m_offset', '500m_trailing'
    ])

    # Get unique configs
    unique_configs = sorted(set(all_config_labels))
    config_counts = {c: all_config_labels.count(c) for c in unique_configs}

    print(f"\nAvailable configurations: {len(unique_configs)}")
    for config in unique_configs:
        print(f"  {config}: {config_counts[config]} samples")

    print(f"\nTotal samples: {len(all_X)}")

    # Results storage
    results = []

    # =========================================================================
    # Test 1: Single configuration performance
    # =========================================================================
    print("\n" + "="*70)
    print("[2/4] TEST 1: Single Configuration Performance")
    print("="*70)
    print("Testing: Can one config predict targets for other configs?\n")

    for train_config in unique_configs:
        print(f"\nTraining on: {train_config} only")
        print("-"*70)

        # Get test configs (all except train config)
        test_configs = [c for c in unique_configs if c != train_config]

        if not test_configs:
            print("  Skipping (no test configs available)")
            continue

        # Split data
        X_train, y_train, X_test, y_test, meta_train, meta_test = split_by_config_groups(
            all_X, all_y, all_metadata, test_configs
        )

        if len(X_train) < 10 or len(X_test) < 5:
            print(f"  Skipping (insufficient data: train={len(X_train)}, test={len(X_test)})")
            continue

        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)} (from {', '.join(test_configs)})")

        # Train and evaluate
        mae = train_and_evaluate(X_train, y_train, X_test, y_test, verbose=0)

        print(f"  → Test MAE: {mae:.2f} m")

        results.append({
            'n_train_configs': 1,
            'train_configs': train_config,
            'test_configs': ', '.join(test_configs),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'test_mae': mae
        })

    # =========================================================================
    # Test 2: Pairs of configurations
    # =========================================================================
    print("\n" + "="*70)
    print("[3/4] TEST 2: Pairs of Configurations")
    print("="*70)
    print("Testing: Which 2 configs give best coverage?\n")

    # Generate all pairs
    for train_pair in combinations(unique_configs, 2):
        train_configs_list = list(train_pair)
        test_configs = [c for c in unique_configs if c not in train_configs_list]

        if not test_configs:
            continue

        print(f"\nTraining on: {' + '.join(train_configs_list)}")

        # Split data
        X_train, y_train, X_test, y_test, meta_train, meta_test = split_by_config_groups(
            all_X, all_y, all_metadata, test_configs
        )

        if len(X_train) < 10 or len(X_test) < 5:
            print(f"  Skipping (insufficient data)")
            continue

        print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples ({', '.join(test_configs)})")

        mae = train_and_evaluate(X_train, y_train, X_test, y_test, verbose=0)

        print(f"  → Test MAE: {mae:.2f} m")

        results.append({
            'n_train_configs': 2,
            'train_configs': ' + '.join(train_configs_list),
            'test_configs': ', '.join(test_configs),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'test_mae': mae
        })

    # =========================================================================
    # Test 3: Triplets (if enough configs)
    # =========================================================================
    if len(unique_configs) >= 4:
        print("\n" + "="*70)
        print("[4/4] TEST 3: Triplets of Configurations")
        print("="*70)
        print("Testing: Does adding a 3rd config improve performance?\n")

        for train_triplet in combinations(unique_configs, 3):
            train_configs_list = list(train_triplet)
            test_configs = [c for c in unique_configs if c not in train_configs_list]

            if not test_configs:
                continue

            print(f"\nTraining on: {' + '.join(train_configs_list)}")

            X_train, y_train, X_test, y_test, meta_train, meta_test = split_by_config_groups(
                all_X, all_y, all_metadata, test_configs
            )

            if len(X_train) < 10 or len(X_test) < 5:
                print(f"  Skipping (insufficient data)")
                continue

            print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples ({', '.join(test_configs)})")

            mae = train_and_evaluate(X_train, y_train, X_test, y_test, verbose=0)

            print(f"  → Test MAE: {mae:.2f} m")

            results.append({
                'n_train_configs': 3,
                'train_configs': ' + '.join(train_configs_list),
                'test_configs': ', '.join(test_configs),
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
                'test_mae': mae
            })

    # =========================================================================
    # Analysis and Summary
    # =========================================================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    df = pd.DataFrame(results)
    df.to_csv('configuration_ablation_results.csv', index=False)
    print("\n✓ Saved: configuration_ablation_results.csv")

    # Best single config
    print("\n1. BEST SINGLE CONFIGURATION:")
    print("-"*70)
    single_results = df[df['n_train_configs'] == 1].sort_values('test_mae')
    if not single_results.empty:
        best_single = single_results.iloc[0]
        print(f"   {best_single['train_configs']}")
        print(f"   → MAE: {best_single['test_mae']:.2f} m")
        print(f"   → Generalizes to: {best_single['test_configs']}")
        print(f"\n   Top 3 single configs:")
        for idx, row in single_results.head(3).iterrows():
            print(f"     {row['train_configs']:<20} MAE: {row['test_mae']:.2f} m")

    # Best pair
    print("\n2. BEST PAIR OF CONFIGURATIONS:")
    print("-"*70)
    pair_results = df[df['n_train_configs'] == 2].sort_values('test_mae')
    if not pair_results.empty:
        best_pair = pair_results.iloc[0]
        print(f"   {best_pair['train_configs']}")
        print(f"   → MAE: {best_pair['test_mae']:.2f} m")
        print(f"   → Generalizes to: {best_pair['test_configs']}")
        print(f"\n   Top 3 pairs:")
        for idx, row in pair_results.head(3).iterrows():
            print(f"     {row['train_configs']:<40} MAE: {row['test_mae']:.2f} m")

    # Best triplet
    triplet_results = df[df['n_train_configs'] == 3].sort_values('test_mae')
    if not triplet_results.empty:
        print("\n3. BEST TRIPLET OF CONFIGURATIONS:")
        print("-"*70)
        best_triplet = triplet_results.iloc[0]
        print(f"   {best_triplet['train_configs']}")
        print(f"   → MAE: {best_triplet['test_mae']:.2f} m")
        print(f"   → Generalizes to: {best_triplet['test_configs']}")

    # Performance vs number of configs
    print("\n4. PERFORMANCE vs NUMBER OF CONFIGURATIONS:")
    print("-"*70)
    for n in sorted(df['n_train_configs'].unique()):
        subset = df[df['n_train_configs'] == n]
        mean_mae = subset['test_mae'].mean()
        std_mae = subset['test_mae'].std()
        best_mae = subset['test_mae'].min()
        print(f"   {n} config(s):  Mean MAE = {mean_mae:.2f} m ± {std_mae:.2f} m  (Best = {best_mae:.2f} m)")

    # Visualization
    print("\n5. Creating visualizations...")
    create_visualizations(df, unique_configs)
    print("   ✓ Saved: configuration_ablation_analysis.png")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    if not single_results.empty and not pair_results.empty:
        best_single_mae = single_results.iloc[0]['test_mae']
        best_pair_mae = pair_results.iloc[0]['test_mae']
        improvement = (best_single_mae - best_pair_mae) / best_single_mae * 100

        print(f"\n→ Running 2 configs vs 1 config:")
        print(f"  Improvement: {improvement:.1f}% (from {best_single_mae:.1f}m to {best_pair_mae:.1f}m)")

        if improvement > 25:
            print(f"  ✓ STRONG benefit - recommend running 2 configs")
        elif improvement > 10:
            print(f"  ⚠ MODERATE benefit - consider cost vs accuracy trade-off")
        else:
            print(f"  ✗ MINIMAL benefit - 1 config may be sufficient")

        if not triplet_results.empty:
            best_triplet_mae = triplet_results.iloc[0]['test_mae']
            improvement_3rd = (best_pair_mae - best_triplet_mae) / best_pair_mae * 100
            print(f"\n→ Running 3 configs vs 2 configs:")
            print(f"  Improvement: {improvement_3rd:.1f}% (from {best_pair_mae:.1f}m to {best_triplet_mae:.1f}m)")

            if improvement_3rd > 15:
                print(f"  ✓ STRONG benefit - 3rd config adds significant value")
            elif improvement_3rd > 5:
                print(f"  ⚠ MODERATE benefit - evaluate cost vs benefit")
            else:
                print(f"  ✗ MINIMAL benefit - 2 configs likely sufficient")

    print("\n" + "="*70)
    print("STUDY COMPLETE!")
    print("="*70)
    return df


def create_visualizations(df, unique_configs):
    """Create visualization plots for ablation results."""
    fig = plt.figure(figsize=(14, 10))

    # 1. MAE by number of configurations
    ax1 = plt.subplot(2, 2, 1)
    for n in sorted(df['n_train_configs'].unique()):
        subset = df[df['n_train_configs'] == n]
        ax1.scatter([n]*len(subset), subset['test_mae'], alpha=0.6, s=100)

    # Add means
    means = df.groupby('n_train_configs')['test_mae'].mean()
    ax1.plot(means.index, means.values, 'r-o', linewidth=2, markersize=10, label='Mean')

    ax1.set_xlabel('Number of Training Configurations', fontsize=12)
    ax1.set_ylabel('Test MAE (m)', fontsize=12)
    ax1.set_title('Performance vs Number of Configurations', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Best combinations by size
    ax2 = plt.subplot(2, 2, 2)
    best_by_n = df.loc[df.groupby('n_train_configs')['test_mae'].idxmin()]
    colors = ['steelblue', 'orange', 'green', 'red', 'purple']
    bars = ax2.bar(range(len(best_by_n)), best_by_n['test_mae'],
                   color=colors[:len(best_by_n)])
    ax2.set_xticks(range(len(best_by_n)))
    ax2.set_xticklabels([f"{int(n)} config(s)" for n in best_by_n['n_train_configs']])
    ax2.set_ylabel('Test MAE (m)', fontsize=12)
    ax2.set_title('Best Performance by Configuration Count', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, best_by_n['test_mae'])):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.1f}m', ha='center', fontsize=10, fontweight='bold')

    # 3. Heatmap of pair performance (if available)
    ax3 = plt.subplot(2, 2, 3)
    pair_results = df[df['n_train_configs'] == 2]

    if not pair_results.empty:
        # Create matrix
        n_configs = len(unique_configs)
        heatmap_data = np.full((n_configs, n_configs), np.nan)

        for idx, row in pair_results.iterrows():
            configs = row['train_configs'].split(' + ')
            if len(configs) == 2:
                i = unique_configs.index(configs[0])
                j = unique_configs.index(configs[1])
                heatmap_data[i, j] = row['test_mae']
                heatmap_data[j, i] = row['test_mae']

        im = ax3.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
        ax3.set_xticks(range(n_configs))
        ax3.set_yticks(range(n_configs))
        ax3.set_xticklabels([c.replace('_offset', '').replace('_trailing', 'T')
                            for c in unique_configs], rotation=45, ha='right')
        ax3.set_yticklabels([c.replace('_offset', '').replace('_trailing', 'T')
                            for c in unique_configs])
        ax3.set_title('Pair Performance Heatmap (MAE in m)', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax3)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for heatmap',
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Pair Performance Heatmap', fontsize=13, fontweight='bold')

    # 4. ROI analysis
    ax4 = plt.subplot(2, 2, 4)
    means_by_n = df.groupby('n_train_configs')['test_mae'].mean().sort_index()

    if len(means_by_n) > 1:
        # Calculate improvement per additional config
        improvements = []
        for i in range(1, len(means_by_n)):
            prev_mae = means_by_n.iloc[i-1]
            curr_mae = means_by_n.iloc[i]
            improvement = (prev_mae - curr_mae) / prev_mae * 100
            improvements.append(improvement)

        x_vals = list(means_by_n.index[1:])
        ax4.bar(range(len(improvements)), improvements, color='teal', alpha=0.7)
        ax4.set_xticks(range(len(improvements)))
        ax4.set_xticklabels([f"{int(x_vals[i]-1)}→{int(x_vals[i])}"
                            for i in range(len(x_vals))])
        ax4.set_xlabel('Adding Configuration', fontsize=12)
        ax4.set_ylabel('Improvement (%)', fontsize=12)
        ax4.set_title('ROI: Improvement per Additional Config', fontsize=13, fontweight='bold')
        ax4.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, val in enumerate(improvements):
            ax4.text(i, val + 1 if val > 0 else val - 1,
                    f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Need more data for ROI analysis',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('ROI Analysis', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig('configuration_ablation_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    results_df = run_configuration_ablation()
