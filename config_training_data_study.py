"""
Configuration Training Data Study
==================================

Answers: "If I only collect training data from 2-3 configurations (not all 5),
will the model still perform well on those configurations?"

This is DIFFERENT from the configuration ablation study:
- OLD: Train on config A, test on config B (cross-config generalization) ❌
- NEW: Train on configs A+B, test on configs A+B (same configs) ✓

Business Question:
- Collecting training data from all 5 configs = expensive field work
- Can I collect from just 2-3 configs and still get good performance?
- What's the performance penalty for having less diverse training data?

Example Test:
- Baseline: Train on all 5 configs (900 samples) → Test on all 5 → MAE: 100m
- Option 1: Train on best 3 configs (540 samples) → Test on those 3 → MAE: 110m
- Option 2: Train on best 2 configs (360 samples) → Test on those 2 → MAE: 125m
- Recommendation: 3 configs = good balance (only 10% worse, 40% less data collection)
"""

import os
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, LSTM, Dense,
                                      Dropout, BatchNormalization, Concatenate,
                                      Multiply, Activation, Bidirectional)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import from main script
import sys
sys.path.insert(0, '.')
from CNN_target_locator import (parse_tem_file, create_feature_profile,
                                 create_raw_channel_profile, FEATURE_MODE,
                                 MAX_STATIONS, RANDOM_SEED)

# Configuration
DATA_DIRECTORY = "."
N_ENSEMBLE = 2  # Reduced for speed
EPOCHS = 50

def build_model(input_shape):
    """Simplified model for faster training."""
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


def load_all_data(base_dir):
    """Load all data with configuration labels."""
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
                    if FEATURE_MODE == 'raw':
                        profile = create_raw_channel_profile(df, metadata)
                    else:
                        profile = create_feature_profile(df, metadata)

                    if profile is not None:
                        config_name = f"{int(metadata['offset'])}m_{metadata['config_type']}"
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


def split_by_config_groups(X, y, metadata, included_configs, test_ratio=0.2):
    """
    Split data using group-based splitting to prevent noise variation leakage.
    Only includes samples from specified configurations.

    Parameters:
    -----------
    X, y, metadata : arrays
        Full dataset
    included_configs : list
        Configs to include (e.g., ['0m_offset', '500m_offset'])
    test_ratio : float
        Fraction for test set

    Returns:
    --------
    tuple : (X_train, y_train, X_test, y_test)
    """
    # Group by (location, config_type, offset) to keep noise variations together
    config_groups = {}
    for i, meta in enumerate(metadata):
        config_name = f"{int(meta['offset'])}m_{meta['config_type']}"

        if config_name not in included_configs:
            continue

        group_key = (meta['true_location'], meta['config_type'], meta['offset'])
        if group_key not in config_groups:
            config_groups[group_key] = []
        config_groups[group_key].append(i)

    # Shuffle groups
    group_keys = list(config_groups.keys())
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(group_keys)

    # Split groups into train/test
    n_test_groups = max(1, int(len(group_keys) * test_ratio))
    test_groups = group_keys[:n_test_groups]
    train_groups = group_keys[n_test_groups:]

    # Create index lists
    train_indices = []
    test_indices = []

    for group_key in train_groups:
        train_indices.extend(config_groups[group_key])
    for group_key in test_groups:
        test_indices.extend(config_groups[group_key])

    return (X[train_indices], y[train_indices],
            X[test_indices], y[test_indices])


def train_and_evaluate(X_train, y_train, X_test, y_test, verbose=0):
    """Train ensemble and return test MAE."""
    if len(X_train) < 10 or len(X_test) < 5:
        return None

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


def run_training_data_study():
    """
    Main study: Test performance when training on different config subsets.
    """
    print("="*70)
    print("CONFIGURATION TRAINING DATA STUDY")
    print("="*70)
    print("\nQuestion: How many configurations do I need to collect training data from?")
    print("Trade-off: More configs = more training diversity but more field work cost\n")

    # Load full dataset
    print("[1/5] Loading full dataset...")
    all_X, all_y, all_metadata, all_config_labels = load_all_data(DATA_DIRECTORY)

    # Get unique configs
    unique_configs = sorted(set(all_config_labels))
    config_counts = {c: all_config_labels.count(c) for c in unique_configs}

    print(f"\nAvailable configurations: {len(unique_configs)}")
    for config in unique_configs:
        print(f"  {config}: {config_counts[config]} samples")

    print(f"\nTotal samples: {len(all_X)}")

    results = []

    # =========================================================================
    # Test 1: Baseline - Train on ALL configurations
    # =========================================================================
    print("\n" + "="*70)
    print("[2/5] BASELINE: Training on ALL configurations")
    print("="*70)

    X_train, y_train, X_test, y_test = split_by_config_groups(
        all_X, all_y, all_metadata, unique_configs, test_ratio=0.2
    )

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Training configurations: {', '.join(unique_configs)}")

    baseline_mae = train_and_evaluate(X_train, y_train, X_test, y_test, verbose=0)
    print(f"  → Baseline MAE: {baseline_mae:.2f} m")

    results.append({
        'n_configs': len(unique_configs),
        'configs': 'ALL (' + ', '.join(unique_configs) + ')',
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'mae': baseline_mae,
        'mae_vs_baseline': 0.0,
        'percent_vs_baseline': 0.0
    })

    # =========================================================================
    # Test 2: Single configurations
    # =========================================================================
    print("\n" + "="*70)
    print("[3/5] TEST: Training on SINGLE configurations")
    print("="*70)

    for config in unique_configs:
        print(f"\nTraining on: {config} only")

        X_train, y_train, X_test, y_test = split_by_config_groups(
            all_X, all_y, all_metadata, [config], test_ratio=0.2
        )

        print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")

        mae = train_and_evaluate(X_train, y_train, X_test, y_test, verbose=0)

        if mae is not None:
            diff = mae - baseline_mae
            percent = (diff / baseline_mae) * 100
            print(f"  → MAE: {mae:.2f} m ({percent:+.1f}% vs baseline)")

            results.append({
                'n_configs': 1,
                'configs': config,
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
                'mae': mae,
                'mae_vs_baseline': diff,
                'percent_vs_baseline': percent
            })

    # =========================================================================
    # Test 3: Pairs of configurations
    # =========================================================================
    print("\n" + "="*70)
    print("[4/5] TEST: Training on PAIRS of configurations")
    print("="*70)

    for config_pair in combinations(unique_configs, 2):
        config_list = list(config_pair)
        print(f"\nTraining on: {' + '.join(config_list)}")

        X_train, y_train, X_test, y_test = split_by_config_groups(
            all_X, all_y, all_metadata, config_list, test_ratio=0.2
        )

        print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")

        mae = train_and_evaluate(X_train, y_train, X_test, y_test, verbose=0)

        if mae is not None:
            diff = mae - baseline_mae
            percent = (diff / baseline_mae) * 100
            print(f"  → MAE: {mae:.2f} m ({percent:+.1f}% vs baseline)")

            results.append({
                'n_configs': 2,
                'configs': ' + '.join(config_list),
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
                'mae': mae,
                'mae_vs_baseline': diff,
                'percent_vs_baseline': percent
            })

    # =========================================================================
    # Test 4: Triplets of configurations
    # =========================================================================
    if len(unique_configs) >= 4:
        print("\n" + "="*70)
        print("[5/5] TEST: Training on TRIPLETS of configurations")
        print("="*70)

        for config_triplet in combinations(unique_configs, 3):
            config_list = list(config_triplet)
            print(f"\nTraining on: {' + '.join(config_list)}")

            X_train, y_train, X_test, y_test = split_by_config_groups(
                all_X, all_y, all_metadata, config_list, test_ratio=0.2
            )

            print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")

            mae = train_and_evaluate(X_train, y_train, X_test, y_test, verbose=0)

            if mae is not None:
                diff = mae - baseline_mae
                percent = (diff / baseline_mae) * 100
                print(f"  → MAE: {mae:.2f} m ({percent:+.1f}% vs baseline)")

                results.append({
                    'n_configs': 3,
                    'configs': ' + '.join(config_list),
                    'n_train_samples': len(X_train),
                    'n_test_samples': len(X_test),
                    'mae': mae,
                    'mae_vs_baseline': diff,
                    'percent_vs_baseline': percent
                })

    # =========================================================================
    # Analysis and Summary
    # =========================================================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    df = pd.DataFrame(results)
    df.to_csv('config_training_data_results.csv', index=False)
    print("\n✓ Saved: config_training_data_results.csv")

    # Best by number of configs
    print("\n1. PERFORMANCE BY NUMBER OF CONFIGURATIONS:")
    print("-"*70)
    print(f"  {'N Configs':<12} {'Mean MAE':<12} {'Std MAE':<12} {'Best MAE':<12} {'vs Baseline'}")
    print("-"*70)

    for n in sorted(df['n_configs'].unique()):
        subset = df[df['n_configs'] == n]
        mean_mae = subset['mae'].mean()
        std_mae = subset['mae'].std()
        best_mae = subset['mae'].min()
        mean_vs_baseline = subset['percent_vs_baseline'].mean()

        print(f"  {n:<12} {mean_mae:<12.1f} {std_mae:<12.1f} {best_mae:<12.1f} {mean_vs_baseline:+.1f}%")

    # Best single config
    print("\n2. BEST SINGLE CONFIGURATION:")
    print("-"*70)
    single_results = df[df['n_configs'] == 1].sort_values('mae')
    if not single_results.empty:
        best = single_results.iloc[0]
        print(f"   {best['configs']}")
        print(f"   → MAE: {best['mae']:.2f} m ({best['percent_vs_baseline']:+.1f}% vs baseline)")
        print(f"\n   Top 3:")
        for idx, row in single_results.head(3).iterrows():
            print(f"     {row['configs']:<20} {row['mae']:>6.1f} m ({row['percent_vs_baseline']:+.1f}%)")

    # Best pair
    print("\n3. BEST PAIR OF CONFIGURATIONS:")
    print("-"*70)
    pair_results = df[df['n_configs'] == 2].sort_values('mae')
    if not pair_results.empty:
        best = pair_results.iloc[0]
        print(f"   {best['configs']}")
        print(f"   → MAE: {best['mae']:.2f} m ({best['percent_vs_baseline']:+.1f}% vs baseline)")
        print(f"\n   Top 3:")
        for idx, row in pair_results.head(3).iterrows():
            print(f"     {row['configs']:<40} {row['mae']:>6.1f} m ({row['percent_vs_baseline']:+.1f}%)")

    # Best triplet
    triplet_results = df[df['n_configs'] == 3].sort_values('mae')
    if not triplet_results.empty:
        print("\n4. BEST TRIPLET OF CONFIGURATIONS:")
        print("-"*70)
        best = triplet_results.iloc[0]
        print(f"   {best['configs']}")
        print(f"   → MAE: {best['mae']:.2f} m ({best['percent_vs_baseline']:+.1f}% vs baseline)")

    # Visualizations
    print("\n5. Creating visualizations...")
    create_visualizations(df, baseline_mae, unique_configs)
    print("   ✓ Saved: config_training_data_analysis.png")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)

    # ROI analysis
    if not single_results.empty and not pair_results.empty:
        best_single = single_results.iloc[0]
        best_pair = pair_results.iloc[0]

        print(f"\n→ Collecting from 2 configs vs 1 config:")
        print(f"  Performance: {best_single['mae']:.1f}m → {best_pair['mae']:.1f}m")
        print(f"  Improvement: {abs(best_pair['percent_vs_baseline'] - best_single['percent_vs_baseline']):.1f}%")
        print(f"  Field work cost: 2× vs 1×")

        improvement = abs(best_pair['mae'] - best_single['mae'])
        if improvement / best_single['mae'] > 0.15:
            print(f"  ✓ STRONG benefit - 2nd config significantly improves performance")
        elif improvement / best_single['mae'] > 0.05:
            print(f"  ⚠ MODERATE benefit - evaluate cost vs accuracy")
        else:
            print(f"  ✗ MINIMAL benefit - 1 config may be sufficient")

        if not triplet_results.empty:
            best_triplet = triplet_results.iloc[0]
            print(f"\n→ Collecting from 3 configs vs 2 configs:")
            print(f"  Performance: {best_pair['mae']:.1f}m → {best_triplet['mae']:.1f}m")
            print(f"  Improvement: {abs(best_triplet['percent_vs_baseline'] - best_pair['percent_vs_baseline']):.1f}%")
            print(f"  Field work cost: 3× vs 2×")

            improvement = abs(best_triplet['mae'] - best_pair['mae'])
            if improvement / best_pair['mae'] > 0.15:
                print(f"  ✓ STRONG benefit - 3rd config adds significant value")
            elif improvement / best_pair['mae'] > 0.05:
                print(f"  ⚠ MODERATE benefit - evaluate cost vs accuracy")
            else:
                print(f"  ✗ MINIMAL benefit - 2 configs likely sufficient")

    # Final recommendation
    print(f"\n→ OPTIMAL CONFIGURATION SET:")
    if not pair_results.empty:
        best_overall = df[df['n_configs'] >= 2].sort_values('mae').iloc[0]
        roi = abs(best_overall['percent_vs_baseline'])
        if roi < 5:
            print(f"   {best_overall['configs']}")
            print(f"   Performance: {best_overall['mae']:.1f}m (only {roi:.1f}% worse than all {len(unique_configs)} configs)")
            print(f"   Cost savings: {100 * (len(unique_configs) - best_overall['n_configs']) / len(unique_configs):.0f}% less field work")
            print(f"   ✓ Excellent trade-off - recommend this configuration set")

    print("\n" + "="*70)
    print("STUDY COMPLETE!")
    print("="*70)

    return df


def create_visualizations(df, baseline_mae, unique_configs):
    """Create visualization plots."""
    fig = plt.figure(figsize=(14, 10))

    # 1. Performance vs number of configs
    ax1 = plt.subplot(2, 2, 1)

    for n in sorted(df['n_configs'].unique()):
        subset = df[df['n_configs'] == n]
        ax1.scatter([n]*len(subset), subset['mae'], alpha=0.6, s=100, label=f'{n} config(s)')

    means = df.groupby('n_configs')['mae'].mean()
    ax1.plot(means.index, means.values, 'r-o', linewidth=3, markersize=12, label='Mean')
    ax1.axhline(baseline_mae, color='green', linestyle='--', linewidth=2,
                label=f'Baseline ({len(unique_configs)} configs)', alpha=0.7)

    ax1.set_xlabel('Number of Configurations (Training Data)', fontsize=12)
    ax1.set_ylabel('Test MAE (m)', fontsize=12)
    ax1.set_title('Performance vs Training Data Diversity', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Percent vs baseline
    ax2 = plt.subplot(2, 2, 2)

    box_data = []
    labels = []
    for n in sorted(df['n_configs'].unique()):
        if n < len(unique_configs):  # Exclude baseline
            subset = df[df['n_configs'] == n]
            box_data.append(subset['percent_vs_baseline'].values)
            labels.append(f'{n} config(s)')

    if box_data:
        ax2.boxplot(box_data, tick_labels=labels)
        ax2.axhline(0, color='green', linestyle='--', linewidth=2, label='Baseline')
        ax2.set_ylabel('Performance vs Baseline (%)', fontsize=12)
        ax2.set_title('Degradation from Using Fewer Configs', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

    # 3. Best options by config count
    ax3 = plt.subplot(2, 2, 3)

    best_by_n = df[df['n_configs'] < len(unique_configs)].loc[
        df[df['n_configs'] < len(unique_configs)].groupby('n_configs')['mae'].idxmin()
    ]

    colors = ['steelblue', 'orange', 'green', 'red']
    bars = ax3.bar(range(len(best_by_n)), best_by_n['mae'], color=colors[:len(best_by_n)])
    ax3.axhline(baseline_mae, color='green', linestyle='--', linewidth=2,
                label=f'Baseline ({baseline_mae:.1f}m)')
    ax3.set_xticks(range(len(best_by_n)))
    ax3.set_xticklabels([f"{int(n)} config(s)" for n in best_by_n['n_configs']])
    ax3.set_ylabel('Best MAE (m)', fontsize=12)
    ax3.set_title('Best Performance by Config Count', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, best_by_n['mae']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.1f}m', ha='center', fontsize=10, fontweight='bold')

    # 4. ROI analysis
    ax4 = plt.subplot(2, 2, 4)

    means_by_n = df.groupby('n_configs')['mae'].mean().sort_index()

    if len(means_by_n) > 1:
        improvements = []
        labels = []

        for i in range(len(means_by_n) - 1):
            curr_mae = means_by_n.iloc[i]
            next_mae = means_by_n.iloc[i+1]
            improvement_pct = ((curr_mae - next_mae) / curr_mae) * 100
            improvements.append(improvement_pct)
            labels.append(f"{int(means_by_n.index[i])}→{int(means_by_n.index[i+1])}")

        colors_roi = ['green' if x > 10 else 'orange' if x > 5 else 'red' for x in improvements]
        bars = ax4.bar(range(len(improvements)), improvements, color=colors_roi, alpha=0.7)
        ax4.set_xticks(range(len(improvements)))
        ax4.set_xticklabels(labels)
        ax4.set_xlabel('Adding Configurations', fontsize=12)
        ax4.set_ylabel('Performance Improvement (%)', fontsize=12)
        ax4.set_title('ROI: Value of Additional Training Data', fontsize=13, fontweight='bold')
        ax4.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, improvements)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('config_training_data_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    results_df = run_training_data_study()
