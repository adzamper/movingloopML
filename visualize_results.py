#!/usr/bin/env python3
"""
Detailed Visualization Script for TEM Target Locator Results
==============================================================
This script reads the exported CSV files from training and generates
detailed individual plots for in-depth analysis.

Generates:
- Configuration-specific performance plots
- Error distribution by target location
- Ensemble agreement plots
- Per-sample prediction analysis
- And more...

Usage:
    python visualize_results.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Create output directory for plots
os.makedirs('detailed_plots', exist_ok=True)

def load_data():
    """Load all CSV files from training results."""
    print("Loading CSV data...")

    predictions = pd.read_csv('predictions_results.csv')
    config_perf = pd.read_csv('configuration_performance.csv')
    noise_robust = pd.read_csv('noise_robustness.csv')
    ensemble = pd.read_csv('ensemble_predictions.csv')
    detailed = pd.read_csv('detailed_test_results.csv')

    print(f"  ✓ Loaded {len(predictions)} predictions")
    return predictions, config_perf, noise_robust, ensemble, detailed


def plot_error_by_config_and_location(predictions):
    """Plot error heatmap: config vs location."""
    print("\n[1] Creating error heatmap by config and location...")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create pivot table
    pivot = predictions.pivot_table(
        values='absolute_error',
        index='config_type',
        columns='true_location',
        aggfunc='mean'
    )

    # Sort by offset
    offset_order = predictions.groupby(['config_type', 'offset']).first().reset_index()
    offset_order = offset_order.sort_values('offset')
    config_order = [f"{int(row['offset'])}m_{row['config_type']}"
                   for _, row in offset_order.iterrows()]

    # Reindex
    pivot_full = predictions.copy()
    pivot_full['config_full'] = pivot_full.apply(
        lambda x: f"{int(x['offset'])}m_{x['config_type']}", axis=1
    )
    pivot = pivot_full.pivot_table(
        values='absolute_error',
        index='config_full',
        columns='true_location',
        aggfunc='mean'
    )

    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
                cbar_kws={'label': 'MAE (m)'}, ax=ax)
    ax.set_xlabel('True Target Location (m)', fontsize=11)
    ax.set_ylabel('Configuration', fontsize=11)
    ax.set_title('Mean Absolute Error by Configuration and Target Location',
                fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('detailed_plots/1_error_heatmap.png', dpi=200)
    plt.close()
    print("  ✓ Saved: detailed_plots/1_error_heatmap.png")


def plot_ensemble_agreement(ensemble):
    """Plot how well ensemble members agree."""
    print("\n[2] Creating ensemble agreement plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Individual model predictions vs truth
    ax1 = axes[0, 0]
    n_models = len([c for c in ensemble.columns if c.startswith('model_')])
    for i in range(n_models):
        ax1.scatter(ensemble['true_location'], ensemble[f'model_{i}'],
                   alpha=0.3, s=20, label=f'Model {i}')
    ax1.plot([ensemble['true_location'].min(), ensemble['true_location'].max()],
            [ensemble['true_location'].min(), ensemble['true_location'].max()],
            'r--', lw=2, label='Perfect')
    ax1.set_xlabel('True Location (m)')
    ax1.set_ylabel('Predicted Location (m)')
    ax1.set_title('Individual Model Predictions')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Ensemble std vs error
    ax2 = axes[0, 1]
    errors = np.abs(ensemble['ensemble_mean'] - ensemble['true_location'])
    ax2.scatter(ensemble['ensemble_std'], errors, alpha=0.5, s=30)
    ax2.set_xlabel('Ensemble Std (σ, m)')
    ax2.set_ylabel('Absolute Error (m)')
    ax2.set_title('Uncertainty vs Error')
    ax2.grid(True, alpha=0.3)

    # Model agreement (pairwise differences)
    ax3 = axes[1, 0]
    model_preds = ensemble[[f'model_{i}' for i in range(n_models)]].values
    pairwise_diff = []
    for i in range(len(model_preds)):
        diffs = []
        for j in range(n_models):
            for k in range(j+1, n_models):
                diffs.append(abs(model_preds[i, j] - model_preds[i, k]))
        pairwise_diff.append(np.mean(diffs))

    ax3.hist(pairwise_diff, bins=30, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Mean Pairwise Difference (m)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Model Agreement Distribution')
    ax3.axvline(np.mean(pairwise_diff), color='r', linestyle='--',
               linewidth=2, label=f'Mean = {np.mean(pairwise_diff):.1f}m')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Uncertainty distribution by location
    ax4 = axes[1, 1]
    unique_locs = sorted(ensemble['true_location'].unique())
    stds_by_loc = [ensemble[ensemble['true_location']==loc]['ensemble_std'].values
                   for loc in unique_locs]
    ax4.boxplot(stds_by_loc, labels=[f'{int(loc)}' for loc in unique_locs])
    ax4.set_xlabel('True Target Location (m)')
    ax4.set_ylabel('Ensemble Uncertainty (σ, m)')
    ax4.set_title('Uncertainty by Target Location')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('detailed_plots/2_ensemble_agreement.png', dpi=200)
    plt.close()
    print("  ✓ Saved: detailed_plots/2_ensemble_agreement.png")


def plot_per_config_analysis(predictions):
    """Detailed analysis for each configuration."""
    print("\n[3] Creating per-configuration analysis...")

    configs = predictions['config_type'].unique()
    offsets = predictions['offset'].unique()

    # Create full config names
    predictions['config_full'] = predictions.apply(
        lambda x: f"{int(x['offset'])}m_{x['config_type']}", axis=1
    )

    unique_configs = sorted(predictions['config_full'].unique())
    n_configs = len(unique_configs)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, config in enumerate(unique_configs):
        if idx >= len(axes):
            break

        ax = axes[idx]
        data = predictions[predictions['config_full'] == config]

        # Scatter plot for this config
        ax.scatter(data['true_location'], data['predicted_location'],
                  alpha=0.6, s=40, c=data['absolute_error'], cmap='RdYlGn_r')
        ax.plot([data['true_location'].min(), data['true_location'].max()],
               [data['true_location'].min(), data['true_location'].max()],
               'r--', lw=2, alpha=0.7)

        mae = data['absolute_error'].mean()
        ax.set_title(f'{config}\nMAE: {mae:.1f}m', fontweight='bold')
        ax.set_xlabel('True Location (m)', fontsize=9)
        ax.set_ylabel('Predicted Location (m)', fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide extra subplots
    for idx in range(n_configs, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Prediction Performance by Configuration', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('detailed_plots/3_per_config_analysis.png', dpi=200)
    plt.close()
    print("  ✓ Saved: detailed_plots/3_per_config_analysis.png")


def plot_worst_predictions(detailed):
    """Analyze worst predictions in detail."""
    print("\n[4] Creating worst predictions analysis...")

    # Get worst 12 predictions
    worst = detailed.nlargest(12, 'error')

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(worst.iterrows()):
        if idx >= 12:
            break

        ax = axes[idx]

        # Create bar chart showing true vs predicted
        labels = ['True', 'Predicted']
        values = [row['true_location'], row['predicted_location']]
        colors = ['green', 'red']

        bars = ax.bar(labels, values, color=colors, alpha=0.7)
        ax.set_ylabel('Location (m)', fontsize=9)
        ax.set_title(f"Error: {row['error']:.1f}m | σ: {row['uncertainty']:.1f}m\n"
                    f"{int(row['offset'])}m_{row['config_type']}",
                    fontsize=9, fontweight='bold')
        ax.set_ylim([min(values)-100, max(values)+100])

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}m', ha='center', va='bottom', fontsize=8)

        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Worst 12 Predictions - Detailed Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('detailed_plots/4_worst_predictions.png', dpi=200)
    plt.close()
    print("  ✓ Saved: detailed_plots/4_worst_predictions.png")


def plot_best_predictions(detailed):
    """Analyze best predictions in detail."""
    print("\n[5] Creating best predictions analysis...")

    # Get best 12 predictions
    best = detailed.nsmallest(12, 'error')

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(best.iterrows()):
        if idx >= 12:
            break

        ax = axes[idx]

        # Create bar chart showing true vs predicted
        labels = ['True', 'Predicted']
        values = [row['true_location'], row['predicted_location']]
        colors = ['green', 'lightgreen']

        bars = ax.bar(labels, values, color=colors, alpha=0.7)
        ax.set_ylabel('Location (m)', fontsize=9)
        ax.set_title(f"Error: {row['error']:.1f}m | σ: {row['uncertainty']:.1f}m\n"
                    f"{int(row['offset'])}m_{row['config_type']}",
                    fontsize=9, fontweight='bold')
        ax.set_ylim([min(values)-50, max(values)+50])

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}m', ha='center', va='bottom', fontsize=8)

        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Best 12 Predictions - Detailed Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('detailed_plots/5_best_predictions.png', dpi=200)
    plt.close()
    print("  ✓ Saved: detailed_plots/5_best_predictions.png")


def plot_augmentation_impact(detailed):
    """Compare performance on original vs augmented samples."""
    print("\n[6] Creating augmentation impact analysis...")

    if 'augmented' not in detailed.columns or not detailed['augmented'].any():
        print("  ⚠ No augmented samples found, skipping...")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    original = detailed[~detailed['augmented']]
    augmented = detailed[detailed['augmented']]

    # Error distribution comparison
    ax1 = axes[0]
    ax1.hist(original['error'], bins=30, alpha=0.6, label='Original', edgecolor='black')
    ax1.hist(augmented['error'], bins=30, alpha=0.6, label='Augmented', edgecolor='black')
    ax1.set_xlabel('Absolute Error (m)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution: Original vs Augmented')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # MAE comparison
    ax2 = axes[1]
    mae_orig = original['error'].mean()
    mae_aug = augmented['error'].mean()

    bars = ax2.bar(['Original', 'Augmented'], [mae_orig, mae_aug],
                  color=['steelblue', 'orange'], alpha=0.7)
    ax2.set_ylabel('Mean Absolute Error (m)')
    ax2.set_title('MAE: Original vs Augmented Samples')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}m', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('detailed_plots/6_augmentation_impact.png', dpi=200)
    plt.close()
    print("  ✓ Saved: detailed_plots/6_augmentation_impact.png")


def main():
    """Main execution."""
    print("="*70)
    print("DETAILED VISUALIZATION GENERATOR")
    print("="*70)

    # Check if CSV files exist
    required_files = [
        'predictions_results.csv',
        'configuration_performance.csv',
        'noise_robustness.csv',
        'ensemble_predictions.csv',
        'detailed_test_results.csv'
    ]

    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"\n✗ Error: Missing CSV files: {missing}")
        print("  Please run CNN_target_locator.py first to generate data.")
        return

    # Load data
    predictions, config_perf, noise_robust, ensemble, detailed = load_data()

    # Generate plots
    print("\nGenerating detailed plots...")
    plot_error_by_config_and_location(predictions)
    plot_ensemble_agreement(ensemble)
    plot_per_config_analysis(predictions)
    plot_worst_predictions(detailed)
    plot_best_predictions(detailed)
    plot_augmentation_impact(detailed)

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\n  ✓ Generated 6 detailed plot files in 'detailed_plots/' directory")
    print(f"    1. Error heatmap by config and location")
    print(f"    2. Ensemble agreement analysis")
    print(f"    3. Per-configuration performance")
    print(f"    4. Worst predictions analysis")
    print(f"    5. Best predictions analysis")
    print(f"    6. Augmentation impact (if applicable)")
    print("="*70)


if __name__ == '__main__':
    main()
