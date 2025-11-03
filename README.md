# TEM Target Locator - CNN-LSTM with Attention

Deep learning model for localizing targets in time-domain electromagnetic (TEM) survey data using an ensemble of CNN-LSTM networks with spatial attention.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify installation
python test_data_loading.py

# 3. Train the model
python CNN_target_locator.py
```

## What This Does

This script trains an ensemble of deep learning models to predict target locations from TEM survey data. It:

- **Learns** to distinguish primary target response from nuisance targets
- **Compares** different survey configurations (0m, 500m, 800m, 1000m offset, trailing)
- **Predicts** target location with uncertainty estimates
- **Visualizes** performance across configurations

## Key Features

✅ **Multi-scale CNN architecture** - Captures anomalies at different spatial scales
✅ **Attention mechanism** - Focuses on anomalous regions
✅ **Bidirectional LSTM** - Models spatial sequences
✅ **Ensemble learning** - Provides uncertainty quantification
✅ **Configuration analysis** - Identifies best survey setup
✅ **Data augmentation** - Improves generalization
✅ **Comprehensive visualization** - 8-plot analysis dashboard

## Data Format

Expected structure:
```
movingloopML/
├── 1700/          # Target at 1700m along profile
│   ├── 0moffset1.tem
│   ├── 0moffset2.tem
│   ├── 500moffset1.tem
│   └── ...
├── 1900/          # Target at 1900m
├── 2100/          # Target at 2100m
└── ...
```

Each folder represents a target location. Each `.tem` file contains survey data with 20 time channels across multiple stations, with 5 different survey configurations and 20 noise variations.

## Output

After training, you'll get:

**Models:**
- `improved_model_0.keras`, `improved_model_1.keras`, etc.
- `improved_scaler.joblib` (for feature normalization)

**Visualizations:**
- `improved_training_results.png` (7-plot comprehensive analysis)

**CSV Data Files (for custom analysis):**
- `predictions_results.csv` - All test predictions with errors and uncertainty
- `configuration_performance.csv` - Performance metrics by configuration
- `noise_robustness.csv` - Noise sensitivity test results
- `ensemble_predictions.csv` - Individual model predictions for analysis
- `detailed_test_results.csv` - Complete test set metadata with predictions

## Performance Metrics

The model reports:
- **MAE** (Mean Absolute Error) - average prediction error in meters
- **Median Error** - robust error metric
- **RMSE** - penalizes large errors
- **Uncertainty (σ)** - ensemble prediction confidence
- **Per-configuration performance** - identifies best survey configuration

## Configuration

Edit these parameters in `CNN_target_locator.py`:

```python
DATA_DIRECTORY = "."              # Path to data folders
N_ENSEMBLE = 3                    # Number of ensemble models
AUGMENTATION_ENABLED = True       # Enable data augmentation
AUGMENTATION_NOISE_LEVEL = 0.05   # 5% noise augmentation
```

## Additional Tools

### Detailed Visualization (`visualize_results.py`)
Generate detailed individual plots from exported CSV files:

```bash
python visualize_results.py
```

Creates 6 detailed plots in `detailed_plots/` folder:
1. Error heatmap by configuration and location
2. Ensemble agreement analysis
3. Per-configuration performance comparison
4. Worst 12 predictions analysis
5. Best 12 predictions analysis
6. Augmentation impact (if applicable)

### Single File Prediction (`predict_single.py`)
Predict target location for a single unseen .tem file with probability curve:

```bash
python predict_single.py <path_to_tem_file>
```

Example:
```bash
python predict_single.py 1700/0moffset1.tem
```

Outputs:
- Ensemble prediction with uncertainty
- Probability distribution curve
- Individual model predictions
- Raw X/Z component profiles
- Spatial feature visualization

### Feature Ablation Study (`feature_ablation_study.py`)
Test which features contribute most to model performance:

```bash
python feature_ablation_study.py
```

Tests 8 different feature combinations:
- **Full**: All ~100 features (baseline)
- **Minimal**: Only late_sum_Z_residual
- **Physics Core**: Decay ratios + key residuals
- **No Residuals**: Raw features only
- **No Gradients**: Excluding spatial derivatives
- **Late Only**: Only late-time channels
- **Ratios Only**: Only decay ratios
- **Z Component Only**: Only vertical component

Outputs:
- `feature_ablation_results.csv` - Performance comparison
- Identifies critical vs redundant features
- Validates if model learns physics vs memorizes

See `FEATURE_RATIONALE.md` for scientific explanation of each feature.

## Documentation

- **README.md** - Project overview and quick start (this file)
- **FEATURE_RATIONALE.md** - Scientific rationale for feature engineering
- **IMPROVEMENTS.md** - Detailed documentation of all improvements
- **requirements.txt** - Python package dependencies
- **test_data_loading.py** - Quick test script to verify setup

## Architecture

```
Input (101 stations × N features)
    ↓
Multi-scale Inception Block (3, 7, 11 kernel sizes)
    ↓
Deep CNN Processing (2 blocks)
    ↓
Spatial Attention Mechanism
    ↓
Bidirectional LSTM
    ↓
Dense Layers
    ↓
Output (predicted target location)
```

## For Team Members

This script is designed to be readable by team members with varying ML experience:

- **Comprehensive docstrings** explain what each function does
- **Inline comments** explain ML concepts
- **Clear section headers** organize the code logically
- **Progress indicators** show training status

## Improvements Made

This version includes major improvements over the original:

1. ✅ Configuration tracking and comparison
2. ✅ Enhanced feature engineering with anomaly metrics
3. ✅ Attention mechanism for better localization
4. ✅ Data augmentation for robustness
5. ✅ Proper train/val/test splits
6. ✅ Comprehensive visualization (8 plots)
7. ✅ Cross-platform compatibility
8. ✅ Extensive documentation
9. ✅ Configuration-specific performance analysis
10. ✅ Improved code readability

See **IMPROVEMENTS.md** for detailed explanations of each improvement.

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn

Full list in `requirements.txt`.

---

**Created by:** Machine Learning Team
**Last Updated:** 2025-11-03
**Purpose:** TEM target localization using deep learning
