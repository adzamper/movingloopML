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

- **Models:** `improved_model_0.keras`, `improved_model_1.keras`, etc.
- **Scaler:** `improved_scaler.joblib` (for feature normalization)
- **Results plot:** `improved_training_results.png` (comprehensive 8-plot analysis)

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

## Documentation

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
