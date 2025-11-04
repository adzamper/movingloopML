# Simple CNN Target Locator - Method Explanation

## The Core Question

**Can a model trained on survey data from certain locations predict conductor position at a completely NEW location?**

This is what the simplified script tests through **location-based splitting**.

---

## What This Script Does

### 1. **The Problem**
- We have TEM (Time-Domain Electromagnetic) survey data from multiple locations
- Each location has multiple configurations (different transmitter-receiver distances: 0m, 500m, 800m, 1000m, trailing)
- Each configuration has ~20 noise variations (same survey, different random noise)
- Goal: Predict the conductor's true location from the survey measurements

### 2. **The Data Split (CRITICAL)**

**Location-Based Split:**
```
Training:   Locations A, B, C, D (all configs at each location)
Validation: Location E (all configs)
Test:       Location F (all configs)
```

**Why this way?**
- Different configurations (0m vs 500m offset) have fundamentally different physics
- You can't train on one config and test on another (proven to fail catastrophically)
- In real deployment: you train at known sites, then deploy at NEW sites
- At each site, you measure ALL configurations

**What this tests:** "Will the model work at a survey site we've never visited before?"

### 3. **Feature Engineering**

We DON'T use raw time-channel data. Instead, we extract **physics-informed features**:

#### Time-Window Features
- **Early channels** (CH1-CH7): Fast decay, near-surface response
- **Mid channels** (CH8-CH14): Medium decay
- **Late channels** (CH15-CH20): Slow decay, deep conductor response

For each component (X, Y, Z), we calculate:
- Sum of early, mid, and late channels
- Total Field Amplitude (TFA) = √(X² + Y² + Z²)
- Decay ratios = late/early (anomaly indicator)

#### Background Removal (CRITICAL)
- Uses Savitzky-Golay filter (smooth polynomial fit)
- Removes regional electromagnetic field trends
- Isolates LOCAL target anomaly
- **This is essential** - without it, performance collapses (1225m MAE → 117m MAE)

#### Anomaly Metrics
- Peak anomaly strength (maximum residual)
- Anomaly energy (sum of squared residuals)

#### What We DON'T Include
- **NO spatial gradients** - Ablation study proved they hurt performance by 22%
- **NO raw channels alone** - They failed catastrophically (1225m MAE)

### 4. **Model Architecture**

Think of the model as having three main parts:

#### A. Multi-Scale CNN (Pattern Recognition)
```
Input: 101 stations × N features
   ↓
Three parallel CNN branches with different "zoom levels":
  - Small kernel (3): Detects sharp, localized anomalies
  - Medium kernel (7): Detects moderate-scale patterns
  - Large kernel (11): Detects broad regional trends
   ↓
Combine all three perspectives
```

**Why?** Conductor anomalies appear at different spatial scales depending on depth and configuration.

#### B. Deep Convolutional Processing
```
Multi-scale features
   ↓
Conv1D (128 filters) → Pool → Dropout
   ↓
Conv1D (256 filters) → Pool → Dropout
```

Progressively extracts higher-level spatial patterns while reducing spatial dimension.

#### C. Bidirectional LSTM (Sequence Modeling)
```
Processed features
   ↓
LSTM (forward pass: station 0 → 100)
LSTM (backward pass: station 100 → 0)
   ↓
Combine both directions
   ↓
Dense layers → Predicted location
```

**Why LSTM?** Survey data has spatial structure - measurements at nearby stations are related. LSTM models this sequential dependency in both directions.

### 5. **Training Process**

1. **Normalize** features to [0, 1] within each survey
2. **Log transform** channels (handles exponential TEM decay)
3. **Scale** to [-1, 1] using MinMaxScaler (fit on training only!)
4. **Train** with early stopping (stops when validation stops improving)
5. **Evaluate** on completely unseen test locations

**Key Hyperparameters:**
- Learning rate: 0.0005 with adaptive reduction
- Batch size: 32
- Dropout: 0.3-0.5 (prevents overfitting)
- Max epochs: 250 (but early stopping usually activates around 80-120)

---

## Key Results and Limitations

### What Works Well
✓ **Interpolation within training range**
  - If test location is BETWEEN training locations, model performs well (~100-200m MAE)
  - Example: Train on 1700-2900m, test at 2300m → Good

### What Doesn't Work
✗ **Extrapolation beyond training range**
  - If test location is OUTSIDE training range, performance degrades (~600-900m MAE)
  - Example: Train on 1700-2900m, test at 3100m → Poor
  - **This is expected** - neural networks cannot extrapolate

✗ **Cross-configuration generalization**
  - Cannot train on 0m offset and test on 500m offset (2212m MAE)
  - Different configs have different EM physics
  - **Solution:** Train with all configs you plan to deploy

### Practical Deployment

**To use this model in the field:**

1. **Collect diverse training data:**
   - Survey at multiple locations spanning the full range you expect to encounter
   - Measure ALL configurations you'll use in deployment
   - Include edge locations (don't just sample the middle)

2. **Test location requirements:**
   - Should be WITHIN the range of training locations (interpolation)
   - Must use configurations that were in training data
   - Cannot expect good performance on configs the model has never seen

3. **Uncertainty estimation:**
   - Large errors often occur at range edges (extrapolation)
   - If deploying near training range boundaries, expect higher error

---

## Why This Approach?

### Why Physics-Informed Features?
- **Raw channels alone failed** (1225m MAE)
- Background removal is CRITICAL for isolating target anomaly
- Time-window aggregation reduces noise and dimensionality
- Domain knowledge (early vs late decay) encoded explicitly

### Why Location-Based Split?
- **Prevents data leakage** (noise variations stay together)
- **Tests realistic deployment** (new survey sites)
- **Honest performance estimate** (not optimistically biased)

### Why CNN + LSTM?
- **CNN:** Learns spatial patterns across stations
- **LSTM:** Models sequential dependencies along survey line
- **Multi-scale:** Captures anomalies at different depths/scales
- **Proven architecture** for spatial sequence data

---

## Files Generated

After running `CNN_target_locator_simple.py`:

1. **simple_cnn_model.keras** - Trained model
2. **simple_cnn_results.png** - Visualization of performance
3. **simple_cnn_predictions.csv** - Detailed predictions for analysis

---

## Quick Start

```bash
# Run training
python CNN_target_locator_simple.py

# Expected output:
# - Training/validation/test split by location
# - Model architecture summary
# - Training progress with early stopping
# - Test performance metrics
# - Visualizations
```

---

## Performance Interpretation

**Good Performance (~100-200m MAE):**
- Test locations within training range ✓
- All configs present in training ✓
- Sufficient training diversity ✓

**Poor Performance (~600-1000m MAE):**
- Test locations outside training range (extrapolation) ✗
- Insufficient training data diversity ✗
- Edge effects (test near range boundaries) ✗

**Failed Performance (~2000m+ MAE):**
- Cross-config testing (train on one config, test on another) ✗
- Critical preprocessing missing (no background removal) ✗
- Data leakage in splits ✗

---

## Comparison to Original Script

**Simplified Version:**
- Single model (no ensemble)
- Streamlined visualization
- ~650 lines (vs 1336 original)
- Focus on core functionality
- Easier to understand and modify

**Original Version:**
- 3-model ensemble (uncertainty estimation)
- Extensive analysis and CSV exports
- Multiple visualization panels
- Configuration-specific noise sensitivity analysis
- Backward compatibility features

**Performance:** Both achieve similar results (~100-200m MAE for interpolation)

---

## Next Steps

1. **Check your test locations:** Are they within or outside training range?
2. **Evaluate interpolation performance:** Use stratified split (test in middle of range)
3. **Collect more data:** If deploying at range edges, include edge training samples
4. **Experiment with features:** Try different time-window definitions
5. **Tune architecture:** Adjust kernel sizes, LSTM units, dropout rates

---

**Bottom Line:** This model works well for predicting conductor locations at NEW survey sites, as long as those sites are within the range of locations used for training and use configurations that were in the training data.
