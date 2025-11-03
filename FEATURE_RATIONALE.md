# Feature Engineering Rationale - Scientific Perspective
## TEM Target Localization

This document explains the **scientific and physics-based rationale** for each feature used in the CNN model, and provides tools to test feature importance.

---

## 📚 TEM Physics Background

**Time-Domain Electromagnetic (TEM)** surveys measure the decay of electromagnetic fields after a transmitter is turned off. Key physics:

1. **Exponential Decay**: EM fields decay exponentially with time
2. **Conductor Response**: Good conductors (targets) decay slower than background
3. **Spatial Anomalies**: Conductive targets create localized anomalies in the response
4. **Component Sensitivity**: X, Y, Z components respond differently based on target geometry

---

## 🔬 Current Feature Categories

### **1. Time-Domain Summations** (Early/Mid/Late)
```python
early_channels = CH1-CH7   # Fast decay (0.1-0.5 ms)
mid_channels = CH8-CH14    # Medium decay (0.5-2 ms)
late_channels = CH15-CH20  # Slow decay (2-6 ms)
```

**Scientific Rationale:**
- **Early times**: Dominated by shallow features and initial transient
- **Mid times**: Transition zone where target response emerges
- **Late times**: Deep conductors dominate; background decays faster
- **Separating time windows** captures different depth/conductivity signatures

**Why this matters:**
- Good conductors (targets) have stronger late-time response
- Background noise decays quickly → disappears in late times
- Ratio of late/early is a **classic TEM interpretation metric**

---

### **2. Component Separation** (X, Y, Z)
```python
for component in ['X', 'Y', 'Z']:
    features[f'early_sum_{component}']
```

**Scientific Rationale:**
- **Z-component** (vertical): Sensitive to horizontal conductors, depth
- **X-component** (inline): Sensitive to target location along profile
- **Y-component** (perpendicular): Sensitive to off-profile targets

**Why this matters:**
- Different survey geometries emphasize different components
- Target shape affects component ratios
- **X-component often shows strongest anomaly** near target location

---

### **3. Total Field Amplitude (TFA)**
```python
tfa = sqrt(X² + Y² + Z²)
```

**Scientific Rationale:**
- Magnitude of response **independent of direction**
- Reduces sensitivity to target orientation
- Used in mineral exploration to identify anomaly strength

**Why this matters:**
- Some targets have strong response but rotated components
- TFA captures anomaly regardless of orientation
- **Good for comparing different configurations**

---

### **4. Decay Ratios** (Late/Early)
```python
ratio = late_sum / (early_sum + ε)
```

**Scientific Rationale:**
- **This is classic TEM interpretation!**
- Good conductors: high late/early ratio (slow decay)
- Resistive features: low late/early ratio (fast decay)
- **Tau (time constant)** of conductor controls this

**Why this matters:**
- **Most physically meaningful feature** for discriminating conductors
- Used by geophysicists for manual interpretation
- Normalizes for anomaly strength (removes amplitude dependency)

---

### **5. Background-Removed Residuals** (Savitzky-Golay Filter)
```python
background = savgol_filter(signal, window=51, polyorder=3)
residual = signal - background
```

**Scientific Rationale:**
- Regional geology creates **long-wavelength background**
- Local targets create **short-wavelength anomalies**
- Polynomial filter removes regional trend, isolates anomalies

**Why this matters:**
- **Critical for anomaly detection**
- Background varies across survey area (geology, topography)
- Without removal, model might learn background patterns instead of targets
- This is **standard practice in geophysical processing**

---

### **6. Spatial Gradients** (Station-to-Station Change)
```python
gradient = diff(feature, axis=stations)
```

**Scientific Rationale:**
- Targets create **localized peaks** in response
- Peaks have **high gradients at edges**
- **Second-derivative analysis** is standard in potential field geophysics

**Why this matters:**
- Helps CNN identify **where** the anomaly is
- Edge detection is important for localization
- Complements absolute values

---

### **7. Anomaly Strength Metrics**
```python
anomaly_peak = max(abs(residuals))
anomaly_energy = sqrt(sum(residuals²))
```

**Scientific Rationale:**
- **Peak**: Maximum anomaly amplitude (target strength)
- **Energy**: Integrated anomaly power (spatial extent × strength)

**Why this matters:**
- Provides **global context** for each station
- Strong anomalies → high confidence
- Weak anomalies → model should be uncertain

---

### **8. Configuration Metadata** (Offset, Config Type)
```python
offset = 0m, 500m, 800m, 1000m
config_type = offset, trailing
```

**Scientific Rationale:**
- **Survey geometry affects response**
- Larger offset → less sensitivity to near-surface, more depth
- Trailing vs offset → different spatial coupling

**Why this matters:**
- Model can learn **which configurations work best**
- Compensates for geometric differences
- Your results show **0m_offset is best** (MAE: 17.71m)

---

## 🤔 Which Features Actually Matter?

This is the key question! Current feature count is **~100+ features per station**. Let's break down:

| Feature Type | Count | Rationale |
|-------------|-------|-----------|
| Time sums (X,Y,Z × 3 windows) | 9 | Core physics |
| TFA (3 windows) | 3 | Orientation-independent |
| Decay ratios (X,Y,Z + TFA) | 4 | **Most important** |
| **Residuals** (16 × base features) | ~16 | Anomaly isolation |
| **Gradients** (16 × base features) | ~16 | Spatial derivatives |
| Anomaly metrics | 2 | Global context |
| Config metadata | 2 | Survey geometry |
| **Total** | **~50-100** | Depends on pivot |

---

## 🧪 Feature Importance Hypothesis

**My scientific hypothesis (to test):**

**Most Important (expect >80% of performance):**
1. **Residuals of late_sum_Z** - vertical component, background-removed
2. **Decay ratios (late/early)** - physics-based conductor indicator
3. **Residuals of TFA_late** - total anomaly strength
4. **Spatial gradients of residuals** - edge detection

**Moderately Important:**
5. Mid-time features - transition zone information
6. X-component residuals - inline sensitivity
7. Configuration offset - geometry correction

**Possibly Redundant:**
8. Raw (non-residual) features - background dominates
9. Y-component features - less sensitivity in your geometry
10. Anomaly peak/energy - may be redundant with CNN's global pooling

---

## 🔬 Testing Feature Importance

I'll create scripts to test:

1. **Feature Ablation**: Remove feature groups, measure performance drop
2. **Minimal Feature Set**: Start with essentials, add incrementally
3. **Feature Importance**: Train with different feature combinations

**Ablation Study Plan:**

| Test | Features Used | Expected MAE | Hypothesis |
|------|--------------|--------------|------------|
| **Full** | All ~100 features | 18.79m | Baseline |
| **Minimal** | late_sum_Z_residual only | ~25m? | Single best feature |
| **Physics-Core** | Decay ratios + Z residuals | ~20m? | Core physics |
| **No Residuals** | Raw features only | ~35m? | Background overwhelms |
| **No Gradients** | Remove all gradients | ~20m? | Less localization |
| **Late-Time Only** | Only late channels | ~22m? | Loses early info |

---

## 💡 Expected Results

**If the model is learning real physics:**
- Decay ratios should be critical
- Residuals should be critical (anomaly isolation)
- Z-component should dominate X, Y
- Configuration metadata helps but isn't essential

**If the model is just memorizing:**
- All features equally important
- Removing any feature causes collapse
- Anomaly metrics (peak, energy) would be most important

---

## 📊 Tools Being Created

1. **`feature_importance_analysis.py`** - Test different feature combinations
2. **`feature_ablation_study.py`** - Systematic removal of feature groups
3. **`minimal_features.py`** - Stripped-back version with only essentials

These will help answer:
- Which features are redundant?
- Does the model learn physics or memorize?
- Can we simplify for better interpretability?

---

## 🎯 Recommendation

Start with **feature ablation** to identify:
1. Critical features (performance drops >20% when removed)
2. Redundant features (performance drops <5% when removed)
3. Minimal feature set that achieves ~90% of full performance

This will give you **scientific confidence** that the model is learning meaningful geophysical patterns, not just overfitting to noise.

---

## References

Standard TEM processing techniques used:
- Time-domain windowing (Fitterman & Stewart, 1986)
- Decay ratio analysis (Barongo & Palacky, 1991)
- Regional-residual separation (Nabighian et al., 2005)
- Multi-component analysis (Smith & West, 1989)

These are **established geophysical methods**, not arbitrary ML features!
