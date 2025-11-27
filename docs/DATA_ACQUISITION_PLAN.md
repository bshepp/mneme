# Bioelectric Data Acquisition Plan

A comprehensive plan for acquiring and analyzing bioelectric time series data to validate Mneme's analysis capabilities before approaching academic collaborators.

---

## Part 1: Publicly Available Datasets

### Tier 1: Immediately Available (PhysioNet)

PhysioNet (physionet.org) provides free access to research-grade physiological data. All datasets can be downloaded using the `wfdb` Python library (already installed).

| Dataset | Type | Size | Sampling | Why It's Useful |
|---------|------|------|----------|-----------------|
| **MIT-BIH Arrhythmia** | ECG | 48 records × 30 min | 360 Hz | ✅ Already tested! Classic benchmark |
| **MIT-BIH Normal Sinus Rhythm** | ECG | 18 records × 24 hr | 128 Hz | Long-term HRV analysis |
| **Apnea-ECG** | ECG + breathing | 70 records × 7-10 hr | 100 Hz | Sleep state dynamics |
| **Sleep-EDF** | EEG/EOG/EMG | 197 records | 100 Hz | Brain state transitions |
| **CHB-MIT Scalp EEG** | EEG | 23 subjects, seizures | 256 Hz | Epileptic attractor dynamics |
| **MIMIC-III** | Multi-modal ICU | >40k patients | Varies | Complex physiological systems |
| **Gait in Aging and Disease** | Gait dynamics | 15 subjects | ~300 Hz | Motor control attractors |

**Download command:**
```python
import wfdb
record = wfdb.rdrecord('record_name', pn_dir='database_name')
```

### Tier 2: Other Public Repositories

| Source | Data Type | Access |
|--------|-----------|--------|
| **OpenNeuro** (openneuro.org) | EEG, fMRI | Free, BIDS format |
| **BNCI Horizon 2020** | BCI datasets | Free registration |
| **Kaggle** | Various competitions | Free account |
| **UCI ML Repository** | EEG, ECG classics | Direct download |
| **Allen Brain Observatory** | Calcium imaging | API access |

### Tier 3: Specialized Biological Data

| Dataset | Organism | Data Type | Source |
|---------|----------|-----------|--------|
| **C. elegans connectome** | Nematode | Neural activity | OpenWorm |
| **Drosophila whole-brain** | Fruit fly | Calcium imaging | Janelia |
| **Zebrafish brain imaging** | Fish | Voltage/calcium | Various labs |

**Note:** Planarian bioelectric data is NOT readily available in public repositories. This is why validation on other biological systems is important first.

---

## Part 2: DIY Data Collection (From Yourself)

### Option A: Consumer EEG (~$200-500)

**Recommended: Muse 2 or Muse S**
- **Price:** $250-400
- **Channels:** 4 EEG + 2 reference
- **Sampling:** 256 Hz
- **Data access:** Via Mind Monitor app → CSV export
- **Pros:** Easy setup, no expertise needed, comfortable
- **Cons:** Limited channels, consumer-grade

**What you can measure:**
- Alpha waves (relaxation)
- Beta waves (focus)
- Meditation states
- Sleep stages (Muse S)

**Analysis potential:**
```python
# Example: Analyze your own meditation data
from mneme.core import compute_lyapunov_spectrum
import numpy as np

# Load exported CSV from Mind Monitor
eeg_data = np.loadtxt('my_meditation_session.csv', delimiter=',')

# Compute Lyapunov spectrum for each channel
for channel in range(4):
    spectrum = compute_lyapunov_spectrum(eeg_data[:, channel], dt=1/256)
    print(f"Channel {channel}: λ₁ = {spectrum[0]:.4f}")
```

### Option B: Research-Grade EEG (~$500-2000)

**OpenBCI Cyton Board**
- **Price:** $500 (8-channel) to $950 (16-channel)
- **Channels:** 8-16 EEG
- **Sampling:** Up to 250 Hz (can be modified)
- **Data access:** Direct streaming, full raw data
- **Pros:** Research-grade, open-source, customizable
- **Cons:** Requires setup expertise, electrode gel

**OpenBCI Ganglion**
- **Price:** $200
- **Channels:** 4
- **Good for:** EMG, ECG, simple EEG

### Option C: Simple Bioelectric Sensors (~$20-100)

**Arduino + AD8232 (ECG)**
- **Price:** ~$25 total
- **What you need:**
  - Arduino Uno (~$15)
  - AD8232 ECG module (~$10)
  - 3 electrode pads (~$5)
- **Sampling:** Up to 500 Hz
- **Quality:** Surprisingly good for research

**Pulse Sensor (PPG)**
- **Price:** ~$25
- **Measures:** Heart rate via optical sensing
- **Use:** HRV analysis without electrodes

**GSR/EDA Sensor**
- **Price:** ~$15-30
- **Measures:** Skin conductance (stress, arousal)
- **Use:** Autonomic nervous system dynamics

### Option D: Combined Approach (Recommended)

**Starter Kit (~$100):**
1. Arduino Uno R3 ($15)
2. AD8232 ECG sensor ($10)
3. Pulse sensor ($25)
4. GSR sensor ($15)
5. Electrodes & leads ($15)
6. Breadboard & wires ($10)

**This gives you:**
- ECG (heart electrical activity)
- PPG (heart rate via light)
- GSR (skin conductance/stress)

All can be analyzed with Lyapunov spectrum!

---

## Part 3: Hardware Comparison Matrix

| Device | Price | Channels | Sample Rate | Ease of Use | Data Quality | Best For |
|--------|-------|----------|-------------|-------------|--------------|----------|
| **Muse 2** | $250 | 4 EEG | 256 Hz | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Quick EEG experiments |
| **OpenBCI Cyton** | $500 | 8 EEG | 250 Hz | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Serious research |
| **Arduino ECG** | $25 | 1 ECG | 500 Hz | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | HRV analysis |
| **Polar H10** | $90 | 1 ECG | 130 Hz | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Long-term HRV |
| **BITalino** | $200 | Multi | 1000 Hz | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Education/research |

---

## Part 4: Recommended Analysis Pipeline

### Phase 1: Public Data Validation (Week 1-2)

```bash
# Download and analyze multiple PhysioNet datasets
python -c "
import wfdb
from mneme.core import compute_lyapunov_spectrum, kaplan_yorke_dimension

# 1. Sleep EEG - brain state transitions
record = wfdb.rdrecord('slp01a', pn_dir='slpdb', sampto=100000)
# Analyze...

# 2. Gait dynamics - motor control
record = wfdb.rdrecord('ga02', pn_dir='gaitdb')
# Analyze...

# 3. More ECG for HRV benchmarking
record = wfdb.rdrecord('16265', pn_dir='nsrdb', sampto=100000)
# Analyze...
"
```

**Goal:** Build a table of Lyapunov spectra across different physiological systems.

### Phase 2: Own Body Data (Week 3-4)

**Experiment ideas:**
1. **Resting vs. Active HRV** - Record ECG during rest, then during mental arithmetic
2. **Meditation dynamics** - Track EEG changes during meditation session
3. **Sleep transitions** - Record overnight data, analyze state changes
4. **Stress response** - GSR during calm vs. stressful tasks

### Phase 3: Cross-System Comparison (Week 5+)

| System | Expected λ₁ | Expected D_KY | Attractor Type |
|--------|-------------|---------------|----------------|
| Resting HRV | +0.1 to +0.2 | 2-3 | Strange |
| Active HRV | Higher? | Higher? | Strange |
| Sleep EEG | State-dependent | Varies by stage | Multiple |
| Meditation | Lower chaos? | Lower dimension? | ? |

---

## Part 5: Concrete Next Steps

### Immediate (Today)

1. ✅ PhysioNet ECG analyzed (already done!)
2. Download 2-3 more PhysioNet datasets
3. Run Lyapunov analysis on each
4. Compare results to published literature

### Short-term (This Week)

1. Download Sleep-EDF dataset
2. Analyze EEG state transitions
3. Download gait dynamics data
4. Build comparison table

### Medium-term (This Month)

**If you want to collect your own data:**

**Budget Option ($50):**
- Arduino + AD8232 ECG kit
- Record your own HRV during different activities
- Compare to PhysioNet benchmarks

**Better Option ($250):**
- Muse 2 headband
- Record EEG during meditation/focus/rest
- Analyze brain state dynamics

**Research Option ($500+):**
- OpenBCI Cyton
- Full research-grade EEG/ECG/EMG capability
- Publication-quality data

---

## Part 6: What This Proves to Academics

When you approach the Levin Lab or other researchers, you can say:

> "I've built a Lyapunov spectrum analysis pipeline and validated it on:
> - PhysioNet ECG data (HRV chaos matches literature)
> - Sleep EEG data (state transitions detected)
> - My own bioelectric recordings
> 
> I'd like to apply these same methods to planarian bioelectric data to characterize attractor dynamics during regeneration."

This demonstrates:
1. **Working tools** - not just theory
2. **Validation** - results match known literature
3. **Personal investment** - you've recorded your own data
4. **Clear application** - specific hypothesis for their data

---

## Appendix: Python Code for Multi-Dataset Analysis

```python
"""
Comprehensive bioelectric analysis script.
Run on multiple PhysioNet datasets to build validation table.
"""

import numpy as np
import wfdb
from mneme.core import compute_lyapunov_spectrum, kaplan_yorke_dimension
from mneme.core.attractors import classify_attractor_by_lyapunov, embed_trajectory

DATASETS = [
    ('100', 'mitdb', 'ECG Arrhythmia'),
    ('slp01a', 'slpdb', 'Sleep Polysomnography'),
    ('ga02', 'gaitdb', 'Gait Dynamics'),
    ('chf01', 'chfdb', 'Congestive Heart Failure'),
]

results = []

for record_name, db_name, description in DATASETS:
    try:
        record = wfdb.rdrecord(record_name, pn_dir=db_name, sampto=50000)
        signal = record.p_signal[:, 0]
        fs = record.fs
        
        # Embed and analyze
        trajectory = embed_trajectory(signal, embedding_dimension=4, time_delay=int(fs/10))
        spectrum = compute_lyapunov_spectrum(trajectory, dt=1/fs)
        
        results.append({
            'dataset': description,
            'lambda_1': spectrum[0],
            'd_ky': kaplan_yorke_dimension(spectrum),
            'type': classify_attractor_by_lyapunov(spectrum)
        })
        print(f"✓ {description}: λ₁={spectrum[0]:.4f}, D_KY={kaplan_yorke_dimension(spectrum):.2f}")
    except Exception as e:
        print(f"✗ {description}: {e}")
```

---

## Summary

| Phase | Data Source | Cost | Time | Value |
|-------|-------------|------|------|-------|
| **Validation** | PhysioNet | Free | 1 week | Proves methods work |
| **Expansion** | OpenNeuro, etc. | Free | 2 weeks | Broader validation |
| **Personal** | Arduino ECG | $50 | 1 week | Hands-on experience |
| **Advanced** | Muse/OpenBCI | $250-500 | 2 weeks | Publication-ready data |

**Recommended path:** Start with PhysioNet (free), then Arduino ECG ($50), then approach academics with validated results.

