# Kneeva OA Triage System — Comprehensive Build Guide

> **Version:** 1.0  
> **Last Updated:** September 2026  
> **Classification:** Internal Engineering Reference  

---

## Table of Contents

1. [System Overview & Architecture Philosophy](#1-system-overview--architecture-philosophy)
2. [Hardware & Sensor Specifications](#2-hardware--sensor-specifications)
3. [Data Collection Protocol](#3-data-collection-protocol)
4. [Layer 1 — Input & Preprocessing](#4-layer-1--input--preprocessing)
5. [Layer 2 — Per-Modality Feature Extraction](#5-layer-2--per-modality-feature-extraction)
6. [Layer 3 — Contextual Meta-Inputs](#6-layer-3--contextual-meta-inputs)
7. [The Architectural Divide — Frozen vs. Trainable](#7-the-architectural-divide--frozen-vs-trainable)
8. [Layer 4 — Fusion & Recalibration](#8-layer-4--fusion--recalibration)
9. [Models to Train — Complete Specification](#9-models-to-train--complete-specification)
10. [Datasets — Authenticated Sources & Usage](#10-datasets--authenticated-sources--usage)
11. [Training Pipeline & Methodology](#11-training-pipeline--methodology)
12. [Evaluation Metrics & Validation](#12-evaluation-metrics--validation)
13. [NER Adaptability — Platt Scaling Deep Dive](#13-ner-adaptability--platt-scaling-deep-dive)
14. [Deployment Architecture](#14-deployment-architecture)
15. [Appendices](#15-appendices)

---

## 1. System Overview & Architecture Philosophy

### 1.1 What Is Kneeva?

Kneeva is a **multi-modal, late-fusion machine learning pipeline** designed to triage knee osteoarthritis (OA) risk in resource-constrained clinical environments. It ingests data from five distinct sensor modalities plus patient questionnaires, extracts clinically validated features through deterministic signal processing, and fuses them in a lightweight gradient-boosted model to produce:

1. **OA Risk Score** — a calibrated probability (0.0–1.0) indicating the likelihood of clinically significant knee OA.
2. **Differential Triage Signal** — a flag indicating whether the patient's symptom profile suggests an alternative pathology (e.g., meniscal tear, ligament injury, referred pain from hip OA, inflammatory arthritis).

### 1.2 Why Late Fusion?

Late fusion means each sensor modality is processed **independently** through its own deterministic pipeline before features are combined at the very end. This design offers:

| Advantage | Explanation |
|-----------|-------------|
| **Modular Degradation** | If one sensor fails or is unavailable, the system can still produce a (degraded) risk score from remaining modalities. |
| **Interpretability** | Each feature entering the fusion model has a clear clinical meaning (e.g., "ROM deficit = 15°"), making the system auditable by clinicians. |
| **Reduced Training Data Requirement** | Only the small fusion head needs labeled data. The feature extractors are algorithmic and require no training. |
| **Regulatory Simplicity** | Frozen, deterministic extractors are easier to validate for medical device certification (IEC 62304, FDA SaMD). |

### 1.3 Architectural Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYER 1: INPUT & PREPROCESSING                   │
│  IMU → GaitPy/AHRS                                                 │
│  Acoustic → DSP Scripts                                             │
│  sEMG → NeuroKit2                                                   │
│  Dynamometer → Peak Detection                                      │
│  Questionnaire → Scoring Formula                                    │
└──────────────┬──────────────────────────────────────────────────────┘
               │
    ┌──────────┴──────────┐
    ▼                     ▼
┌────────────┐    ┌────────────────┐    ┌──────────────────────┐
│  TIER A:   │    │    TIER B:      │    │  TIER C:             │
│  OA Signals│    │  Differential  │    │  Contextual Meta     │
│            │    │  Signals       │    │  (BMI, Age, Sex...)  │
│  • ROM     │    │  • Gait Asym.  │    │                      │
│  • Crepitus│    │  • Neuromusc.  │    │  ─ ─ ─ ─ (bypass) ─ ┤
│  • Co-contr│    │  • Gen. Weak.  │    │                      │
│  • Str.Def.│    │  • Injury Hx   │    └──────────┬───────────┘
└──────┬─────┘    └──────┬─────────┘               │
       │                 │                         │
═══════╪═════════════════╪═════════════════════════╪═══ FREEZE LINE ═══
       │                 │                         │
       ▼                 ▼                         ▼
  ┌──────────────────────────────────────────────────────┐
  │        FUSION LAYER (CatBoost / XGBoost)             │
  │                     │                                │
  │                     ▼                                │
  │        Platt Scaling Recalibration (NER)             │
  │                     │                                │
  │           ┌─────────┴─────────┐                      │
  │           ▼                   ▼                      │
  │      OA Risk Score    Differential Signal            │
  └──────────────────────────────────────────────────────┘
```

---

## 2. Hardware & Sensor Specifications

### 2.1 IMU (Inertial Measurement Unit)

The IMU captures tri-axial acceleration and angular velocity, enabling reconstruction of knee joint kinematics.

#### Recommended Hardware

| Tier | Device | Cost | Specs | Notes |
|------|--------|------|-------|-------|
| **Budget/DIY** | MPU-6050 (with Arduino/ESP32) | $3–5 per unit | 6-axis (3-accel + 3-gyro), 16-bit ADC, up to 1 kHz | Adequate for research prototyping. Requires custom PCB for reliability. |
| **Mid-Range** | BNO055 (Bosch) | $15–25 per unit | 9-axis (accel + gyro + mag), onboard sensor fusion, quaternion output | Built-in AHRS reduces preprocessing burden. |
| **Clinical-Grade** | Xsens DOT / APDM Opal | $500–1,500 per unit | 9-axis, calibrated, wireless BLE, ±2000°/s gyro range, 120 Hz default | Gold standard for biomechanics research. Battery life ~6 hrs. |

#### Placement Protocol

- **Location:** Strapped to the lateral aspect of the proximal tibia (shank/lower leg), approximately 10 cm below the tibial plateau. Aligned parallel to the tibial axis.
- **Why the shank?:** Gait tracking algorithms (like GaitPy) rely heavily on the vertical acceleration of the lower leg to identify initial contact (heel strike) and final contact (toe-off).
- **Fixation:** Elastic Velcro straps with anti-slip silicone lining. Units must be tightly secured to minimize soft-tissue artifact (skin motion artifact is the primary noise source in IMU-based gait analysis).

#### Raw Data Format

```
Sampling Rate: 100–200 Hz (minimum 100 Hz for gait analysis)
Channels: 6 per unit (ax, ay, az, gx, gy, gz)
Output Format: CSV or binary log file

Example Row:
timestamp_ms, ax, ay, az, gx, gy, gz
1694736000000, 0.12, -9.78, 0.34, 1.2, -0.5, 0.8
1694736000005, 0.13, -9.77, 0.35, 1.1, -0.6, 0.9
```

#### Data Volume Estimate

| Task | Duration | Data Points (at 200 Hz) | File Size (approx.) |
|------|----------|------------------------|---------------------|
| Sit-to-stand (5 reps) | ~60 sec | 12,000 rows × 6 channels | ~0.75 MB |
| 6-minute walk test | 360 sec | 72,000 rows × 6 channels | ~4.5 MB |
| Stair climbing (10 steps) | ~30 sec | 6,000 rows × 6 channels | ~0.4 MB |
| **Total per leg** | ~8 min | ~90,000 rows | **~5.6 MB** |

---

### 2.2 Acoustic Sensor (Crepitus Microphone)

Captures vibroacoustic emissions from the knee joint during movement — grinding, clicking, popping sounds collectively termed **crepitus**, which are strongly correlated with cartilage degradation.

#### Recommended Hardware

| Tier | Device | Cost | Specs | Notes |
|------|--------|------|-------|-------|
| **✅ Purchased** | **Piezo Vibration Sensor – Large** (PVDF film) + **MCP6002 Op-Amp** for signal conditioning | ₹1,379 + ₹138 (2× MCP6002) | PVDF piezoelectric polymer, 30×13 mm, frequency response up to 90 Hz (+3 dB), output up to ±90V raw (stepped down via resistor divider for ADC). MCP6002: Dual-channel, 1 MHz bandwidth, 0.6 V/µs slew rate, 1.8–6V supply, DIP-8 | PVDF film must be coupled with ultrasound gel for acoustic impedance matching. MCP6002 used as a buffer/amplifier stage to condition the piezo output for microcontroller ADC input. |
| **Mid-Range** | INMP441 MEMS microphone (with ESP32 I2S) | $3–8 | Digital output, 60 Hz – 15 kHz, SNR 61 dB | Compact, integrable into a wearable knee brace. |
| **Clinical-Grade** | Thinklabs One Digital Stethoscope | $400–500 | 20 Hz – 2 kHz (tunable), 50x amplification, 3.5 mm audio out | Designed for body sound capture. Excellent low-frequency response. |

> **⚠️ Note on Purchased Piezo Sensor:** The PVDF sensor's native frequency response rolls off at ~90 Hz (+3 dB), which covers the low-frequency grinding component of crepitus but may miss higher-frequency clicking events (200–1000 Hz). The MCP6002 op-amp circuit should include a gain stage to amplify weak crepitus signals before ADC digitization. A voltage divider / clamping circuit is also needed to bring the raw ±90V piezo output down to 0–3.3V ADC range. Consider upgrading to the INMP441 MEMS mic in future iterations for broader frequency coverage.

#### Placement Protocol

- **Location:** Directly over the **patella** (kneecap), centered on the patellar surface.
- **Coupling:** Thin layer of ultrasound transmission gel between sensor and skin to ensure acoustic impedance matching.
- **Fixation:** Medical-grade adhesive ring or elastic bandage. The microphone must remain stationary — any sensor sliding generates friction artifacts indistinguishable from crepitus.
- **Environment:** Recording must occur in a **quiet room** (ambient noise < 40 dB). No talking during recording.

#### Raw Data Format

```
Sampling Rate: 16,000 Hz (minimum) — 44,100 Hz (preferred)
Bit Depth: 16-bit PCM
Channels: 1 (mono)
Output Format: .WAV (uncompressed)

Duration per recording: 30–60 seconds (5–10 slow flexion-extension cycles)
File Size: ~1.3 MB per 30 sec at 44.1 kHz / 16-bit
```

#### Recording Protocol

1. Patient seated with leg hanging freely (knee at ~90° flexion).
2. Clinician applies acoustic gel and attaches microphone to patella.
3. Patient performs **slow, controlled** knee extension from 90° to full extension (~0°), then back to 90°.
4. Speed: approximately **5 seconds per full cycle** (extension + flexion).
5. Repeat for **10 cycles** = ~50 seconds of recording.
6. **Critical:** Patient must not speak. Clinician must not touch the leg during recording.

---

### 2.3 sEMG (Surface Electromyography)

Captures the electrical activity of muscles surrounding the knee joint, enabling quantification of muscle activation patterns, co-contraction ratios, and neuromuscular control deficits.

#### Recommended Hardware

| Tier | Device | Cost | Specs | Notes |
|------|--------|------|-------|-------|
| **✅ Purchased** | **Advancer Technologies EMG Muscle Sensor V3.0** (with cable & electrodes) | ₹940 per channel | sEMG sensor board, 1" × 1" form factor, analog output (0–Vs), onboard instrumentation amplifier with filtering & rectification, adjustable gain potentiometer, 3.5 mm electrode jack, electrode pad diameter 52 mm. Requires ±3.5V to ±9V dual supply (or single 3.3–5V on some variants). | **Single-channel** — for 4-muscle coverage (VM, VL, RF, BF), you will need **4 units** or time-multiplex recordings. Currently purchased: 1 unit. Includes cable and Ag/AgCl electrode pads. |
| **Mid-Range** | BITalino (r)evolution Board | $200–400 | 6-channel EMG, 10-bit ADC, 1 kHz sampling, BLE | Excellent for prototyping. Open-source API (Python). |
| **Clinical-Grade** | Delsys Trigno Avanti / Noraxon Ultium | $5,000–15,000 | 16-channel, 2 kHz sampling, wireless, <1 µV baseline noise | Research gold standard. FDA-listed for clinical research. |

> **⚠️ Note on Purchased EMG Sensor:** The V3.0 is a single-channel board. The guide's protocol requires simultaneous recording from 4 muscles (VM, VL, RF, BF). With 1 unit, you must record each muscle sequentially during separate MVC and task repetitions, which increases session time and prevents simultaneous co-contraction measurement. **Recommendation:** Purchase 3 additional units (₹2,820 total) for true 4-channel simultaneous recording, or adopt a time-multiplexed protocol as an interim measure.

#### Electrode Placement (SENIAM Guidelines)

Follow the **SENIAM (Surface EMG for Non-Invasive Assessment of Muscles)** electrode placement recommendations:

| Channel | Muscle | Placement | Clinical Rationale |
|---------|--------|-----------|-------------------|
| CH1 | **Vastus Medialis (VM)** | ~4 cm superior and medial to the superomedial border of the patella, oriented at 55° from the femoral axis | Primary stabilizer of the patella; preferentially weakens in OA |
| CH2 | **Vastus Lateralis (VL)** | ~8 cm superior and lateral to the superolateral border of the patella | Lateral quadriceps; co-contraction with VM indicates patellar instability |
| CH3 | **Rectus Femoris (RF)** | Midpoint between ASIS (anterior superior iliac spine) and superior patellar border | Primary hip flexor and knee extensor; compensatory overactivation in OA |
| CH4 | **Biceps Femoris (BF)** | Midpoint between ischial tuberosity and lateral epicondyle of the tibia | Primary hamstring; co-contraction ratio (quad:ham) is a key OA biomarker |

#### Skin Preparation

1. **Shave** hair at electrode sites (if necessary).
2. **Abrade** skin lightly with fine sandpaper or abrasive gel to reduce impedance.
3. **Clean** with isopropyl alcohol wipe. Allow to dry completely.
4. **Apply** Ag/AgCl disposable gel electrodes with inter-electrode distance of **20 mm**.
5. **Target impedance:** < 10 kΩ (verify with impedance meter if available).

#### Raw Data Format

```
Sampling Rate: 1,000 Hz (minimum) — 2,000 Hz (preferred)
Resolution: 16-bit ADC minimum
Channels: 4 (VM, VL, RF, BF)
Units: Microvolts (µV)
Output Format: CSV

Example Row:
timestamp_ms, VM_uV, VL_uV, RF_uV, BF_uV
1694736000000, 45.2, 38.7, 22.1, 15.8
1694736000001, 47.8, 40.1, 23.5, 16.2
```

#### Data Volume Estimate

| Task | Duration | Data Points (at 1 kHz) | File Size |
|------|----------|----------------------|-----------|
| Isometric MVC (3 reps × 5 sec each) | ~30 sec | 30,000 rows × 4 ch | ~1.2 MB |
| Walking (6-min walk test) | 360 sec | 360,000 rows × 4 ch | ~14 MB |
| Sit-to-stand (5 reps) | ~60 sec | 60,000 rows × 4 ch | ~2.4 MB |
| **Total per patient** | ~8 min | ~450,000 rows | **~17.6 MB** |

---

### 2.4 Dynamometer

Measures peak isometric force production of the knee extensors and flexors, enabling quantification of absolute strength and strength deficit ratios.

#### Recommended Hardware

| Tier | Device | Cost | Specs | Notes |
|------|--------|------|-------|-------|
| **Budget/DIY** | Load Cell (50 kg) + HX711 ADC + Arduino | $10–20 | 24-bit ADC, 80 Hz sampling, ±0.05% linearity | Requires custom jig for consistent positioning. Adequate for isometric force measurement. |
| **Mid-Range** | MicroFET2 Handheld Dynamometer | $1,200–1,800 | 0–135 kg range, ±1% accuracy, digital display, USB data export | Clinical standard for manual muscle testing. Portable. |
| **Clinical-Grade** | Biodex System 4 Isokinetic Dynamometer | $50,000+ | Isokinetic + isometric, torque/angle/velocity curves, built-in software | Research gold standard. Not portable — requires dedicated space. |

#### Measurement Protocol

**For the Budget/Mid-Range Setup (Isometric Testing):**

1. **Position:** Patient seated on a standard chair with hip at 90° and knee at **60° flexion** (measured with goniometer). The ankle is positioned against the dynamometer pad/load cell.
2. **Stabilization:** Strap the thigh firmly to the chair to prevent hip compensation.

**Tests performed:**

| Test | Description | Repetitions | Rest Between |
|------|-------------|-------------|-------------|
| **Knee Extension MVC** | Patient pushes ankle anteriorly against the pad as hard as possible | 3 reps × 5 seconds each | 30 seconds |
| **Knee Flexion MVC** | Patient pulls ankle posteriorly against the pad | 3 reps × 5 seconds each | 30 seconds |

3. **Verbal encouragement** must be standardized: "Push/pull as hard as you can! Keep going! Harder!" throughout each 5-second effort.
4. **Warm-up:** 2 submaximal practice contractions before recorded trials.

#### Raw Data Format

```
For load cell setup:
  Sampling Rate: 80 Hz
  Output: Force in Newtons (after calibration)
  Format: CSV → [timestamp_ms, force_N]

For handheld dynamometer:
  Output: Single peak force value (Newtons) per trial
  Format: Manual entry or USB export → [trial_id, direction, peak_force_N]
```

#### Derived Metrics

| Metric | Formula | Clinical Meaning |
|--------|---------|-----------------|
| **Peak Extension Force** | max(force_N) across 3 trials | Quadriceps maximal strength |
| **Peak Flexion Force** | max(force_N) across 3 trials | Hamstring maximal strength |
| **Extension Deficit (%)** | `(1 - affected_ext / unaffected_ext) × 100` | Bilateral strength asymmetry |
| **H:Q Ratio** | `peak_flexion / peak_extension` | Hamstring-to-quadriceps ratio; normal ~0.5–0.8 |

---

### 2.5 Questionnaire

Captures patient-reported outcomes (PROs) and subjective history that cannot be measured by sensors.

#### Instruments Used

| Instrument | Items | Scoring | What It Measures |
|------------|-------|---------|-----------------|
| **WOMAC** (Western Ontario & McMaster Universities OA Index) | 24 items across 3 subscales | Likert 0–4 per item; Pain (0–20), Stiffness (0–8), Function (0–68). Normalized to 0–100. | OA-specific pain, stiffness, and physical function |
| **KOOS** (Knee Injury & OA Outcome Score) | 42 items across 5 subscales | Likert 0–4 per item; each subscale normalized to 0–100 (100 = no problems) | Pain, Symptoms, ADL, Sport/Recreation, Quality of Life |
| **VAS Pain Scale** | 1 item | 0–100 mm visual analog scale | Current knee pain intensity |
| **Injury History Form** | ~10 structured items | Binary (yes/no) and categorical | Previous knee injuries, surgeries, diagnoses |
| **Activity Level** | Tegner Activity Scale | 0–10 ordinal scale | Current physical activity level |

#### Scoring Formulas

**WOMAC Scoring (Likert version):**

```
Pain subscale:      Sum of items 1–5       → range 0–20
Stiffness subscale: Sum of items 6–7       → range 0–8
Function subscale:  Sum of items 8–24      → range 0–68
Total WOMAC:        Sum of all 24 items    → range 0–96

Normalized:  (raw_score / max_possible) × 100
Higher score = worse symptoms
```

**KOOS Scoring:**

```
Each subscale:
  Normalized_score = 100 - (sum_of_items / (num_items × 4)) × 100

Subscales:
  Pain:           9 items
  Symptoms:       7 items
  ADL:           17 items
  Sport/Rec:      5 items
  QoL:            4 items

Higher score = fewer problems (opposite of WOMAC)
```

#### Data Format

```json
{
  "patient_id": "KNV-NER-0042",
  "timestamp": "2026-09-15T10:30:00+05:30",
  "womac": {
    "pain": [3, 2, 2, 1, 3],
    "stiffness": [2, 1],
    "function": [2, 1, 3, 2, 1, 1, 2, 3, 2, 1, 0, 1, 2, 1, 2, 1, 3]
  },
  "koos": { ... },
  "vas_pain_mm": 62,
  "injury_history": {
    "previous_knee_injury": true,
    "type": "meniscal_tear",
    "year": 2019,
    "surgery": true,
    "surgery_type": "arthroscopic_partial_meniscectomy"
  },
  "tegner_activity_level": 4
}
```

---

## 3. Data Collection Protocol

### 3.1 Visit Overview

Each patient visit follows a **standardized 45-minute protocol**:

```
Time    Activity                            Sensors Active
─────   ──────────────────────────────────  ──────────────────
0:00    Consent, demographics, BMI          None (tablet entry)
0:05    Questionnaires (WOMAC, KOOS, etc.)  None (tablet entry)
0:15    Skin prep, sensor attachment        None
0:20    MVC calibration (sEMG normalization) sEMG + Dynamometer
0:25    Seated knee flexion-extension        IMU + Acoustic + sEMG
0:30    Sit-to-stand (5 reps)               IMU + sEMG
0:33    6-minute walk test                  IMU + sEMG
0:40    Isometric strength testing          Dynamometer + sEMG
0:43    Sensor removal, data verification   None
0:45    End
```

### 3.2 Bilateral Assessment

**Critical:** All tests are performed on **both knees** (affected and unaffected). This enables:

- Within-subject normalization (affected vs. contralateral)
- Calculation of asymmetry indices
- Reduced confounding from inter-individual variability

### 3.3 Data Storage — Offline-First Paradigm

```
Collection Device (Tablet/Laptop)
    │
    ├── /data/raw/{patient_id}/{session_date}/
    │       ├── imu_shank.csv
    │       ├── acoustic_knee.wav
    │       ├── semg_4ch.csv
    │       ├── flex_sensor.csv
    │       ├── dynamometer.csv
    │       ├── questionnaire.json
    │       └── metadata.json    ← device IDs, calibration, clinician notes
    │
    ├── [Local SQLite DB for indexing]
    │
    └── [Sync to cloud when connectivity available]
            │
            ▼
        Cloud Storage (GCS / S3 / Azure Blob)
            │
            ▼
        Preprocessing Server
```

**Why offline-first?**
- NER (North-Eastern Region) deployment sites may have intermittent internet.
- Raw sensor data must be preserved at full fidelity — no lossy compression or edge quantization.
- Local storage acts as the source of truth; cloud is a replicated backup.

### 3.4 Data Integrity Checks

Before a session is marked "complete," the following automated checks run on the collection device:

| Check | Rule | Action on Failure |
|-------|------|-------------------|
| IMU completeness | Shank file exists, >10,000 rows | Block session completion |
| Acoustic quality | WAV file >10 seconds, peak amplitude > noise floor × 3 | Flag for manual review |
| sEMG baseline noise | Baseline RMS < 20 µV (measured during 5-sec rest) | Re-prep electrodes |
| Dynamometer trials | ≥3 valid extension trials, ≥3 valid flexion trials | Repeat failed trials |
| Questionnaire | No missing required fields | Block submission |

---

## 4. Layer 1 — Input & Preprocessing

This layer transforms raw sensor signals into clean, segmented, and normalized data ready for feature extraction. **All processing in this layer is deterministic and algorithmic — no ML models are trained.**

### 4.1 IMU Preprocessing — GaitPy & AHRS

#### Step 1: Orientation Estimation (AHRS)

**Library:** `ahrs` (Python, open-source)  
**Algorithm:** Madgwick filter (preferred) or Mahony complementary filter

```python
import ahrs

# For 9-axis IMU (with magnetometer)
madgwick = ahrs.filters.Madgwick(
    gyr=gyroscope_data,    # shape: (N, 3) in rad/s
    acc=accelerometer_data, # shape: (N, 3) in m/s²
    mag=magnetometer_data,  # shape: (N, 3) in µT (if available)
    frequency=200.0         # sampling rate in Hz
)
quaternions = madgwick.Q  # shape: (N, 4) — orientation quaternions

# For 6-axis IMU (no magnetometer)
madgwick = ahrs.filters.Madgwick(
    gyr=gyroscope_data,
    acc=accelerometer_data,
    frequency=200.0
)
quaternions = madgwick.Q
```

**Purpose:** Converts raw accelerometer + gyroscope data into orientation quaternions, which are then used to calculate joint angles (knee flexion-extension angle).

#### Step 2: Joint Angle Calculation

```python
import numpy as np
from scipy.spatial.transform import Rotation

def calculate_knee_angle(q_thigh, q_shank):
    """
    Calculate knee flexion-extension angle from thigh and shank quaternions.
    
    Parameters:
        q_thigh: (N, 4) array of thigh orientation quaternions [w, x, y, z]
        q_shank: (N, 4) array of shank orientation quaternions [w, x, y, z]
    
    Returns:
        knee_angle: (N,) array of knee angles in degrees
    """
    r_thigh = Rotation.from_quat(q_thigh[:, [1, 2, 3, 0]])  # scipy uses [x,y,z,w]
    r_shank = Rotation.from_quat(q_shank[:, [1, 2, 3, 0]])
    
    # Relative rotation: shank relative to thigh
    r_knee = r_thigh.inv() * r_shank
    
    # Extract Euler angles (sagittal plane = flexion-extension)
    euler = r_knee.as_euler('ZXY', degrees=True)
    knee_angle = euler[:, 1]  # X-axis rotation = flexion-extension
    
    return knee_angle
```

#### Step 3: Gait Cycle Segmentation (GaitPy)

**Library:** `gaitpy` (Python, open-source, developed by Pfizer)

```python
from gaitpy import gaitpy

# GaitPy expects vertical acceleration from a lumbar/shank sensor
gp = gaitpy.Gaitpy(
    raw_data=vertical_accel_df,  # DataFrame with 'timestamps' and 'y' columns
    sample_rate=200.0,
    v_acc_col_name='y',
    ts_col_name='timestamps',
    v_acc_units='m/s^2',
    ts_units='ms'
)

# Detect gait events (heel strikes and toe-offs)
gait_events = gp.extract_features(
    ic_prom=15,      # Initial contact prominence threshold
    fc_prom=15,      # Final contact prominence threshold
)
```

**Output:** DataFrame of gait cycle events (initial contact, final contact timestamps) from which stride-level features are extracted.

#### Step 4: Signal Filtering

```python
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply zero-phase Butterworth bandpass filter."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

# IMU preprocessing filters
imu_filtered = bandpass_filter(raw_imu, lowcut=0.5, highcut=20.0, fs=200)
```

---

### 4.2 Acoustic Preprocessing — DSP Scripts

#### Step 1: Load & Validate Audio

```python
import librosa
import numpy as np

# Load WAV file
signal, sr = librosa.load('acoustic_knee.wav', sr=44100, mono=True)

# Validate
assert len(signal) / sr >= 10, "Recording too short (<10 seconds)"
assert np.max(np.abs(signal)) > 0.01, "Signal too quiet — check microphone coupling"
```

#### Step 2: Noise Reduction

```python
import noisereduce as nr

# Spectral gating noise reduction
# Use the first 0.5 seconds (before movement starts) as noise profile
noise_profile = signal[:int(0.5 * sr)]
signal_clean = nr.reduce_noise(
    y=signal,
    sr=sr,
    y_noise=noise_profile,
    stationary=True,
    prop_decrease=0.8
)
```

#### Step 3: Time-Frequency Representation

```python
import librosa

# Short-Time Fourier Transform (STFT)
stft = librosa.stft(signal_clean, n_fft=2048, hop_length=512)
magnitude = np.abs(stft)
phase = np.angle(stft)

# Mel-frequency spectrogram (for crepitus characterization)
mel_spec = librosa.feature.melspectrogram(
    y=signal_clean,
    sr=sr,
    n_mels=128,
    fmin=50,       # Crepitus typically 50–1000 Hz
    fmax=2000,
    n_fft=2048,
    hop_length=512
)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
```

#### Step 4: Crepitus Event Detection

```python
from scipy.signal import find_peaks

# Compute spectral flux (onset detection function)
spectral_flux = np.sqrt(np.mean(np.diff(magnitude, axis=1)**2, axis=0))

# Detect crepitus events as peaks in spectral flux
peaks, properties = find_peaks(
    spectral_flux,
    height=np.mean(spectral_flux) + 2 * np.std(spectral_flux),  # threshold
    distance=int(0.2 * sr / 512),  # minimum 200ms between events
    prominence=0.1
)

crepitus_count = len(peaks)
crepitus_rate = crepitus_count / (len(signal) / sr)  # events per second
```

---

### 4.3 sEMG Preprocessing — NeuroKit2

**Library:** `neurokit2` (Python, open-source)

#### Complete sEMG Preprocessing Pipeline

```python
import neurokit2 as nk
import pandas as pd

def preprocess_semg_channel(raw_signal, sampling_rate=1000):
    """
    Full sEMG preprocessing pipeline for a single channel.
    
    Steps:
        1. Bandpass filter (20–450 Hz) — remove motion artifact and high-freq noise
        2. Notch filter (50 Hz) — remove powerline interference
        3. Full-wave rectification
        4. Linear envelope (low-pass at 6 Hz)
    """
    # Step 1: Bandpass filter
    filtered = nk.emg_clean(raw_signal, sampling_rate=sampling_rate, method='biosppy')
    
    # Step 2: Amplitude extraction (rectification + smoothing)
    amplitude = nk.emg_amplitude(filtered)
    
    # Step 3: Activation detection (onset/offset)
    info = nk.emg_activation(amplitude, sampling_rate=sampling_rate, method='threshold')
    activation_signal = info[0]['EMG_Activity']  # binary: 0 = rest, 1 = active
    
    return {
        'filtered': filtered,
        'amplitude': amplitude,
        'activation': activation_signal,
        'onsets': info[1].get('EMG_Onsets', []),
        'offsets': info[1].get('EMG_Offsets', [])
    }

# Process all 4 channels
channels = ['VM', 'VL', 'RF', 'BF']
semg_processed = {}
for ch_name, ch_data in zip(channels, [vm_raw, vl_raw, rf_raw, bf_raw]):
    semg_processed[ch_name] = preprocess_semg_channel(ch_data, sampling_rate=1000)
```

#### MVC Normalization

All sEMG amplitudes must be normalized to the Maximum Voluntary Contraction (MVC) recorded during the calibration phase:

```python
def normalize_to_mvc(amplitude, mvc_value):
    """Normalize sEMG amplitude as percentage of MVC."""
    return (amplitude / mvc_value) * 100  # %MVC

# MVC values are the peak amplitude from the MVC calibration trials
mvc_values = {
    'VM': np.max(mvc_vm_amplitude),
    'VL': np.max(mvc_vl_amplitude),
    'RF': np.max(mvc_rf_amplitude),
    'BF': np.max(mvc_bf_amplitude)
}

for ch in channels:
    semg_processed[ch]['amplitude_mvc'] = normalize_to_mvc(
        semg_processed[ch]['amplitude'], mvc_values[ch]
    )
```

---

### 4.4 Dynamometer Preprocessing — Peak Detection

```python
from scipy.signal import find_peaks
import numpy as np

def extract_peak_force(force_signal, sampling_rate=80, n_trials=3):
    """
    Extract peak force values from an isometric dynamometer recording.
    
    The recording contains multiple MVC trials separated by rest periods.
    """
    # Smooth the signal to remove noise
    from scipy.ndimage import uniform_filter1d
    force_smooth = uniform_filter1d(force_signal, size=int(0.1 * sampling_rate))
    
    # Detect peaks (MVC peaks)
    peaks, properties = find_peaks(
        force_smooth,
        height=np.max(force_smooth) * 0.3,  # at least 30% of max
        distance=int(3 * sampling_rate),       # at least 3 sec apart
        prominence=10                           # at least 10N prominence
    )
    
    # Take the top n_trials peaks
    peak_forces = sorted(properties['peak_heights'], reverse=True)[:n_trials]
    
    return {
        'peak_force_N': np.max(peak_forces),
        'mean_peak_force_N': np.mean(peak_forces),
        'cv_percent': (np.std(peak_forces) / np.mean(peak_forces)) * 100
    }

extension_results = extract_peak_force(extension_force_data)
flexion_results = extract_peak_force(flexion_force_data)
```

---

### 4.5 Questionnaire Preprocessing — Published Scoring Formulas

```python
def score_womac(responses: dict) -> dict:
    """
    Score WOMAC questionnaire using the published Likert scoring formula.
    
    Input: dict with keys 'pain' (5 items), 'stiffness' (2 items), 'function' (17 items)
           Each item scored 0–4 (None/Mild/Moderate/Severe/Extreme)
    """
    pain_raw = sum(responses['pain'])           # 0–20
    stiffness_raw = sum(responses['stiffness']) # 0–8
    function_raw = sum(responses['function'])   # 0–68
    total_raw = pain_raw + stiffness_raw + function_raw  # 0–96
    
    return {
        'womac_pain': pain_raw,
        'womac_pain_normalized': (pain_raw / 20) * 100,
        'womac_stiffness': stiffness_raw,
        'womac_stiffness_normalized': (stiffness_raw / 8) * 100,
        'womac_function': function_raw,
        'womac_function_normalized': (function_raw / 68) * 100,
        'womac_total': total_raw,
        'womac_total_normalized': (total_raw / 96) * 100
    }

def score_koos(responses: dict) -> dict:
    """
    Score KOOS questionnaire.
    Each subscale: 100 - (mean_of_items / 4) × 100
    Higher = better (opposite of WOMAC)
    """
    subscales = {}
    for subscale_name, items in responses.items():
        mean_score = np.mean(items)
        subscales[f'koos_{subscale_name}'] = 100 - (mean_score / 4) * 100
    return subscales
```

---

## 5. Layer 2 — Per-Modality Feature Extraction

### 5.1 Tier A: OA-Associated Signals

These features are the **primary biomarkers** directly indicative of knee osteoarthritis.

#### Feature A1: Range of Motion (ROM)

**Source:** IMU pipeline  
**Clinical Significance:** Reduced knee ROM (especially loss of full extension) is a hallmark of OA. Normal ROM: 0° extension to 135° flexion.

```python
def extract_rom_features(knee_angle_timeseries):
    """
    Extract ROM features from knee angle time series.
    
    Parameters:
        knee_angle_timeseries: 1D array of knee flexion-extension angles (degrees)
                               0° = full extension, positive = flexion
    """
    max_flexion = np.max(knee_angle_timeseries)
    min_extension = np.min(knee_angle_timeseries)
    total_rom = max_flexion - min_extension
    
    # Extension deficit: how far from 0° the knee can extend
    extension_deficit = max(0, min_extension - 0)  # should be ~0° in healthy knee
    
    # Flexion deficit: difference from normal 135°
    flexion_deficit = max(0, 135 - max_flexion)
    
    return {
        'rom_total_deg': total_rom,
        'rom_max_flexion_deg': max_flexion,
        'rom_min_extension_deg': min_extension,
        'rom_extension_deficit_deg': extension_deficit,
        'rom_flexion_deficit_deg': flexion_deficit
    }
```

**Reference ranges:**

| Measure | Normal | Mild OA | Moderate OA | Severe OA |
|---------|--------|---------|-------------|-----------|
| Total ROM | 130–140° | 110–130° | 90–110° | <90° |
| Extension Deficit | 0° | 0–5° | 5–15° | >15° |

---

#### Feature A2: Crepitus Score

**Source:** Acoustic DSP pipeline  
**Clinical Significance:** Audible or palpable crepitus during knee movement correlates with cartilage surface irregularity (fibrillation, erosion).

```python
def extract_crepitus_features(signal_clean, sr, crepitus_peaks, mel_spec_db):
    """
    Extract crepitus features from acoustic analysis.
    """
    duration = len(signal_clean) / sr
    
    # Count-based features
    crepitus_count = len(crepitus_peaks)
    crepitus_rate = crepitus_count / duration  # events per second
    
    # Frequency-domain features of crepitus events
    crepitus_energies = []
    for peak_idx in crepitus_peaks:
        # Extract 100ms window around each crepitus event
        window_start = max(0, int((peak_idx * 512 / sr - 0.05) * sr))
        window_end = min(len(signal_clean), int((peak_idx * 512 / sr + 0.05) * sr))
        segment = signal_clean[window_start:window_end]
        crepitus_energies.append(np.sum(segment**2))
    
    # Spectral centroid of crepitus events (characterizes grinding vs clicking)
    spectral_centroids = librosa.feature.spectral_centroid(y=signal_clean, sr=sr)[0]
    
    return {
        'crepitus_count': crepitus_count,
        'crepitus_rate_per_sec': crepitus_rate,
        'crepitus_mean_energy': np.mean(crepitus_energies) if crepitus_energies else 0,
        'crepitus_spectral_centroid_hz': np.mean(spectral_centroids),
        'crepitus_present': int(crepitus_count > 2)  # binary flag
    }
```

**Clinical interpretation:**

| Crepitus Rate | Interpretation |
|--------------|----------------|
| 0 events/sec | No crepitus |
| 0–1 events/sec | Mild — possible early cartilage changes |
| 1–3 events/sec | Moderate — likely cartilage fibrillation |
| >3 events/sec | Severe — significant cartilage damage |

---

#### Feature A3: Co-Contraction Index

**Source:** sEMG NeuroKit2 pipeline  
**Clinical Significance:** In OA, patients exhibit increased co-contraction (simultaneous activation of agonist and antagonist muscles) as a protective stiffening mechanism. This increases joint loading and accelerates degeneration.

```python
def calculate_cocontraction_index(agonist_amplitude, antagonist_amplitude):
    """
    Calculate the co-contraction index (CCI) using the Winter (1990) method.
    
    CCI = (2 × overlap_area) / (agonist_area + antagonist_area) × 100
    
    Parameters:
        agonist_amplitude: Linear envelope of agonist (e.g., quadriceps during extension)
        antagonist_amplitude: Linear envelope of antagonist (e.g., hamstring during extension)
    """
    # Calculate overlap area (minimum of the two signals at each time point)
    overlap = np.minimum(agonist_amplitude, antagonist_amplitude)
    
    overlap_area = np.trapezoid(overlap)
    total_area = np.trapezoid(agonist_amplitude) + np.trapezoid(antagonist_amplitude)
    
    if total_area == 0:
        return 0.0
    
    cci = (2 * overlap_area / total_area) * 100  # percentage
    return cci

# Calculate CCI for knee extension phase
cci_extension = calculate_cocontraction_index(
    agonist_amplitude=semg_processed['RF']['amplitude_mvc'],    # quadriceps
    antagonist_amplitude=semg_processed['BF']['amplitude_mvc']  # hamstring
)

# CCI during walking (per gait cycle)
cci_walking_cycles = []
for cycle_start, cycle_end in gait_cycles:
    cci = calculate_cocontraction_index(
        agonist_amplitude=semg_processed['RF']['amplitude_mvc'][cycle_start:cycle_end],
        antagonist_amplitude=semg_processed['BF']['amplitude_mvc'][cycle_start:cycle_end]
    )
    cci_walking_cycles.append(cci)
```

**Reference ranges:**

| CCI (%) | Interpretation |
|---------|----------------|
| 10–25% | Normal — efficient reciprocal activation |
| 25–40% | Elevated — possible early OA compensatory pattern |
| >40% | High — significant arthrogenic muscle inhibition |

---

#### Feature A4: Strength Deficit

**Source:** Dynamometer pipeline  
**Clinical Significance:** Quadriceps weakness is both a risk factor for and consequence of knee OA. A bilateral strength deficit >20% is clinically significant.

```python
def extract_strength_features(ext_affected, ext_unaffected, flex_affected, flex_unaffected, body_weight_kg):
    """
    Extract strength deficit features from dynamometer data.
    """
    # Bilateral extension deficit
    ext_deficit_pct = ((ext_unaffected - ext_affected) / ext_unaffected) * 100
    
    # Bilateral flexion deficit
    flex_deficit_pct = ((flex_unaffected - flex_affected) / flex_unaffected) * 100
    
    # H:Q ratio (affected side)
    hq_ratio = flex_affected / ext_affected if ext_affected > 0 else 0
    
    # Strength normalized to body weight
    ext_normalized = ext_affected / (body_weight_kg * 9.81)  # N / body_weight_N
    
    return {
        'strength_ext_deficit_pct': ext_deficit_pct,
        'strength_flex_deficit_pct': flex_deficit_pct,
        'strength_hq_ratio': hq_ratio,
        'strength_ext_normalized': ext_normalized,
        'strength_ext_affected_N': ext_affected,
        'strength_ext_unaffected_N': ext_unaffected
    }
```

---

### 5.2 Tier B: General / Differential Signals

These features capture broader biomechanical patterns that help **rule in or rule out alternative pathologies**.

#### Feature B1: Gait Asymmetry

**Source:** IMU pipeline (GaitPy)

```python
def extract_gait_asymmetry_features(gait_features_df):
    """
    Extract gait asymmetry features from GaitPy output.
    
    Gait asymmetry can indicate OA (antalgic gait) but also hip pathology,
    neurological conditions, or leg length discrepancy.
    """
    # Stride time asymmetry
    left_stride_times = gait_features_df[gait_features_df['side'] == 'left']['stride_time']
    right_stride_times = gait_features_df[gait_features_df['side'] == 'right']['stride_time']
    
    stride_time_asymmetry = abs(
        left_stride_times.mean() - right_stride_times.mean()
    ) / (0.5 * (left_stride_times.mean() + right_stride_times.mean())) * 100
    
    # Stance time asymmetry
    left_stance = gait_features_df[gait_features_df['side'] == 'left']['stance_time']
    right_stance = gait_features_df[gait_features_df['side'] == 'right']['stance_time']
    
    stance_time_asymmetry = abs(
        left_stance.mean() - right_stance.mean()
    ) / (0.5 * (left_stance.mean() + right_stance.mean())) * 100
    
    # Gait speed
    gait_speed = gait_features_df['stride_length'].mean() / gait_features_df['stride_time'].mean()
    
    # Step time variability (CV)
    step_time_cv = gait_features_df['step_time'].std() / gait_features_df['step_time'].mean() * 100
    
    return {
        'gait_stride_time_asymmetry_pct': stride_time_asymmetry,
        'gait_stance_time_asymmetry_pct': stance_time_asymmetry,
        'gait_speed_m_per_s': gait_speed,
        'gait_step_time_cv_pct': step_time_cv,
        'gait_cadence_steps_per_min': 60 / gait_features_df['step_time'].mean()
    }
```

#### Feature B2: Neuromuscular Activation Pattern

**Source:** sEMG pipeline

```python
def extract_neuromuscular_features(semg_processed, gait_cycles):
    """
    Extract neuromuscular activation features that indicate central nervous
    system control quality — relevant for differential diagnosis.
    """
    features = {}
    
    for ch in ['VM', 'VL', 'RF', 'BF']:
        amplitude = semg_processed[ch]['amplitude_mvc']
        
        # Mean activation level during walking (%MVC)
        features[f'neuromusc_{ch}_mean_activation_pct_mvc'] = np.mean(amplitude)
        
        # Peak activation during walking
        features[f'neuromusc_{ch}_peak_activation_pct_mvc'] = np.max(amplitude)
        
        # Activation timing variability across gait cycles
        onset_times = []
        for start, end in gait_cycles:
            cycle_activation = semg_processed[ch]['activation'][start:end]
            first_onset = np.argmax(cycle_activation > 0)
            onset_times.append(first_onset / (end - start))  # normalized to cycle
        
        features[f'neuromusc_{ch}_onset_variability'] = np.std(onset_times)
    
    # VM:VL ratio (medial-lateral activation balance)
    features['neuromusc_vm_vl_ratio'] = (
        np.mean(semg_processed['VM']['amplitude_mvc']) /
        max(np.mean(semg_processed['VL']['amplitude_mvc']), 1e-6)
    )
    
    return features
```

#### Feature B3: General Weakness

**Source:** Dynamometer pipeline

```python
def extract_general_weakness_features(ext_force_N, flex_force_N, body_weight_kg, age, sex):
    """
    Assess general lower limb weakness by comparing to population norms.
    General weakness can indicate systemic conditions (sarcopenia, deconditioning,
    neurological disease) rather than localized OA.
    """
    # Normalize to body weight
    ext_bw_ratio = ext_force_N / (body_weight_kg * 9.81)
    flex_bw_ratio = flex_force_N / (body_weight_kg * 9.81)
    
    # Population norms (Andrews et al., 1996; Bohannon, 1997)
    # Simplified — actual implementation would use age/sex stratified lookup tables
    norms = {
        'male':   {'extension_bw': {30: 0.65, 50: 0.55, 70: 0.40}},
        'female': {'extension_bw': {30: 0.55, 50: 0.45, 70: 0.35}}
    }
    
    # Find closest age bracket
    age_bracket = min(norms[sex]['extension_bw'].keys(), key=lambda x: abs(x - age))
    expected_ratio = norms[sex]['extension_bw'][age_bracket]
    
    weakness_z_score = (ext_bw_ratio - expected_ratio) / (expected_ratio * 0.15)  # assume 15% CV
    
    return {
        'general_weakness_ext_bw_ratio': ext_bw_ratio,
        'general_weakness_flex_bw_ratio': flex_bw_ratio,
        'general_weakness_z_score': weakness_z_score,
        'general_weakness_below_norm': int(weakness_z_score < -1.5)
    }
```

#### Feature B4: Injury History

**Source:** Questionnaire pipeline

```python
def extract_injury_history_features(questionnaire_data):
    """
    Extract structured injury history features from questionnaire responses.
    """
    injury = questionnaire_data.get('injury_history', {})
    
    return {
        'injury_previous_knee_injury': int(injury.get('previous_knee_injury', False)),
        'injury_type_meniscal': int(injury.get('type') == 'meniscal_tear'),
        'injury_type_acl': int(injury.get('type') in ['acl_tear', 'acl_reconstruction']),
        'injury_type_fracture': int(injury.get('type') == 'fracture'),
        'injury_surgery_history': int(injury.get('surgery', False)),
        'injury_years_since': (
            2026 - injury.get('year', 2026) if injury.get('previous_knee_injury') else 0
        ),
        'injury_tegner_activity': questionnaire_data.get('tegner_activity_level', 5),
        'injury_womac_pain_norm': questionnaire_data.get('womac_pain_normalized', 0),
        'injury_koos_qol': questionnaire_data.get('koos_qol', 100)
    }
```

---

## 6. Layer 3 — Contextual Meta-Inputs

### 6.1 Tier C: Contextual Features

These are **static patient variables** that bypass the feature extraction pipelines entirely and feed directly into the fusion model via dashed connector lines.

| Feature | Source | Type | Range | Clinical Rationale |
|---------|--------|------|-------|-------------------|
| **BMI** | Measured (height + weight) | Continuous | 15–50+ kg/m² | Obesity is the strongest modifiable risk factor for knee OA. BMI >30 = 4× increased risk. |
| **Age** | Patient record | Continuous | 18–100 years | OA prevalence increases exponentially after age 45. |
| **Sex** | Patient record | Binary | 0/1 (M/F) | Female sex confers ~1.8× higher OA risk (hormonal, biomechanical factors). |
| **Previous Injury** | Questionnaire | Binary | 0/1 | Prior ACL/meniscal injury increases OA risk 4–6× in that knee. |
| **Activity Level** | Questionnaire (Tegner) | Ordinal | 0–10 | Both very low (deconditioning) and very high (overuse) levels increase risk. |

```python
def extract_contextual_features(patient_metadata, questionnaire_data):
    """
    Extract Tier C contextual features.
    These bypass Layer 2 and go directly to the fusion model.
    """
    height_m = patient_metadata['height_cm'] / 100
    weight_kg = patient_metadata['weight_kg']
    
    return {
        'ctx_bmi': weight_kg / (height_m ** 2),
        'ctx_age': patient_metadata['age'],
        'ctx_sex_female': int(patient_metadata['sex'] == 'female'),
        'ctx_previous_injury': int(questionnaire_data.get('injury_history', {}).get('previous_knee_injury', False)),
        'ctx_activity_level': questionnaire_data.get('tegner_activity_level', 5)
    }
```

### 6.2 Why These Bypass Layer 2

The architectural decision to bypass the per-modality experts is deliberate:

1. **No signal processing needed** — these are already clean, discrete values.
2. **Prevent information dilution** — if demographic data were mixed with biomechanical features before fusion, the tree model might underweight them.
3. **Enable interaction effects** — the fusion model can learn interactions like "ROM deficit + high BMI + age > 60 = high risk" directly.

---

## 7. The Architectural Divide — Frozen vs. Trainable

### 7.1 The Freeze Line

A conceptual horizontal boundary divides the entire system:

```
╔═══════════════════════════════════════════════════════════════════╗
║                    FROZEN (Layers 1, 2, 3)                       ║
║                                                                   ║
║  • All signal processing (GaitPy, AHRS, DSP, NeuroKit2)         ║
║  • All feature extraction (ROM, crepitus, CCI, strength)         ║
║  • All questionnaire scoring (WOMAC, KOOS)                       ║
║  • Contextual feature extraction (BMI calculation, etc.)          ║
║                                                                   ║
║  These are DETERMINISTIC algorithms.                              ║
║  They DO NOT have learnable parameters.                           ║
║  They DO NOT change during training.                              ║
║  They CAN be validated independently against gold standards.      ║
╠═══════════════════════════════════════════════════════════════════╣
║                 TRAINABLE (Layer 4)                                ║
║                                                                   ║
║  • Fusion Model (CatBoost/XGBoost) — learns feature weights      ║
║  • Platt Scaling — learns calibration parameters                  ║
║  • Differential classifier — learns pathology boundaries          ║
║                                                                   ║
║  These HAVE learnable parameters.                                 ║
║  They ARE updated during training.                                ║
║  They REQUIRE labeled data.                                       ║
╚═══════════════════════════════════════════════════════════════════╝
```

### 7.2 Why This Separation Matters

| Concern | Benefit of Frozen Extractors |
|---------|------------------------------|
| **Data Efficiency** | Only need labeled data for the ~20-feature fusion model, not for training complex signal processing |
| **Reproducibility** | Same raw signal always produces the same features, regardless of when it's processed |
| **Regulatory Compliance** | Frozen algorithms can be validated once and locked; only the fusion head needs re-validation on model updates |
| **Debugging** | If the final prediction is wrong, you can inspect the intermediate features and identify whether the issue is in signal processing or in the fusion model |
| **Transfer Learning** | The frozen extractors work universally across populations; only the fusion head needs NER-specific adaptation |

---

## 8. Layer 4 — Fusion & Recalibration

### 8.1 Feature Vector Assembly

Before entering the fusion model, all features from Tiers A, B, and C are concatenated into a single feature vector per patient:

```python
def assemble_feature_vector(tier_a, tier_b, tier_c):
    """
    Assemble the final feature vector for the fusion model.
    
    Returns a dict with all features, ready for DataFrame conversion.
    """
    features = {}
    features.update(tier_a)  # ROM, crepitus, co-contraction, strength deficit
    features.update(tier_b)  # gait asymmetry, neuromuscular, general weakness, injury hx
    features.update(tier_c)  # BMI, age, sex, previous injury, activity level
    
    return features

# Example feature vector (one patient)
feature_vector = {
    # Tier A (OA-specific)
    'rom_total_deg': 112.5,
    'rom_extension_deficit_deg': 5.2,
    'rom_flexion_deficit_deg': 17.3,
    'crepitus_count': 7,
    'crepitus_rate_per_sec': 1.4,
    'crepitus_mean_energy': 0.023,
    'crepitus_present': 1,
    'cocontraction_cci_extension': 35.2,
    'cocontraction_cci_walking_mean': 28.7,
    'strength_ext_deficit_pct': 22.1,
    'strength_hq_ratio': 0.58,
    'strength_ext_normalized': 0.42,
    
    # Tier B (Differential)
    'gait_stride_time_asymmetry_pct': 8.3,
    'gait_speed_m_per_s': 1.05,
    'gait_step_time_cv_pct': 5.7,
    'neuromusc_vm_vl_ratio': 0.78,
    'neuromusc_RF_mean_activation_pct_mvc': 32.1,
    'general_weakness_z_score': -0.8,
    'injury_previous_knee_injury': 1,
    'injury_type_meniscal': 1,
    'injury_years_since': 7,
    'injury_womac_pain_norm': 45.0,
    'injury_koos_qol': 38.0,
    
    # Tier C (Contextual)
    'ctx_bmi': 28.3,
    'ctx_age': 57,
    'ctx_sex_female': 1,
    'ctx_previous_injury': 1,
    'ctx_activity_level': 4
}
# Total: ~27 features
```

### 8.2 Fusion Model — CatBoost/XGBoost

#### Why Gradient-Boosted Trees (Not Deep Learning)?

| Factor | Gradient Boosted Trees | Deep Learning |
|--------|----------------------|---------------|
| **Tabular data performance** | State-of-the-art (Grinsztajn et al., 2022) | Generally inferior on tabular data |
| **Sample size requirement** | Works well with N=200–2,000 | Needs N>10,000 typically |
| **Interpretability** | Feature importance, SHAP values | Black box |
| **Training time** | Minutes on CPU | Hours on GPU |
| **Missing data handling** | Native (CatBoost) | Requires imputation |
| **Regulatory pathway** | Easier to explain to FDA/regulatory bodies | Harder to justify |

#### CatBoost Configuration

```python
from catboost import CatBoostClassifier, Pool
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Prepare data
X = pd.DataFrame([patient_features for patient_features in all_patients])
y = pd.Series([patient_label for patient_label in all_labels])  # 0 = no OA, 1 = OA

# Define categorical features (if any)
cat_features = ['ctx_sex_female', 'injury_previous_knee_injury', 
                'injury_type_meniscal', 'injury_type_acl', 'crepitus_present']

# CatBoost model
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    min_data_in_leaf=5,
    random_seed=42,
    eval_metric='AUC',
    use_best_model=True,
    early_stopping_rounds=50,
    verbose=100,
    
    # Handle class imbalance (OA is common in older populations)
    auto_class_weights='Balanced',
    
    # CatBoost-specific advantages
    cat_features=cat_features,
    nan_mode='Min',  # Native missing value handling
    
    # Regularization to prevent overfitting on small datasets
    subsample=0.8,
    colsample_bylevel=0.8,
    random_strength=1.0,
    bagging_temperature=0.5
)

# Cross-validation training
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)
    
    model.fit(train_pool, eval_set=val_pool)
    
    # Get raw logits (before sigmoid) for Platt scaling
    raw_logits = model.predict(X_val, prediction_type='RawFormulaVal')
```

#### XGBoost Alternative Configuration

```python
import xgboost as xgb

model_xgb = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=5,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=len(y[y==0]) / len(y[y==1]),  # class balance
    eval_metric='auc',
    early_stopping_rounds=50,
    random_state=42,
    use_label_encoder=False
)
```

### 8.3 NER Adaptability — Platt Scaling Recalibration

#### What Is Platt Scaling?

Platt Scaling is a **post-hoc calibration technique** that fits a logistic regression model to the raw outputs (logits) of a classifier, transforming them into well-calibrated probabilities.

**Why is it needed?**
- The fusion model may be trained on OAI data (predominantly white, North American population).
- The NER (North Eastern Region of India) population has different:
  - OA prevalence rates
  - BMI distributions
  - Activity patterns (manual labor, terrain)
  - Genetic predispositions
- Platt scaling adjusts the decision boundary without retraining the entire model.

#### Mathematical Formulation

Given raw logit $f(x)$ from the fusion model, Platt scaling learns parameters $A$ and $B$:

$$P(\text{OA} \mid x) = \frac{1}{1 + \exp(A \cdot f(x) + B)}$$

Where $A$ (slope) and $B$ (intercept) are fit on a **small NER-specific calibration dataset** using maximum likelihood estimation.

#### Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

class PlattScalingRecalibrator:
    """
    Platt Scaling for NER population adaptation.
    
    Learns A and B parameters to recalibrate raw logits from the
    base fusion model to match NER-specific OA prevalence.
    """
    
    def __init__(self):
        self.calibrator = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            C=1e10  # Minimal regularization — we want to fit the calibration curve
        )
        self.A = None
        self.B = None
    
    def fit(self, raw_logits, true_labels):
        """
        Fit Platt scaling parameters on NER calibration data.
        
        Parameters:
            raw_logits: (N,) array — raw outputs from fusion model
            true_labels: (N,) array — true OA labels from NER cohort
        
        Minimum recommended N: 50–100 labeled NER samples
        """
        # Reshape for sklearn
        X = raw_logits.reshape(-1, 1)
        y = true_labels
        
        self.calibrator.fit(X, y)
        self.A = self.calibrator.coef_[0][0]
        self.B = self.calibrator.intercept_[0]
        
        print(f"Platt Scaling Parameters: A={self.A:.4f}, B={self.B:.4f}")
        return self
    
    def predict_proba(self, raw_logits):
        """
        Apply Platt scaling to produce calibrated probabilities.
        """
        X = raw_logits.reshape(-1, 1)
        return self.calibrator.predict_proba(X)[:, 1]  # P(OA)
    
    def predict_risk_score(self, raw_logits):
        """
        Produce final OA Risk Score (0.0 — 1.0).
        """
        return self.predict_proba(raw_logits)

# Usage
recalibrator = PlattScalingRecalibrator()
recalibrator.fit(
    raw_logits=base_model_logits_on_ner_data,  # From fusion model
    true_labels=ner_ground_truth_labels          # From NER clinical validation
)

# Final calibrated predictions
oa_risk_scores = recalibrator.predict_risk_score(new_patient_logits)
```

### 8.4 Differential Triage Signal

The differential triage signal is a **secondary output** that flags whether the patient's presentation suggests something other than primary knee OA.

```python
def compute_differential_triage(tier_b_features, oa_risk_score):
    """
    Rule-based differential triage signal.
    
    Flags patients whose symptom pattern is atypical for primary knee OA,
    suggesting further investigation for:
    - Inflammatory arthritis (RA, gout)
    - Meniscal/ligament pathology
    - Hip-referred pain
    - Neurological conditions
    - Systemic weakness / sarcopenia
    """
    flags = []
    
    # Flag 1: High gait asymmetry with low OA score → possible neurological or hip issue
    if (tier_b_features['gait_stride_time_asymmetry_pct'] > 15 and oa_risk_score < 0.3):
        flags.append('NEURO_OR_HIP_REFERRAL')
    
    # Flag 2: General weakness without OA signs → sarcopenia or systemic condition
    if (tier_b_features['general_weakness_z_score'] < -2.0 and oa_risk_score < 0.4):
        flags.append('SYSTEMIC_WEAKNESS')
    
    # Flag 3: Acute injury pattern (recent injury, young, high VAS pain)
    if (tier_b_features['injury_years_since'] < 1 and tier_b_features.get('ctx_age', 50) < 40):
        flags.append('ACUTE_INJURY_PATTERN')
    
    # Flag 4: VM:VL ratio abnormality → patellofemoral issues
    if tier_b_features.get('neuromusc_vm_vl_ratio', 1.0) < 0.5:
        flags.append('PATELLOFEMORAL_DYSFUNCTION')
    
    return {
        'differential_signal': len(flags) > 0,
        'differential_flags': flags,
        'differential_confidence': min(len(flags) / 3, 1.0)
    }
```

### 8.5 Terminal Outputs

The system produces two final outputs per patient:

```python
@dataclass
class KneevaOutput:
    """Final output of the Kneeva OA Triage System."""
    
    # Primary output
    oa_risk_score: float        # 0.0 — 1.0, calibrated probability
    oa_risk_category: str       # 'low' / 'moderate' / 'high' / 'very_high'
    
    # Secondary output
    differential_signal: bool   # True if alternative pathology suspected
    differential_flags: list    # List of suspected conditions
    
    # Supporting data
    feature_importance: dict    # Top contributing features (SHAP values)
    confidence_interval: tuple  # (lower, upper) 95% CI on risk score
    
    @property
    def risk_category(self):
        if self.oa_risk_score < 0.2:
            return 'low'
        elif self.oa_risk_score < 0.5:
            return 'moderate'
        elif self.oa_risk_score < 0.8:
            return 'high'
        else:
            return 'very_high'
    
    @property
    def clinical_action(self):
        actions = {
            'low': 'Reassurance + lifestyle advice. No immediate referral needed.',
            'moderate': 'Primary care follow-up. Conservative management (exercise, weight management).',
            'high': 'Refer to orthopedics. Consider imaging (X-ray, MRI).',
            'very_high': 'Urgent orthopedic referral. Likely advanced OA requiring intervention.'
        }
        return actions[self.risk_category]
```

---

## 9. Models to Train — Complete Specification

### Summary Table

| # | Model Name | Type | Parameters | Training Data Requirement | What It Learns |
|---|-----------|------|------------|--------------------------|----------------|
| **1** | **Fusion Model** | CatBoost or XGBoost (GBDT) | ~1,000 trees × depth 6 = ~50K parameters | N ≥ 200 (ideal: 500–2,000) labeled patients | Feature interactions → OA probability |
| **2** | **Platt Scaling Recalibrator** | Logistic Regression | **2 parameters** (A, B) | N ≥ 50 NER-specific labeled patients | Population-specific calibration |
| **3** | **Differential Triage** (optional ML) | Logistic Regression or small XGBoost | 10–50 parameters | N ≥ 100 patients with pathology labels | OA vs. alternative pathology boundary |

### What You Do NOT Train

| Component | Why Not Trained |
|-----------|----------------|
| GaitPy / AHRS orientation filters | Mathematical algorithms (Kalman/Madgwick filters) — no learnable weights |
| DSP scripts (spectral analysis, FFT) | Pure mathematical transforms |
| NeuroKit2 sEMG processing | Rule-based signal processing (thresholds, filters) |
| Peak detection algorithms | SciPy's `find_peaks` — parameter-based, not learned |
| WOMAC/KOOS scoring | Published, standardized formulas |
| ROM calculation | Trigonometry on quaternions |
| Co-contraction index | Published formula (Winter, 1990) |
| Strength deficit | Simple ratio calculation |

---

## 10. Datasets — Authenticated Sources & Usage

### 10.1 Primary Datasets for Labels & Clinical Data

#### OAI (Osteoarthritis Initiative) ⭐ PRIMARY

| Attribute | Detail |
|-----------|--------|
| **URL** | https://nda.nih.gov/oai |
| **Size** | 4,796 participants, 8+ years of longitudinal follow-up |
| **Population** | Age 45–79, USA, multi-site (Baltimore, Pittsburgh, Columbus, Providence) |
| **Clinical Data** | KL grades (X-ray OA severity 0–4), WOMAC, KOOS, physical exam, comorbidities |
| **Imaging** | X-ray (all visits), MRI (subset), available separately |
| **Demographics** | Age, sex, race, BMI, education, occupation, activity level |
| **Physical Performance** | Chair stand time, 20m walk speed, grip strength, knee exam findings |
| **Access** | Free registration required. Approval takes ~1–2 weeks. |
| **Usage in Kneeva** | **Ground truth labels** (KL grade), contextual features (BMI, age, sex), questionnaire validation, baseline clinical data |

**How to use OAI for Kneeva training:**

1. Download the clinical datasets (AllClinical files).
2. Extract KL grade as the primary label:
   - KL 0–1 → OA Negative (label = 0)
   - KL 2–4 → OA Positive (label = 1)
3. Extract WOMAC, physical exam features, demographics.
4. You will NOT have sensor data from OAI — this dataset provides labels and clinical benchmarks only.
5. You will need to collect your own sensor data and match it with KL-grade-verified labels.

---

#### MOST (Multicenter Osteoarthritis Study)

| Attribute | Detail |
|-----------|--------|
| **URL** | https://most.ucsf.edu |
| **Size** | 3,026 participants |
| **Population** | Age 50–79, high risk for knee OA, USA (Birmingham AL, Iowa City IA) |
| **Clinical Data** | KL grades, WOMAC, physical exam, strength testing, gait speed |
| **Strength Data** | Isokinetic knee extension/flexion strength (Biodex) |
| **Access** | Application to MOST Coordinating Center (UCSF) |
| **Usage in Kneeva** | Validation of strength deficit features, cross-validation of fusion model trained on OAI |

---

#### KNHANES (Korea National Health & Nutrition Examination Survey)

| Attribute | Detail |
|-----------|--------|
| **URL** | https://knhanes.kdca.go.kr/knhanes/eng/ |
| **Size** | ~10,000 participants per survey year (Asian population) |
| **Clinical Data** | OA diagnosis codes, BMI, demographics, physical function |
| **Relevance** | Asian population data — closer to NER demographics than OAI |
| **Access** | Free, online download |
| **Usage in Kneeva** | Population-level OA prevalence for NER calibration baseline |

---

### 10.2 Sensor-Specific Validation Datasets

#### For IMU / Gait Analysis

| Dataset | Description | Size | Access |
|---------|-------------|------|--------|
| **GaitRec** | Ground reaction forces + clinical gait analysis from ~2,000 sessions | 2,084 sessions | Free — PhysioNet |
| **MAREA** | IMU data during walking, jogging, stairs (wrist + ankle sensors) | 20 subjects | Free — UCI ML Repository |
| **Gutenberg Gait Database** | 3D motion capture + IMU (healthy + pathological gait) | 350 subjects | Application required |

**URL for GaitRec:** https://physionet.org/content/gaitrec/1.0.0/

**Usage:** Validate that your GaitPy pipeline produces gait parameters (stride time, stance time, asymmetry) consistent with established motion capture gold standards.

---

#### For sEMG

| Dataset | Description | Size | Access |
|---------|-------------|------|--------|
| **Ninapro DB2** | sEMG from 12 electrodes during hand movements | 40 subjects | Free — http://ninapro.hevs.ch |
| **Ninapro DB5** | Dual Myo armband sEMG data | 10 subjects | Free |
| **CASIA sEMG** | Lower limb sEMG during gait (limited) | 20 subjects | Application required |

> **⚠️ Important:** No large public dataset exists for lower-limb sEMG specifically in knee OA patients. You will need to collect your own sEMG data as part of the Kneeva data collection protocol.

**Usage:** Validate that your NeuroKit2 pipeline produces physiologically reasonable MVC-normalized amplitudes and co-contraction indices.

---

#### For Acoustic / Crepitus

| Dataset | Description | Size | Access |
|---------|-------------|------|--------|
| **No major public dataset exists** | Knee acoustic emission / crepitus detection is an emerging field | — | — |
| **Relevant papers:** | Mascaro et al. (2009), Shark et al. (2011), Krishnan et al. (2019) | — | — |

> **⚠️ Critical Gap:** You must collect your own knee acoustic data. No public dataset with labeled crepitus recordings is available. This is a unique data asset that Kneeva will build.

**Workaround:** You can initially validate your DSP pipeline on synthetic crepitus signals (impulse + noise at known frequencies) to ensure the detection algorithm works before collecting real patient data.

---

#### For Dynamometer / Strength

| Dataset | Description | Size | Access |
|---------|-------------|------|--------|
| **OAI physical performance data** | Includes chair stand time, 20m walk speed (proxy for lower limb strength) | 4,796 subjects | Free (same OAI access) |
| **MOST strength data** | Isokinetic knee ext/flex torque (Biodex) | 3,026 subjects | Application required |
| **Bohannon (2012) norms** | Published population norms for knee extension strength by age/sex | Reference values | Published literature |

**Usage:** Validate your peak detection algorithm against MOST isokinetic data. Use Bohannon norms for the general weakness z-score calculation.

---

### 10.3 Dataset Usage Matrix

| Dataset | Labels | Tier A Features | Tier B Features | Tier C Features | Calibration |
|---------|--------|----------------|-----------------|-----------------|-------------|
| **OAI** | ✅ KL grades | ❌ (no sensors) | Partial (gait speed, strength) | ✅ Full demographics | ❌ (not NER) |
| **MOST** | ✅ KL grades | ❌ (no sensors) | ✅ (strength data) | ✅ Full demographics | ❌ (not NER) |
| **KNHANES** | ✅ OA diagnosis | ❌ | ❌ | ✅ BMI, age, sex | Partial (Asian population) |
| **GaitRec** | ❌ | Partial (gait only) | ✅ (gait asymmetry) | ❌ | ❌ |
| **Your NER Data** | ✅ (must collect) | ✅ Full sensor suite | ✅ Full | ✅ Full | ✅ (THIS is what Platt scaling uses) |

---

## 11. Training Pipeline & Methodology

### 11.1 Phase 1: Feature Extractor Validation (No ML Training)

**Goal:** Verify that each deterministic pipeline produces clinically valid features.

```
For each modality:
  1. Run pipeline on known/synthetic test data
  2. Compare output features to published reference ranges
  3. Test edge cases (missing data, noisy signals, sensor disconnection)
  4. Document validation results
```

### 11.2 Phase 2: Base Fusion Model Training (OAI Data)

Since OAI does not have multi-modal sensor data, the initial base model is trained on **OAI clinical features that approximate your sensor-derived features:**

| Kneeva Sensor Feature | OAI Proxy Variable | OAI Variable Name |
|-----------------------|--------------------|--------------------|
| ROM (from IMU) | Clinical ROM measurement | `P01KNKEXR`, `P01KNKFLR` |
| Crepitus (from acoustic) | Clinical crepitus exam finding | `P01KCRPRL`, `P01KCRPLL` |
| Strength deficit | Chair stand time (proxy) | `P01CHRSPD` |
| Gait speed | 20m walk pace | `P01WSPD20` |
| WOMAC scores | Self-reported WOMAC | `WOMTSL`, `WOMTSR` |
| BMI, Age, Sex | Demographics | `P01BMI`, `V00AGE`, `P02SEX` |

```python
# Phase 2: Train base model on OAI proxy features
import pandas as pd
from catboost import CatBoostClassifier

# Load OAI data
oai_data = pd.read_csv('oai_clinical_merged.csv')

# Define features (OAI proxy variables)
feature_cols = [
    'P01KNKEXR', 'P01KNKFLR',     # ROM
    'P01KCRPRL',                    # Crepitus (clinical exam)
    'P01CHRSPD',                    # Chair stand (strength proxy)
    'P01WSPD20',                    # Gait speed
    'WOMTSL',                       # WOMAC total
    'P01BMI', 'V00AGE', 'P02SEX'   # Demographics
]

# Label: KL grade ≥ 2
oai_data['oa_label'] = (oai_data['V00XRKL'] >= 2).astype(int)

X = oai_data[feature_cols].dropna()
y = oai_data.loc[X.index, 'oa_label']

# Train base model
base_model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=5)
base_model.fit(X, y)
```

### 11.3 Phase 3: Transfer to Sensor Features (Kneeva Data Collection)

Once you collect your own multi-modal sensor data from NER patients:

```python
# Phase 3: Retrain fusion model on actual sensor-derived features

# Your collected data
kneeva_data = pd.read_csv('kneeva_ner_pilot_data.csv')

# Full feature vector (from Tiers A, B, C)
sensor_feature_cols = [
    # Tier A
    'rom_total_deg', 'rom_extension_deficit_deg', 'crepitus_count',
    'crepitus_rate_per_sec', 'cocontraction_cci_walking_mean',
    'strength_ext_deficit_pct', 'strength_hq_ratio',
    
    # Tier B
    'gait_stride_time_asymmetry_pct', 'gait_speed_m_per_s',
    'neuromusc_vm_vl_ratio', 'general_weakness_z_score',
    'injury_previous_knee_injury', 'injury_womac_pain_norm',
    
    # Tier C
    'ctx_bmi', 'ctx_age', 'ctx_sex_female',
    'ctx_previous_injury', 'ctx_activity_level'
]

X_sensor = kneeva_data[sensor_feature_cols]
y_sensor = kneeva_data['oa_label']  # From clinical verification (X-ray KL grade)

# Train final fusion model
final_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    auto_class_weights='Balanced',
    early_stopping_rounds=50
)

# Use 5-fold cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(final_model, X_sensor, y_sensor, cv=5, scoring='roc_auc')
print(f"Cross-validated AUC: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
```

### 11.4 Phase 4: Platt Scaling Calibration (NER-Specific)

```python
# Phase 4: Calibrate for NER population
# Use a held-out NER validation set (N ≥ 50)

# Get raw logits from the trained fusion model
ner_logits = final_model.predict(X_ner_holdout, prediction_type='RawFormulaVal')

# Fit Platt scaling
recalibrator = PlattScalingRecalibrator()
recalibrator.fit(ner_logits, y_ner_holdout)

# Verify calibration
from sklearn.calibration import calibration_curve
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_ner_holdout,
    recalibrator.predict_risk_score(ner_logits),
    n_bins=10
)
```

---

## 12. Evaluation Metrics & Validation

### 12.1 Primary Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| **AUC-ROC** | ≥ 0.85 | Discrimination ability across all thresholds |
| **Sensitivity (Recall)** | ≥ 0.90 at chosen threshold | Must not miss OA cases (triage system — false negatives are costly) |
| **Specificity** | ≥ 0.70 at chosen threshold | Acceptable false positive rate for triage |
| **Positive Predictive Value** | ≥ 0.75 | Of those flagged, most should have OA |
| **Calibration (Brier Score)** | ≤ 0.15 | Predicted probabilities must be accurate (especially post-Platt scaling) |

### 12.2 Validation Strategy

```
┌─────────────────────────────────────────┐
│     Training Data (OAI + Kneeva NER)    │
│                                         │
│  ┌─── 5-Fold Cross-Validation ───────┐  │
│  │  Fold 1: Train 80% → Val 20%     │  │
│  │  Fold 2: Train 80% → Val 20%     │  │
│  │  Fold 3: Train 80% → Val 20%     │  │
│  │  Fold 4: Train 80% → Val 20%     │  │
│  │  Fold 5: Train 80% → Val 20%     │  │
│  │  → Report mean ± std AUC         │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌─── Held-Out NER Test Set ─────────┐  │
│  │  N ≥ 50 NER patients              │  │
│  │  → Final performance estimate     │  │
│  │  → Platt scaling calibration      │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### 12.3 SHAP Explainability

```python
import shap

# SHAP analysis for model interpretability
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_sensor)

# Summary plot (feature importance ranking)
shap.summary_plot(shap_values, X_sensor, feature_names=sensor_feature_cols)

# Individual patient explanation
shap.force_plot(explainer.expected_value, shap_values[0], X_sensor.iloc[0])
```

---

## 13. NER Adaptability — Platt Scaling Deep Dive

### 13.1 Why NER Needs Separate Calibration

The North Eastern Region of India differs from Western populations (OAI) in several epidemiologically significant ways:

| Factor | OAI Population | NER Population | Impact on Model |
|--------|---------------|----------------|-----------------|
| **OA Prevalence** | ~30% (age 45–79) | Estimated 20–40% (limited data) | Prior probability shift |
| **BMI Distribution** | Mean ~28–30 | Mean ~22–25 (lower obesity rates) | BMI feature weight changes |
| **Activity Pattern** | Sedentary to moderate | Higher manual labor, hilly terrain | Different gait patterns |
| **Genetic Factors** | Predominantly Caucasian | Tibeto-Burman, Indo-Aryan, Austroasiatic | Different susceptibility |
| **Healthcare Access** | Regular checkups | Limited — patients present later | Different severity distribution |
| **Diet** | Western | Rice-based, lower dairy | Different metabolic profile |

### 13.2 Minimum NER Data Required

| Purpose | Minimum N | Ideal N | Data Needed |
|---------|-----------|---------|-------------|
| Platt scaling calibration | 50 | 100–200 | Full sensor data + confirmed KL grade |
| Calibration validation | 30 | 50–100 | Same as above (held-out set) |
| **Total NER data collection** | **80** | **150–300** | — |

### 13.3 Calibration Verification

After Platt scaling, verify with a **reliability diagram** (calibration curve):

```python
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# Before Platt scaling
prob_before = 1 / (1 + np.exp(-ner_logits))  # sigmoid of raw logits

# After Platt scaling
prob_after = recalibrator.predict_risk_score(ner_logits)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Before
frac_pos_b, mean_pred_b = calibration_curve(y_ner_holdout, prob_before, n_bins=10)
ax1.plot(mean_pred_b, frac_pos_b, 's-', label='Before Platt Scaling')
ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
ax1.set_title('Before NER Recalibration')
ax1.set_xlabel('Predicted Probability')
ax1.set_ylabel('Observed Frequency')
ax1.legend()

# After
frac_pos_a, mean_pred_a = calibration_curve(y_ner_holdout, prob_after, n_bins=10)
ax2.plot(mean_pred_a, frac_pos_a, 's-', label='After Platt Scaling', color='green')
ax2.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
ax2.set_title('After NER Recalibration')
ax2.set_xlabel('Predicted Probability')
ax2.set_ylabel('Observed Frequency')
ax2.legend()

plt.tight_layout()
plt.savefig('calibration_comparison.png', dpi=150)
```

---

## 14. Deployment Architecture

### 14.1 Data Flow in Production

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Clinical Site   │     │  Local Processing │     │  Cloud / Server    │
│  (NER Hospital)  │     │  (Edge Device)    │     │                    │
│                  │     │                    │     │                    │
│  Sensors → Data  │────►│  Raw data stored   │────►│  Feature pipelines │
│  Collection App  │     │  locally (SQLite)  │     │  (Layers 1-2-3)    │
│                  │     │                    │     │                    │
│                  │     │  Sync when online  │     │  Fusion model       │
│                  │     │                    │     │  (Layer 4)          │
│                  │     │                    │     │                    │
│                  │◄────│                    │◄────│  Risk Score +       │
│  Display result  │     │                    │     │  Triage Signal      │
└─────────────────┘     └──────────────────┘     └────────────────────┘
```

### 14.2 Model Serialization

```python
# Save the trained models
import joblib

# Fusion model (CatBoost has native serialization)
final_model.save_model('models/fusion_catboost_v1.cbm')

# Platt scaling recalibrator
joblib.dump(recalibrator, 'models/platt_scaler_ner_v1.pkl')

# Feature configuration (feature names, order, expected ranges)
import json
config = {
    'feature_names': sensor_feature_cols,
    'feature_ranges': {col: {'min': float(X_sensor[col].min()), 
                              'max': float(X_sensor[col].max())} 
                       for col in sensor_feature_cols},
    'model_version': '1.0.0',
    'training_date': '2026-09-15',
    'training_samples': len(X_sensor),
    'auc_cv_mean': float(np.mean(scores)),
    'platt_A': float(recalibrator.A),
    'platt_B': float(recalibrator.B)
}
with open('models/config_v1.json', 'w') as f:
    json.dump(config, f, indent=2)
```

### 14.3 Inference Pipeline

```python
def kneeva_inference(raw_data_paths: dict, patient_metadata: dict) -> KneevaOutput:
    """
    Full Kneeva inference pipeline: Raw sensor data → OA Risk Score.
    
    Parameters:
        raw_data_paths: dict with keys 'imu_thigh', 'imu_shank', 'acoustic',
                        'semg', 'dynamometer', 'questionnaire'
        patient_metadata: dict with 'age', 'sex', 'height_cm', 'weight_kg'
    """
    # --- LAYER 1: Preprocessing ---
    imu_data = preprocess_imu(raw_data_paths['imu_thigh'], raw_data_paths['imu_shank'])
    acoustic_data = preprocess_acoustic(raw_data_paths['acoustic'])
    semg_data = preprocess_semg(raw_data_paths['semg'])
    dyno_data = preprocess_dynamometer(raw_data_paths['dynamometer'])
    questionnaire_data = load_questionnaire(raw_data_paths['questionnaire'])
    
    # --- LAYER 2: Feature Extraction ---
    # Tier A
    tier_a = {}
    tier_a.update(extract_rom_features(imu_data['knee_angle']))
    tier_a.update(extract_crepitus_features(*acoustic_data))
    tier_a.update({'cocontraction_cci_walking_mean': calculate_cocontraction_index(
        semg_data['RF']['amplitude_mvc'], semg_data['BF']['amplitude_mvc']
    )})
    tier_a.update(extract_strength_features(*dyno_data, patient_metadata['weight_kg']))
    
    # Tier B
    tier_b = {}
    tier_b.update(extract_gait_asymmetry_features(imu_data['gait_features']))
    tier_b.update(extract_neuromuscular_features(semg_data, imu_data['gait_cycles']))
    tier_b.update(extract_general_weakness_features(
        *dyno_data, patient_metadata['weight_kg'],
        patient_metadata['age'], patient_metadata['sex']
    ))
    tier_b.update(extract_injury_history_features(questionnaire_data))
    
    # --- LAYER 3: Contextual Features ---
    tier_c = extract_contextual_features(patient_metadata, questionnaire_data)
    
    # --- LAYER 4: Fusion & Recalibration ---
    feature_vector = assemble_feature_vector(tier_a, tier_b, tier_c)
    X = pd.DataFrame([feature_vector])[sensor_feature_cols]  # ensure correct column order
    
    # Fusion model prediction
    raw_logit = final_model.predict(X, prediction_type='RawFormulaVal')[0]
    
    # Platt scaling recalibration
    oa_risk_score = recalibrator.predict_risk_score(np.array([raw_logit]))[0]
    
    # Differential triage
    diff_result = compute_differential_triage(tier_b, oa_risk_score)
    
    # SHAP explanation
    explainer = shap.TreeExplainer(final_model)
    shap_vals = explainer.shap_values(X)[0]
    top_features = sorted(
        zip(sensor_feature_cols, shap_vals),
        key=lambda x: abs(x[1]), reverse=True
    )[:5]
    
    return KneevaOutput(
        oa_risk_score=oa_risk_score,
        oa_risk_category=get_risk_category(oa_risk_score),
        differential_signal=diff_result['differential_signal'],
        differential_flags=diff_result['differential_flags'],
        feature_importance=dict(top_features),
        confidence_interval=(max(0, oa_risk_score - 0.1), min(1, oa_risk_score + 0.1))
    )
```

---

## 15. Appendices

### Appendix A: Python Dependencies

```
# requirements.txt
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
scikit-learn>=1.3.0
catboost>=1.2.0
xgboost>=2.0.0
shap>=0.43.0
neurokit2>=0.2.7
ahrs>=0.3.1
gaitpy>=1.6.1
librosa>=0.10.0
noisereduce>=3.0.0
matplotlib>=3.7.0
joblib>=1.3.0
```

### Appendix B: Reference Literature

| # | Reference | Relevance |
|---|-----------|-----------|
| 1 | Kellgren & Lawrence (1957). Radiological assessment of OA. | KL grading system — gold standard labels |
| 2 | Winter (1990). Biomechanics and Motor Control. | Co-contraction index formula |
| 3 | Platt (1999). Probabilistic outputs for SVMs. | Platt scaling methodology |
| 4 | Bohannon (2012). Knee extension strength norms. | Strength deficit reference values |
| 5 | Madgwick et al. (2011). IMU orientation estimation. | AHRS filter algorithm |
| 6 | Bellamy et al. (1988). WOMAC OA Index. | Questionnaire scoring |
| 7 | Roos et al. (1998). KOOS development. | Questionnaire scoring |
| 8 | Grinsztajn et al. (2022). Tree-based models vs. deep learning on tabular data. | Justification for GBDT over neural networks |
| 9 | Shark et al. (2011). Knee acoustic emission detection. | Crepitus DSP methodology |
| 10 | SENIAM project. Surface EMG electrode placement. | sEMG placement protocol |

### Appendix C: Glossary

| Term | Definition |
|------|-----------|
| **AHRS** | Attitude and Heading Reference System — algorithm to estimate 3D orientation from IMU data |
| **CCI** | Co-Contraction Index — measure of simultaneous agonist-antagonist muscle activation |
| **DSP** | Digital Signal Processing — mathematical transformation of digital signals |
| **GBDT** | Gradient Boosted Decision Trees — ensemble ML algorithm (CatBoost/XGBoost) |
| **H:Q Ratio** | Hamstring-to-Quadriceps strength ratio |
| **IMU** | Inertial Measurement Unit — accelerometer + gyroscope (± magnetometer) |
| **KL Grade** | Kellgren-Lawrence radiographic OA severity grade (0–4) |
| **KOOS** | Knee Injury and Osteoarthritis Outcome Score |
| **MVC** | Maximum Voluntary Contraction — used for sEMG normalization |
| **NER** | North Eastern Region (of India) — target deployment population |
| **OAI** | Osteoarthritis Initiative — NIH-funded longitudinal cohort study |
| **ROM** | Range of Motion — angular excursion of a joint |
| **sEMG** | Surface Electromyography — non-invasive measurement of muscle electrical activity |
| **SENIAM** | Surface EMG for Non-Invasive Assessment of Muscles — electrode placement guidelines |
| **SHAP** | SHapley Additive exPlanations — game-theoretic approach to explain ML predictions |
| **WOMAC** | Western Ontario and McMaster Universities Osteoarthritis Index |

---

> **Document End**  
> For questions, contact the Kneeva engineering team.
