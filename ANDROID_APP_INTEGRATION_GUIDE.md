# Kneeva Android App Integration Guide

Welcome to the Kneeva Mobile Integration guide! The AI and Machine Learning backend is 100% complete and ready to accept traffic. Your job is to build the Android User Interface, read the phone/wearable sensors, and send a correctly formatted JSON payload to the API.

---

## 📱 1. App UI Flow & Data Collection

To feed the AI what it needs, your Android app must guide the clinician/patient through these 3 steps:

### Step 1: Patient Profile & Questionnaire (Manual Input)
You need to collect basic demographics and mountain-specific lifestyle data.
*   **Profile**: Age, Sex (`'male'` or `'female'`), Height (cm), Weight (kg)
*   **Questionnaire**: 
    *   *How many kilograms do you carry on average daily?* (carried load)
    *   *How many hours do you spend walking on steep inclines daily?* (daily incline hours)
    *   *How difficult is squatting?* (0-4 scale)
    *   *Have you had a previous knee injury?* (Yes/No)
    *   *What is your activity level?* (1=Sedentary to 4=Heavy Manual Labor)

### Step 2: Clinical Examination (Manual Input)
The clinician manually tests the knee and enters the following into the app:
*   **ROM (Range of Motion)**: Active and Passive flexion/extension in degrees.
*   **Strength**: Maximum push/pull force measured using a Dynamometer (in Newtons).
*   **Crepitus (Acoustic)**: Count of clicking sounds heard during movement, and their estimated intensity.

### Step 3: The 2-Part Functional Walk Test (IMU Sensor Input)
This is the most critical part. You must record accelerometer and gyroscope data from the Android device (or Bluetooth wearable).
1.  **Test 1 (Flat Walking)**: Have the patient walk on a flat surface for 60 seconds.
2.  **Test 2 (Stair Climbing)**: Have the patient climb stairs for 60 seconds.

---

## ⚙️ 2. Processing IMU Data on the Device

The backend API **does not** accept raw sensor arrays (like `[0.01, 0.02, ...]`). It expects you to calculate the cadence and stride variability *on the Android device* and just send the final numbers.

### How to calculate Cadence and Stride CV in Java/Kotlin:
1.  Read the **Y-axis Gyroscope** or **Z-axis Accelerometer** data (whichever axis corresponds to the forward swing of the leg).
2.  Apply a simple peak-detection algorithm to find the "heel strikes" (each time the foot hits the ground is a peak).
3.  **Cadence**: `(Total Number of Peaks / Total Time in Minutes)`. E.g., 90 steps per minute.
4.  **Stride Time CV**: 
    *   Calculate the time difference (in seconds) between each consecutive peak. This is your array of step times.
    *   `CV = Standard Deviation(step_times) / Mean(step_times)`.
5.  Do this twice: once for the 60s Flat test, and once for the 60s Climbing test.

---

## 🌐 3. The API Contract

**Endpoint**: `POST /api/v1/triage`  
**Content-Type**: `application/json`

Your Android app must send a payload exactly like this:

```json
{
  "patient_id": "PT-10045",
  "patient_metadata": {
    "age": 55,
    "sex": "female",
    "height_cm": 158.0,
    "weight_kg": 62.0
  },
  "questionnaire": {
    "carried_load_kg": 15.0,
    "daily_incline_hours": 2.5,
    "squatting_difficulty": 3,
    "previous_injury": 0,
    "activity_level": 3
  },
  "sensor_features": {
    "rom_active_flexion_deg": 115.0,
    "rom_active_extension_deficit_deg": 5.0,
    "rom_passive_flexion_deg": 120.0,
    "rom_flexion_deficit_deg": 25.0,
    "crepitus_event_count": 4.0,
    "crepitus_total_energy": 18.5,
    "crepitus_mean_energy": 4.6,
    "crepitus_presence": 1.0,
    "cocontraction_cci_walking_mean": 0.45,
    "strength_ext_peak_n": 220.0,
    "strength_flex_peak_n": 140.0,
    "strength_ext_bw_ratio": 3.5,
    "strength_hq_ratio": 0.63,
    "gait_step_time_asymmetry": 0.12,
    "gait_swing_time_asymmetry": 0.08,
    
    "flat_gait_cadence": 98.5,
    "flat_gait_stride_time_cv": 0.04,
    "climbing_cadence": 82.0,
    "climbing_stride_time_cv": 0.11,
    "gait_speed_ms": 0.95,
    
    "neuro_rf_activation_duration_pct": 42.0,
    "neuro_bf_activation_duration_pct": 38.0,
    "neuro_onset_emg_to_heelstrike_ms": 110.0,
    "strength_ext_bw_ratio_general": 3.5,
    "strength_general_weakness_flag": 1.0,
    "strength_general_z_score": -1.2,
    "injury_previous_knee_injury": 0.0,
    "injury_acl_history": 0.0,
    "injury_meniscal_history": 0.0
  }
}
```

> **Note:** Do not worry about calculating `ctx_bmi` or `ctx_effective_bmi`. The backend calculates these automatically using the `patient_metadata` and `questionnaire` objects you provide!

---

## 🎯 4. The Response

The server will reply with the clinical diagnosis. Your Android app should display the `oa_risk_score` (as a percentage), the `oa_risk_category` (e.g., HIGH), and the `clinical_explanation` to the doctor.

```json
{
  "patient_id": "PT-10045",
  "oa_risk_score": 0.825,
  "oa_risk_category": "high",
  "confidence_interval": [0.75, 0.89],
  "feature_importance": {
    "flat_gait_stride_time_cv": 0.15,
    "climbing_cadence": 0.12
  },
  "clinical_explanation": "Patient demonstrates significantly elevated risk (82.5%). Primary drivers are high stride variability during flat walking and slow climbing cadence.",
  "clinical_action": "Refer to orthopedic specialist for immediate X-ray and conservative management.",
  "differential_signal": false,
  "differential_flags": [],
  "missing_modality_count": 0,
  "effective_bmi": 28.4
}
```
