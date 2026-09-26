# Kneeva: AI-Powered Knee Osteoarthritis Triage System 🏔️

**Kneeva** is an intelligent triage and screening system designed specifically for early detection of Knee Osteoarthritis (OA) in the Indian Mountainous Regions (e.g., Himachal Pradesh, Uttarakhand, Northeast India). 

It fuses data from a multimodal wearable sleeve (Acoustic, IMU, sEMG) with clinical history to generate a calibrated **Risk Score** and a **Clinical Explanation** for doctors.

---

## 🏗️ System Architecture

The Kneeva ecosystem consists of three main components:

1. **Hardware (Knee Sleeve):** Contains 5 sensor modalities (Microphone, IMU, sEMG, Dynamometer proxy, Goniometer). Streams raw data via Bluetooth (BLE).
2. **Android App (Frontend):** Collects BLE data, runs on-device feature extraction (e.g., peak detection, signal processing), collects patient questionnaire data, and sends a JSON payload to the backend.
3. **FastAPI Server (Backend - *This Codebase*):** Receives the JSON payload, handles missing modalities gracefully, runs the CatBoost AI model, calibrates the output using Platt Scaling, and generates SHAP-based clinical narratives.

---

## 📱 App Integration Guide

This section is for **Android/Frontend Developers** building the app around this backend.

### Running the Server

To start the local development server:
```bash
# Ensure you are in the project root
pip install -r requirements.txt
uvicorn src.server:app --host 0.0.0.0 --port 8000 --reload
```
The server will be available at `http://<your-ip>:8000`. You can view the interactive API documentation at `http://<your-ip>:8000/docs`.

---

### 🏃‍♂️ App Testing Protocol (2-Part Clinical Test)

To provide the AI with the cleanest possible biomechanical data (especially crucial for mountain populations), the app should guide the patient through two distinct tests:

1. **Test 1 (Flat):** 60 seconds of walking on flat ground. 
2. **Test 2 (Incline):** 60 seconds of climbing stairs or walking uphill.

**App-Side Feature Extraction Note:** 
By separating the tests, the app's signal processing can safely calculate cadence and variance (CV) for each terrain without false spikes. The app will then send **both** sets of features in the JSON payload, allowing the AI to see exactly how the knee degrades under load.

---

### 📡 API Contract

The Android app must send a `POST` request to the `/triage/` endpoint containing the extracted features. **Kneeva is highly fault-tolerant.** If a sensor is disconnected or a test is skipped (e.g. they couldn't find stairs), simply omit that field or send `null`. The AI will still generate the best possible prediction using the remaining data.

#### Endpoint
`POST /triage/`

#### Request Payload (JSON)

```json
{
  "patient_id": "PT-10495",
  "tier_a": {
    "rom_active_flexion_deg": 110.5,
    "crepitus_presence": 1.0,
    "strength_ext_peak_n": 245.0
  },
  "tier_b": {
    "flat_gait_cadence": 105.0,
    "flat_gait_stride_time_cv": 0.04,
    "climbing_cadence": 85.0,
    "climbing_stride_time_cv": 0.12,
    "gait_step_time_asymmetry": null, 
    "gait_swing_time_asymmetry": null
  },
  "questionnaire": {
    "age": 58,
    "weight_kg": 75.0,
    "height_cm": 165.0,
    "sex_female": 1,
    "activity_level": 6,
    "previous_injury": 1,
    "daily_carried_load_kg": 15.0,
    "daily_incline_walking_hours": 3.5
  }
}
```

*Note on Questionnaire:* The backend automatically calculates complex features like `effective_bmi` (accounting for daily carried load in mountain regions) and maps them to the Asian-Indian BMI scale.

#### Response Payload (JSON)

The app will receive the risk score and a human-readable explanation to display to the user or doctor.

```json
{
  "patient_id": "PT-10495",
  "risk_score": 0.824,
  "risk_category": "High Risk",
  "confidence_interval": [0.75, 0.89],
  "missing_modalities": ["goniometer"],
  "clinical_explanation": {
    "narrative": "High Risk OA Profile identified. The primary driving factors are elevated age (58 yrs), high effective biomechanical load (BMI + 15.0kg daily load), and reduced active flexion (110.5°). Protective factors included good baseline strength (245.0N).",
    "top_risk_factors": [
      {"feature": "ctx_age", "contribution_value": 1.5, "clinical_interpretation": "Age increases baseline risk"},
      {"feature": "ctx_effective_bmi", "contribution_value": 0.9, "clinical_interpretation": "High load stress on joint"}
    ]
  },
  "differential_flags": {
    "meniscal_involvement_suspected": false,
    "acl_laxity_suspected": false
  }
}
```

---

### 📝 Exact App Questionnaire (Tier C)

The app UI must ask the patient the following questions to populate the `questionnaire` JSON object:

1. **Age:** (Integer) e.g., `58`
2. **Weight:** (Float in kg) e.g., `75.0`
3. **Height:** (Float in cm) e.g., `165.0`
4. **Sex:** (Boolean/Binary) `1` for Female, `0` for Male
5. **Activity Level:** (Integer 1-10) Tegner Activity Score (1 = Sedentary, 10 = Elite Athlete).
6. **Previous Knee Injury:** (Boolean/Binary) `1` if they have had a prior major knee injury (ACL, meniscus), `0` otherwise.
7. **Daily Carried Load:** (Float in kg) How much weight they carry daily (e.g., carrying crops, water, heavy bags). Crucial for mountain populations. Enter `0` if none.
8. **Daily Incline Walking:** (Float in hours) How many hours per day they spend walking uphill/downhill. Enter `0` if flat terrain.

---

### 🦿 Single-Leg Prototype Guidance

If you are currently building a prototype with **only one knee sensor**, you will not be able to calculate comparison features between the left and right legs (e.g., `gait_step_time_asymmetry` and `gait_swing_time_asymmetry`). 

**How the AI handles this:**
CatBoost natively handles missing data. If you only have one sensor, calculate the features you *can* (like cadence, speed, stride time CV) and send `null` for the asymmetry features in the JSON payload:

```json
  "tier_b": {
    "gait_cadence": 95.0,
    "gait_stride_time_cv": 0.08,
    "gait_step_time_asymmetry": null, 
    "gait_swing_time_asymmetry": null
  }
```
The model will safely ignore the missing asymmetry data and generate the risk score based purely on the absolute measurements of the single leg.

---

## 🧠 AI Training & Data

If you are a Data Scientist working on the model, please refer to the project documentation for instructions on preparing the **LASI Wave 1** clinical dataset and the **Mendeley MPU6050** sensor dataset.

Training the model on combined CSV data:
```bash
python -m src.train --data data/processed/combined_dataset.csv --model-type catboost --out-dir models
```
(If no `--data` flag is provided, the script runs a synthetic data validation test to verify pipeline integrity.)

---

## 🛠️ Project Structure

- `src/server.py`: The FastAPI server connecting the app to the AI.
- `src/models/fusion.py`: The CatBoost model wrapper handling missing data.
- `src/models/calibration.py`: Platt Scaling implementation to convert raw AI logits into true clinical probabilities (0.0 - 1.0).
- `src/models/explainability.py`: SHAP implementation generating the natural language narrative.
- `src/features/tier_c.py`: Indian-specific clinical logic (e.g., Asian-Indian BMI scales, mountain load factors).
- `src/train.py`: The training and cross-validation pipeline.
- `tests/`: Pytest suite (Run with `python -m pytest tests/`).
