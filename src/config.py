"""
Configuration dataclasses for the Kneeva OA Triage System.

All tunable parameters are defined here with defaults sourced directly
from the Kneeva_OA_Triage_System_Build_Guide.md.  No numerical constants
should appear in the processing modules — they read from these configs.

To override defaults, either:
  1. Instantiate a config with keyword arguments, or
  2. Load from a JSON file via PreprocessingConfig.from_json(path).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IMU  (Build Guide §4.1)
# ---------------------------------------------------------------------------
@dataclass
class IMUConfig:
    """Configuration for IMU preprocessing pipeline (MPU6050)."""

    # Sampling rate in Hz — guide specifies 100–200 Hz minimum
    sampling_rate: float = 200.0

    # CSV column mapping (matches guide §2.1 raw data format)
    timestamp_col: str = "timestamp_ms"
    accel_cols: Tuple[str, ...] = ("ax", "ay", "az")
    gyro_cols: Tuple[str, ...] = ("gx", "gy", "gz")

    # Which index in accel_cols is the vertical axis for GaitPy
    vertical_accel_index: int = 1  # ay

    # --- MPU6050-specific: raw I2C value scaling ---
    # Set raw_input=True when the CSV contains raw 16-bit signed I2C
    # values instead of pre-scaled m/s² and rad/s.
    raw_input: bool = False

    # Accelerometer sensitivity (LSB per g) based on configured FS_SEL:
    #   ±2g  → 16384,  ±4g → 8192,  ±8g → 4096,  ±16g → 2048
    accel_sensitivity_lsb_per_g: float = 16384.0

    # Gyroscope sensitivity (LSB per °/s) based on configured FS_SEL:
    #   ±250°/s → 131,  ±500 → 65.5,  ±1000 → 32.8,  ±2000 → 16.4
    gyro_sensitivity_lsb_per_dps: float = 131.0

    # Butterworth bandpass filter (§4.1 Step 4)
    filter_lowcut_hz: float = 0.5
    filter_highcut_hz: float = 20.0
    filter_order: int = 4

    # GaitPy parameters (§4.1 Step 3)
    gaitpy_ic_prominence: float = 15.0
    gaitpy_fc_prominence: float = 15.0
    gaitpy_v_acc_units: str = "m/s^2"
    gaitpy_ts_units: str = "ms"


# ---------------------------------------------------------------------------
# Acoustic / Crepitus  (Build Guide §4.2)
# ---------------------------------------------------------------------------
@dataclass
class AcousticConfig:
    """Configuration for acoustic crepitus preprocessing."""

    # Audio loading (§4.2 Step 1) — used when input_from_adc=False
    target_sample_rate: int = 44100
    mono: bool = True

    # --- Piezo vibration sensor ADC input mode ---
    # The purchased Piezo Vibration Sensor – Large outputs raw AC voltage
    # (self-generated, up to ±90 V raw, clamped via 1 MΩ load resistor +
    # MCP6002 conditioning circuit to ADC-safe levels).  When True, input
    # is read from a CSV of ADC voltage samples instead of a WAV file.
    input_from_adc: bool = False
    adc_sample_rate: int = 1000    # Hz — typical microcontroller ADC rate
    adc_reference_voltage: float = 3.3  # V — ADC reference voltage
    adc_resolution_bits: int = 10       # 10-bit ADC (Arduino default)
    adc_voltage_col: str = "voltage"    # CSV column name for voltage
    adc_timestamp_col: str = "timestamp_ms"  # CSV column for timestamps

    # Validation thresholds
    min_duration_sec: float = 10.0
    min_peak_amplitude: float = 0.01

    # Noise reduction — noisereduce (§4.2 Step 2)
    noise_profile_duration_sec: float = 0.5
    noise_prop_decrease: float = 0.8
    noise_stationary: bool = True

    # STFT / Mel spectrogram (§4.2 Step 3)
    stft_n_fft: int = 2048
    stft_hop_length: int = 512
    mel_n_mels: int = 128
    mel_fmin_hz: float = 50.0   # Crepitus typically 50–1000 Hz
    mel_fmax_hz: float = 2000.0

    # Crepitus event detection (§4.2 Step 4)
    crepitus_threshold_std_multiplier: float = 2.0
    crepitus_min_distance_sec: float = 0.2
    crepitus_peak_prominence: float = 0.1


# ---------------------------------------------------------------------------
# sEMG  (Build Guide §4.3)
# ---------------------------------------------------------------------------
@dataclass
class SEMGConfig:
    """Configuration for sEMG preprocessing."""

    sampling_rate: int = 1000  # Hz — guide specifies 1 kHz minimum

    # Channel labels (matches §2.3 SENIAM placement table)
    channels: Tuple[str, ...] = ("VM", "VL", "RF", "BF")

    # NeuroKit2 processing methods (used only when input_is_envelope=False)
    clean_method: str = "biosppy"       # Bandpass 20–450 Hz + notch 50 Hz
    activation_method: str = "threshold"

    # --- Hardware-specific: Advancer Technologies EMG Muscle Sensor V3.0 ---
    # The V3.0 board outputs a pre-processed envelope (filtered, rectified,
    # amplified) as a 0–Vs analog voltage.  When True, the NeuroKit2
    # emg_clean() and emg_amplitude() steps are SKIPPED — the ADC signal
    # is treated directly as the amplitude envelope.
    input_is_envelope: bool = False

    # Supply voltage of the sensor board (V).  Used to convert raw ADC
    # readings (0–Vs) into a normalised 0–1 range when input_is_envelope
    # is True.  Set to 0 to skip normalisation (data already in µV/volts).
    sensor_supply_voltage: float = 0.0


# ---------------------------------------------------------------------------
# Dynamometer  (Build Guide §4.4)
# ---------------------------------------------------------------------------
@dataclass
class DynamometerConfig:
    """Configuration for dynamometer peak-force extraction."""

    sampling_rate: float = 80.0  # Hz — load-cell / HX711 default

    # Smoothing window (§4.4: uniform_filter1d size = 0.1 * sr)
    smooth_window_sec: float = 0.1

    # Peak detection thresholds
    peak_height_fraction: float = 0.3   # At least 30 % of max
    peak_min_distance_sec: float = 3.0  # At least 3 s between MVC peaks
    peak_prominence_n: float = 10.0     # At least 10 N prominence
    n_trials: int = 3


# ---------------------------------------------------------------------------
# Questionnaire  (Build Guide §4.5)
# ---------------------------------------------------------------------------
@dataclass
class WOMACConfig:
    """WOMAC scoring parameters (Bellamy et al., 1988)."""
    pain_max: int = 20       # 5 items × 0–4
    stiffness_max: int = 8   # 2 items × 0–4
    function_max: int = 68   # 17 items × 0–4
    total_max: int = 96      # 24 items × 0–4


@dataclass
class KOOSConfig:
    """KOOS scoring parameters (Roos et al., 1998)."""
    max_item_score: int = 4
    subscale_items: Dict[str, int] = field(default_factory=lambda: {
        "pain": 9,
        "symptoms": 7,
        "adl": 17,
        "sport_rec": 5,
        "qol": 4,
    })


@dataclass
class QuestionnaireConfig:
    """Aggregated questionnaire scoring configuration."""
    womac: WOMACConfig = field(default_factory=WOMACConfig)
    koos: KOOSConfig = field(default_factory=KOOSConfig)


# ---------------------------------------------------------------------------
# Flex Sensor  (2.2" Bend Sensor — variable resistor)
# ---------------------------------------------------------------------------
@dataclass
class FlexSensorConfig:
    """Configuration for the 2.2" Flex Bend Sensor.

    The sensor is a variable resistor: ~10 KΩ flat, ~60–110 KΩ fully bent.
    Paired with a fixed resistor in a voltage divider, the microcontroller
    ADC reads a voltage proportional to bend angle.
    """

    sampling_rate: float = 100.0  # Hz

    # CSV column mapping
    timestamp_col: str = "timestamp_ms"
    voltage_col: str = "flex_v"  # ADC voltage column

    # Voltage divider circuit parameters
    supply_voltage: float = 3.3        # V_in to the divider
    fixed_resistor_ohms: float = 10000.0  # R_fixed in divider (match R_flat)

    # Flex sensor resistance characteristics (from datasheet)
    flat_resistance_ohms: float = 10000.0   # ~10 KΩ when flat
    max_bend_resistance_ohms: float = 110000.0  # ~110 KΩ at max bend

    # Angle calibration (linear mapping from resistance to degrees)
    flat_angle_deg: float = 180.0   # fully straight leg
    max_bend_angle_deg: float = 45.0  # deep bend

    # ADC parameters
    adc_reference_voltage: float = 3.3
    adc_resolution_bits: int = 10

    # Smoothing — moving average window in seconds
    smooth_window_sec: float = 0.05


# ===========================================================================
# LAYER 2 — Feature Extraction  (Build Guide §5)
# ===========================================================================

# ---------------------------------------------------------------------------
# Tier A: OA-Associated Signal Features  (§5.1)
# ---------------------------------------------------------------------------
@dataclass
class ROMConfig:
    """Feature A1: Range of Motion parameters."""

    # Normal full flexion angle (degrees) — deficit measured from this
    normal_max_flexion_deg: float = 135.0

    # Normal full extension angle (degrees) — should be ~0
    normal_full_extension_deg: float = 0.0


@dataclass
class CrepitusFeatureConfig:
    """Feature A2: Crepitus scoring parameters."""

    # Window (seconds) around each crepitus peak for energy calculation
    event_window_sec: float = 0.05

    # Minimum crepitus events to flag presence (binary feature)
    presence_threshold_count: int = 2


@dataclass
class StrengthConfig:
    """Feature A4: Strength deficit parameters."""

    # Body-weight normalisation uses gravity constant
    gravity_ms2: float = 9.81


@dataclass
class GeneralWeaknessConfig:
    """Feature B3: Population-norm strength comparison.

    Extension body-weight ratio norms by sex and age bracket
    (Andrews et al., 1996; Bohannon, 1997).
    """
    # {sex: {age_bracket: expected_ext_bw_ratio}}
    norms: Dict[str, Dict[int, float]] = field(default_factory=lambda: {
        "male":   {30: 0.65, 50: 0.55, 70: 0.40},
        "female": {30: 0.55, 50: 0.45, 70: 0.35},
    })

    # Assumed coefficient of variation for z-score calculation
    norm_cv: float = 0.15

    # Z-score threshold for flagging weakness
    weakness_z_threshold: float = -1.5


@dataclass
class FeatureExtractionConfig:
    """Top-level configuration for Layer 2 feature extraction."""

    rom: ROMConfig = field(default_factory=ROMConfig)
    crepitus: CrepitusFeatureConfig = field(default_factory=CrepitusFeatureConfig)
    strength: StrengthConfig = field(default_factory=StrengthConfig)
    general_weakness: GeneralWeaknessConfig = field(default_factory=GeneralWeaknessConfig)


# ===========================================================================
# LAYER 3 — Contextual Meta-Inputs  (Build Guide §6)
# ===========================================================================

@dataclass
class ContextualConfig:
    """Configuration for Layer 3 contextual feature extraction.

    Addresses the BMI Underestimation Trap for Indian Mountainous / NER
    populations by using ICMR Asian-Indian cutoffs and an Effective Knee
    Load Index that accounts for carried loads and incline transit.
    """

    # --- ICMR / WHO Asian-Indian BMI thresholds ---
    # (WHO Expert Consultation, Lancet 2004; ICMR guidelines)
    bmi_underweight: float = 18.5
    bmi_normal_upper: float = 22.9
    bmi_overweight_upper: float = 24.9
    # >= 25.0 is obese under Asian-Indian classification

    # --- Effective Knee Load Index ---
    # Incline load coefficient: per-hour multiplier for daily slope transit
    incline_load_alpha: float = 0.10

    # Default carried load (kg) when patient does not report
    default_carried_load_kg: float = 0.0

    # Default daily incline hours when patient does not report
    default_incline_hours: float = 0.0

    # --- Regional baseline prevalence (from LASI Wave-1 hill states) ---
    # Used to initialise Platt scaling intercept B
    regional_oa_prevalence: float = 0.18


@dataclass
class CalibrationConfig:
    """Platt Scaling recalibration parameters.

    Equation: P(OA=1 | s) = 1 / (1 + exp(A * s + B))
    where s is the raw GBDT logit output.

    A and B are initialised from the regional prevalence prior and
    refined when local labeled data (50–80 patients) becomes available.
    """

    # Slope — controls confidence stretch/shrink (negative: higher logit → higher risk)
    platt_A: float = -1.0

    # Intercept — shifts the prior; default derived from regional prevalence
    #   B = ln((1 - P0) / P0) where P0 = 0.18 → B ≈ 1.516
    platt_B: float = 1.516

    # L2 regularisation strength for fit()
    l2_lambda: float = 1e-4

    # Convergence criteria for fit()
    max_iter: int = 200
    tol: float = 1e-6


@dataclass
class FusionConfig:
    """Layer 4 fusion model and output configuration."""

    # Risk tier boundaries (inclusive lower, exclusive upper)
    tier_low_upper: float = 0.20
    tier_moderate_upper: float = 0.50
    tier_high_upper: float = 0.80
    # >= 0.80 is very_high

    # Clinical action text per tier
    clinical_actions: Dict[str, str] = field(default_factory=lambda: {
        "low": (
            "Reassurance + lifestyle advice (weight management, "
            "regular low-impact exercise). No immediate specialist referral."
        ),
        "moderate": (
            "Primary care follow-up within 4 weeks. Conservative management: "
            "physiotherapy, load modification, analgesics PRN."
        ),
        "high": (
            "Refer to orthopedics. Recommend weight-bearing AP X-ray. "
            "Consider MRI if mechanical symptoms present."
        ),
        "very_high": (
            "Urgent orthopedic referral. High likelihood of structural "
            "joint degeneration requiring intervention."
        ),
    })

    # CI half-width base (widened per missing modality)
    ci_base_half_width: float = 0.08
    ci_missing_modality_penalty: float = 0.04

    # SHAP — number of top features to report
    top_k_features: int = 5


# ---------------------------------------------------------------------------
# Top-level aggregator
# ---------------------------------------------------------------------------
@dataclass
class PreprocessingConfig:
    """Top-level configuration containing all Layer 1–4 parameters."""

    imu: IMUConfig = field(default_factory=IMUConfig)
    acoustic: AcousticConfig = field(default_factory=AcousticConfig)
    semg: SEMGConfig = field(default_factory=SEMGConfig)
    dynamometer: DynamometerConfig = field(default_factory=DynamometerConfig)
    questionnaire: QuestionnaireConfig = field(default_factory=QuestionnaireConfig)
    flex_sensor: FlexSensorConfig = field(default_factory=FlexSensorConfig)

    # Layer 3 — Contextual
    contextual: ContextualConfig = field(default_factory=ContextualConfig)

    # Layer 4 — Fusion & Calibration
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_json(cls, path: str | Path) -> "PreprocessingConfig":
        """Load configuration overrides from a JSON file.

        Only keys present in the JSON are overridden; everything else
        keeps the Build-Guide defaults.
        """
        with open(path, "r") as fh:
            data: Dict[str, Any] = json.load(fh)

        return cls(
            imu=IMUConfig(**data.get("imu", {})),
            acoustic=AcousticConfig(**data.get("acoustic", {})),
            semg=SEMGConfig(**data.get("semg", {})),
            dynamometer=DynamometerConfig(**data.get("dynamometer", {})),
            questionnaire=QuestionnaireConfig(
                womac=WOMACConfig(
                    **data.get("questionnaire", {}).get("womac", {})
                ),
                koos=KOOSConfig(
                    **data.get("questionnaire", {}).get("koos", {})
                ),
            ),
            flex_sensor=FlexSensorConfig(
                **data.get("flex_sensor", {})
            ),
            contextual=ContextualConfig(
                **data.get("contextual", {})
            ),
            calibration=CalibrationConfig(
                **data.get("calibration", {})
            ),
            fusion=FusionConfig(
                **{k: v for k, v in data.get("fusion", {}).items()
                   if k != "clinical_actions"},
                **({"clinical_actions": data["fusion"]["clinical_actions"]}
                   if "clinical_actions" in data.get("fusion", {}) else {}),
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Recursively serialise to a plain dict (JSON-safe)."""
        import dataclasses
        return dataclasses.asdict(self)

    def save_json(self, path: str | Path) -> None:
        """Persist current config to a JSON file."""
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info("Config saved to %s", path)
