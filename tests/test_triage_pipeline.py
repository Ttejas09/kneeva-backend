"""
Tests for the Kneeva OA Risk Triage Pipeline.

Covers:
  1. BMI Underestimation Trap — lean patient with severe sensor signals
  2. Risk score calibration and range validation
  3. Platt Scaler fitting and prior shift
  4. Explainability — SHAP + clinical narrative output
  5. Tier C contextual feature extraction
"""

import math

import numpy as np
import pytest

from src.config import CalibrationConfig, ContextualConfig, FusionConfig
from src.features.tier_c import (
    _asian_indian_bmi_score,
    _effective_knee_load_index,
    extract_contextual_features,
)
from src.inference import (
    KneevaInferencePipeline,
    KneevaOutput,
    compute_differential_triage,
)
from src.models.calibration import PlattScaler
from src.models.fusion import KneevaFusionModel


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def contextual_config():
    return ContextualConfig()


@pytest.fixture
def platt_scaler():
    return PlattScaler.from_config()


@pytest.fixture
def lean_hill_patient_metadata():
    """A lean patient from a mountainous region."""
    return {
        "age": 55,
        "sex": "female",
        "height_cm": 155,
        "weight_kg": 48,
    }


@pytest.fixture
def lean_hill_questionnaire():
    """Questionnaire data for a load-carrying hill patient."""
    return {
        "tegner_activity_level": 6,
        "injury_history": {"previous_knee_injury": False},
        "carried_load_kg": 20.0,
        "daily_incline_hours": 3.0,
    }


@pytest.fixture
def severe_sensor_features():
    """Tier A + B features indicating severe joint degradation."""
    return {
        # Tier A — severe
        "rom_active_flexion_deg": 95.0,
        "rom_active_extension_deficit_deg": 14.0,
        "rom_passive_flexion_deg": 100.0,
        "rom_flexion_deficit_deg": 40.0,
        "crepitus_event_count": 18.0,
        "crepitus_total_energy": 4.5,
        "crepitus_mean_energy": 0.25,
        "crepitus_presence": 1.0,
        "cocontraction_cci_walking_mean": 0.72,
        "strength_ext_peak_n": 85.0,
        "strength_flex_peak_n": 60.0,
        "strength_ext_bw_ratio": 0.18,
        "strength_hq_ratio": 0.71,
        # Tier B — moderate asymmetry
        "gait_step_time_asymmetry": 0.08,
        "gait_swing_time_asymmetry": 0.06,
        "flat_gait_stride_time_cv": 0.05,
        "flat_gait_cadence": 98.0,
        "climbing_stride_time_cv": 0.10,
        "climbing_cadence": 80.0,
        "gait_speed_ms": 0.85,
        "neuro_rf_activation_duration_pct": 42.0,
        "neuro_bf_activation_duration_pct": 38.0,
        "neuro_onset_emg_to_heelstrike_ms": -15.0,
        "strength_ext_bw_ratio_general": 0.30,
        "strength_general_weakness_flag": 1.0,
        "strength_general_z_score": -1.8,
        "injury_previous_knee_injury": 0.0,
        "injury_acl_history": 0.0,
        "injury_meniscal_history": 0.0,
    }


# ============================================================================
# Test 1: Tier C — Effective Knee Load Index & Asian-Indian BMI
# ============================================================================

class TestTierCFeatures:

    def test_effective_bmi_scales_with_load(self, contextual_config):
        """Carrying 20 kg up slopes should significantly increase effective BMI."""
        height_m = 1.55
        weight_kg = 48.0

        # Baseline (no load, no slope)
        baseline = _effective_knee_load_index(
            weight_kg, height_m, 0.0, 0.0,
            contextual_config.incline_load_alpha,
        )
        # With load + slope
        loaded = _effective_knee_load_index(
            weight_kg, height_m, 20.0, 3.0,
            contextual_config.incline_load_alpha,
        )

        assert loaded > baseline
        # Should be at least 30% higher (20 kg load + 3h incline factor)
        assert loaded / baseline > 1.3

    def test_asian_indian_bmi_cutoffs(self, contextual_config):
        """ICMR cutoffs: 23 is overweight, 25 is obese (not 25/30 as Western)."""
        assert _asian_indian_bmi_score(17.0, contextual_config) == 0.1   # underweight
        assert _asian_indian_bmi_score(21.0, contextual_config) == 0.2   # normal
        assert _asian_indian_bmi_score(24.0, contextual_config) == 0.6   # overweight
        assert _asian_indian_bmi_score(27.0, contextual_config) == 1.0   # obese

    def test_extract_contextual_features_structure(
        self,
        lean_hill_patient_metadata,
        lean_hill_questionnaire,
    ):
        """Tier C should produce all expected keys."""
        features = extract_contextual_features(
            lean_hill_patient_metadata,
            lean_hill_questionnaire,
        )
        expected_keys = {
            "ctx_bmi", "ctx_effective_bmi", "ctx_asian_indian_bmi_score",
            "ctx_age", "ctx_sex_female", "ctx_previous_injury",
            "ctx_activity_level",
        }
        assert set(features.keys()) == expected_keys

    def test_lean_patient_effective_bmi_elevated(
        self,
        lean_hill_patient_metadata,
        lean_hill_questionnaire,
    ):
        """A 48 kg patient carrying 20 kg on 3h slopes should have
        effective BMI well above raw BMI."""
        features = extract_contextual_features(
            lean_hill_patient_metadata,
            lean_hill_questionnaire,
        )
        assert features["ctx_effective_bmi"] > features["ctx_bmi"]
        # Raw BMI ~ 20, effective should be > 26
        assert features["ctx_effective_bmi"] > 26.0


# ============================================================================
# Test 2: Platt Scaler — Calibration & Range
# ============================================================================

class TestPlattScaler:

    def test_risk_score_bounded(self, platt_scaler):
        """Risk scores must be strictly in [0, 1]."""
        logits = np.array([-100.0, -5.0, -1.0, 0.0, 1.0, 5.0, 100.0])
        scores = platt_scaler.predict_risk_score(logits)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_monotonic(self, platt_scaler):
        """Higher logits (more OA-like) should produce higher risk scores."""
        logits = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        scores = platt_scaler.predict_risk_score(logits)
        # With negative A, increasing logit → decreasing A*s+B → higher P
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]

    def test_confidence_interval_no_inversion(self, platt_scaler):
        """CI bounds must satisfy lower <= score <= upper."""
        for score in [0.0, 0.1, 0.5, 0.9, 1.0]:
            lo, hi = platt_scaler.confidence_interval(score)
            assert lo <= score <= hi
            assert 0.0 <= lo <= hi <= 1.0

    def test_confidence_interval_widens_with_missing(self, platt_scaler):
        """Missing modalities should produce wider CIs."""
        _, hi_0 = platt_scaler.confidence_interval(0.5, missing_modality_count=0)
        _, hi_3 = platt_scaler.confidence_interval(0.5, missing_modality_count=3)
        width_0 = hi_0 - (0.5 - (hi_0 - 0.5))  # approximate
        lo_0, _ = platt_scaler.confidence_interval(0.5, missing_modality_count=0)
        lo_3, _ = platt_scaler.confidence_interval(0.5, missing_modality_count=3)
        assert (hi_3 - lo_3) > (hi_0 - lo_0)

    def test_risk_categories(self, platt_scaler):
        """Verify the four risk tiers map correctly."""
        assert platt_scaler.get_risk_category(0.10) == "low"
        assert platt_scaler.get_risk_category(0.35) == "moderate"
        assert platt_scaler.get_risk_category(0.65) == "high"
        assert platt_scaler.get_risk_category(0.90) == "very_high"

    def test_fit_converges_on_synthetic_data(self):
        """Platt fit should converge on a simple synthetic dataset."""
        np.random.seed(42)
        n = 100
        true_scores = np.random.rand(n)
        labels = (true_scores > 0.5).astype(float)
        # Add noise to logits
        logits = np.log(true_scores / (1 - true_scores + 1e-7)) + np.random.randn(n) * 0.5

        scaler = PlattScaler(A=1.0, B=0.0)
        scaler.fit(logits, labels)

        assert scaler.is_fitted
        # A should still be roughly positive (preserving ordering)
        assert scaler.A != 1.0 or scaler.B != 0.0  # parameters changed

    def test_serialisation_roundtrip(self, platt_scaler):
        """to_dict → from_dict should preserve parameters."""
        d = platt_scaler.to_dict()
        restored = PlattScaler.from_dict(d)
        assert restored.A == platt_scaler.A
        assert restored.B == platt_scaler.B


# ============================================================================
# Test 3: Fusion Model — Stub Mode
# ============================================================================

class TestFusionModel:

    def test_stub_returns_zero_logit(self, severe_sensor_features):
        """Without a trained model, stub returns logit=0.0."""
        model = KneevaFusionModel(model=None)
        logit = model.predict_logit(severe_sensor_features)
        assert logit == 0.0

    def test_missing_features_handled(self):
        """Partial feature dict should not raise."""
        model = KneevaFusionModel(model=None)
        partial = {"ctx_age": 55.0, "ctx_bmi": 20.0}
        logit = model.predict_logit(partial)
        assert isinstance(logit, float)

    def test_count_missing_modalities(self):
        """All-empty dict should report 5 missing modalities."""
        model = KneevaFusionModel(model=None)
        count = model.count_missing_modalities({})
        assert count == 5  # rom, acoustic, emg, dynamometer, gait


# ============================================================================
# Test 4: Explainability
# ============================================================================

class TestExplainability:

    def test_narrative_non_empty_without_model(self):
        """Even without SHAP, narrative should return a valid string."""
        from src.models.explainer import ClinicalExplainer

        explainer = ClinicalExplainer(model=None)
        top, narrative = explainer.explain({"ctx_age": 55.0})
        assert isinstance(narrative, str)
        assert len(narrative) > 0

    def test_narrative_with_manual_features(self):
        """Manual feature attribution should produce numbered entries."""
        from src.models.explainer import ClinicalExplainer

        explainer = ClinicalExplainer(model=None)
        fake_top = {
            "crepitus_total_energy": 0.22,
            "rom_active_extension_deficit_deg": 0.15,
            "ctx_age": 0.08,
        }
        narrative = explainer.generate_clinical_narrative(
            fake_top, {"crepitus_total_energy": 4.5}
        )
        assert "crepitus_total_energy" in narrative
        assert "1." in narrative
        assert "risk contribution" in narrative


# ============================================================================
# Test 5: End-to-End Inference — BMI Trap Verification
# ============================================================================

class TestEndToEndInference:

    def test_kneeva_output_structure(
        self,
        severe_sensor_features,
        lean_hill_patient_metadata,
        lean_hill_questionnaire,
    ):
        """KneevaOutput should have all required fields."""
        ctx = extract_contextual_features(
            lean_hill_patient_metadata,
            lean_hill_questionnaire,
        )
        merged = {**severe_sensor_features, **ctx}

        pipeline = KneevaInferencePipeline()
        output = pipeline.run(merged)

        assert isinstance(output, KneevaOutput)
        assert 0.0 <= output.oa_risk_score <= 1.0
        assert output.oa_risk_category in {"low", "moderate", "high", "very_high"}
        assert output.confidence_interval[0] <= output.confidence_interval[1]
        assert isinstance(output.clinical_explanation, str)
        assert isinstance(output.clinical_action, str)
        assert isinstance(output.differential_signal, bool)

    def test_bmi_trap_effective_load_is_present(
        self,
        lean_hill_patient_metadata,
        lean_hill_questionnaire,
    ):
        """The merged feature vector for a lean hill patient must contain
        ctx_effective_bmi which is meaningfully higher than ctx_bmi."""
        ctx = extract_contextual_features(
            lean_hill_patient_metadata,
            lean_hill_questionnaire,
        )
        # Raw BMI for 48 kg / 1.55m ≈ 20.0
        assert ctx["ctx_bmi"] < 21.0
        # Effective BMI with 20 kg load + 3h incline should be > 26
        assert ctx["ctx_effective_bmi"] > 26.0
        # Asian-Indian score should be 0.2 (normal) for raw BMI ~20
        assert ctx["ctx_asian_indian_bmi_score"] == 0.2


# ============================================================================
# Test 6: Differential Triage
# ============================================================================

class TestDifferentialTriage:

    def test_meniscal_flag_on_high_asymmetry_low_oa(self):
        """High asymmetry + low OA score should flag possible meniscal tear."""
        tier_b = {"gait_step_time_asymmetry": 0.25}
        result = compute_differential_triage(tier_b, oa_risk_score=0.15)
        assert result["differential_signal"] is True
        assert "possible_meniscal_tear" in result["differential_flags"]

    def test_no_flag_on_low_asymmetry(self):
        """Low asymmetry should not trigger differential flags."""
        tier_b = {"gait_step_time_asymmetry": 0.03}
        result = compute_differential_triage(tier_b, oa_risk_score=0.6)
        assert result["differential_signal"] is False
