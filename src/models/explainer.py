"""
Clinical Explainability Engine — Build Guide §12.3

Provides two layers of explanation for each OA risk score:

1. **Quantitative (SHAP):** Top-K feature attributions showing which
   biomechanical signals push the score up or down.
2. **Narrative (Clinical):** Human-readable text mapping each feature
   to its clinical interpretation, suitable for clinician reports.
"""

from __future__ import annotations
import shap
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature → Clinical interpretation mapping
# ---------------------------------------------------------------------------
_FEATURE_CLINICAL_MAP: Dict[str, Dict[str, str]] = {
    # --- Tier A: OA-associated ---
    "rom_active_flexion_deg": {
        "up": "Reduced active flexion indicates joint stiffness or capsular restriction.",
        "down": "Good active flexion range — no significant capsular restriction.",
    },
    "rom_active_extension_deficit_deg": {
        "up": "Extension lag suggests effusion, osteophyte block, or flexion contracture.",
        "down": "Full active extension achieved — no mechanical block.",
    },
    "rom_flexion_deficit_deg": {
        "up": "Flexion deficit from normal ({value:.0f}°) suggests joint degeneration.",
        "down": "Flexion within normal range.",
    },
    "crepitus_event_count": {
        "up": "High crepitus event count indicates cartilage surface irregularity.",
        "down": "Low crepitus count — smooth articular surface likely.",
    },
    "crepitus_total_energy": {
        "up": "Elevated crepitus energy indicates cartilage fibrillation or roughening.",
        "down": "Low crepitus energy — minimal cartilage roughening.",
    },
    "crepitus_mean_energy": {
        "up": "High mean energy per crepitus event suggests deeper cartilage lesions.",
        "down": "Low per-event energy — surface fibrillation only.",
    },
    "crepitus_presence": {
        "up": "Crepitus detected — consistent with chondral damage.",
        "down": "No significant crepitus detected.",
    },
    "cocontraction_cci_walking_mean": {
        "up": "Elevated co-contraction index indicates neuromuscular guarding (pain-avoidance).",
        "down": "Normal co-contraction — no compensatory guarding pattern.",
    },
    "strength_ext_peak_n": {
        "up": "Low extension peak force indicates quadriceps weakness.",
        "down": "Adequate quadriceps extension strength.",
    },
    "strength_flex_peak_n": {
        "up": "Low flexion peak force indicates hamstring weakness.",
        "down": "Adequate hamstring flexion strength.",
    },
    "strength_ext_bw_ratio": {
        "up": "Low body-weight-normalised extension strength — quadriceps insufficiency.",
        "down": "Adequate body-weight-normalised extension strength.",
    },
    "strength_hq_ratio": {
        "up": "Abnormal hamstring:quadriceps ratio indicates muscle imbalance.",
        "down": "Balanced hamstring:quadriceps strength ratio.",
    },
    # --- Tier B: General / Differential ---
    "gait_step_time_asymmetry": {
        "up": "Step time asymmetry suggests antalgic gait (pain-related limping).",
        "down": "Symmetric step timing — no antalgic gait.",
    },
    "gait_swing_time_asymmetry": {
        "up": "Swing time asymmetry indicates compensatory weight-bearing shift.",
        "down": "Symmetric swing phase — balanced weight-bearing.",
    },
    "flat_gait_stride_time_cv": {
        "up": "High flat-walking stride variability indicates unstable or painful gait on level ground.",
        "down": "Consistent flat-walking stride timing — stable gait pattern on level ground.",
    },
    "flat_gait_cadence": {
        "up": "Abnormal flat-walking cadence may indicate compensatory gait.",
        "down": "Normal flat-walking cadence.",
    },
    "climbing_stride_time_cv": {
        "up": "High stair-climbing stride variability — knee instability worsens under load.",
        "down": "Consistent stair-climbing stride timing — knee stable under load.",
    },
    "climbing_cadence": {
        "up": "Abnormal stair-climbing cadence may indicate pain or weakness under load.",
        "down": "Normal stair-climbing cadence.",
    },
    "gait_speed_ms": {
        "up": "Reduced gait speed is a strong functional decline marker.",
        "down": "Normal gait speed — adequate functional mobility.",
    },
    # --- Tier C: Contextual ---
    "ctx_bmi": {
        "up": "Elevated BMI increases mechanical load on weight-bearing joints.",
        "down": "BMI within range.",
    },
    "ctx_effective_bmi": {
        "up": "High effective load (body weight + carried loads on slopes) increases joint stress.",
        "down": "Effective mechanical load within manageable range.",
    },
    "ctx_asian_indian_bmi_score": {
        "up": "Asian-Indian BMI classification indicates elevated metabolic risk.",
        "down": "Asian-Indian BMI classification within normal range.",
    },
    "ctx_age": {
        "up": "Advanced age is a primary non-modifiable risk factor for OA.",
        "down": "Younger age lowers baseline OA risk.",
    },
    "ctx_sex_female": {
        "up": "Female sex confers ~1.8x higher OA risk (hormonal, biomechanical).",
        "down": "Male sex — lower baseline hormonal OA risk.",
    },
    "ctx_previous_injury": {
        "up": "Prior knee injury (ACL/meniscal) increases OA risk 4 - 6x.",
        "down": "No prior knee injury history.",
    },
    "ctx_activity_level": {
        "up": "Activity level outside the protective mid-range.",
        "down": "Moderate activity level — protective for joint health.",
    },
}


class ClinicalExplainer:
    """SHAP-based explainability with clinical narrative generation.

    Parameters
    ----------
    model : object
        The underlying GBDT model (CatBoost/XGBoost/sklearn).
        If None, SHAP explanations are skipped and only narrative
        stubs are generated.
    feature_names : list[str]
        Canonical feature ordering matching the fusion model.
    top_k : int
        Number of top features to include in explanations.
    """

    def __init__(
        self,
        model: Any = None,
        feature_names: Optional[List[str]] = None,
        top_k: int = 5,
    ):
        self.model = model
        self.feature_names = feature_names or []
        self.top_k = top_k
        self._explainer: Any = None

    def _ensure_explainer(self) -> None:
        """Lazily initialise the SHAP TreeExplainer."""
        if self._explainer is not None or self.model is None:
            return
        try:
            self._explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP TreeExplainer initialised.")
        except Exception as exc:
            logger.warning("Could not create SHAP explainer: %s", exc)
            self._explainer = None

    # ------------------------------------------------------------------
    # SHAP attribution
    # ------------------------------------------------------------------
    def compute_shap_values(
        self,
        feature_vector: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute per-feature SHAP values for a single patient.

        Returns a dict of feature_name → SHAP value, sorted by
        absolute magnitude (descending), limited to top_k.
        """
        import pandas as pd

        if self.model is None:
            logger.info("No model loaded — returning empty SHAP values.")
            return {}

        self._ensure_explainer()
        if self._explainer is None:
            return {}

        row = {f: feature_vector.get(f, np.nan) for f in self.feature_names}
        X = pd.DataFrame([row], columns=self.feature_names)

        shap_vals = self._explainer.shap_values(X)
        # Handle multi-output (binary classification may return list)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

        sv = shap_vals[0] if shap_vals.ndim > 1 else shap_vals

        pairs = sorted(
            zip(self.feature_names, sv),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return {name: float(val) for name, val in pairs[: self.top_k]}

    # ------------------------------------------------------------------
    # Clinical narrative
    # ------------------------------------------------------------------
    def generate_clinical_narrative(
        self,
        top_features: Dict[str, float],
        feature_values: Optional[Dict[str, float]] = None,
    ) -> str:
        """Generate a clinician-facing natural language explanation.

        Parameters
        ----------
        top_features : dict
            Feature name → SHAP value (from ``compute_shap_values``).
        feature_values : dict, optional
            Actual feature values for richer text (e.g., "ROM = 110°").

        Returns
        -------
        str
            Multi-line clinical narrative suitable for a triage report.
        """
        if not top_features:
            return "Insufficient data for detailed clinical explanation."

        feature_values = feature_values or {}
        lines = ["**Primary Risk Drivers:**", ""]

        for rank, (feat, shap_val) in enumerate(top_features.items(), 1):
            direction = "up" if shap_val > 0 else "down"
            magnitude = abs(shap_val)

            # Look up clinical text
            clinical = _FEATURE_CLINICAL_MAP.get(feat, {})
            explanation = clinical.get(
                direction,
                f"{'Increases' if direction == 'up' else 'Decreases'} OA risk.",
            )

            # Inject actual value if available
            actual = feature_values.get(feat)
            value_str = f" (value: {actual:.2f})" if actual is not None else ""

            sign = "+" if shap_val > 0 else "−"
            lines.append(
                f"{rank}. **{feat}**{value_str}: "
                f"{explanation} ({sign}{magnitude:.3f} risk contribution)"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Combined explanation
    # ------------------------------------------------------------------
    def explain(
        self,
        feature_vector: Dict[str, float],
    ) -> Tuple[Dict[str, float], str]:
        """Compute SHAP values and generate clinical narrative in one call.

        Returns
        -------
        tuple[dict, str]
            (top_features dict, clinical_narrative string)
        """
        top = self.compute_shap_values(feature_vector)
        narrative = self.generate_clinical_narrative(top, feature_vector)
        return top, narrative
