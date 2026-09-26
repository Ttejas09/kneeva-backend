"""
Kneeva Inference Pipeline — Build Guide §14.3

End-to-end orchestrator: Raw sensor data → Calibrated OA Risk Score
with clinical explanation and differential triage signal.

Output is a ``KneevaOutput`` dataclass containing:
  • oa_risk_score (0.0–1.0 calibrated probability)
  • oa_risk_category (low / moderate / high / very_high)
  • confidence_interval (95% CI, widened per missing modality)
  • feature_importance (top-K SHAP values)
  • clinical_explanation (clinician-facing narrative)
  • clinical_action (recommended next step)
  • differential_signal / differential_flags
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import (
    CalibrationConfig,
    ContextualConfig,
    FusionConfig,
    PreprocessingConfig,
)
from src.models.calibration import PlattScaler
from src.models.explainer import ClinicalExplainer
from src.models.fusion import KneevaFusionModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------
@dataclass
class KneevaOutput:
    """Final output of the Kneeva OA Triage System.

    This is a calibrated risk assessment with clinical explanation,
    NOT a binary classification.
    """

    # Primary output — calibrated probability
    oa_risk_score: float
    """Calibrated OA risk probability (0.00–1.00)."""

    oa_risk_category: str
    """Risk tier: 'low', 'moderate', 'high', or 'very_high'."""

    confidence_interval: Tuple[float, float]
    """Approximate 95% CI on the risk score."""

    # Explainability
    feature_importance: Dict[str, float]
    """Top-K SHAP feature attributions (feature → SHAP value)."""

    clinical_explanation: str
    """Clinician-facing narrative summarising the risk drivers."""

    clinical_action: str
    """Recommended clinical action for this risk tier."""

    # Differential triage
    differential_signal: bool
    """True if feature pattern suggests an alternative pathology."""

    differential_flags: List[str]
    """Suspected alternative conditions (e.g., 'possible_meniscal_tear')."""

    # Raw components (for audit / debugging)
    raw_logit: float = 0.0
    """Raw GBDT margin before Platt scaling."""

    missing_modality_count: int = 0
    """Number of sensor modalities absent from this assessment."""


# ---------------------------------------------------------------------------
# Differential triage (rule-based, Build Guide §8)
# ---------------------------------------------------------------------------
def compute_differential_triage(
    tier_b_features: Dict[str, float],
    oa_risk_score: float,
) -> Dict[str, Any]:
    """Check for biomechanical patterns suggesting alternative pathology.

    Returns
    -------
    dict
        'differential_signal': bool
        'differential_flags': list[str]
    """
    flags: List[str] = []

    # High asymmetry + low OA score → possible acute unilateral injury
    asymmetry = tier_b_features.get("gait_step_time_asymmetry", 0.0)
    if asymmetry > 0.15 and oa_risk_score < 0.4:
        flags.append("possible_meniscal_tear")

    # ACL / meniscal history + high instability markers
    if tier_b_features.get("injury_acl_history", 0) > 0:
        flags.append("acl_injury_history")
    if tier_b_features.get("injury_meniscal_history", 0) > 0:
        flags.append("meniscal_injury_history")

    # Marked general weakness without joint-specific findings
    if (tier_b_features.get("strength_general_weakness_flag", 0) > 0
            and oa_risk_score < 0.3):
        flags.append("possible_systemic_weakness_non_oa")

    return {
        "differential_signal": len(flags) > 0,
        "differential_flags": flags,
    }


# ---------------------------------------------------------------------------
# Inference orchestrator
# ---------------------------------------------------------------------------
class KneevaInferencePipeline:
    """Configurable inference pipeline.

    Holds references to the fusion model, Platt scaler, and explainer.
    Call ``run()`` with a merged feature dictionary (Tier A + B + C)
    to produce a ``KneevaOutput``.
    """

    def __init__(
        self,
        fusion_model: Optional[KneevaFusionModel] = None,
        platt_scaler: Optional[PlattScaler] = None,
        explainer: Optional[ClinicalExplainer] = None,
        config: Optional[PreprocessingConfig] = None,
    ):
        cfg = config or PreprocessingConfig()
        self.fusion_model = fusion_model or KneevaFusionModel()
        self.platt_scaler = platt_scaler or PlattScaler.from_config(
            cfg.calibration, cfg.fusion,
        )
        self.explainer = explainer or ClinicalExplainer(
            model=self.fusion_model.model,
            feature_names=self.fusion_model.feature_names,
            top_k=cfg.fusion.top_k_features,
        )

    def run(self, feature_vector: Dict[str, float]) -> KneevaOutput:
        """Execute the full inference pipeline.

        Parameters
        ----------
        feature_vector : dict
            Merged feature dictionary from Tier A, B, and C extractors.

        Returns
        -------
        KneevaOutput
        """
        # --- Layer 4a: Fusion model → raw logit ---
        raw_logit = self.fusion_model.predict_logit(feature_vector)
        missing = self.fusion_model.count_missing_modalities(feature_vector)

        # --- Layer 4b: Platt scaling → calibrated probability ---
        risk_score = float(
            self.platt_scaler.predict_risk_score(np.array([raw_logit]))[0]
        )
        risk_score = float(np.clip(risk_score, 0.0, 1.0))
        category = self.platt_scaler.get_risk_category(risk_score)
        ci = self.platt_scaler.confidence_interval(risk_score, missing)
        action = self.platt_scaler.get_clinical_action(category)

        # --- Explainability ---
        top_features, narrative = self.explainer.explain(feature_vector)

        # --- Differential triage ---
        # Extract Tier B features for differential checks
        tier_b = {k: v for k, v in feature_vector.items()
                  if k.startswith(("gait_", "neuro_", "injury_",
                                   "strength_general"))}
        diff = compute_differential_triage(tier_b, risk_score)

        return KneevaOutput(
            oa_risk_score=round(risk_score, 4),
            oa_risk_category=category,
            confidence_interval=ci,
            feature_importance=top_features,
            clinical_explanation=narrative,
            clinical_action=action,
            differential_signal=diff["differential_signal"],
            differential_flags=diff["differential_flags"],
            raw_logit=round(raw_logit, 6),
            missing_modality_count=missing,
        )
