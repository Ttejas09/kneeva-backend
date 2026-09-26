"""
Layer 3 — Contextual Meta-Input Feature Extraction (Build Guide §6)

Extracts Tier C contextual features that bypass Layer 2 and feed
directly into the fusion model.  Addresses the BMI Underestimation Trap
for Indian Mountainous / NER populations by computing:

  1. Raw BMI
  2. ICMR Asian-Indian BMI risk score (categorical ordinal)
  3. Effective Knee Load Index (body weight + carried load × incline factor)

All thresholds come from ContextualConfig — no magic numbers here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.config import ContextualConfig

logger = logging.getLogger(__name__)


def _asian_indian_bmi_score(bmi: float, cfg: ContextualConfig) -> float:
    """Map raw BMI to an ordinal risk score using ICMR cutoffs.

    Returns a value in {0.1, 0.2, 0.6, 1.0} representing increasing
    metabolic/mechanical risk.  Unlike Western cutoffs (obese ≥30),
    ICMR classifies ≥25 as obese for Asian-Indian populations.

    Also assigns 0.1 for underweight (frailty / sarcopenia risk).
    """
    if bmi < cfg.bmi_underweight:
        return 0.1   # underweight — frailty risk
    elif bmi <= cfg.bmi_normal_upper:
        return 0.2   # normal
    elif bmi <= cfg.bmi_overweight_upper:
        return 0.6   # overweight (Asian-Indian)
    else:
        return 1.0   # obese (Asian-Indian, ≥25)


def _effective_knee_load_index(
    weight_kg: float,
    height_m: float,
    carried_load_kg: float,
    daily_incline_hours: float,
    alpha: float,
) -> float:
    """Compute the Effective Knee Load Index.

    Formula
    -------
    effective_weight = body_weight + carried_load
    effective_bmi = (effective_weight / height_m²) × (1 + α × incline_hours)

    This captures two biomechanical realities missed by raw BMI:
      • Load carrying (headband/tumpline baskets, water, firewood)
      • Eccentric quadriceps stress from repetitive slope descent
    """
    if height_m <= 0:
        logger.warning("Invalid height_m=%.2f; returning 0.0", height_m)
        return 0.0

    effective_weight = weight_kg + carried_load_kg
    incline_factor = 1.0 + alpha * daily_incline_hours
    return (effective_weight / (height_m ** 2)) * incline_factor


def extract_contextual_features(
    patient_metadata: Dict[str, Any],
    questionnaire_data: Any,
    config: Optional[ContextualConfig] = None,
) -> Dict[str, float]:
    """Extract Tier C contextual features for the fusion model.

    Parameters
    ----------
    patient_metadata : dict
        Must contain ``age``, ``sex`` (``'male'``/``'female'``),
        ``height_cm``, ``weight_kg``.
    questionnaire_data
        Either a dict or a ``QuestionnaireResult`` dataclass instance.
        Used for ``tegner_activity_level``, ``injury_history``,
        ``carried_load_kg``, ``daily_incline_hours``.
    config : ContextualConfig, optional
        Defaults are ICMR Asian-Indian thresholds with hill-region
        incline coefficient.

    Returns
    -------
    dict[str, float]
        Feature dictionary with keys prefixed ``ctx_``.
    """
    if config is None:
        config = ContextualConfig()

    height_m = patient_metadata["height_cm"] / 100.0
    weight_kg = patient_metadata["weight_kg"]
    raw_bmi = weight_kg / (height_m ** 2) if height_m > 0 else 0.0

    # --- Retrieve mountainous-region questionnaire fields ---
    # Support both dict and dataclass (QuestionnaireResult) inputs
    if hasattr(questionnaire_data, "carried_load_kg"):
        carried_load = float(questionnaire_data.carried_load_kg)
        incline_hours = float(questionnaire_data.daily_incline_hours)
        tegner = questionnaire_data.tegner_activity_level
        injury_hist = questionnaire_data.injury_history
    else:
        carried_load = float(
            questionnaire_data.get("carried_load_kg", config.default_carried_load_kg)
        )
        incline_hours = float(
            questionnaire_data.get("daily_incline_hours", config.default_incline_hours)
        )
        tegner = questionnaire_data.get("tegner_activity_level", 5)
        injury_hist = questionnaire_data.get("injury_history", {})

    # --- Compute features ---
    effective_bmi = _effective_knee_load_index(
        weight_kg, height_m, carried_load, incline_hours,
        config.incline_load_alpha,
    )
    asian_score = _asian_indian_bmi_score(raw_bmi, config)

    previous_injury = int(
        injury_hist.get("previous_knee_injury", False)
        if isinstance(injury_hist, dict) else bool(injury_hist)
    )

    features = {
        "ctx_bmi": round(raw_bmi, 2),
        "ctx_effective_bmi": round(effective_bmi, 2),
        "ctx_asian_indian_bmi_score": asian_score,
        "ctx_age": float(patient_metadata["age"]),
        "ctx_sex_female": float(int(
            str(patient_metadata.get("sex", "")).lower() == "female"
        )),
        "ctx_previous_injury": float(previous_injury),
        "ctx_activity_level": float(tegner if tegner is not None else 5),
    }

    logger.info(
        "Tier C features: BMI=%.1f  Eff.BMI=%.1f  Asian-Score=%.1f",
        features["ctx_bmi"], features["ctx_effective_bmi"],
        features["ctx_asian_indian_bmi_score"],
    )
    return features
