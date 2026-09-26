"""
Questionnaire Preprocessing Module — Layer 1, Build Guide §4.5

Implements published scoring formulas:
  - WOMAC  (Bellamy et al., 1988)
  - KOOS   (Roos et al., 1998)

Plus pass-through for VAS pain, Tegner activity level, and injury history.

All scoring parameters come from QuestionnaireConfig.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import QuestionnaireConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class WOMACScores:
    """WOMAC scoring output.  Higher score = worse symptoms."""

    pain_raw: int
    pain_normalized: float          # 0–100

    stiffness_raw: int
    stiffness_normalized: float     # 0–100

    function_raw: int
    function_normalized: float      # 0–100

    total_raw: int
    total_normalized: float         # 0–100


@dataclass
class KOOSScores:
    """KOOS scoring output.  Higher score = fewer problems."""

    subscales: Dict[str, float]
    """Mapping of subscale name → normalised score (0–100)."""


@dataclass
class QuestionnaireResult:
    """Structured output of the questionnaire preprocessing pipeline."""

    womac: Optional[WOMACScores] = None
    koos: Optional[KOOSScores] = None
    vas_pain_mm: Optional[float] = None
    tegner_activity_level: Optional[int] = None
    injury_history: Optional[Dict[str, Any]] = None

    # --- Indian Mountainous / NER region fields ---
    carried_load_kg: float = 0.0
    """Average daily carried load in kg (e.g. firewood, water, crops)."""

    daily_incline_hours: float = 0.0
    """Hours per day spent walking on steep slopes / stairs."""

    squatting_difficulty: int = 0
    """Difficulty squatting / sitting cross-legged (0–4 Likert)."""


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------
def score_womac(
    responses: Dict[str, List[int]],
    config: QuestionnaireConfig,
) -> WOMACScores:
    """Score WOMAC using the published Likert formula.

    Parameters
    ----------
    responses : dict
        Keys: ``pain`` (5 items), ``stiffness`` (2 items),
        ``function`` (17 items).  Each item scored 0–4.
    """
    womac = config.womac

    pain_raw = sum(responses["pain"])
    stiffness_raw = sum(responses["stiffness"])
    function_raw = sum(responses["function"])
    total_raw = pain_raw + stiffness_raw + function_raw

    scores = WOMACScores(
        pain_raw=pain_raw,
        pain_normalized=(pain_raw / womac.pain_max) * 100,
        stiffness_raw=stiffness_raw,
        stiffness_normalized=(stiffness_raw / womac.stiffness_max) * 100,
        function_raw=function_raw,
        function_normalized=(function_raw / womac.function_max) * 100,
        total_raw=total_raw,
        total_normalized=(total_raw / womac.total_max) * 100,
    )

    logger.info(
        "WOMAC scored: total=%d/%d (%.1f%%).",
        total_raw, womac.total_max, scores.total_normalized,
    )
    return scores


def score_koos(
    responses: Dict[str, List[int]],
    config: QuestionnaireConfig,
) -> KOOSScores:
    """Score KOOS using the published formula.

    Formula per subscale::

        100 - (mean_of_items / max_item_score) × 100

    Higher score = fewer problems (opposite of WOMAC).
    """
    koos = config.koos
    subscales: Dict[str, float] = {}

    for subscale_name, items in responses.items():
        mean_score = float(np.mean(items))
        normalized = 100.0 - (mean_score / koos.max_item_score) * 100.0
        subscales[subscale_name] = normalized

    logger.info(
        "KOOS scored: %s",
        {k: f"{v:.1f}" for k, v in subscales.items()},
    )
    return KOOSScores(subscales=subscales)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def preprocess_questionnaire(
    questionnaire_data: Dict[str, Any],
    config: Optional[QuestionnaireConfig] = None,
) -> QuestionnaireResult:
    """Run the full questionnaire preprocessing pipeline.

    Parameters
    ----------
    questionnaire_data : dict
        Raw questionnaire payload (typically loaded from JSON).
        Expected keys: ``womac``, ``koos``, ``vas_pain_mm``,
        ``tegner_activity_level``, ``injury_history``.
    config : QuestionnaireConfig, optional
        Scoring configuration.  Build-Guide defaults used when ``None``.
    """
    if config is None:
        config = QuestionnaireConfig()

    result = QuestionnaireResult()

    if "womac" in questionnaire_data:
        result.womac = score_womac(questionnaire_data["womac"], config)

    if "koos" in questionnaire_data:
        result.koos = score_koos(questionnaire_data["koos"], config)

    # Pass-through fields — no transformation needed
    result.vas_pain_mm = questionnaire_data.get("vas_pain_mm")
    result.tegner_activity_level = questionnaire_data.get("tegner_activity_level")
    result.injury_history = questionnaire_data.get("injury_history")

    # Indian Mountainous / NER region fields
    result.carried_load_kg = float(
        questionnaire_data.get("carried_load_kg", 0.0)
    )
    result.daily_incline_hours = float(
        questionnaire_data.get("daily_incline_hours", 0.0)
    )
    result.squatting_difficulty = int(
        questionnaire_data.get("squatting_difficulty", 0)
    )

    return result
