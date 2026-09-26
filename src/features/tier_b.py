"""
Tier B — General / Differential Signal Feature Extraction (Build Guide §5.2)

Broader biomechanical patterns that help rule in or rule out alternative
pathologies:
  B1. Gait Asymmetry            — from IMU / GaitPy pipeline
  B2. Neuromuscular Activation   — from sEMG pipeline
  B3. General Weakness           — from Dynamometer + population norms
  B4. Injury History             — from Questionnaire pipeline

All tuneable numbers come from FeatureExtractionConfig — nothing is
hardcoded here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import GeneralWeaknessConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# B1 — Gait Asymmetry
# ---------------------------------------------------------------------------
def extract_gait_asymmetry_features(
    gait_features_df: Optional[pd.DataFrame],
) -> Dict[str, float]:
    """Extract gait asymmetry features from GaitPy output.

    Gait asymmetry can indicate OA (antalgic gait) but also hip pathology,
    neurological conditions, or leg length discrepancy — making it a
    **differential** signal rather than a pure OA biomarker.

    Parameters
    ----------
    gait_features_df : pd.DataFrame | None
        GaitPy ``extract_features`` output.  If ``None`` or empty, all
        features default to ``NaN`` so downstream models can handle
        gracefully via missingness-aware imputation.

    Returns
    -------
    dict[str, float]
    """
    nan_defaults = {
        "gait_stride_time_mean_s": float("nan"),
        "gait_stride_time_cv_pct": float("nan"),
        "gait_stance_time_mean_s": float("nan"),
        "gait_stance_time_cv_pct": float("nan"),
        "gait_step_time_mean_s": float("nan"),
        "gait_step_time_cv_pct": float("nan"),
        "gait_cadence_steps_per_min": float("nan"),
    }

    if gait_features_df is None or gait_features_df.empty:
        logger.warning("Gait features DataFrame is empty — returning NaN defaults.")
        return nan_defaults

    features: Dict[str, float] = {}

    # Helper: safe CV calculation
    def _cv(series: pd.Series) -> float:
        m = series.mean()
        return float(series.std() / m * 100.0) if m > 0 else 0.0

    # Stride time
    if "stride_time" in gait_features_df.columns:
        st = gait_features_df["stride_time"].dropna()
        features["gait_stride_time_mean_s"] = float(st.mean())
        features["gait_stride_time_cv_pct"] = _cv(st)
    else:
        features["gait_stride_time_mean_s"] = float("nan")
        features["gait_stride_time_cv_pct"] = float("nan")

    # Stance time
    if "stance_time" in gait_features_df.columns:
        st = gait_features_df["stance_time"].dropna()
        features["gait_stance_time_mean_s"] = float(st.mean())
        features["gait_stance_time_cv_pct"] = _cv(st)
    else:
        features["gait_stance_time_mean_s"] = float("nan")
        features["gait_stance_time_cv_pct"] = float("nan")

    # Step time
    if "step_time" in gait_features_df.columns:
        st = gait_features_df["step_time"].dropna()
        features["gait_step_time_mean_s"] = float(st.mean())
        features["gait_step_time_cv_pct"] = _cv(st)
        features["gait_cadence_steps_per_min"] = (
            60.0 / st.mean() if st.mean() > 0 else float("nan")
        )
    else:
        features["gait_step_time_mean_s"] = float("nan")
        features["gait_step_time_cv_pct"] = float("nan")
        features["gait_cadence_steps_per_min"] = float("nan")

    logger.info(
        "Gait features extracted from %d rows.",
        len(gait_features_df),
    )
    return features


# ---------------------------------------------------------------------------
# B2 — Neuromuscular Activation Pattern
# ---------------------------------------------------------------------------
def extract_neuromuscular_features(
    semg_channels: Dict[str, Dict[str, np.ndarray]],
    gait_cycles: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, float]:
    """Extract neuromuscular activation features from sEMG data.

    Captures central nervous system control quality — relevant for
    differential diagnosis (OA vs. neurological vs. deconditioning).

    Parameters
    ----------
    semg_channels : dict
        Mapping ``{channel_name: {"amplitude_mvc": ndarray, "activation": ndarray, ...}}``.
    gait_cycles : list of (start, end), optional
        Gait cycle boundaries for onset-timing variability.

    Returns
    -------
    dict[str, float]
    """
    features: Dict[str, float] = {}

    channel_names = ["VM", "VL", "RF", "BF"]

    for ch in channel_names:
        ch_data = semg_channels.get(ch)
        if ch_data is None:
            logger.warning("sEMG channel '%s' not available — skipping.", ch)
            continue

        amplitude = ch_data.get("amplitude_mvc")
        if amplitude is None:
            continue

        # Mean and peak activation during walking (%MVC)
        features[f"neuromusc_{ch}_mean_pct_mvc"] = float(np.mean(amplitude))
        features[f"neuromusc_{ch}_peak_pct_mvc"] = float(np.max(amplitude))

        # Activation timing variability across gait cycles
        activation = ch_data.get("activation")
        if activation is not None and gait_cycles:
            onset_fractions: List[float] = []
            for start, end in gait_cycles:
                if end <= len(activation):
                    cycle_act = activation[start:end]
                    first_onset_idx = int(np.argmax(cycle_act > 0))
                    cycle_length = end - start
                    if cycle_length > 0:
                        onset_fractions.append(
                            first_onset_idx / cycle_length
                        )
            if onset_fractions:
                features[f"neuromusc_{ch}_onset_variability"] = float(
                    np.std(onset_fractions)
                )

    # VM:VL ratio (medial-lateral activation balance)
    vm_mean = features.get("neuromusc_VM_mean_pct_mvc", 0.0)
    vl_mean = features.get("neuromusc_VL_mean_pct_mvc", 0.0)
    features["neuromusc_vm_vl_ratio"] = (
        vm_mean / vl_mean if vl_mean > 1e-6 else float("nan")
    )

    logger.info(
        "Neuromuscular features extracted for %d channels.",
        sum(1 for ch in channel_names if ch in semg_channels),
    )
    return features


# ---------------------------------------------------------------------------
# B3 — General Weakness
# ---------------------------------------------------------------------------
def extract_general_weakness_features(
    ext_force_n: float,
    flex_force_n: float,
    body_weight_kg: float,
    age: int,
    sex: str,
    config: Optional[GeneralWeaknessConfig] = None,
) -> Dict[str, float]:
    """Assess general lower-limb weakness against population norms.

    General weakness can indicate systemic conditions (sarcopenia,
    deconditioning, neurological disease) rather than localised OA.

    Parameters
    ----------
    ext_force_n : float
        Peak extension force in Newtons (affected side).
    flex_force_n : float
        Peak flexion force in Newtons (affected side).
    body_weight_kg : float
        Patient body weight in kg.
    age : int
        Patient age in years.
    sex : str
        ``"male"`` or ``"female"``.
    config : GeneralWeaknessConfig, optional
        Population-norm lookup table and z-score thresholds.

    Returns
    -------
    dict[str, float]
    """
    if config is None:
        config = GeneralWeaknessConfig()

    gravity = 9.81
    body_weight_n = body_weight_kg * gravity

    ext_bw_ratio = ext_force_n / body_weight_n if body_weight_n > 0 else 0.0
    flex_bw_ratio = flex_force_n / body_weight_n if body_weight_n > 0 else 0.0

    # Find closest age bracket in the norms table
    sex_key = sex.lower()
    sex_norms = config.norms.get(sex_key, {})
    if sex_norms:
        age_bracket = min(sex_norms.keys(), key=lambda x: abs(x - age))
        expected_ratio = sex_norms[age_bracket]
    else:
        expected_ratio = 0.5  # fallback
        logger.warning("No norm data for sex='%s' — using fallback.", sex)

    # Z-score relative to population norm
    norm_std = expected_ratio * config.norm_cv
    weakness_z = (
        (ext_bw_ratio - expected_ratio) / norm_std if norm_std > 0 else 0.0
    )

    features = {
        "general_weakness_ext_bw_ratio": ext_bw_ratio,
        "general_weakness_flex_bw_ratio": flex_bw_ratio,
        "general_weakness_z_score": weakness_z,
        "general_weakness_below_norm": float(
            weakness_z < config.weakness_z_threshold
        ),
    }

    logger.info(
        "General weakness: ext_bw=%.3f, z=%.2f, below_norm=%s.",
        ext_bw_ratio, weakness_z, bool(features["general_weakness_below_norm"]),
    )
    return features


# ---------------------------------------------------------------------------
# B4 — Injury History
# ---------------------------------------------------------------------------
def extract_injury_history_features(
    questionnaire_data: Dict[str, Any],
) -> Dict[str, float]:
    """Extract structured injury history features from questionnaire data.

    Parameters
    ----------
    questionnaire_data : dict
        Raw questionnaire JSON including ``injury_history``,
        ``tegner_activity_level``, and scored subscales.

    Returns
    -------
    dict[str, float]
    """
    injury = questionnaire_data.get("injury_history", {})

    features = {
        "injury_previous_knee_injury": float(
            injury.get("previous_knee_injury", False)
        ),
        "injury_type_meniscal": float(
            injury.get("type") == "meniscal_tear"
        ),
        "injury_type_acl": float(
            injury.get("type") in ("acl_tear", "acl_reconstruction")
        ),
        "injury_type_fracture": float(
            injury.get("type") == "fracture"
        ),
        "injury_surgery_history": float(
            injury.get("surgery", False)
        ),
        "injury_years_since": float(
            2026 - injury.get("year", 2026)
            if injury.get("previous_knee_injury")
            else 0
        ),
        "injury_tegner_activity": float(
            questionnaire_data.get("tegner_activity_level", 5)
        ),
        "injury_womac_pain_norm": float(
            questionnaire_data.get("womac_pain_normalized", 0)
        ),
        "injury_koos_qol": float(
            questionnaire_data.get("koos_qol", 100)
        ),
    }

    logger.info(
        "Injury history: prev_injury=%s, years_since=%.0f, tegner=%.0f.",
        bool(features["injury_previous_knee_injury"]),
        features["injury_years_since"],
        features["injury_tegner_activity"],
    )
    return features
