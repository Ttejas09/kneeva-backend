"""
Fusion Model Wrapper — Build Guide §9

Wraps a gradient-boosted decision tree (CatBoost preferred, XGBoost
fallback) for the Layer 4 late-fusion head.

Key responsibilities:
  • Enforce canonical feature column ordering across Tier A, B, C.
  • Output raw decision margins (logits) for downstream Platt scaling.
  • Support model persistence (save / load).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical feature order — must match training pipeline exactly
TIER_A_FEATURES: List[str] = [
    "rom_active_flexion_deg",
    "rom_active_extension_deficit_deg",
    "rom_passive_flexion_deg",
    "rom_flexion_deficit_deg",
    "crepitus_event_count",
    "crepitus_total_energy",
    "crepitus_mean_energy",
    "crepitus_presence",
    "cocontraction_cci_walking_mean",
    "strength_ext_peak_n",
    "strength_flex_peak_n",
    "strength_ext_bw_ratio",
    "strength_hq_ratio",
]

TIER_B_FEATURES: List[str] = [
    "gait_step_time_asymmetry",
    "gait_swing_time_asymmetry",
    "flat_gait_stride_time_cv",
    "flat_gait_cadence",
    "climbing_stride_time_cv",
    "climbing_cadence",
    "gait_speed_ms",
    "neuro_rf_activation_duration_pct",
    "neuro_bf_activation_duration_pct",
    "neuro_onset_emg_to_heelstrike_ms",
    "strength_ext_bw_ratio_general",
    "strength_general_weakness_flag",
    "strength_general_z_score",
    "injury_previous_knee_injury",
    "injury_acl_history",
    "injury_meniscal_history",
]

TIER_C_FEATURES: List[str] = [
    "ctx_bmi",
    "ctx_effective_bmi",
    "ctx_asian_indian_bmi_score",
    "ctx_age",
    "ctx_sex_female",
    "ctx_previous_injury",
    "ctx_activity_level",
]

ALL_FEATURES: List[str] = TIER_A_FEATURES + TIER_B_FEATURES + TIER_C_FEATURES


class KneevaFusionModel:
    """Late-fusion GBDT wrapper for the Kneeva OA risk prediction head.

    Parameters
    ----------
    model : object, optional
        A pre-trained CatBoost, XGBoost, or sklearn GBDT model.
        If None, a stub is created that returns logit 0.0 (prior-only mode).
    feature_names : list[str], optional
        Canonical feature ordering.  Defaults to ``ALL_FEATURES``.
    """

    def __init__(
        self,
        model: Any = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.feature_names = feature_names or ALL_FEATURES
        self._model_type = self._detect_model_type()

    def _detect_model_type(self) -> str:
        if self.model is None:
            return "stub"
        cls_name = type(self.model).__name__
        if "CatBoost" in cls_name:
            return "catboost"
        elif "XGB" in cls_name or "Booster" in cls_name:
            return "xgboost"
        else:
            return "sklearn"

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_logit(self, feature_dict: Dict[str, float]) -> float:
        """Predict a raw decision margin (logit) from a feature dictionary.

        Missing features are filled with 0.0 (neutral) and logged as
        warnings.  The fusion model should be trained to handle this via
        missing-value-aware splitting (CatBoost/XGBoost native support).

        Parameters
        ----------
        feature_dict : dict
            Feature name → value mapping (Tier A + B + C merged).

        Returns
        -------
        float
            Raw logit / margin for downstream Platt scaling.
        """
        row = {}
        missing = []
        for feat in self.feature_names:
            if feat in feature_dict:
                row[feat] = feature_dict[feat]
            else:
                row[feat] = np.nan  # Let tree handle natively
                missing.append(feat)

        if missing:
            logger.warning(
                "%d features missing (filled NaN): %s",
                len(missing), ", ".join(missing[:5]),
            )

        X = pd.DataFrame([row], columns=self.feature_names)

        if self._model_type == "stub":
            logger.info("Stub model — returning logit=0.0 (prior-only).")
            return 0.0

        if self._model_type == "catboost":
            return float(
                self.model.predict(X, prediction_type="RawFormulaVal")[0]
            )
        elif self._model_type == "xgboost":
            import xgboost as xgb
            dmat = xgb.DMatrix(X, feature_names=self.feature_names)
            return float(self.model.predict(dmat, output_margin=True)[0])
        else:
            # sklearn interface (decision_function or predict)
            if hasattr(self.model, "decision_function"):
                return float(self.model.decision_function(X)[0])
            else:
                # Fallback: log-odds from predict_proba
                proba = self.model.predict_proba(X)[0, 1]
                proba = np.clip(proba, 1e-7, 1 - 1e-7)
                return float(np.log(proba / (1 - proba)))

    def count_missing_modalities(
        self, feature_dict: Dict[str, float]
    ) -> int:
        """Count how many sensor modalities are entirely absent.

        A modality is considered missing if ALL of its Tier A/B features
        are absent from the feature dictionary.
        """
        modality_prefixes = {
            "rom": ["rom_"],
            "acoustic": ["crepitus_"],
            "emg": ["cocontraction_", "neuro_"],
            "dynamometer": ["strength_"],
            "gait": ["gait_"],
        }
        missing_count = 0
        for _mod, prefixes in modality_prefixes.items():
            has_any = any(
                feat in feature_dict
                for feat in self.feature_names
                if any(feat.startswith(p) for p in prefixes)
            )
            if not has_any:
                missing_count += 1
        return missing_count

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save the underlying model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self._model_type == "catboost":
            self.model.save_model(str(path))
        elif self._model_type == "xgboost":
            self.model.save_model(str(path))
        else:
            import joblib
            joblib.dump(self.model, str(path))
        logger.info("Model saved to %s", path)

    @classmethod
    def load(
        cls,
        path: str | Path,
        model_type: str = "catboost",
        feature_names: Optional[List[str]] = None,
    ) -> "KneevaFusionModel":
        """Load a persisted model from disk.

        Parameters
        ----------
        path : str or Path
            Path to the saved model file.
        model_type : str
            One of ``'catboost'``, ``'xgboost'``, ``'sklearn'``.
        feature_names : list, optional
            Canonical feature ordering.
        """
        path = Path(path)
        if model_type == "catboost":
            from catboost import CatBoostClassifier
            model = CatBoostClassifier()
            model.load_model(str(path))
        elif model_type == "xgboost":
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(str(path))
        else:
            import joblib
            model = joblib.load(str(path))

        return cls(model=model, feature_names=feature_names)
