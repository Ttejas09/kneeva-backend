"""
Kneeva Triage API Server — FastAPI Microservice (Build Guide §14)

Acts as the backend server that clinical mobile/tablet apps connect to.
Ingests patient questionnaires, metadata, and sensor features; executes
the multi-modal late-fusion pipeline and Platt calibration; and returns
calibrated OA risk scores, confidence intervals, and clinical explanations.

Endpoints:
  • GET  /health            - Health check, model status, and metadata
  • POST /api/v1/triage     - Primary triage endpoint for mobile/edge app
  • POST /api/v1/triage/batch - Batch triage for multiple patient records
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import PreprocessingConfig
from src.features.tier_c import extract_contextual_features
from src.inference import KneevaInferencePipeline, KneevaOutput
from src.models.calibration import PlattScaler
from src.models.explainer import ClinicalExplainer
from src.models.fusion import ALL_FEATURES, KneevaFusionModel

logger = logging.getLogger("kneeva.server")

# Project root (d:\Kneeva) resolved from this file's location
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# FastAPI Application & CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Kneeva OA Triage Server",
    description=(
        "Medical triage backend providing calibrated knee osteoarthritis (OA) "
        "risk scoring and clinical explanations for rural & mountainous regions."
    ),
    version="1.0.0",
)

# CORS — in production, replace with your actual app domain(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # wildcard origins cannot use credentials
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Request / Response Schemas
# ---------------------------------------------------------------------------
class PatientMetadataModel(BaseModel):
    age: float = Field(..., ge=18, le=110, description="Patient age in years")
    sex: str = Field(..., description="'female' or 'male'")
    height_cm: float = Field(..., ge=80, le=240, description="Height in centimetres")
    weight_kg: float = Field(..., ge=25, le=220, description="Weight in kilograms")


class QuestionnaireModel(BaseModel):
    carried_load_kg: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Average daily carried weight (kg)"
    )
    daily_incline_hours: float = Field(
        default=0.0, ge=0.0, le=18.0, description="Hours/day walking on mountain slopes"
    )
    squatting_difficulty: int = Field(
        default=0, ge=0, le=4, description="Difficulty squatting (0=none to 4=extreme)"
    )
    previous_injury: int = Field(
        default=0, ge=0, le=1, description="Prior knee trauma / surgery (0=no, 1=yes)"
    )
    activity_level: int = Field(
        default=2, ge=1, le=4, description="Activity level (1=sedentary to 4=heavy manual)"
    )


class TriageRequest(BaseModel):
    patient_id: Optional[str] = Field(
        default="ANON", description="Unique or anonymised patient identifier"
    )
    patient_metadata: PatientMetadataModel
    questionnaire: QuestionnaireModel
    sensor_features: Dict[str, float] = Field(
        default_factory=dict,
        description="Dictionary of sensor-extracted features (Tier A + Tier B).",
    )


class TriageResponse(BaseModel):
    patient_id: str
    oa_risk_score: float = Field(
        ..., description="Calibrated OA risk probability (0.00 to 1.00)"
    )
    oa_risk_category: str = Field(
        ..., description="'low', 'moderate', 'high', or 'very_high'"
    )
    confidence_interval: Tuple[float, float] = Field(
        ..., description="95% confidence interval on risk score"
    )
    feature_importance: Dict[str, float] = Field(
        ..., description="Top drivers of risk (SHAP attributions)"
    )
    clinical_explanation: str = Field(
        ..., description="Clinician-facing explanation of risk drivers"
    )
    clinical_action: str = Field(
        ..., description="Recommended clinical decision / triage protocol"
    )
    differential_signal: bool = Field(
        ..., description="True if alternative pathology suspected (e.g. meniscal/ACL)"
    )
    differential_flags: List[str] = Field(
        ..., description="List of suspected alternative conditions"
    )
    missing_modality_count: int = Field(
        ..., description="Number of sensor modalities absent from assessment"
    )
    effective_bmi: float = Field(
        ..., description="Mechanically corrected BMI accounting for carried load & slope"
    )


# ---------------------------------------------------------------------------
# Questionnaire Schema Translation
# ---------------------------------------------------------------------------
def _translate_questionnaire_for_tier_c(q_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Map the server's QuestionnaireModel fields to the keys that
    ``extract_contextual_features()`` expects.

    Server schema → tier_c.py expectation:
      activity_level   → tegner_activity_level
      previous_injury  → injury_history.previous_knee_injury
    """
    return {
        "carried_load_kg": q_dict.get("carried_load_kg", 0.0),
        "daily_incline_hours": q_dict.get("daily_incline_hours", 0.0),
        "tegner_activity_level": q_dict.get("activity_level", 2),
        "injury_history": {
            "previous_knee_injury": bool(q_dict.get("previous_injury", 0)),
        },
    }


# ---------------------------------------------------------------------------
# Global Pipeline State
# ---------------------------------------------------------------------------
_pipeline: Optional[KneevaInferencePipeline] = None
_model_source: str = "stub"


def get_inference_pipeline() -> KneevaInferencePipeline:
    """Lazy-initialise or return singleton Kneeva inference pipeline."""
    global _pipeline, _model_source
    if _pipeline is not None:
        return _pipeline

    cfg = PreprocessingConfig()
    model_dir = _PROJECT_ROOT / "models"
    model_path_cbm = model_dir / "fusion_catboost_v1.cbm"
    model_path_xgb = model_dir / "fusion_xgboost_v1.json"

    # --- Load GBDT model ---
    if model_path_cbm.exists():
        try:
            logger.info("Loading pre-trained CatBoost model from %s", model_path_cbm)
            fusion_model = KneevaFusionModel.load(
                model_path_cbm, model_type="catboost", feature_names=ALL_FEATURES
            )
            _model_source = f"catboost ({model_path_cbm.name})"
        except Exception as exc:
            logger.warning("Failed loading CatBoost model: %s. Using stub.", exc)
            fusion_model = KneevaFusionModel()
            _model_source = "stub"
    elif model_path_xgb.exists():
        try:
            logger.info("Loading pre-trained XGBoost model from %s", model_path_xgb)
            fusion_model = KneevaFusionModel.load(
                model_path_xgb, model_type="xgboost", feature_names=ALL_FEATURES
            )
            _model_source = f"xgboost ({model_path_xgb.name})"
        except Exception as exc:
            logger.warning("Failed loading XGBoost model: %s. Using stub.", exc)
            fusion_model = KneevaFusionModel()
            _model_source = "stub"
    else:
        logger.info("No saved model checkpoint found. Initialising prior-only stub.")
        fusion_model = KneevaFusionModel()
        _model_source = "stub (prior-only)"

    # --- Load trained Platt Scaler from saved artifact ---
    platt_path = model_dir / "platt_scaler_v1.json"
    meta_path = model_dir / "model_metadata.json"

    if platt_path.exists():
        try:
            with open(platt_path) as f:
                platt_dict = json.load(f)
            platt_scaler = PlattScaler.from_dict(platt_dict)
            platt_scaler._fusion_config = cfg.fusion
            platt_scaler._cal_config = cfg.calibration
            logger.info(
                "Loaded trained Platt scaler (A=%.4f, B=%.4f) from %s",
                platt_scaler.A, platt_scaler.B, platt_path.name,
            )
        except Exception as exc:
            logger.warning("Failed loading Platt scaler: %s. Using config defaults.", exc)
            platt_scaler = PlattScaler.from_config(cfg.calibration, cfg.fusion)
    elif meta_path.exists():
        # Fallback: read A, B from model_metadata.json
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            cal = meta.get("calibration", {})
            platt_scaler = PlattScaler(
                A=cal["A"], B=cal["B"], is_fitted=True,
                _cal_config=cfg.calibration, _fusion_config=cfg.fusion,
            )
            logger.info(
                "Loaded Platt scaler from metadata (A=%.4f, B=%.4f)",
                platt_scaler.A, platt_scaler.B,
            )
        except Exception as exc:
            logger.warning("Failed reading metadata calibration: %s. Using defaults.", exc)
            platt_scaler = PlattScaler.from_config(cfg.calibration, cfg.fusion)
    else:
        logger.info("No trained Platt scaler found. Using prior-only defaults.")
        platt_scaler = PlattScaler.from_config(cfg.calibration, cfg.fusion)

    explainer = ClinicalExplainer(
        model=fusion_model.model,
        feature_names=fusion_model.feature_names,
        top_k=cfg.fusion.top_k_features,
    )

    _pipeline = KneevaInferencePipeline(
        fusion_model=fusion_model,
        platt_scaler=platt_scaler,
        explainer=explainer,
        config=cfg,
    )
    return _pipeline


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/", tags=["Info"])
def root_info():
    """Service landing page and documentation link."""
    return {
        "service": "Kneeva OA Triage Server",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Verify server status, active model family, and feature count."""
    pipeline = get_inference_pipeline()
    return {
        "status": "healthy",
        "model_backend": _model_source,
        "feature_count": len(pipeline.fusion_model.feature_names),
        "calibration": {
            "platt_A": pipeline.platt_scaler.A,
            "platt_B": pipeline.platt_scaler.B,
            "is_fitted": pipeline.platt_scaler.is_fitted,
        },
    }


@app.post(
    "/api/v1/triage",
    response_model=TriageResponse,
    status_code=status.HTTP_200_OK,
    tags=["Triage"],
)
def triage_patient(req: TriageRequest):
    """Run full triage pipeline on a single patient record.

    Extracts Tier C contextual features (incorporating the Effective Knee Load
    Index for mountain terrain load carriage), combines them with sensor
    features, feeds them into the calibrated fusion model, and produces
    the clinical report.
    """
    try:
        pipeline = get_inference_pipeline()

        # 1. Extract Tier C contextual features
        p_dict = req.patient_metadata.model_dump()
        q_dict = req.questionnaire.model_dump()
        # Translate server schema → tier_c.py expected keys
        q_for_tier_c = _translate_questionnaire_for_tier_c(q_dict)

        tier_c = extract_contextual_features(
            patient_metadata=p_dict,
            questionnaire_data=q_for_tier_c,
        )

        # 2. Merge with Tier A and Tier B sensor features from mobile client
        merged_features: Dict[str, float] = {}
        merged_features.update(req.sensor_features)
        merged_features.update(tier_c)

        # 3. Execute inference pipeline
        output: KneevaOutput = pipeline.run(merged_features)

        return TriageResponse(
            patient_id=req.patient_id or "ANON",
            oa_risk_score=output.oa_risk_score,
            oa_risk_category=output.oa_risk_category,
            confidence_interval=output.confidence_interval,
            feature_importance=output.feature_importance,
            clinical_explanation=output.clinical_explanation,
            clinical_action=output.clinical_action,
            differential_signal=output.differential_signal,
            differential_flags=output.differential_flags,
            missing_modality_count=output.missing_modality_count,
            effective_bmi=tier_c.get("ctx_effective_bmi", tier_c.get("ctx_bmi", 0.0)),
        )
    except Exception as exc:
        logger.exception("Error processing triage request: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        )


@app.post(
    "/api/v1/triage/batch",
    response_model=List[TriageResponse],
    status_code=status.HTTP_200_OK,
    tags=["Triage"],
)
def triage_batch(requests: List[TriageRequest]):
    """Batch assessment endpoint for processing multiple patient records.

    Returns a result for each patient. If an individual patient fails,
    a partial error is recorded in the response for that patient without
    aborting the rest of the batch.
    """
    results = []
    for r in requests:
        try:
            results.append(triage_patient(r))
        except HTTPException:
            # Return a sentinel error response for this patient
            results.append(
                TriageResponse(
                    patient_id=r.patient_id or "ANON",
                    oa_risk_score=-1.0,
                    oa_risk_category="error",
                    confidence_interval=(0.0, 0.0),
                    feature_importance={},
                    clinical_explanation="Error processing this patient record.",
                    clinical_action="Retry or submit individually.",
                    differential_signal=False,
                    differential_flags=[],
                    missing_modality_count=-1,
                    effective_bmi=0.0,
                )
            )
    return results
