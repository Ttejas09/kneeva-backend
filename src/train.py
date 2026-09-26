"""
Kneeva Fusion Model Training Pipeline — Build Guide §11

Trains the Layer 4 GBDT fusion model (CatBoost or XGBoost) and fits
the Platt Scaler calibration head on multimodal feature vectors.

Supports:
  • Training on CSV datasets (from OAI proxy features or collected pilot data).
  • Synthetic benchmark dataset generation for end-to-end pipeline verification.
  • K-fold cross-validation with ROC-AUC, Brier score, and calibration metrics.
  • Serialization of trained model, Platt scaler, and metadata.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import CalibrationConfig, ContextualConfig, FusionConfig
from src.models.calibration import PlattScaler
from src.models.fusion import ALL_FEATURES, KneevaFusionModel

logger = logging.getLogger("kneeva.train")


# ---------------------------------------------------------------------------
# Synthetic feature helpers — replicate the real tier_c.py formulas
# ---------------------------------------------------------------------------
def _synth_effective_bmi(
    weight_kg: float,
    height_m: float,
    carried_load_kg: float,
    daily_incline_hours: float,
    alpha: float = 0.10,
) -> float:
    """Replicate the Effective Knee Load Index from tier_c.py exactly."""
    if height_m <= 0:
        return 0.0
    effective_weight = weight_kg + carried_load_kg
    incline_factor = 1.0 + alpha * daily_incline_hours
    return (effective_weight / (height_m ** 2)) * incline_factor


def _synth_asian_indian_bmi_score(bmi: float) -> float:
    """Replicate the 4-level ICMR ordinal score from tier_c.py exactly."""
    cfg = ContextualConfig()
    if bmi < cfg.bmi_underweight:       # < 18.5
        return 0.1   # underweight — frailty risk
    elif bmi <= cfg.bmi_normal_upper:   # <= 22.9
        return 0.2   # normal
    elif bmi <= cfg.bmi_overweight_upper:  # <= 24.9
        return 0.6   # overweight (Asian-Indian)
    else:
        return 1.0   # obese (Asian-Indian, >= 25)


def generate_synthetic_dataset(
    n_samples: int = 300,
    prevalence: float = 0.30,
    random_state: int = 42,
    lasi_csv: Optional[str] = "data/processed/lasi_mountain_clinical.csv",
    mendeley_csv: Optional[str] = "data/processed/mendeley_imu_features.csv",
) -> pd.DataFrame:
    """Generate a biologically plausible synthetic feature dataset.
    
    If lasi_csv and mendeley_csv exist, it will use real-world distributions
    from the Indian mountain populations and raw IMU sensors as seeds for the
    synthetic data, providing a highly realistic proxy dataset.
    """
    rng = np.random.default_rng(random_state)
    n_pos = int(n_samples * prevalence)
    n_neg = n_samples - n_pos

    labels = np.array([0] * n_neg + [1] * n_pos)
    rng.shuffle(labels)
    
    # Load Real Datasets if they exist
    lasi_df = None
    mendeley_df = None
    if lasi_csv and Path(lasi_csv).exists():
        lasi_df = pd.read_csv(lasi_csv)
        lasi_pos = lasi_df[lasi_df['oa_positive'] == 1]
        lasi_neg = lasi_df[lasi_df['oa_positive'] == 0]
        
    if mendeley_csv and Path(mendeley_csv).exists():
        mendeley_df = pd.read_csv(mendeley_csv)

    rows = []
    for y in labels:
        is_oa = (y == 1)

        # 1. SAMPLE DEMOGRAPHICS & CLINICAL FROM LASI (if available)
        if lasi_df is not None:
            source_df = lasi_pos if is_oa else lasi_neg
            if len(source_df) > 0:
                sample = source_df.sample(1, random_state=rng.integers(0, 10000)).iloc[0]
                age = sample['ctx_age']
                female = sample['ctx_sex_female']
                raw_bmi = sample['ctx_bmi']
                ext_peak = sample['strength_ext_peak_n']
                gait_speed = sample['gait_speed_ms']
            else:
                age, female, raw_bmi, ext_peak, gait_speed = 55, 0.5, 24.0, 250.0, 1.1
        else:
            age = rng.normal(57 if is_oa else 48, 12)
            female = rng.choice([0, 1], p=[0.40, 0.60] if is_oa else [0.50, 0.50])
            height_m = rng.normal(1.57 if female else 1.66, 0.07)
            weight_kg = rng.normal(58 if is_oa else 55, 12)
            raw_bmi = weight_kg / (height_m ** 2)
            ext_peak = rng.normal(230 if is_oa else 340, 80)
            gait_speed = rng.normal(1.00 if is_oa else 1.25, 0.22)

        # Add some noise to prevent identical samples
        age = np.clip(age + rng.normal(0, 2), 25, 85)
        raw_bmi = np.clip(raw_bmi + rng.normal(0, 1), 16, 40)
        ext_peak = np.clip(ext_peak + rng.normal(0, 20), 50, 600)
        gait_speed = np.clip(gait_speed + rng.normal(0, 0.1), 0.4, 2.0)

        # Reverse engineer height/weight for Effective BMI formula
        height_m = rng.normal(1.57 if female else 1.66, 0.07)
        weight_kg = raw_bmi * (height_m ** 2)

        # 2. SAMPLE GAIT FEATURES FROM MENDELEY (if available)
        if mendeley_df is not None and len(mendeley_df) > 0:
            sample = mendeley_df.sample(1, random_state=rng.integers(0, 10000)).iloc[0]
            flat_cadence = sample['flat_gait_cadence']
            flat_stride_cv = sample['flat_gait_stride_time_cv']
            climbing_cadence = sample['climbing_cadence']
            climbing_stride_cv = sample['climbing_stride_time_cv']
            
            # Apply OA degradation manually since Mendeley is all healthy
            if is_oa:
                flat_cadence *= rng.uniform(0.75, 0.90)
                flat_stride_cv *= rng.uniform(1.2, 2.0)
                climbing_cadence *= rng.uniform(0.65, 0.85)
                climbing_stride_cv *= rng.uniform(1.5, 2.5)
        else:
            flat_cadence = rng.normal(96 if is_oa else 108, 12)
            flat_stride_cv = rng.normal(0.05 if is_oa else 0.03, 0.015)
            climbing_cadence = rng.normal(78 if is_oa else 92, 14)
            climbing_stride_cv = rng.normal(0.10 if is_oa else 0.05, 0.025)

        # Mountainous-region load carriage
        carried_load_kg = rng.uniform(0, 30)
        incline_hours = rng.uniform(0, 4.0)

        effective_bmi = _synth_effective_bmi(weight_kg, height_m, carried_load_kg, incline_hours)
        asian_score = _synth_asian_indian_bmi_score(raw_bmi)

        # ROM features
        active_flex = rng.normal(115 if is_oa else 130, 15)
        active_ext_deficit = max(0.0, rng.normal(6 if is_oa else 2, 4))

        # Acoustic crepitus
        crepitus_count = rng.poisson(7 if is_oa else 2)
        crepitus_energy = rng.exponential(25 if is_oa else 8) if crepitus_count > 0 else 0.0
        crepitus_pres = 1.0 if crepitus_count > 0 else 0.0

        # sEMG Co-contraction
        cci = np.clip(rng.normal(0.42 if is_oa else 0.30, 0.10), 0.05, 0.95)

        # Strength Ratios
        flex_peak = rng.normal(130 if is_oa else 190, 50)
        bw_ratio = ext_peak / max(40.0, weight_kg)
        hq_ratio = flex_peak / max(1.0, ext_peak)

        # Other gait features
        step_asym = np.clip(rng.normal(0.09 if is_oa else 0.04, 0.05), 0.0, 0.40)

        # Injury history
        prev_injury = rng.choice([0, 1], p=[0.65, 0.35] if is_oa else [0.88, 0.12])
        acl_hist = rng.choice([0, 1], p=[0.85, 0.15] if is_oa else [0.96, 0.04])
        meniscal_hist = rng.choice([0, 1], p=[0.80, 0.20] if is_oa else [0.95, 0.05])

        row = {
            # Tier A
            "rom_active_flexion_deg": round(float(active_flex), 1),
            "rom_active_extension_deficit_deg": round(float(active_ext_deficit), 1),
            "rom_passive_flexion_deg": round(float(active_flex + rng.normal(5, 2)), 1),
            "rom_flexion_deficit_deg": round(float(max(0.0, 140.0 - active_flex)), 1),
            "crepitus_event_count": float(crepitus_count),
            "crepitus_total_energy": round(float(crepitus_energy), 2),
            "crepitus_mean_energy": round(float(crepitus_energy / max(1, crepitus_count)), 2),
            "crepitus_presence": float(crepitus_pres),
            "cocontraction_cci_walking_mean": round(float(cci), 3),
            "strength_ext_peak_n": round(float(ext_peak), 1),
            "strength_flex_peak_n": round(float(flex_peak), 1),
            "strength_ext_bw_ratio": round(float(bw_ratio), 2),
            "strength_hq_ratio": round(float(hq_ratio), 2),
            # Tier B
            "gait_step_time_asymmetry": round(float(step_asym), 3),
            "gait_swing_time_asymmetry": round(float(step_asym * rng.uniform(0.7, 1.0)), 3),
            "flat_gait_stride_time_cv": round(float(flat_stride_cv), 3),
            "flat_gait_cadence": round(float(flat_cadence), 1),
            "climbing_stride_time_cv": round(float(climbing_stride_cv), 3),
            "climbing_cadence": round(float(climbing_cadence), 1),
            "gait_speed_ms": round(float(gait_speed), 2),
            "neuro_rf_activation_duration_pct": round(float(rng.normal(45 if is_oa else 38, 8)), 1),
            "neuro_bf_activation_duration_pct": round(float(rng.normal(40 if is_oa else 33, 7)), 1),
            "neuro_onset_emg_to_heelstrike_ms": round(float(rng.normal(105 if is_oa else 80, 25)), 1),
            "strength_ext_bw_ratio_general": round(float(bw_ratio), 2),
            "strength_general_weakness_flag": 1.0 if bw_ratio < 4.0 else 0.0,
            "strength_general_z_score": round(float(rng.normal(-0.8 if is_oa else 0.2, 1.0)), 2),
            "injury_previous_knee_injury": float(prev_injury),
            "injury_acl_history": float(acl_hist),
            "injury_meniscal_history": float(meniscal_hist),
            # Tier C
            "ctx_bmi": round(float(raw_bmi), 1),
            "ctx_effective_bmi": round(float(effective_bmi), 1),
            "ctx_asian_indian_bmi_score": float(asian_score),
            "ctx_age": round(float(age), 0),
            "ctx_sex_female": float(female),
            "ctx_previous_injury": float(prev_injury),
            "ctx_activity_level": float(rng.choice([1, 2, 3, 4])),
            # Label
            "oa_label": int(y),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def train_fusion_model(
    df: pd.DataFrame,
    target_col: str = "oa_label",
    model_type: str = "catboost",
    output_dir: Optional[str | Path] = None,
    calib_cfg: Optional[CalibrationConfig] = None,
    fusion_cfg: Optional[FusionConfig] = None,
) -> Tuple[KneevaFusionModel, PlattScaler, Dict[str, float]]:
    """Train the GBDT fusion head and fit the Platt scaler.

    Performs 5-fold stratified cross-validation, trains final model on full
    data, fits PlattScaler on out-of-fold logits, and computes performance.

    Parameters
    ----------
    df : pd.DataFrame
        Training dataframe with ALL_FEATURES and target_col.
    target_col : str
        Name of binary ground-truth label (1 = OA, 0 = Non-OA).
    model_type : str
        'catboost' (recommended) or 'xgboost'.
    output_dir : Path or str, optional
        Directory to save model artifacts.
    calib_cfg : CalibrationConfig, optional
    fusion_cfg : FusionConfig, optional

    Returns
    -------
    (KneevaFusionModel, PlattScaler, dict of metrics)
    """
    calib = calib_cfg or CalibrationConfig()
    fusion = fusion_cfg or FusionConfig()

    X = df[ALL_FEATURES].copy()
    y = df[target_col].values.astype(int)

    # 1. Stratified 5-Fold Cross Validation for unbiased out-of-fold logits
    random_seed = 42
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    oof_logits = np.zeros(len(df))
    cv_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        if model_type == "catboost":
            from catboost import CatBoostClassifier

            fold_model = CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=5,
                l2_leaf_reg=3.0,
                random_seed=random_seed + fold,
                verbose=False,
            )
            fold_model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
            val_logits = fold_model.predict(X_va, prediction_type="RawFormulaVal")
        else:
            import xgboost as xgb

            fold_model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                random_state=random_seed + fold,
                eval_metric="logloss",
                use_label_encoder=False,
            )
            fold_model.fit(X_tr, y_tr, verbose=False)
            # Use sklearn API for margin prediction (avoids DMatrix mismatch)
            val_logits = fold_model.predict(X_va, output_margin=True)

        oof_logits[val_idx] = val_logits
        auc = roc_auc_score(y_va, val_logits)
        cv_aucs.append(auc)

    mean_auc = float(np.mean(cv_aucs))
    logger.info("5-Fold CV Mean ROC-AUC: %.4f (std: %.4f)", mean_auc, np.std(cv_aucs))

    # 2. Fit Platt Scaler on Out-of-Fold logits
    platt_scaler = PlattScaler.from_config(calib, fusion)
    platt_scaler.fit(oof_logits, y)
    calibrated_probs = platt_scaler.predict_risk_score(oof_logits)
    brier = float(brier_score_loss(y, calibrated_probs))
    logger.info("Platt calibration: A=%.4f, B=%.4f, Brier=%.4f",
                platt_scaler.A, platt_scaler.B, brier)

    # 3. Train final model on entire dataset
    if model_type == "catboost":
        from catboost import CatBoostClassifier

        final_gbdt = CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=5,
            l2_leaf_reg=3.0,
            random_seed=random_seed,
            verbose=False,
        )
        final_gbdt.fit(X, y, verbose=False)
    else:
        import xgboost as xgb

        final_gbdt = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=random_seed,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        final_gbdt.fit(X, y, verbose=False)

    fusion_model = KneevaFusionModel(model=final_gbdt, feature_names=ALL_FEATURES)

    metrics = {
        "cv_roc_auc_mean": round(mean_auc, 4),
        "cv_roc_auc_std": round(float(np.std(cv_aucs)), 4),
        "brier_score": round(brier, 4),
        "platt_a": round(platt_scaler.A, 4),
        "platt_b": round(platt_scaler.B, 4),
        "n_samples": len(df),
        "n_positive": int(y.sum()),
        "prevalence": round(float(y.mean()), 4),
    }

    # 4. Optional saving
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        model_filename = (
            "fusion_catboost_v1.cbm"
            if model_type == "catboost"
            else "fusion_xgboost_v1.json"
        )
        fusion_model.save(out_path / model_filename)

        # Save Platt scaler parameters alongside model
        platt_dict = platt_scaler.to_dict()
        with open(out_path / "platt_scaler_v1.json", "w") as f:
            json.dump(platt_dict, f, indent=2)

        # Save metadata and calibration
        meta = {
            "model_type": model_type,
            "feature_names": ALL_FEATURES,
            "metrics": metrics,
            "calibration": {
                "A": platt_scaler.A,
                "B": platt_scaler.B,
                "dataset_prevalence": metrics["prevalence"],
            },
        }
        with open(out_path / "model_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Saved trained artifacts to %s", out_path)

    return fusion_model, platt_scaler, metrics


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Kneeva Fusion Model Trainer")
    parser.add_argument("--data", type=str, help="Path to CSV dataset")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic benchmark cohort for training",
    )
    parser.add_argument(
        "--samples", type=int, default=300, help="Number of synthetic samples"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="catboost",
        choices=["catboost", "xgboost"],
        help="GBDT model family",
    )
    parser.add_argument(
        "--out-dir", type=str, default="models", help="Output artifact directory"
    )
    args = parser.parse_args()

    if args.data:
        logger.info("Loading training data from %s", args.data)
        df = pd.read_csv(args.data)
    elif args.synthetic:
        logger.info("Generating %d synthetic patient profiles...", args.samples)
        df = generate_synthetic_dataset(n_samples=args.samples)
    else:
        logger.info("No --data provided. Using synthetic fallback (%d samples).", args.samples)
        df = generate_synthetic_dataset(n_samples=args.samples)

    model, scaler, metrics = train_fusion_model(
        df, model_type=args.model_type, output_dir=args.out_dir
    )
    print("\n" + "=" * 50)
    print("  Kneeva Model Training Complete")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
