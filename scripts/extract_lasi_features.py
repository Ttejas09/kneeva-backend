import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Mountain states mapped to LASI stateid (from LASI Data User Guide)
# 2: Himachal Pradesh, 5: Uttarakhand, 10-18: North Eastern States + Sikkim
MOUNTAIN_STATES = [2, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18]

def extract_lasi(dataset_dir: str, output_path: str):
    base_dir = Path(dataset_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"LASI directory not found: {base_dir}")

    # File paths
    cv_file = base_dir / "5_LASI_W1_CV_memberfile.dta"
    ind_file = base_dir / "3_LASI_W1_Individual_v4.dta"
    bio_file = base_dir / "4_LASI_W1_Biomarker.dta"

    logger.info("Loading LASI CV member file (Sex, Age) via chunks...")
    cv_chunks = []
    for chunk in pd.read_stata(cv_file, convert_categoricals=False, columns=['prim_key', 'stateid', 'cv005', 'cv006'], chunksize=5000):
        # Pre-filter for mountain states to save memory
        chunk = chunk[chunk['stateid'].isin(MOUNTAIN_STATES)]
        cv_chunks.append(chunk)
    cv_df = pd.concat(cv_chunks, ignore_index=True)

    logger.info("Loading LASI Individual file (Arthritis, Walking Difficulty) via chunks...")
    ind_chunks = []
    for chunk in pd.read_stata(ind_file, convert_categoricals=False, columns=['prim_key', 'stateid', 'ht004', 'ht020'], chunksize=5000):
        chunk = chunk[chunk['stateid'].isin(MOUNTAIN_STATES)]
        # We only need prim_key, ht004, ht020 for the merge
        ind_chunks.append(chunk[['prim_key', 'ht004', 'ht020']])
    ind_df = pd.concat(ind_chunks, ignore_index=True)

    logger.info("Loading LASI Biomarker file (Height, Weight, Grip, Timed Walk) via chunks...")
    bio_chunks = []
    for chunk in pd.read_stata(bio_file, convert_categoricals=False, columns=['prim_key', 'stateid', 'bm066', 'bm071', 'bm072', 'bm028', 'bm029', 'bm030', 'bm056'], chunksize=5000):
        chunk = chunk[chunk['stateid'].isin(MOUNTAIN_STATES)]
        bio_chunks.append(chunk[['prim_key', 'bm066', 'bm071', 'bm072', 'bm028', 'bm029', 'bm030', 'bm056']])
    bio_df = pd.concat(bio_chunks, ignore_index=True)

    logger.info("Merging datasets on prim_key...")
    # Outer join to keep as much data as possible, though we need at least CV + Ind for labels
    merged_df = cv_df.merge(ind_df, on='prim_key', how='inner')
    merged_df = merged_df.merge(bio_df, on='prim_key', how='left')

    logger.info(f"Total rows after merge: {len(merged_df)}")

    # Filter for Mountain States
    merged_df = merged_df[merged_df['stateid'].isin(MOUNTAIN_STATES)].copy()
    logger.info(f"Rows after filtering for Mountain States: {len(merged_df)}")

    logger.info("Extracting and mapping Kneeva features...")
    kneeva_df = pd.DataFrame()

    # 1. Demographics
    # cv006: Age in years
    kneeva_df['ctx_age'] = pd.to_numeric(merged_df['cv006'], errors='coerce')
    
    # cv005: Sex (1 = Male, 2 = Female) -> Kneeva: (1.0 = Female, 0.0 = Male)
    sex = pd.to_numeric(merged_df['cv005'], errors='coerce')
    kneeva_df['ctx_sex_female'] = np.where(sex == 2, 1.0, np.where(sex == 1, 0.0, np.nan))

    # 2. BMI
    # bm066: Height (cm)
    height_cm = pd.to_numeric(merged_df['bm066'], errors='coerce')
    # bm071/bm072: Weight (kg). Take the average if both exist, else the first valid one.
    w1 = pd.to_numeric(merged_df['bm071'], errors='coerce')
    w2 = pd.to_numeric(merged_df['bm072'], errors='coerce')
    weight_kg = w1.combine_first(w2) # Use w1, fallback to w2
    
    height_m = height_cm / 100.0
    # Mask invalid heights to prevent division by zero or unrealistic BMI
    valid_mask = (height_m > 1.0) & (weight_kg > 20.0)
    kneeva_df['ctx_bmi'] = np.where(valid_mask, weight_kg / (height_m ** 2), np.nan)

    # 3. Strength (Grip Strength Proxy)
    # bm028/029: Right hand (T1, T2) in kg. bm030: Left hand in kg.
    g1 = pd.to_numeric(merged_df['bm028'], errors='coerce')
    g2 = pd.to_numeric(merged_df['bm029'], errors='coerce')
    g3 = pd.to_numeric(merged_df['bm030'], errors='coerce')
    
    # Take the absolute maximum recorded grip strength across trials and hands
    max_grip_kg = pd.concat([g1, g2, g3], axis=1).max(axis=1)
    
    # Convert kg-force to Newtons (1 kgf ~ 9.81 N)
    kneeva_df['strength_ext_peak_n'] = max_grip_kg * 9.81

    # 4. Gait Speed
    # bm056: Timed walk duration (seconds) for 4 metres.
    walk_sec = pd.to_numeric(merged_df['bm056'], errors='coerce')
    kneeva_df['gait_speed_ms'] = np.where(
        (walk_sec > 0) & (walk_sec < 100), # sanity bounds
        4.0 / walk_sec, 
        np.nan
    )

    # 5. Target Labels
    # ht004: Diagnosed with Arthritis (1 = Yes, 2 = No)
    arthritis = pd.to_numeric(merged_df['ht004'], errors='coerce')
    
    # ht020: Difficulty walking 100 metres (1-5 scale, 4/5 is severe)
    walk_diff = pd.to_numeric(merged_df['ht020'], errors='coerce')

    # Positive class: Diagnosed with arthritis OR severe walking difficulty
    # This addresses the massive under-reporting of clinical diagnoses in mountain areas.
    kneeva_df['oa_positive'] = np.where(
        (arthritis == 1) | (walk_diff >= 4), 
        1, 
        0
    )

    # Drop rows where we have absolutely no target label information
    valid_labels = merged_df['ht004'].notna() | merged_df['ht020'].notna()
    kneeva_df = kneeva_df[valid_labels].copy()

    # Drop rows that are completely missing all features
    features = ['ctx_age', 'ctx_sex_female', 'ctx_bmi', 'strength_ext_peak_n', 'gait_speed_ms']
    kneeva_df = kneeva_df.dropna(subset=features, how='all')

    logger.info(f"Final dataset shape: {kneeva_df.shape}")
    logger.info("Class distribution (oa_positive):")
    logger.info(kneeva_df['oa_positive'].value_counts(normalize=True).to_string())

    # Save to CSV
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kneeva_df.to_csv(out_path, index=False)
    logger.info(f"Saved extracted features to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract LASI clinical features")
    parser.add_argument("--dataset-dir", type=str, default="dataset/65dc53fc88cd61708938236", help="Path to LASI folder")
    parser.add_argument("--output", type=str, default="data/processed/lasi_mountain_clinical.csv", help="Output CSV path")
    
    args = parser.parse_args()
    
    try:
        extract_lasi(args.dataset_dir, args.output)
    except Exception as e:
        logger.error(f"Failed to extract LASI data: {e}", exc_info=True)
        exit(1)
