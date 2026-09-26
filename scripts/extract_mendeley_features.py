import argparse
import ast
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Mendeley IMU Sampling Rate is typically 50Hz or 100Hz for MPU6050
# Based on 200+ samples per row, assuming ~50Hz for 4-5 seconds of walking
FS = 50.0

def calculate_gait_features(row, sensor_cols):
    """
    Parses raw MPU6050 sensor data and calculates Cadence and Stride Time CV.
    We use the primary Gyroscope axis (usually Data_4 or Data_5 for sagittal plane knee flexion)
    to detect peaks (heel strikes/toe offs).
    """
    try:
        # We assume Data_4 (Gyro X) or Data_5 (Gyro Y) captures the main swing phase.
        # Let's use Data_5 as the primary signal for peak detection.
        # Parse the stringified list: "[123, 456, 789]" -> [123, 456, 789]
        signal_str = row[sensor_cols[4]]
        if pd.isna(signal_str):
            return np.nan, np.nan
            
        signal = np.array(ast.literal_eval(signal_str))
        
        # Smooth signal slightly
        signal = pd.Series(signal).rolling(window=5, center=True).mean().bfill().ffill().values
        
        # Find peaks (representing gait events)
        # Distance of 25 samples at 50Hz = 0.5s minimum between steps
        peaks, _ = find_peaks(signal, distance=25, prominence=np.std(signal)*0.5)
        
        if len(peaks) < 3:
            return np.nan, np.nan
            
        # Time between peaks (stride/step time) in seconds
        step_times = np.diff(peaks) / FS
        
        # Cadence: (Steps / Total Time) * 60
        total_time = len(signal) / FS
        cadence = (len(peaks) / total_time) * 60.0
        
        # Stride Time CV (Coefficient of Variation) = std / mean
        mean_step_time = np.mean(step_times)
        if mean_step_time > 0:
            stride_cv = np.std(step_times) / mean_step_time
        else:
            stride_cv = np.nan
            
        return cadence, stride_cv
    except Exception as e:
        return np.nan, np.nan

def process_file(filepath, terrain_prefix):
    logger.info(f"Processing {filepath.name}...")
    df = pd.read_csv(filepath)
    
    # Demographics act as the Subject ID in this dataset
    df['subject_id'] = df['Gender'].astype(str) + "_" + df['Age'].astype(str) + "_" + df['BMI'].astype(str)
    
    # Process each row
    cadences = []
    cvs = []
    
    # Data_1 to Data_6 are Sensor 1. Data_7 to Data_12 are Sensor 2.
    sensor1_cols = [f'Data_{i}' for i in range(1, 7)]
    
    for idx, row in df.iterrows():
        cadence, cv = calculate_gait_features(row, sensor1_cols)
        cadences.append(cadence)
        cvs.append(cv)
        
    df[f'{terrain_prefix}_cadence'] = cadences
    df[f'{terrain_prefix}_stride_time_cv'] = cvs
    
    # Group by subject to get average features per person
    # (since each person has ~70 rows/trials)
    subject_df = df.groupby('subject_id')[[f'{terrain_prefix}_cadence', f'{terrain_prefix}_stride_time_cv']].mean().reset_index()
    
    # Extract demographics back out
    subject_df['Gender'] = subject_df['subject_id'].apply(lambda x: x.split('_')[0])
    subject_df['Age'] = subject_df['subject_id'].apply(lambda x: int(x.split('_')[1]))
    subject_df['BMI'] = subject_df['subject_id'].apply(lambda x: int(x.split('_')[2]))
    
    return subject_df

def extract_mendeley(dataset_dir: str, output_path: str):
    base_dir = Path(dataset_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Mendeley directory not found: {base_dir}")

    walk_file = base_dir / "Walking_Data.csv"
    climb_file = base_dir / "Climbing_data.csv"

    if not walk_file.exists() or not climb_file.exists():
        raise FileNotFoundError("Mendeley Walking_Data.csv or Climbing_data.csv not found.")

    walk_df = process_file(walk_file, "flat_gait")
    climb_df = process_file(climb_file, "climbing")

    logger.info("Merging Flat and Climbing features...")
    merged_df = walk_df.merge(climb_df[['subject_id', 'climbing_cadence', 'climbing_stride_time_cv']], on='subject_id', how='inner')
    
    logger.info(f"Extracted features for {len(merged_df)} unique subjects.")
    
    # Map to Kneeva features
    kneeva_df = pd.DataFrame()
    kneeva_df['ctx_age'] = merged_df['Age']
    kneeva_df['ctx_sex_female'] = np.where(merged_df['Gender'] == 'F', 1.0, 0.0)
    kneeva_df['ctx_bmi'] = merged_df['BMI']
    kneeva_df['flat_gait_cadence'] = merged_df['flat_gait_cadence']
    kneeva_df['flat_gait_stride_time_cv'] = merged_df['flat_gait_stride_time_cv']
    kneeva_df['climbing_cadence'] = merged_df['climbing_cadence']
    kneeva_df['climbing_stride_time_cv'] = merged_df['climbing_stride_time_cv']
    
    # Drop NaNs
    kneeva_df = kneeva_df.dropna()
    
    logger.info(f"Final Mendeley IMU dataset shape: {kneeva_df.shape}")
    
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kneeva_df.to_csv(out_path, index=False)
    logger.info(f"Saved extracted features to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Mendeley IMU gait features")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Path to Mendeley CSVs")
    parser.add_argument("--output", type=str, default="data/processed/mendeley_imu_features.csv", help="Output CSV path")
    
    args = parser.parse_args()
    
    try:
        extract_mendeley(args.dataset_dir, args.output)
    except Exception as e:
        logger.error(f"Failed to extract Mendeley data: {e}", exc_info=True)
        exit(1)
