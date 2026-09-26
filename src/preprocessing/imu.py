"""
IMU Preprocessing Module — Layer 1, Build Guide §4.1

Deterministic pipeline:
  1. Butterworth bandpass filter on raw accel/gyro  (§4.1 Step 4)
  2. Madgwick AHRS orientation estimation → quaternions  (§4.1 Step 1)
  3. Gait cycle segmentation via GaitPy  (§4.1 Step 3)

All tuneable numbers come from IMUConfig — nothing is hardcoded here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import ahrs
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

from src.config import IMUConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class IMUResult:
    """Structured output of the IMU preprocessing pipeline."""

    quaternions: np.ndarray
    """(N, 4) IMU orientation quaternions [w, x, y, z]."""

    gait_features: Optional[pd.DataFrame]
    """GaitPy extracted features DataFrame (stride time, stance time, etc.)."""

    gait_cycles: List[Tuple[int, int]]
    """List of (start_sample, end_sample) index pairs for each detected gait cycle."""


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------
def bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    data : np.ndarray
        Input signal, shape ``(N,)`` or ``(N, channels)``.
    lowcut, highcut : float
        Cutoff frequencies in Hz.
    fs : float
        Sampling rate in Hz.
    order : int
        Filter order.
    """
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data, axis=0)


def estimate_orientation(
    gyr: np.ndarray,
    acc: np.ndarray,
    frequency: float,
    mag: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate orientation with Madgwick AHRS filter.

    Parameters
    ----------
    gyr : np.ndarray
        Gyroscope readings ``(N, 3)`` in rad/s.
    acc : np.ndarray
        Accelerometer readings ``(N, 3)`` in m/s².
    frequency : float
        Sampling rate in Hz.
    mag : np.ndarray, optional
        Magnetometer readings ``(N, 3)`` in µT.  Enables 9-axis fusion.

    Returns
    -------
    np.ndarray
        Orientation quaternions ``(N, 4)`` in ``[w, x, y, z]`` convention.
    """
    kwargs: dict = dict(gyr=gyr, acc=acc, frequency=frequency)
    if mag is not None:
        kwargs["mag"] = mag

    madgwick = ahrs.filters.Madgwick(**kwargs)
    return madgwick.Q  # (N, 4) — [w, x, y, z]


def segment_gait_cycles(
    vertical_accel: np.ndarray,
    timestamps_ms: np.ndarray,
    config: IMUConfig,
) -> Tuple[Optional[pd.DataFrame], List[Tuple[int, int]]]:
    """Detect gait events and convert to sample-index cycle boundaries.

    Parameters
    ----------
    vertical_accel : np.ndarray
        Vertical acceleration ``(N,)`` in m/s².
    timestamps_ms : np.ndarray
        Corresponding timestamps ``(N,)`` in milliseconds.
    config : IMUConfig
        Pipeline configuration.

    Returns
    -------
    gait_features : pd.DataFrame | None
        GaitPy output (stride/stance/step parameters per cycle).
    gait_cycles : list[tuple[int, int]]
        ``(start_idx, end_idx)`` sample-index pairs.
    """
    from gaitpy import gaitpy as gaitpy_module

    v_accel_df = pd.DataFrame({
        "timestamps": timestamps_ms,
        "y": vertical_accel,
    })

    gp = gaitpy_module.Gaitpy(
        raw_data=v_accel_df,
        sample_rate=config.sampling_rate,
        v_acc_col_name="y",
        ts_col_name="timestamps",
        v_acc_units=config.gaitpy_v_acc_units,
        ts_units=config.gaitpy_ts_units,
    )

    try:
        gait_features_df = gp.extract_features(
            ic_prom=config.gaitpy_ic_prominence,
            fc_prom=config.gaitpy_fc_prominence,
        )
    except Exception as exc:
        logger.warning("GaitPy could not detect gait events: %s", exc)
        return None, []

    # Convert timestamp-based IC events → sample-index cycle boundaries
    gait_cycles: List[Tuple[int, int]] = []
    if gait_features_df is not None and not gait_features_df.empty:
        # GaitPy provides IC (initial-contact) timestamps per stride
        ic_col = None
        for candidate in ("IC", "ic", "initial_contact"):
            if candidate in gait_features_df.columns:
                ic_col = candidate
                break

        if ic_col is not None:
            ic_times = gait_features_df[ic_col].values
            for i in range(len(ic_times) - 1):
                start_idx = int(np.argmin(np.abs(timestamps_ms - ic_times[i])))
                end_idx = int(np.argmin(np.abs(timestamps_ms - ic_times[i + 1])))
                if end_idx > start_idx:
                    gait_cycles.append((start_idx, end_idx))

    logger.info("Detected %d gait cycles.", len(gait_cycles))
    return gait_features_df, gait_cycles


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def preprocess_imu(
    imu_file: str,
    config: Optional[IMUConfig] = None,
) -> IMUResult:
    """Run the full IMU preprocessing pipeline.

    Pipeline order
    --------------
    1. Load CSV data for the IMU sensor.
    2. Optional: Convert MPU6050 raw I2C values to physical units.
    3. Bandpass-filter raw accelerometer / gyroscope signals.
    4. Estimate orientation via Madgwick AHRS → quaternions.
    5. Segment gait cycles via GaitPy.

    Parameters
    ----------
    imu_file : str
        Path to the IMU CSV file.
    config : IMUConfig, optional
        Pipeline configuration.  Build-Guide defaults used when ``None``.
    """
    if config is None:
        config = IMUConfig()

    logger.info("Loading IMU data: %s", imu_file)
    imu_df = pd.read_csv(imu_file)

    accel_cols = list(config.accel_cols)
    gyro_cols = list(config.gyro_cols)

    acc_raw = imu_df[accel_cols].values.astype(float)
    gyr_raw = imu_df[gyro_cols].values.astype(float)
    timestamps_ms = imu_df[config.timestamp_col].values

    # Step 0 — MPU6050 raw I2C scaling (if needed)
    if config.raw_input:
        G_TO_MS2 = 9.80665
        logger.info(
            "Scaling raw MPU6050 values (accel: %.0f LSB/g, gyro: %.1f LSB/°/s).",
            config.accel_sensitivity_lsb_per_g,
            config.gyro_sensitivity_lsb_per_dps,
        )
        acc_raw = (acc_raw / config.accel_sensitivity_lsb_per_g) * G_TO_MS2
        gyr_raw = np.deg2rad(gyr_raw / config.gyro_sensitivity_lsb_per_dps)

    gyr = gyr_raw

    # Step 1 — bandpass filter raw IMU
    logger.info(
        "Bandpass filtering (%.1f–%.1f Hz, order %d).",
        config.filter_lowcut_hz,
        config.filter_highcut_hz,
        config.filter_order,
    )
    acc = bandpass_filter(
        acc_raw,
        config.filter_lowcut_hz,
        config.filter_highcut_hz,
        config.sampling_rate,
        config.filter_order,
    )

    # Step 2 — AHRS orientation estimation
    logger.info("Running Madgwick AHRS at %.0f Hz.", config.sampling_rate)
    q = estimate_orientation(gyr, acc, config.sampling_rate)

    # Step 3 — gait cycle segmentation
    vertical_accel = acc[:, config.vertical_accel_index]
    gait_features, gait_cycles = segment_gait_cycles(
        vertical_accel, timestamps_ms, config,
    )

    return IMUResult(
        quaternions=q,
        gait_features=gait_features,
        gait_cycles=gait_cycles,
    )
