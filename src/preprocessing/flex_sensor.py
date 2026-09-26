"""
Flex Sensor Preprocessing Module — Layer 1

Converts ADC voltage readings from a 2.2" Flex Bend Sensor (variable
resistor in a voltage divider circuit) into knee bend angle.

Pipeline:
  1. Read ADC voltage from CSV
  2. Convert voltage → resistance via the voltage divider formula
  3. Map resistance → bend angle via linear interpolation
  4. Smooth the angle signal (moving average)

The flex sensor provides a direct ROM (Range of Motion) measurement
complementary to the IMU-derived knee angle.

All tuneable numbers come from FlexSensorConfig.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from src.config import FlexSensorConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class FlexSensorResult:
    """Structured output of the flex sensor preprocessing pipeline."""

    voltage: np.ndarray
    """Raw ADC voltage readings ``(N,)``."""

    resistance_ohms: np.ndarray
    """Computed flex sensor resistance ``(N,)`` in Ohms."""

    knee_angle_deg: np.ndarray
    """Knee bend angle ``(N,)`` in degrees (smoothed)."""

    knee_angle_raw_deg: np.ndarray
    """Knee bend angle ``(N,)`` in degrees (before smoothing)."""

    rom_deg: float
    """Range of motion: max_angle − min_angle (degrees)."""

    timestamps_ms: Optional[np.ndarray] = None
    """Timestamps ``(N,)`` in milliseconds, if available."""


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------
def voltage_to_resistance(
    v_out: np.ndarray,
    v_in: float,
    r_fixed: float,
) -> np.ndarray:
    """Convert voltage divider output to flex sensor resistance.

    Circuit: V_out = V_in × R_fixed / (R_flex + R_fixed)
    Solving:  R_flex = R_fixed × (V_in / V_out − 1)

    Parameters
    ----------
    v_out : np.ndarray
        ADC voltage readings ``(N,)``.
    v_in : float
        Supply voltage to the voltage divider.
    r_fixed : float
        Fixed resistor value in the divider (Ohms).
    """
    # Clamp to avoid division by zero / negative resistance
    v_out_safe = np.clip(v_out, 0.001, v_in - 0.001)
    resistance = r_fixed * (v_in / v_out_safe - 1.0)
    return resistance


def resistance_to_angle(
    resistance: np.ndarray,
    r_flat: float,
    r_max_bend: float,
    angle_flat: float,
    angle_max_bend: float,
) -> np.ndarray:
    """Map flex sensor resistance to bend angle via linear interpolation.

    Parameters
    ----------
    resistance : np.ndarray
        Sensor resistance in Ohms.
    r_flat : float
        Resistance when sensor is flat (~10 KΩ).
    r_max_bend : float
        Resistance at maximum bend (~110 KΩ).
    angle_flat : float
        Angle in degrees when flat (e.g., 180° = straight leg).
    angle_max_bend : float
        Angle in degrees at max bend (e.g., 45° = deep squat).
    """
    angle = np.interp(
        resistance,
        [r_flat, r_max_bend],
        [angle_flat, angle_max_bend],
    )
    return angle


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def preprocess_flex_sensor(
    csv_file: str,
    config: Optional[FlexSensorConfig] = None,
) -> FlexSensorResult:
    """Run the full flex sensor preprocessing pipeline.

    Parameters
    ----------
    csv_file : str
        Path to CSV file containing ADC voltage readings from the flex
        sensor voltage divider circuit.
    config : FlexSensorConfig, optional
        Pipeline configuration.  Defaults used when ``None``.
    """
    if config is None:
        config = FlexSensorConfig()

    logger.info("Loading flex sensor data: %s", csv_file)
    df = pd.read_csv(csv_file)

    voltage = df[config.voltage_col].values.astype(float)
    timestamps_ms = (
        df[config.timestamp_col].values
        if config.timestamp_col in df.columns
        else None
    )

    # Step 1 — voltage → resistance
    resistance = voltage_to_resistance(
        voltage,
        config.supply_voltage,
        config.fixed_resistor_ohms,
    )
    logger.info(
        "Resistance range: %.0f – %.0f Ω.",
        float(np.min(resistance)),
        float(np.max(resistance)),
    )

    # Step 2 — resistance → angle
    angle_raw = resistance_to_angle(
        resistance,
        config.flat_resistance_ohms,
        config.max_bend_resistance_ohms,
        config.flat_angle_deg,
        config.max_bend_angle_deg,
    )

    # Step 3 — smooth
    smooth_size = max(1, int(config.smooth_window_sec * config.sampling_rate))
    angle_smooth = uniform_filter1d(angle_raw, size=smooth_size)

    rom = float(np.max(angle_smooth) - np.min(angle_smooth))
    logger.info(
        "Knee angle: %.1f° – %.1f° (ROM=%.1f°).",
        float(np.min(angle_smooth)),
        float(np.max(angle_smooth)),
        rom,
    )

    return FlexSensorResult(
        voltage=voltage,
        resistance_ohms=resistance,
        knee_angle_deg=angle_smooth,
        knee_angle_raw_deg=angle_raw,
        rom_deg=rom,
        timestamps_ms=timestamps_ms,
    )
