"""
Dynamometer Preprocessing Module — Layer 1, Build Guide §4.4

Two input modes:
  A. **Manual entry** (default) — User reads peak force from the device
     display and enters it in the app.  No signal processing needed.
  B. **Load-cell CSV** — Raw time-series from a load-cell + HX711.
     Smoothing + peak detection + trial selection applied.

All tuneable numbers come from DynamometerConfig.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from src.config import DynamometerConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class ForceResult:
    """Peak-force summary for one direction (extension or flexion)."""

    peak_force_n: float
    """Maximum peak force across top trials (Newtons)."""

    mean_peak_force_n: float
    """Mean of the top-N trial peaks (Newtons)."""

    cv_percent: float
    """Coefficient of variation across top trials (%)."""

    all_peaks_n: List[float]
    """All detected peak values (sorted descending, Newtons)."""


@dataclass
class DynamometerResult:
    """Structured output of the dynamometer preprocessing pipeline."""

    extension: Optional[ForceResult] = None
    flexion: Optional[ForceResult] = None


# ---------------------------------------------------------------------------
# Manual-entry helper
# ---------------------------------------------------------------------------
def force_result_from_manual(
    peak_values_n: List[float],
) -> ForceResult:
    """Create a ForceResult from manually-entered peak force values.

    Parameters
    ----------
    peak_values_n : list[float]
        One or more peak-force readings (Newtons) entered by the clinician
        from the dynamometer's display.  Typically 3 trials.
    """
    if not peak_values_n:
        return ForceResult(
            peak_force_n=0.0, mean_peak_force_n=0.0,
            cv_percent=0.0, all_peaks_n=[],
        )

    sorted_peaks = sorted(peak_values_n, reverse=True)
    mean_p = float(np.mean(sorted_peaks))
    std_p = float(np.std(sorted_peaks)) if len(sorted_peaks) > 1 else 0.0
    cv = (std_p / mean_p * 100.0) if mean_p > 0 else 0.0

    return ForceResult(
        peak_force_n=sorted_peaks[0],
        mean_peak_force_n=mean_p,
        cv_percent=cv,
        all_peaks_n=sorted_peaks,
    )


# ---------------------------------------------------------------------------
# Signal-based extraction (load-cell CSV path)
# ---------------------------------------------------------------------------
def extract_peak_force(
    force_signal: np.ndarray,
    config: DynamometerConfig,
) -> ForceResult:
    """Extract peak force values from a raw isometric MVC recording.

    Only used when raw load-cell data is available (load-cell + HX711).
    For manual-entry workflows, use ``force_result_from_manual`` instead.
    """
    smooth_size = max(1, int(config.smooth_window_sec * config.sampling_rate))
    force_smooth = uniform_filter1d(
        force_signal.astype(float), size=smooth_size,
    )

    max_force = float(np.max(force_smooth))
    if max_force <= 0:
        logger.warning("No positive force detected in signal.")
        return ForceResult(
            peak_force_n=0.0, mean_peak_force_n=0.0,
            cv_percent=0.0, all_peaks_n=[],
        )

    min_distance = max(
        1, int(config.peak_min_distance_sec * config.sampling_rate),
    )

    peaks, properties = find_peaks(
        force_smooth,
        height=max_force * config.peak_height_fraction,
        distance=min_distance,
        prominence=config.peak_prominence_n,
    )

    if len(peaks) == 0:
        logger.warning("No force peaks detected above threshold.")
        return ForceResult(
            peak_force_n=0.0, mean_peak_force_n=0.0,
            cv_percent=0.0, all_peaks_n=[],
        )

    peak_heights = sorted(
        [float(h) for h in properties["peak_heights"]], reverse=True,
    )
    top_peaks = peak_heights[: config.n_trials]

    mean_peak = float(np.mean(top_peaks))
    std_peak = float(np.std(top_peaks)) if len(top_peaks) > 1 else 0.0
    cv = (std_peak / mean_peak * 100.0) if mean_peak > 0 else 0.0

    logger.info(
        "Detected %d peaks, top %d: max=%.1f N, mean=%.1f N, CV=%.1f%%.",
        len(peaks), len(top_peaks), top_peaks[0], mean_peak, cv,
    )

    return ForceResult(
        peak_force_n=top_peaks[0],
        mean_peak_force_n=mean_peak,
        cv_percent=cv,
        all_peaks_n=top_peaks,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def preprocess_dynamometer(
    extension_peaks_n: Optional[List[float]] = None,
    flexion_peaks_n: Optional[List[float]] = None,
    extension_signal: Optional[np.ndarray] = None,
    flexion_signal: Optional[np.ndarray] = None,
    config: Optional[DynamometerConfig] = None,
) -> DynamometerResult:
    """Run the dynamometer preprocessing pipeline.

    Supports two input modes:

    **Manual entry** (preferred for deployed device):
        Pass ``extension_peaks_n`` / ``flexion_peaks_n`` as lists of
        peak-force values (Newtons) read from the device display and
        entered in the app.

    **Load-cell CSV**:
        Pass ``extension_signal`` / ``flexion_signal`` as raw numpy arrays.
        Peak detection is run automatically.

    Manual-entry values take priority if both are provided.
    """
    if config is None:
        config = DynamometerConfig()

    # Extension
    if extension_peaks_n is not None:
        logger.info("Extension: using %d manual entries.", len(extension_peaks_n))
        extension = force_result_from_manual(extension_peaks_n)
    elif extension_signal is not None:
        logger.info("Extension: processing raw signal.")
        extension = extract_peak_force(extension_signal, config)
    else:
        extension = None

    # Flexion
    if flexion_peaks_n is not None:
        logger.info("Flexion: using %d manual entries.", len(flexion_peaks_n))
        flexion = force_result_from_manual(flexion_peaks_n)
    elif flexion_signal is not None:
        logger.info("Flexion: processing raw signal.")
        flexion = extract_peak_force(flexion_signal, config)
    else:
        flexion = None

    return DynamometerResult(extension=extension, flexion=flexion)
