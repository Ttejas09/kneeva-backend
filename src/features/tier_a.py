"""
Tier A — OA-Associated Signal Feature Extraction (Build Guide §5.1)

Primary biomarkers directly indicative of knee osteoarthritis:
  A1. Range of Motion (ROM)       — from Flex Sensor
  A2. Crepitus Score              — from Acoustic / Piezo pipeline
  A3. Co-Contraction Index (CCI)  — from sEMG pipeline
  A4. Strength Deficit            — from Dynamometer pipeline

All tuneable numbers come from FeatureExtractionConfig — nothing is
hardcoded here.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import (
    CrepitusFeatureConfig,
    FeatureExtractionConfig,
    ROMConfig,
    StrengthConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A1 — Range of Motion
# ---------------------------------------------------------------------------
def extract_rom_features(
    knee_angle_timeseries: np.ndarray,
    config: Optional[ROMConfig] = None,
) -> Dict[str, float]:
    """Extract ROM features from knee angle time series.

    Parameters
    ----------
    knee_angle_timeseries : np.ndarray
        1-D array of knee flexion-extension angles in degrees.
        Convention: 180° = full extension (straight leg),
        decreasing values = increasing flexion.
        (This matches the Flex Sensor output where flat = 180°.)
    config : ROMConfig, optional
        Clinical reference values.  Build-Guide defaults used when ``None``.

    Returns
    -------
    dict[str, float]
        ROM feature dictionary.

    Reference Ranges (Build Guide §5.1)
    ------------------------------------
    | Measure       | Normal    | Mild OA   | Moderate | Severe |
    |---------------|-----------|-----------|----------|--------|
    | Total ROM     | 130–140°  | 110–130°  | 90–110°  | <90°   |
    | Ext. Deficit  | 0°        | 0–5°      | 5–15°    | >15°   |
    """
    if config is None:
        config = ROMConfig()

    max_extension = float(np.max(knee_angle_timeseries))  # closest to 180°
    max_flexion = float(np.min(knee_angle_timeseries))    # furthest from 180°

    total_rom = max_extension - max_flexion

    # Extension deficit: how far from full extension (180°) can they reach?
    extension_deficit = max(0.0, config.normal_max_flexion_deg - max_extension)

    # Flexion deficit: how far from expected full flexion?
    # Expected max flexion angle = 180 - normal_max_flexion = 180 - 135 = 45°
    expected_max_flexion_angle = (
        config.normal_full_extension_deg
        + (config.normal_full_extension_deg - config.normal_max_flexion_deg)
    )
    # In our convention: smaller angle = more flexion
    flexion_deficit = max(0.0, max_flexion - expected_max_flexion_angle)

    logger.info(
        "ROM features: total=%.1f°, ext_def=%.1f°, flex_def=%.1f°.",
        total_rom, extension_deficit, flexion_deficit,
    )

    return {
        "rom_total_deg": total_rom,
        "rom_max_extension_deg": max_extension,
        "rom_max_flexion_deg": max_flexion,
        "rom_extension_deficit_deg": extension_deficit,
        "rom_flexion_deficit_deg": flexion_deficit,
    }


# ---------------------------------------------------------------------------
# A2 — Crepitus Score
# ---------------------------------------------------------------------------
def extract_crepitus_features(
    signal_clean: np.ndarray,
    sample_rate: int,
    crepitus_peaks: np.ndarray,
    mel_spec_db: np.ndarray,
    stft_hop_length: int,
    config: Optional[CrepitusFeatureConfig] = None,
) -> Dict[str, float]:
    """Extract crepitus features from acoustic analysis.

    Parameters
    ----------
    signal_clean : np.ndarray
        Cleaned 1-D audio / vibration signal.
    sample_rate : int
        Sampling rate of ``signal_clean``.
    crepitus_peaks : np.ndarray
        Indices into the spectral-flux vector where crepitus events were
        detected (from Layer 1 ``AcousticResult``).
    mel_spec_db : np.ndarray
        Mel spectrogram in dB ``(n_mels, T)`` — used for spectral shape.
    stft_hop_length : int
        Hop length used in the STFT (needed to map peak indices back to
        time-domain sample positions).
    config : CrepitusFeatureConfig, optional
        Scoring parameters.

    Returns
    -------
    dict[str, float]

    Clinical Interpretation (Build Guide §5.1)
    -------------------------------------------
    | Rate         | Meaning                            |
    |--------------|------------------------------------|
    | 0 /sec       | No crepitus                        |
    | 0–1 /sec     | Mild — possible early changes      |
    | 1–3 /sec     | Moderate — likely fibrillation     |
    | >3 /sec      | Severe — significant damage        |
    """
    if config is None:
        config = CrepitusFeatureConfig()

    import librosa  # lazy import to avoid top-level dep

    duration = len(signal_clean) / sample_rate

    # Count-based features
    crepitus_count = len(crepitus_peaks)
    crepitus_rate = crepitus_count / duration if duration > 0 else 0.0

    # Energy of each crepitus event (window around peak)
    window_samples = int(config.event_window_sec * sample_rate)
    crepitus_energies: List[float] = []
    for peak_idx in crepitus_peaks:
        # Convert spectral-flux index → time-domain sample index
        centre_sample = int(peak_idx * stft_hop_length)
        win_start = max(0, centre_sample - window_samples)
        win_end = min(len(signal_clean), centre_sample + window_samples)
        segment = signal_clean[win_start:win_end]
        crepitus_energies.append(float(np.sum(segment ** 2)))

    # Spectral centroid (grinding = low, clicking = high)
    spectral_centroids = librosa.feature.spectral_centroid(
        y=signal_clean, sr=sample_rate,
    )[0]

    features = {
        "crepitus_count": float(crepitus_count),
        "crepitus_rate_per_sec": crepitus_rate,
        "crepitus_mean_energy": (
            float(np.mean(crepitus_energies)) if crepitus_energies else 0.0
        ),
        "crepitus_spectral_centroid_hz": float(np.mean(spectral_centroids)),
        "crepitus_present": float(
            crepitus_count > config.presence_threshold_count
        ),
    }

    logger.info(
        "Crepitus features: count=%d, rate=%.2f/s, present=%s.",
        crepitus_count, crepitus_rate, bool(features["crepitus_present"]),
    )
    return features


# ---------------------------------------------------------------------------
# A3 — Co-Contraction Index
# ---------------------------------------------------------------------------
def calculate_cocontraction_index(
    agonist_amplitude: np.ndarray,
    antagonist_amplitude: np.ndarray,
) -> float:
    """Calculate the Co-Contraction Index using the Winter (1990) method.

    CCI = (2 × overlap_area) / (agonist_area + antagonist_area) × 100

    Parameters
    ----------
    agonist_amplitude : np.ndarray
        Linear envelope of the agonist (e.g., quadriceps during extension).
    antagonist_amplitude : np.ndarray
        Linear envelope of the antagonist (e.g., hamstring during extension).

    Returns
    -------
    float
        CCI as a percentage (0–100).

    Reference Ranges (Build Guide §5.1)
    ------------------------------------
    | CCI (%) | Interpretation                                |
    |---------|-----------------------------------------------|
    | 10–25   | Normal — efficient reciprocal activation       |
    | 25–40   | Elevated — possible early OA compensation     |
    | >40     | High — significant arthrogenic inhibition      |
    """
    overlap = np.minimum(agonist_amplitude, antagonist_amplitude)

    overlap_area = float(np.trapezoid(overlap))
    total_area = float(
        np.trapezoid(agonist_amplitude) + np.trapezoid(antagonist_amplitude)
    )

    if total_area == 0:
        return 0.0

    cci = (2.0 * overlap_area / total_area) * 100.0
    return cci


def extract_cci_features(
    semg_channels: Dict[str, Dict[str, np.ndarray]],
    gait_cycles: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, float]:
    """Extract co-contraction features from sEMG data.

    Calculates:
    - CCI for knee extension (quad agonist, ham antagonist)
    - Mean CCI across gait cycles (if provided)

    Parameters
    ----------
    semg_channels : dict
        Mapping ``{channel_name: {"amplitude_mvc": np.ndarray, ...}}``.
        Requires at least ``RF`` (agonist) and ``BF`` (antagonist).
    gait_cycles : list of (start, end), optional
        Gait cycle sample-index boundaries for per-cycle CCI.
    """
    features: Dict[str, float] = {}

    # Overall CCI (full recording)
    rf = semg_channels.get("RF", {})
    bf = semg_channels.get("BF", {})

    rf_amp = rf.get("amplitude_mvc")
    bf_amp = bf.get("amplitude_mvc")

    if rf_amp is not None and bf_amp is not None:
        min_len = min(len(rf_amp), len(bf_amp))
        cci_overall = calculate_cocontraction_index(
            rf_amp[:min_len], bf_amp[:min_len],
        )
        features["cci_extension_overall"] = cci_overall
        logger.info("CCI overall (extension): %.1f%%.", cci_overall)

        # Per-gait-cycle CCI
        if gait_cycles:
            cycle_ccis = []
            for start, end in gait_cycles:
                if end <= min_len:
                    c = calculate_cocontraction_index(
                        rf_amp[start:end], bf_amp[start:end],
                    )
                    cycle_ccis.append(c)
            if cycle_ccis:
                features["cci_extension_gait_mean"] = float(np.mean(cycle_ccis))
                features["cci_extension_gait_std"] = float(np.std(cycle_ccis))
                features["cci_extension_gait_max"] = float(np.max(cycle_ccis))
    else:
        logger.warning(
            "CCI features skipped — RF and/or BF amplitude_mvc not available."
        )

    return features


# ---------------------------------------------------------------------------
# A4 — Strength Deficit
# ---------------------------------------------------------------------------
def extract_strength_features(
    ext_affected: float,
    ext_unaffected: float,
    flex_affected: float,
    flex_unaffected: float,
    body_weight_kg: float,
    config: Optional[StrengthConfig] = None,
) -> Dict[str, float]:
    """Extract strength deficit features from dynamometer data.

    Parameters
    ----------
    ext_affected, ext_unaffected : float
        Peak extension force (N) for the affected and unaffected legs.
    flex_affected, flex_unaffected : float
        Peak flexion force (N) for the affected and unaffected legs.
    body_weight_kg : float
        Patient body weight in kg.
    config : StrengthConfig, optional
        Constants (gravity, etc.).

    Returns
    -------
    dict[str, float]

    Clinical Note
    -------------
    A bilateral extension deficit >20% is clinically significant.
    """
    if config is None:
        config = StrengthConfig()

    # Bilateral extension deficit
    ext_deficit_pct = (
        ((ext_unaffected - ext_affected) / ext_unaffected) * 100.0
        if ext_unaffected > 0 else 0.0
    )

    # Bilateral flexion deficit
    flex_deficit_pct = (
        ((flex_unaffected - flex_affected) / flex_unaffected) * 100.0
        if flex_unaffected > 0 else 0.0
    )

    # Hamstring:Quadriceps ratio (affected side)
    hq_ratio = (
        flex_affected / ext_affected if ext_affected > 0 else 0.0
    )

    # Body-weight-normalised extension strength
    body_weight_n = body_weight_kg * config.gravity_ms2
    ext_normalised = ext_affected / body_weight_n if body_weight_n > 0 else 0.0

    features = {
        "strength_ext_deficit_pct": ext_deficit_pct,
        "strength_flex_deficit_pct": flex_deficit_pct,
        "strength_hq_ratio": hq_ratio,
        "strength_ext_normalised_bw": ext_normalised,
        "strength_ext_affected_N": ext_affected,
        "strength_ext_unaffected_N": ext_unaffected,
    }

    logger.info(
        "Strength features: ext_def=%.1f%%, H:Q=%.2f, ext_norm=%.3f BW.",
        ext_deficit_pct, hq_ratio, ext_normalised,
    )
    return features
