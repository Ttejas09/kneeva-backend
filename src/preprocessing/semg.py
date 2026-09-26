"""
sEMG Preprocessing Module — Layer 1, Build Guide §4.3

Deterministic pipeline using NeuroKit2:
  1. Bandpass filter (20–450 Hz) + notch 50 Hz  via ``emg_clean``
  2. Full-wave rectification + linear envelope (6 Hz LP)  via ``emg_amplitude``
  3. Activation detection (onset / offset)  via ``emg_activation``
  4. MVC normalisation  (§4.3 MVC Normalization)

All tuneable numbers come from SEMGConfig.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import neurokit2 as nk
import numpy as np

from src.config import SEMGConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class ChannelResult:
    """Preprocessing result for a single sEMG channel."""

    filtered: np.ndarray
    """Bandpass + notch filtered signal ``(N,)``."""

    amplitude: np.ndarray
    """Rectified and smoothed envelope ``(N,)``."""

    activation: np.ndarray
    """Binary activation signal ``(N,)``: 0 = rest, 1 = active."""

    onsets: List[int]
    """Sample indices of activation onsets."""

    offsets: List[int]
    """Sample indices of activation offsets."""

    amplitude_mvc: Optional[np.ndarray] = None
    """MVC-normalised amplitude (% MVC). Populated after ``normalize_to_mvc``."""


@dataclass
class SEMGResult:
    """Structured output of the sEMG preprocessing pipeline."""

    channels: Dict[str, ChannelResult]
    """Mapping of channel label → processing result."""


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------
def preprocess_channel(
    raw_signal: np.ndarray,
    sampling_rate: int,
    clean_method: str,
    activation_method: str,
    input_is_envelope: bool = False,
    sensor_supply_voltage: float = 0.0,
) -> ChannelResult:
    """Full sEMG preprocessing for one channel (Build Guide §4.3).

    Two modes of operation
    ----------------------
    **Raw EMG** (``input_is_envelope=False``, default — clinical-grade sensors):

    1. Bandpass filter (20–450 Hz) — remove motion artifact / high-freq noise
    2. Notch filter (50 Hz) — remove powerline interference
    3. Full-wave rectification
    4. Linear envelope (low-pass at 6 Hz)
    5. Activation detection (onset / offset)

    **Pre-processed envelope** (``input_is_envelope=True`` — Advancer
    Technologies EMG Muscle Sensor V3.0 and similar boards):

    The sensor board already outputs a filtered, rectified, amplified
    analog voltage (0–Vs) that tracks muscle effort.  Steps 1–4 are
    **skipped** because applying them again would destroy the signal.
    Only activation detection (step 5) is run on the envelope.
    """
    if input_is_envelope:
        # ---------------------------------------------------------------
        # V3.0 path: signal is ALREADY an envelope
        # ---------------------------------------------------------------
        logger.info("  Input is pre-processed envelope — skipping NK2 clean/amplitude.")
        amplitude = np.asarray(raw_signal, dtype=float)

        # Optional: normalise ADC voltage to 0–1 range
        if sensor_supply_voltage > 0:
            amplitude = amplitude / sensor_supply_voltage

        # The "filtered" output is the raw envelope itself (no filtering
        # was applied by us — the hardware did it).
        filtered = amplitude.copy()

    else:
        # ---------------------------------------------------------------
        # Standard path: raw EMG → full NeuroKit2 pipeline
        # ---------------------------------------------------------------
        # Steps 1–2: clean signal via NeuroKit2
        filtered = nk.emg_clean(
            raw_signal, sampling_rate=sampling_rate, method=clean_method,
        )

        # Steps 3–4: amplitude envelope
        amplitude_raw = nk.emg_amplitude(filtered)
        amplitude = (
            amplitude_raw.values
            if hasattr(amplitude_raw, "values")
            else np.asarray(amplitude_raw)
        )

    # Step 5: activation detection (runs in both modes)
    signals_df, info_dict = nk.emg_activation(
        amplitude, sampling_rate=sampling_rate, method=activation_method,
    )
    activation = signals_df["EMG_Activity"].values
    onsets = list(info_dict.get("EMG_Onsets", []))
    offsets = list(info_dict.get("EMG_Offsets", []))

    return ChannelResult(
        filtered=np.asarray(filtered),
        amplitude=amplitude,
        activation=activation,
        onsets=onsets,
        offsets=offsets,
    )


def normalize_to_mvc(
    amplitude: np.ndarray,
    mvc_value: float,
) -> np.ndarray:
    """Normalise sEMG amplitude as percentage of MVC.

    Parameters
    ----------
    amplitude : np.ndarray
        Envelope amplitude values.
    mvc_value : float
        Peak amplitude recorded during MVC calibration trial.
    """
    if mvc_value <= 0:
        raise ValueError(f"MVC value must be positive, got {mvc_value}")
    return (amplitude / mvc_value) * 100.0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def preprocess_semg(
    channel_data: Dict[str, np.ndarray],
    config: Optional[SEMGConfig] = None,
    mvc_values: Optional[Dict[str, float]] = None,
) -> SEMGResult:
    """Run the full sEMG preprocessing pipeline for all channels.

    Parameters
    ----------
    channel_data : dict[str, np.ndarray]
        Mapping of channel label → raw 1-D signal array.
        Expected labels: ``VM``, ``VL``, ``RF``, ``BF`` (configurable).
    config : SEMGConfig, optional
        Pipeline configuration.  Build-Guide defaults used when ``None``.
    mvc_values : dict[str, float], optional
        Mapping of channel label → MVC peak amplitude for normalisation.
        When provided, ``amplitude_mvc`` is populated in each result.
    """
    if config is None:
        config = SEMGConfig()

    results: Dict[str, ChannelResult] = {}

    for ch_name in config.channels:
        if ch_name not in channel_data:
            logger.warning(
                "Channel '%s' not found in input data — skipping.", ch_name,
            )
            continue

        logger.info("Processing sEMG channel: %s", ch_name)
        result = preprocess_channel(
            raw_signal=channel_data[ch_name],
            sampling_rate=config.sampling_rate,
            clean_method=config.clean_method,
            activation_method=config.activation_method,
            input_is_envelope=config.input_is_envelope,
            sensor_supply_voltage=config.sensor_supply_voltage,
        )

        if mvc_values is not None and ch_name in mvc_values:
            result.amplitude_mvc = normalize_to_mvc(
                result.amplitude, mvc_values[ch_name],
            )
            logger.info("  MVC-normalised (MVC=%.2f µV).", mvc_values[ch_name])

        results[ch_name] = result

    logger.info(
        "sEMG preprocessing complete: %d / %d channels processed.",
        len(results),
        len(config.channels),
    )
    return SEMGResult(channels=results)
