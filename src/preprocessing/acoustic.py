"""
Acoustic / Crepitus Preprocessing Module — Layer 1, Build Guide §4.2

Deterministic DSP pipeline:
  1. Load and validate WAV audio  (§4.2 Step 1)
  2. Spectral-gating noise reduction  (§4.2 Step 2)
  3. Time-frequency representation (STFT + Mel spectrogram)  (§4.2 Step 3)
  4. Crepitus event detection via spectral-flux peaks  (§4.2 Step 4)

All tuneable numbers come from AcousticConfig.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import librosa
import noisereduce as nr
import numpy as np
from scipy.signal import find_peaks

from src.config import AcousticConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class AcousticResult:
    """Structured output of the acoustic preprocessing pipeline."""

    signal_clean: np.ndarray
    """Noise-reduced audio signal ``(N,)``."""

    sample_rate: int
    """Audio sample rate in Hz."""

    crepitus_peaks: np.ndarray
    """Indices into the spectral-flux vector where crepitus events were detected."""

    mel_spec_db: np.ndarray
    """Mel spectrogram in dB ``(n_mels, T)``."""

    stft_magnitude: np.ndarray
    """STFT magnitude matrix ``(freq_bins, T)``."""


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------
def load_from_adc_csv(
    csv_file: str,
    config: AcousticConfig,
) -> tuple[np.ndarray, int]:
    """Load vibration data from an ADC CSV file.

    The piezo vibration sensor outputs raw AC voltage which, after
    conditioning (1 MΩ load resistor + MCP6002 op-amp), is digitised
    by the microcontroller's ADC and saved as CSV.

    The voltage is centred around zero and normalised to ``-1..+1``
    (like audio) so the downstream DSP pipeline works identically.
    """
    import pandas as pd

    df = pd.read_csv(csv_file)
    voltage = df[config.adc_voltage_col].values.astype(float)

    # Centre the signal (remove DC offset from ADC mid-point)
    voltage_centred = voltage - np.mean(voltage)

    # Normalise to -1..+1 by the ADC reference voltage / 2
    half_ref = config.adc_reference_voltage / 2.0
    signal = voltage_centred / half_ref if half_ref > 0 else voltage_centred

    sr = config.adc_sample_rate
    return signal, sr


def load_and_validate_audio(
    audio_file: str,
    config: AcousticConfig,
) -> tuple[np.ndarray, int]:
    """Load audio/vibration data and validate against Build Guide thresholds.

    Supports two input modes:
    - **WAV file** (``input_from_adc=False``): loaded via ``librosa``.
    - **ADC CSV** (``input_from_adc=True``): loaded from a CSV of
      voltage samples captured from the piezo vibration sensor.
    """
    if config.input_from_adc:
        signal, sr = load_from_adc_csv(audio_file, config)
    else:
        signal, sr = librosa.load(
            audio_file, sr=config.target_sample_rate, mono=config.mono,
        )

    duration = len(signal) / sr
    if duration < config.min_duration_sec:
        raise ValueError(
            f"Recording too short ({duration:.1f}s < "
            f"{config.min_duration_sec}s)"
        )

    peak_amplitude = float(np.max(np.abs(signal)))
    if peak_amplitude < config.min_peak_amplitude:
        raise ValueError(
            f"Signal too quiet (peak={peak_amplitude:.4f} < "
            f"{config.min_peak_amplitude}) — check sensor coupling"
        )

    logger.info(
        "Audio loaded: %.1f s, sr=%d Hz, peak=%.4f (from_adc=%s).",
        duration, sr, peak_amplitude, config.input_from_adc,
    )
    return signal, sr


def reduce_noise(
    signal: np.ndarray,
    sr: int,
    config: AcousticConfig,
) -> np.ndarray:
    """Spectral-gating noise reduction.

    Uses the first ``noise_profile_duration_sec`` of audio (before patient
    movement begins) as the noise profile.
    """
    noise_samples = int(config.noise_profile_duration_sec * sr)
    noise_profile = signal[:noise_samples]

    signal_clean: np.ndarray = nr.reduce_noise(
        y=signal,
        sr=sr,
        y_noise=noise_profile,
        stationary=config.noise_stationary,
        prop_decrease=config.noise_prop_decrease,
    )

    logger.info(
        "Noise reduction applied (profile: first %.1f s, decrease: %.0f%%).",
        config.noise_profile_duration_sec,
        config.noise_prop_decrease * 100,
    )
    return signal_clean


def compute_time_frequency(
    signal_clean: np.ndarray,
    sr: int,
    config: AcousticConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute STFT magnitude and Mel spectrogram (dB)."""
    stft = librosa.stft(
        signal_clean,
        n_fft=config.stft_n_fft,
        hop_length=config.stft_hop_length,
    )
    magnitude = np.abs(stft)

    mel_spec = librosa.feature.melspectrogram(
        y=signal_clean,
        sr=sr,
        n_mels=config.mel_n_mels,
        fmin=config.mel_fmin_hz,
        fmax=config.mel_fmax_hz,
        n_fft=config.stft_n_fft,
        hop_length=config.stft_hop_length,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return magnitude, mel_spec_db


def detect_crepitus_events(
    magnitude: np.ndarray,
    sr: int,
    config: AcousticConfig,
) -> np.ndarray:
    """Detect crepitus events as peaks in the spectral-flux function."""
    spectral_flux = np.sqrt(
        np.mean(np.diff(magnitude, axis=1) ** 2, axis=0)
    )

    threshold = (
        np.mean(spectral_flux)
        + config.crepitus_threshold_std_multiplier * np.std(spectral_flux)
    )
    min_distance_frames = max(
        1, int(config.crepitus_min_distance_sec * sr / config.stft_hop_length)
    )

    peaks, _ = find_peaks(
        spectral_flux,
        height=threshold,
        distance=min_distance_frames,
        prominence=config.crepitus_peak_prominence,
    )

    logger.info("Detected %d crepitus events.", len(peaks))
    return peaks


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def preprocess_acoustic(
    audio_file: str,
    config: Optional[AcousticConfig] = None,
) -> AcousticResult:
    """Run the full acoustic / crepitus preprocessing pipeline.

    Parameters
    ----------
    audio_file : str
        Path to the knee acoustic ``.wav`` recording.
    config : AcousticConfig, optional
        Pipeline configuration.  Build-Guide defaults used when ``None``.
    """
    if config is None:
        config = AcousticConfig()

    signal, sr = load_and_validate_audio(audio_file, config)
    signal_clean = reduce_noise(signal, sr, config)
    magnitude, mel_spec_db = compute_time_frequency(signal_clean, sr, config)
    crepitus_peaks = detect_crepitus_events(magnitude, sr, config)

    return AcousticResult(
        signal_clean=signal_clean,
        sample_rate=sr,
        crepitus_peaks=crepitus_peaks,
        mel_spec_db=mel_spec_db,
        stft_magnitude=magnitude,
    )
