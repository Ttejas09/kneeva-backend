"""
Layer 2 & 3 — Per-Modality Feature Extraction

Re-exports Tier A (OA-associated), Tier B (general/differential),
and Tier C (contextual meta-inputs) feature extraction functions::

    from src.features import extract_rom_features, extract_contextual_features
"""

from src.features.tier_a import (
    calculate_cocontraction_index,
    extract_cci_features,
    extract_crepitus_features,
    extract_rom_features,
    extract_strength_features,
)
from src.features.tier_b import (
    extract_gait_asymmetry_features,
    extract_general_weakness_features,
    extract_injury_history_features,
    extract_neuromuscular_features,
)
from src.features.tier_c import (
    extract_contextual_features,
)

__all__ = [
    # Tier A
    "extract_rom_features",
    "extract_crepitus_features",
    "calculate_cocontraction_index",
    "extract_cci_features",
    "extract_strength_features",
    # Tier B
    "extract_gait_asymmetry_features",
    "extract_neuromuscular_features",
    "extract_general_weakness_features",
    "extract_injury_history_features",
    # Tier C
    "extract_contextual_features",
]
