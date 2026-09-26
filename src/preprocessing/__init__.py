"""
Layer 1 — Input & Preprocessing (Frozen, Deterministic)

Re-exports the six modality-specific preprocessing pipelines and their
structured result types for convenient top-level imports::

    from src.preprocessing import preprocess_imu, IMUResult
"""

from src.preprocessing.acoustic import AcousticResult, preprocess_acoustic
from src.preprocessing.dynamometer import DynamometerResult, preprocess_dynamometer
from src.preprocessing.flex_sensor import FlexSensorResult, preprocess_flex_sensor
from src.preprocessing.imu import IMUResult, preprocess_imu
from src.preprocessing.questionnaire import QuestionnaireResult, preprocess_questionnaire
from src.preprocessing.semg import SEMGResult, preprocess_semg

__all__ = [
    "preprocess_imu",
    "IMUResult",
    "preprocess_acoustic",
    "AcousticResult",
    "preprocess_semg",
    "SEMGResult",
    "preprocess_dynamometer",
    "DynamometerResult",
    "preprocess_questionnaire",
    "QuestionnaireResult",
    "preprocess_flex_sensor",
    "FlexSensorResult",
]
