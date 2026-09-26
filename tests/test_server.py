"""
Tests for Kneeva FastAPI Server and Client App Integration.
"""

import pytest
from fastapi.testclient import TestClient

from src.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Kneeva" in data["service"]


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["feature_count"] == 36


def test_triage_endpoint_lean_hill_patient(client):
    """Test full triage payload from a client app for a lean hill carrier."""
    payload = {
        "patient_id": "NER_HILL_001",
        "patient_metadata": {
            "age": 55,
            "sex": "female",
            "height_cm": 155,
            "weight_kg": 48,
        },
        "questionnaire": {
            "carried_load_kg": 20.0,
            "daily_incline_hours": 3.0,
            "squatting_difficulty": 3,
            "previous_injury": 0,
            "activity_level": 3,
        },
        "sensor_features": {
            "rom_active_flexion_deg": 108.0,
            "rom_active_extension_deficit_deg": 8.5,
            "crepitus_event_count": 9.0,
            "crepitus_total_energy": 42.0,
            "cocontraction_cci_walking_mean": 0.44,
            "strength_ext_peak_n": 195.0,
            "gait_step_time_asymmetry": 0.11,
            "gait_speed_ms": 0.92,
        },
    }

    response = client.post("/api/v1/triage", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["patient_id"] == "NER_HILL_001"
    assert 0.0 <= data["oa_risk_score"] <= 1.0
    assert data["oa_risk_category"] in ["low", "moderate", "high", "very_high"]
    assert len(data["confidence_interval"]) == 2
    assert "effective_bmi" in data
    # Effective BMI should be significantly higher than raw BMI (48 / 1.55^2 ≈ 20.0)
    assert data["effective_bmi"] > 25.0
    assert "clinical_explanation" in data
    assert "clinical_action" in data
