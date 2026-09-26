"""
Platt Scaling Recalibration — Build Guide §13

Maps raw GBDT logits to calibrated OA risk probabilities via a
2-parameter sigmoid:

    P(OA=1 | s) = 1 / (1 + exp(A * s + B))

Key design decisions:
  • Default A=1.0, B derived from LASI hill-state OA prevalence (~18%).
  • fit() uses Newton's method on the NLL with optional L2 reg.
  • confidence_interval() widens CIs when sensor modalities are missing,
    ensuring modular degradation rather than false confidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from src.config import CalibrationConfig, FusionConfig

logger = logging.getLogger(__name__)


@dataclass
class PlattScaler:
    """Two-parameter Platt scaling recalibrator.

    Attributes
    ----------
    A : float
        Slope parameter — stretches or compresses logit confidence.
    B : float
        Intercept — shifts the prior probability baseline.
    is_fitted : bool
        True after ``fit()`` has been called with local labeled data.
    """

    A: float = -1.0
    B: float = 1.516  # ln((1 - 0.18) / 0.18)
    is_fitted: bool = False

    _cal_config: CalibrationConfig = field(
        default_factory=CalibrationConfig, repr=False
    )
    _fusion_config: FusionConfig = field(
        default_factory=FusionConfig, repr=False
    )

    @classmethod
    def from_config(
        cls,
        cal_config: Optional[CalibrationConfig] = None,
        fusion_config: Optional[FusionConfig] = None,
    ) -> "PlattScaler":
        """Create a PlattScaler from configuration objects."""
        cal = cal_config or CalibrationConfig()
        fus = fusion_config or FusionConfig()
        return cls(A=cal.platt_A, B=cal.platt_B,
                   _cal_config=cal, _fusion_config=fus)

    # ------------------------------------------------------------------
    # Core transform
    # ------------------------------------------------------------------
    def predict_risk_score(self, raw_logits: np.ndarray) -> np.ndarray:
        """Map raw GBDT logits to calibrated probabilities [0, 1].

        Parameters
        ----------
        raw_logits : ndarray, shape (n,)
            Raw margin output from the fusion GBDT model.

        Returns
        -------
        ndarray, shape (n,)
            Calibrated OA risk probabilities.
        """
        logits = np.asarray(raw_logits, dtype=np.float64)
        z = np.clip(self.A * logits + self.B, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(z))

    # ------------------------------------------------------------------
    # Fitting (for when local NER/hill data becomes available)
    # ------------------------------------------------------------------
    def fit(
        self,
        raw_logits: np.ndarray,
        true_labels: np.ndarray,
    ) -> "PlattScaler":
        """Optimise A, B on local labeled data using Newton's method.

        Implements the Platt (1999) / Lin et al. (2007) algorithm with
        target probabilities adjusted to avoid overfitting on extreme
        counts, plus optional L2 regularisation.

        Parameters
        ----------
        raw_logits : ndarray, shape (n,)
            Raw GBDT logit outputs for each patient.
        true_labels : ndarray, shape (n,)
            Binary ground truth (1 = OA confirmed, 0 = no OA).

        Returns
        -------
        self
        """
        s = np.asarray(raw_logits, dtype=np.float64)
        y = np.asarray(true_labels, dtype=np.float64)
        n = len(s)

        if n < 10:
            logger.warning(
                "Only %d samples for Platt fit — results may be unreliable. "
                "Recommend ≥50 samples.", n
            )

        # Target probabilities (Lin et al., 2007 smoothing)
        n_pos = y.sum()
        n_neg = n - n_pos
        t = np.where(y > 0.5,
                     (n_pos + 1) / (n_pos + 2),
                     1.0 / (n_neg + 2))

        A, B = self.A, self.B
        lam = self._cal_config.l2_lambda

        # Maximum step size per iteration to prevent divergence on
        # highly-separable data (AUC > 0.99).
        max_step_A = 10.0
        max_step_B = 10.0

        for iteration in range(self._cal_config.max_iter):
            fApB = np.clip(A * s + B, -500.0, 500.0)
            # Numerically stable sigmoid
            p = np.where(
                fApB >= 0,
                np.exp(-fApB) / (1.0 + np.exp(-fApB)),
                1.0 / (1.0 + np.exp(fApB)),
            )

            # Gradient
            d1 = t - p
            g_A = -(d1 * s).sum() + lam * A
            g_B = -d1.sum() + lam * B

            # Hessian diagonal (approximate)
            pqr = p * (1.0 - p)
            h_AA = (pqr * s * s).sum() + lam
            h_BB = pqr.sum() + lam
            h_AB = (pqr * s).sum()

            # Newton step (2×2 system)
            det = h_AA * h_BB - h_AB * h_AB
            if abs(det) < 1e-15:
                logger.warning("Singular Hessian at iteration %d", iteration)
                break

            dA = -(h_BB * g_A - h_AB * g_B) / det
            dB = -(h_AA * g_B - h_AB * g_A) / det

            # Damped step — prevents explosion on separable data
            dA = np.clip(dA, -max_step_A, max_step_A)
            dB = np.clip(dB, -max_step_B, max_step_B)

            A += dA
            B += dB

            if abs(dA) < self._cal_config.tol and abs(dB) < self._cal_config.tol:
                logger.info(
                    "Platt fit converged at iteration %d: A=%.4f, B=%.4f",
                    iteration, A, B,
                )
                break
        else:
            logger.warning(
                "Platt fit did not converge in %d iterations (A=%.4f, B=%.4f)",
                self._cal_config.max_iter, A, B,
            )

        # Sanity check: degenerate parameters indicate data is too
        # separable for 2-parameter Platt scaling.  Fall back to
        # sklearn LogisticRegression on the logits.
        if abs(A) > 100.0 or abs(B) > 100.0:
            logger.warning(
                "Platt parameters degenerate (A=%.2f, B=%.2f). "
                "Falling back to sklearn LogisticRegression on logits.",
                A, B,
            )
            try:
                from sklearn.linear_model import LogisticRegression

                lr = LogisticRegression(
                    C=1.0, solver="lbfgs", max_iter=1000
                )
                lr.fit(s.reshape(-1, 1), y)
                # Map LR coef/intercept back to Platt form:
                # LR: P = 1/(1+exp(-(w*s+b)))
                # Platt: P = 1/(1+exp(A*s+B))
                # So A = -w, B = -b
                A = float(-lr.coef_[0, 0])
                B = float(-lr.intercept_[0])
                logger.info(
                    "Fallback LR calibration: A=%.4f, B=%.4f", A, B
                )
            except Exception as exc:
                logger.error("Fallback LR failed: %s. Using prior.", exc)
                A = -1.0
                B = 1.516

        self.A = A
        self.B = B
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Confidence interval
    # ------------------------------------------------------------------
    def confidence_interval(
        self,
        risk_score: float,
        missing_modality_count: int = 0,
    ) -> Tuple[float, float]:
        """Compute approximate 95% CI for a calibrated risk score.

        The CI base half-width is expanded by a fixed penalty per
        missing sensor modality, ensuring the system communicates
        increased uncertainty under modular degradation.

        Parameters
        ----------
        risk_score : float
            Calibrated probability in [0, 1].
        missing_modality_count : int
            Number of sensor modalities unavailable (0–5).

        Returns
        -------
        tuple[float, float]
            (lower, upper) bounds clamped to [0, 1].
        """
        half = (
            self._fusion_config.ci_base_half_width
            + self._fusion_config.ci_missing_modality_penalty * missing_modality_count
        )
        lower = max(0.0, risk_score - half)
        upper = min(1.0, risk_score + half)
        return (round(lower, 4), round(upper, 4))

    # ------------------------------------------------------------------
    # Risk tiering
    # ------------------------------------------------------------------
    def get_risk_category(self, risk_score: float) -> str:
        """Map a calibrated risk score to a clinical tier label."""
        cfg = self._fusion_config
        if risk_score < cfg.tier_low_upper:
            return "low"
        elif risk_score < cfg.tier_moderate_upper:
            return "moderate"
        elif risk_score < cfg.tier_high_upper:
            return "high"
        else:
            return "very_high"

    def get_clinical_action(self, risk_category: str) -> str:
        """Return clinical action text for a given risk tier."""
        return self._fusion_config.clinical_actions.get(
            risk_category, "Consult clinician for assessment."
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialise the fitted parameters."""
        return {"A": self.A, "B": self.B, "is_fitted": self.is_fitted}

    @classmethod
    def from_dict(cls, data: dict) -> "PlattScaler":
        """Restore from serialised parameters."""
        return cls(A=data["A"], B=data["B"], is_fitted=data.get("is_fitted", True))
