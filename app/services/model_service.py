"""
Machine Learning Model Service for Fraud Inference.
Encapsulates artifact management, business rule validations, feature scaling, and prediction.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd

logger = logging.getLogger("fraud_api.model_service")


class ModelArtifactError(Exception):
    """Raised when model or scaler artifacts cannot be loaded or used."""
    pass


class ModelService:
    """Production service for managing fraud detection inference artifacts and predictions."""

    def __init__(
        self,
        model_path: Path,
        scaler_path: Path,
        threshold: float = 0.60,
        expected_columns: Optional[List[str]] = None,
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.threshold = threshold
        self.expected_columns = expected_columns or [
            "Receivables-Net(t-1)",
            "Cash-Gen(t)",
            "SalesGenAdmExpen - R&Dexpense(t-1)",
            "Sales(t)",
            "AQI",
            "DEPI",
            "SGI",
            "DSRI",
            "TATA",
            "GMI",
            "SGAI",
            "LVGI",
        ]
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None

    def load_artifacts(self) -> None:
        """Loads model and scaler joblib artifacts into memory."""
        try:
            logger.info(f"Loading ML model artifact from: {self.model_path}")
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found at {self.model_path}")
            self.model = joblib.load(self.model_path)

            logger.info(f"Loading Scaler artifact from: {self.scaler_path}")
            if not self.scaler_path.exists():
                raise FileNotFoundError(f"Scaler file not found at {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)

            logger.info(
                f"Model artifacts loaded successfully. Decision threshold set to {self.threshold:.2f}."
            )
        except Exception as e:
            logger.critical(f"Failed to initialize model artifacts: {str(e)}", exc_info=True)
            raise ModelArtifactError(f"Startup failed: Unable to load model artifacts: {e}")

    def validate_business_constraints(self, features: Dict[str, float]) -> None:
        """
        Validates business domain constraints for corporate financial data.
        Raises ValueError if constraints are violated.
        """
        # Business Rule 1: Gross Sales revenue cannot be negative under standard accounting
        sales = features.get("Sales(t)", 0.0)
        if sales < 0:
            raise ValueError(
                f"Constraint violated: Reported Sales(t) must be non-negative (received {sales})."
            )

        # Business Rule 2: Trade accounts receivable cannot be negative
        receivables = features.get("Receivables-Net(t-1)", 0.0)
        if receivables < 0:
            raise ValueError(
                f"Constraint violated: Net Receivables(t-1) cannot be negative (received {receivables})."
            )

        # Business Rule 3: Key financial ratios (indices) must be non-negative
        non_negative_indices = ["DSRI", "AQI", "DEPI", "SGI", "GMI", "SGAI", "LVGI"]
        for index_name in non_negative_indices:
            val = features.get(index_name, 0.0)
            if val < 0:
                raise ValueError(
                    f"Constraint violated: Index '{index_name}' cannot be negative (received {val})."
                )

    def predict(self, features: Dict[str, float]) -> Tuple[float, bool, float]:
        """
        Executes model inference on validated financial feature dictionary.

        Args:
            features: Dictionary containing all 12 expected financial metrics.

        Returns:
            Tuple of (fraud_probability, is_fraudulent, latency_ms).
        """
        if self.model is None or self.scaler is None:
            raise ModelArtifactError("Model or scaler is not loaded into memory.")

        start_time = time.perf_counter()

        try:
            # Enforce exact column order required by StandardScaler
            input_df = pd.DataFrame([features], columns=self.expected_columns)

            # Apply StandardScaler transformation
            scaled_features = self.scaler.transform(input_df)

            # Generate prediction probability for Class 1 (Fraudulent)
            raw_probability = float(self.model.predict_proba(scaled_features)[0, 1])
            fraud_probability = round(raw_probability, 4)
            is_fraudulent = bool(fraud_probability >= self.threshold)

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

            return fraud_probability, is_fraudulent, latency_ms

        except Exception as e:
            logger.error(f"Inference execution failed: {str(e)}", exc_info=True)
            raise

    def is_healthy(self) -> bool:
        """Returns True if artifacts are loaded and ready for inference."""
        return self.model is not None and self.scaler is not None

