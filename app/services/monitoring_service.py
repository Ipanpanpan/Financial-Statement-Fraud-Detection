"""
Comet ML Production Model Monitoring Service.
Streams real-time inference inputs (features) and outputs (fraud probabilities) to Comet ML.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("fraud_api.monitoring_service")

# Lazy import flag for comet_ml
try:
    import comet_ml
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False


class CometMonitoringService:
    """
    Manages production model monitoring via Comet ML.
    Enables tracking of inference metrics, feature drift, latency, and prediction distributions.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_name: str = "financial-fraud-detection",
        workspace: Optional[str] = None,
        disabled: bool = False,
    ):
        self.api_key = api_key
        self.project_name = project_name
        self.workspace = workspace
        self.disabled = disabled or (not api_key)
        self.experiment: Optional[Any] = None
        self.step: int = 0
        self.is_active: bool = False

    def initialize(self) -> None:
        """Initializes the Comet ML experiment session for production inference logging."""
        if self.disabled:
            logger.warning(
                "COMET_API_KEY is not configured or COMET_DISABLED is True. "
                "Comet ML monitoring is operating in disabled/mock mode."
            )
            self.is_active = False
            return

        if not COMET_AVAILABLE:
            logger.warning(
                "`comet_ml` library is not installed. Monitoring will run in mock mode."
            )
            self.is_active = False
            return

        try:
            logger.info(
                f"Initializing Comet ML experiment (project='{self.project_name}', "
                f"workspace='{self.workspace or 'default'}')..."
            )
            self.experiment = comet_ml.Experiment(
                api_key=self.api_key,
                project_name=self.project_name,
                workspace=self.workspace,
                auto_metric_logging=False,
                log_code=False,
                log_graph=False,
                parse_args=False,
            )

            # Register production deployment metadata
            self.experiment.add_tags(["production", "cloud-run", "l1-logistic-regression"])
            self.experiment.log_parameters({
                "model_type": "LogisticRegression_L1",
                "inference_engine": "FastAPI",
                "framework": "scikit-learn",
                "decision_threshold": 0.60,
                "input_features_count": 12,
            })

            self.is_active = True
            logger.info("Comet ML experiment initialized successfully for production monitoring.")

        except Exception as e:
            logger.error(
                f"Failed to initialize Comet ML experiment: {str(e)}. "
                "Inference will proceed without cloud monitoring.",
                exc_info=True,
            )
            self.is_active = False

    def log_inference(
        self,
        features: Dict[str, float],
        fraud_probability: float,
        is_fraudulent: bool,
        latency_ms: float,
        request_id: Optional[str] = None,
    ) -> None:
        """
        Logs inference inputs (features) and outputs (fraud probabilities) to Comet ML.

        Args:
            features: Dictionary containing the 12 input financial metrics.
            fraud_probability: Model predicted fraud probability [0.0 - 1.0].
            is_fraudulent: Binary threshold classification decision.
            latency_ms: Inference execution time in milliseconds.
            request_id: Optional unique request tracking identifier.
        """
        if not self.is_active or self.experiment is None:
            logger.debug(
                f"Mock monitoring log: prob={fraud_probability:.4f}, "
                f"fraud={is_fraudulent}, latency={latency_ms:.2f}ms"
            )
            return

        try:
            self.step += 1

            # Prepare output monitoring metrics
            metrics_payload: Dict[str, Any] = {
                "fraud_probability": float(fraud_probability),
                "is_fraudulent": int(is_fraudulent),
                "inference_latency_ms": float(latency_ms),
            }

            # Prepare feature input metrics for drift & distribution monitoring
            for feat_name, feat_val in features.items():
                clean_name = feat_name.translate(str.maketrans({" ": "_", "(": "", ")": "", "-": "_", "&": "and"}))
                sanitized_key = f"input_{clean_name}"
                metrics_payload[sanitized_key] = float(feat_val)

            # Stream metrics to Comet ML
            self.experiment.log_metrics(metrics_payload, step=self.step)

            logger.debug(
                f"Streamed inference metrics to Comet ML at step {self.step} "
                f"(probability={fraud_probability:.4f})"
            )

        except Exception as e:
            # Monitoring should NEVER crash primary inference pipeline
            logger.error(f"Error logging inference to Comet ML: {str(e)}", exc_info=True)

    def close(self) -> None:
        """Gracefully ends the Comet ML experiment on application shutdown."""
        if self.is_active and self.experiment is not None:
            try:
                logger.info("Ending Comet ML experiment session...")
                self.experiment.end()
                self.is_active = False
            except Exception as e:
                logger.error(f"Error closing Comet ML experiment: {str(e)}", exc_info=True)
