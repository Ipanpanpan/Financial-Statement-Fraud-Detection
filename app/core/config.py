"""
Application Configuration and Environment Settings.
"""

import os
from pathlib import Path
from typing import Optional


class Settings:
    """Production configuration settings for the Fraud Inference API."""

    # Project Directories
    APP_DIR: Path = Path(__file__).resolve().parent.parent
    ROOT_DIR: Path = APP_DIR.parent

    # Service Metadata
    APP_TITLE: str = "Corporate Fraud Inference API"
    APP_DESCRIPTION: str = (
        "Production MLOps inference service with Comet ML model monitoring "
        "and Beneish M-Score risk evaluation."
    )
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")

    # Networking & Cloud Run
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8080"))

    # Model & Feature Artifacts
    MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", str(ROOT_DIR / "lr_fraud_model.joblib")))
    SCALER_PATH: Path = Path(os.getenv("SCALER_PATH", str(ROOT_DIR / "scaler.joblib")))
    FRAUD_THRESHOLD: float = float(os.getenv("FRAUD_THRESHOLD", "0.60"))

    # Expected feature columns in strict order required by StandardScaler
    FEATURE_COLUMNS = [
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

    # Comet ML Model Monitoring Settings
    COMET_API_KEY: Optional[str] = os.getenv("COMET_API_KEY")
    COMET_PROJECT_NAME: str = os.getenv("COMET_PROJECT_NAME", "financial-fraud-detection")
    COMET_WORKSPACE: Optional[str] = os.getenv("COMET_WORKSPACE")
    COMET_DISABLED: bool = (
        os.getenv("COMET_DISABLED", "false").lower() in ("true", "1", "yes")
        or not os.getenv("COMET_API_KEY")
    )

    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_JSON_FORMAT: bool = os.getenv("LOG_JSON_FORMAT", "false").lower() in (
        "true",
        "1",
        "yes",
    )


settings = Settings()

