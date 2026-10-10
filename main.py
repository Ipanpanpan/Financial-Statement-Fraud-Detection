"""
Production-Grade Financial Fraud Inference FastAPI Application.
Equipped with Comet ML model monitoring, structured logging, and modular inference architecture.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging_config import configure_logging
from app.schemas.fraud import (
    FinancialStatementInput,
    FraudPredictionResponse,
    HealthCheckResponse,
)
from app.services.model_service import ModelArtifactError, ModelService
from app.services.monitoring_service import CometMonitoringService

# Configure structured logging for production
logger = configure_logging(
    log_level=settings.LOG_LEVEL,
    json_format=settings.LOG_JSON_FORMAT,
)

# Initialize service singletons
model_service = ModelService(
    model_path=settings.MODEL_PATH,
    scaler_path=settings.SCALER_PATH,
    threshold=settings.FRAUD_THRESHOLD,
    expected_columns=settings.FEATURE_COLUMNS,
)

monitoring_service = CometMonitoringService(
    api_key=settings.COMET_API_KEY,
    project_name=settings.COMET_PROJECT_NAME,
    workspace=settings.COMET_WORKSPACE,
    disabled=settings.COMET_DISABLED,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.
    Handles startup artifact initialization and graceful shutdown of monitoring streams.
    """
    logger.info(f"Starting {settings.APP_TITLE} (version: {settings.APP_VERSION})...")

    # 1. Load ML model and scaler artifacts into memory
    try:
        model_service.load_artifacts()
    except ModelArtifactError as err:
        logger.critical(f"Application failed to start due to missing artifacts: {err}")
        raise RuntimeError(f"Startup aborted: {err}")

    # 2. Initialize Comet ML monitoring experiment
    monitoring_service.initialize()

    logger.info("Application startup sequence completed successfully.")
    yield

    # 3. Graceful shutdown sequence
    logger.info("Initiating application shutdown sequence...")
    monitoring_service.close()
    logger.info("Application shutdown completed.")


# Initialize FastAPI application
app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Enable CORS for cross-origin frontend dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    tags=["Root"],
    summary="Service Information",
)
async def root():
    """Returns basic service information and API status."""
    return {
        "service": settings.APP_TITLE,
        "version": settings.APP_VERSION,
        "status": "online",
        "docs_url": "/docs",
        "health_check": "/health",
    }


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["Health"],
    summary="Health and Readiness Probe",
)
async def health_check() -> HealthCheckResponse:
    """
    Container health check endpoint for Cloud Run and Kubernetes readiness probes.
    Verifies that model artifacts are loaded in memory and monitoring is active.
    """
    is_ready = model_service.is_healthy()
    status_str = "healthy" if is_ready else "unhealthy"

    return HealthCheckResponse(
        status=status_str,
        model_loaded=model_service.model is not None,
        scaler_loaded=model_service.scaler is not None,
        monitoring_active=monitoring_service.is_active,
        environment=settings.ENVIRONMENT,
        version=settings.APP_VERSION,
    )


@app.post(
    "/predict",
    response_model=FraudPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
    summary="Evaluate Financial Statement for Corporate Fraud Risk",
)
async def predict_fraud(features: FinancialStatementInput) -> FraudPredictionResponse:
    """
    Production inference endpoint for financial statement fraud risk scoring.

    - Validates accounting business constraints and input schemas
    - Generates fraud probability and classification via L1-Logistic Regression
    - Logs features and predictions to Comet ML for production model monitoring
    - Computes 8-variable Beneish M-Score for econometric risk comparison
    """
    try:
        # 1. Extract feature dictionary
        features_dict = features.to_features_dict()

        # 2. Business domain validation rules
        model_service.validate_business_constraints(features_dict)

        logger.info(
            f"Received prediction request with Beneish M-Score: {features.beneish_m_score:.4f} "
            f"(Sales: ${features.Sales_t:,.2f}, Cash-Gen: ${features.Cash_Gen_t:,.2f})"
        )

        # 3. Execute model inference
        probability, is_fraud, latency_ms = model_service.predict(features_dict)

        # 4. Stream inference inputs and outputs to Comet ML for MLOps monitoring
        monitoring_service.log_inference(
            features=features_dict,
            fraud_probability=probability,
            is_fraudulent=is_fraud,
            latency_ms=latency_ms,
        )

        logger.info(
            f"Inference complete: probability={probability:.4f}, "
            f"fraudulent={is_fraud}, latency={latency_ms:.2f}ms"
        )

        # 5. Return standardized response schema
        return FraudPredictionResponse(
            fraud_probability=probability,
            is_fraudulent=is_fraud,
            inference_latency_ms=latency_ms,
            beneish_m_score=features.beneish_m_score,
        )

    except ValueError as ve:
        logger.warning(f"Business constraint validation failure: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except ModelArtifactError as me:
        logger.critical(f"Model service artifact error: {str(me)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference service unavailable: Model artifacts uninitialized.",
        )
    except Exception as e:
        logger.error(f"Inference failure during prediction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during fraud prediction.",
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )
