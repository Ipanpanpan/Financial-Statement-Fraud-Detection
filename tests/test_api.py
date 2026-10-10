"""
Automated Test Suite for Financial Statement Fraud Detection API.
Validates health probes, inference correctness, business domain rules, and Comet monitoring.
"""

from typing import AsyncGenerator
import pytest
from httpx import ASGITransport, AsyncClient

from main import app, model_service, monitoring_service

SAMPLE_VALID_PAYLOAD = {
    "Receivables-Net(t-1)": 2450.0,
    "Cash-Gen(t)": -410.0,
    "SalesGenAdmExpen - R&Dexpense(t-1)": 1180.0,
    "Sales(t)": 8900.0,
    "AQI": 1.65,
    "DEPI": 1.22,
    "SGI": 1.45,
    "DSRI": 1.88,
    "TATA": 0.19,
    "GMI": 1.42,
    "SGAI": 1.18,
    "LVGI": 1.38,
}


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Ensure ML model artifacts and mock monitoring are initialized for test execution."""
    model_service.load_artifacts()
    monitoring_service.initialize()


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Yields asynchronous HTTP client configured for ASGI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        yield test_client


@pytest.mark.anyio
async def test_root_endpoint(client: AsyncClient):
    """Verify service information endpoint."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data


@pytest.mark.anyio
async def test_health_check_endpoint(client: AsyncClient):
    """Verify readiness and health probe returns healthy status and artifact states."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["scaler_loaded"] is True


@pytest.mark.anyio
async def test_successful_prediction(client: AsyncClient):
    """Verify inference pipeline produces probability, decision, latency, and M-Score."""
    response = await client.post("/predict", json=SAMPLE_VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert "is_fraudulent" in data
    assert isinstance(data["is_fraudulent"], bool)
    assert "inference_latency_ms" in data
    assert data["inference_latency_ms"] > 0
    assert "beneish_m_score" in data
    assert isinstance(data["beneish_m_score"], float)


@pytest.mark.anyio
async def test_business_constraint_negative_sales(client: AsyncClient):
    """Verify business rule constraint: Sales(t) must be non-negative."""
    invalid_payload = SAMPLE_VALID_PAYLOAD.copy()
    invalid_payload["Sales(t)"] = -500.0

    response = await client.post("/predict", json=invalid_payload)
    assert response.status_code == 400
    assert "Sales(t) must be non-negative" in response.json()["detail"]


@pytest.mark.anyio
async def test_business_constraint_negative_receivables(client: AsyncClient):
    """Verify business rule constraint: Receivables-Net(t-1) must be non-negative."""
    invalid_payload = SAMPLE_VALID_PAYLOAD.copy()
    invalid_payload["Receivables-Net(t-1)"] = -100.0

    response = await client.post("/predict", json=invalid_payload)
    assert response.status_code == 400
    assert "Net Receivables(t-1) cannot be negative" in response.json()["detail"]


@pytest.mark.anyio
async def test_business_constraint_negative_index(client: AsyncClient):
    """Verify business rule constraint: Key indices like DSRI cannot be negative."""
    invalid_payload = SAMPLE_VALID_PAYLOAD.copy()
    invalid_payload["DSRI"] = -1.5

    response = await client.post("/predict", json=invalid_payload)
    assert response.status_code == 400
    assert "cannot be negative" in response.json()["detail"]


@pytest.mark.anyio
async def test_missing_feature_validation(client: AsyncClient):
    """Verify Pydantic validation returns 422 if required features are missing."""
    incomplete_payload = {
        "Sales(t)": 5000.0,
        "Cash-Gen(t)": 100.0,
    }
    response = await client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422

