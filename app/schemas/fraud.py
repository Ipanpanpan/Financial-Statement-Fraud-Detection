"""
Data validation schemas for financial statement fraud detection.
"""

import math
from typing import Dict, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class FinancialStatementInput(BaseModel):
    """
    Validated input features representing corporate financial statement metrics.
    Supports both raw accounting figures and Beneish M-Score indices.
    """

    Receivables_Net_t_minus_1: float = Field(
        ...,
        alias="Receivables-Net(t-1)",
        description="Net trade receivables from prior fiscal period (t-1).",
    )
    Cash_Gen_t: float = Field(
        ...,
        alias="Cash-Gen(t)",
        description="Operating cash generated during the current fiscal period (t).",
    )
    SalesGenAdmExpen_R_Dexpense_t_minus_1: float = Field(
        ...,
        alias="SalesGenAdmExpen - R&Dexpense(t-1)",
        description="SG&A expenditures excluding R&D in prior period (t-1).",
    )
    Sales_t: float = Field(
        ...,
        alias="Sales(t)",
        description="Gross sales/revenue reported in current period (t).",
    )
    AQI: float = Field(
        ...,
        description="Asset Quality Index. Ratio of non-current assets other than PP&E.",
    )
    DEPI: float = Field(
        ...,
        description="Depreciation Index. Prior depreciation rate vs current rate.",
    )
    SGI: float = Field(
        ...,
        description="Sales Growth Index. Ratio of current sales to prior sales.",
    )
    DSRI: float = Field(
        ...,
        description="Days Sales in Receivables Index. Ratio of receivables to sales growth.",
    )
    TATA: float = Field(
        ...,
        description="Total Accruals to Total Assets. (Net Income - Operating Cash) / Assets.",
    )
    GMI: float = Field(
        ...,
        description="Gross Margin Index. Ratio of prior gross margin to current gross margin.",
    )
    SGAI: float = Field(
        ...,
        description="Sales, General and Administrative expenses Index.",
    )
    LVGI: float = Field(
        ...,
        description="Leverage Index. Ratio of total debt to total assets.",
    )

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
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
        },
    )

    @field_validator(
        "Receivables_Net_t_minus_1",
        "Cash_Gen_t",
        "SalesGenAdmExpen_R_Dexpense_t_minus_1",
        "Sales_t",
        "AQI",
        "DEPI",
        "SGI",
        "DSRI",
        "TATA",
        "GMI",
        "SGAI",
        "LVGI",
    )
    @classmethod
    def validate_finite_numbers(cls, value: float, info) -> float:
        """Ensure all input numerical features are finite floats (not NaN or Inf)."""
        if not math.isfinite(value):
            raise ValueError(
                f"Feature '{info.field_name}' must be a finite numerical value, got {value}."
            )
        return float(value)

    @property
    def beneish_m_score(self) -> float:
        """
        Calculates the standard 8-variable Beneish M-Score:
        M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
            + 0.115*DEPI - 0.172*SGAI + 4.037*TATA + 0.0327*LVGI
        """
        m_score = (
            -4.84
            + (0.920 * self.DSRI)
            + (0.528 * self.GMI)
            + (0.404 * self.AQI)
            + (0.892 * self.SGI)
            + (0.115 * self.DEPI)
            - (0.172 * self.SGAI)
            + (4.037 * self.TATA)
            + (0.0327 * self.LVGI)
        )
        return round(m_score, 4)

    def to_features_dict(self) -> Dict[str, float]:
        """Returns input dictionary mapped to exact column headers expected by StandardScaler."""
        return {
            "Receivables-Net(t-1)": self.Receivables_Net_t_minus_1,
            "Cash-Gen(t)": self.Cash_Gen_t,
            "SalesGenAdmExpen - R&Dexpense(t-1)": self.SalesGenAdmExpen_R_Dexpense_t_minus_1,
            "Sales(t)": self.Sales_t,
            "AQI": self.AQI,
            "DEPI": self.DEPI,
            "SGI": self.SGI,
            "DSRI": self.DSRI,
            "TATA": self.TATA,
            "GMI": self.GMI,
            "SGAI": self.SGAI,
            "LVGI": self.LVGI,
        }


class FraudPredictionResponse(BaseModel):
    """Production response schema for corporate fraud prediction."""

    fraud_probability: float = Field(
        ...,
        description="Predicted probability of earnings manipulation [0.0 - 1.0].",
    )
    is_fraudulent: bool = Field(
        ...,
        description="Binary decision flag based on calibrated optimal business threshold.",
    )
    inference_latency_ms: float = Field(
        ...,
        description="Inference execution latency in milliseconds.",
    )
    beneish_m_score: Optional[float] = Field(
        default=None,
        description="Classic 8-variable Beneish M-Score metric (cutoff: > -1.78 flags manipulation).",
    )


class HealthCheckResponse(BaseModel):
    """Structured health check response for container orchestration."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    scaler_loaded: bool
    monitoring_active: bool
    environment: str
    version: str
