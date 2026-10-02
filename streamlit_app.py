"""
Streamlit SaaS Dashboard for Corporate Financial Statement Fraud Detection.
Integrates FastAPI Beneish M-Score / L1 Logistic Regression Inference API
with Google Gemini Agentic Forensic Analysis.
"""

import io
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import urllib3

# Suppress InsecureRequestWarning when testing against endpoints with self-signed SSL certs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

# Optional Google Gemini import with fallback
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ==============================================================================
# 1. CONSTANTS & METRIC SPECIFICATIONS
# ==============================================================================

DEFAULT_PRODUCTION_API = "https://api.gloryatk.com"
DEFAULT_LOCAL_API = "http://localhost:8000"
FRAUD_THRESHOLD = 0.60
BENEISH_THRESHOLD = -1.78

EXPECTED_COLUMNS = [
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

# Aliases mapping to support minor column naming variations in user files
COLUMN_ALIASES: Dict[str, List[str]] = {
    "Receivables-Net(t-1)": [
        "receivables-net(t-1)", "receivables_net_t_minus_1", "receivables_net(t-1)",
        "receivables_net_t_1", "receivables(t-1)", "receivables_t_minus_1"
    ],
    "Cash-Gen(t)": [
        "cash-gen(t)", "cash_gen_t", "cash_gen(t)", "cash_generation_t",
        "cash_gen", "cashgen_t"
    ],
    "SalesGenAdmExpen - R&Dexpense(t-1)": [
        "salesgenadmexpen - r&dexpense(t-1)", "salesgenadmexpen_r_dexpense_t_minus_1",
        "salesgenadmexpen_r_dexpense(t-1)", "sg&a_minus_rd(t-1)", "sga_rd_t_1"
    ],
    "Sales(t)": [
        "sales(t)", "sales_t", "sales", "revenue(t)", "revenue_t", "revenue"
    ],
    "AQI": ["aqi", "asset_quality_index"],
    "DEPI": ["depi", "depreciation_index"],
    "SGI": ["sgi", "sales_growth_index"],
    "DSRI": ["dsri", "days_sales_in_receivables_index"],
    "TATA": ["tata", "total_accruals_to_total_assets"],
    "GMI": ["gmi", "gross_margin_index"],
    "SGAI": ["sgai", "sga_expense_index", "sg&a_index"],
    "LVGI": ["lvgi", "leverage_index"],
}

METRIC_METADATA: Dict[str, Dict[str, Any]] = {
    "DSRI": {
        "name": "Days Sales in Receivables Index",
        "category": "Beneish Metric",
        "benchmark": "≤ 1.03",
        "red_flag": "> 1.20",
        "threshold": 1.20,
        "direction": "higher_is_worse",
        "desc": "Ratio of days sales in receivables. An abnormal surge (>1.2) suggests aggressive revenue recognition, channel stuffing, or fictitious sales.",
    },
    "GMI": {
        "name": "Gross Margin Index",
        "category": "Beneish Metric",
        "benchmark": "≤ 1.01",
        "red_flag": "> 1.20",
        "threshold": 1.20,
        "direction": "higher_is_worse",
        "desc": "Ratio of prior year gross margin to current year. Values >1 denote deteriorating profit margins, amplifying incentive to manipulate.",
    },
    "AQI": {
        "name": "Asset Quality Index",
        "category": "Beneish Metric",
        "benchmark": "≤ 1.04",
        "red_flag": "> 1.25",
        "threshold": 1.25,
        "direction": "higher_is_worse",
        "desc": "Ratio of non-current assets other than PP&E. A spike (>1.25) suggests capitalizing operating expenses onto the balance sheet.",
    },
    "SGI": {
        "name": "Sales Growth Index",
        "category": "Beneish Metric",
        "benchmark": "≤ 1.13",
        "red_flag": "> 1.30",
        "threshold": 1.30,
        "direction": "higher_is_worse",
        "desc": "Current sales to prior sales. Rapid growth companies face extreme pressure to preserve earnings growth trajectory.",
    },
    "DEPI": {
        "name": "Depreciation Index",
        "category": "Beneish Metric",
        "benchmark": "≤ 1.00",
        "red_flag": "> 1.10",
        "threshold": 1.10,
        "direction": "higher_is_worse",
        "desc": "Prior depreciation rate to current rate. >1 denotes slower depreciation, potentially extending useful lives to inflate net income.",
    },
    "SGAI": {
        "name": "SG&A Expense Index",
        "category": "Beneish Metric",
        "benchmark": "≤ 1.04",
        "red_flag": "> 1.20",
        "threshold": 1.20,
        "direction": "higher_is_worse",
        "desc": "Ratio of SG&A expenses to sales. Elevated ratios denote declining operating efficiency and fixed overhead burdens.",
    },
    "LVGI": {
        "name": "Leverage Index",
        "category": "Beneish Metric",
        "benchmark": "≤ 1.04",
        "red_flag": "> 1.25",
        "threshold": 1.25,
        "direction": "higher_is_worse",
        "desc": "Total debt to total assets. Rising leverage signals tightening debt covenants and financial distress.",
    },
    "TATA": {
        "name": "Total Accruals to Total Assets",
        "category": "Beneish Metric",
        "benchmark": "≤ 0.02",
        "red_flag": "> 0.05",
        "threshold": 0.05,
        "direction": "higher_is_worse",
        "desc": "(Net Income - Operating Cash Flow) / Total Assets. Positive values indicate earnings are driven by non-cash accounting accruals.",
    },
    "Receivables-Net(t-1)": {
        "name": "Net Receivables (t-1)",
        "category": "Raw Financials ($K)",
        "benchmark": "Baseline",
        "red_flag": "N/A",
        "threshold": None,
        "direction": "neutral",
        "desc": "Net trade accounts receivable carried from prior fiscal period.",
    },
    "Cash-Gen(t)": {
        "name": "Cash Generation (t)",
        "category": "Raw Financials ($K)",
        "benchmark": "> 0",
        "red_flag": "< 0",
        "threshold": 0.0,
        "direction": "lower_is_worse",
        "desc": "Operating cash generated during the current period. Negative values indicate cash burn despite reported profits.",
    },
    "SalesGenAdmExpen - R&Dexpense(t-1)": {
        "name": "SG&A minus R&D (t-1)",
        "category": "Raw Financials ($K)",
        "benchmark": "Baseline",
        "red_flag": "N/A",
        "threshold": None,
        "direction": "neutral",
        "desc": "Baseline prior period selling, general & administrative overhead excluding R&D.",
    },
    "Sales(t)": {
        "name": "Total Sales Revenue (t)",
        "category": "Raw Financials ($K)",
        "benchmark": "Baseline",
        "red_flag": "N/A",
        "threshold": None,
        "direction": "neutral",
        "desc": "Current period top-line revenue reported on the income statement.",
    },
}

# Fictitious Demo Dataset for instant testing
DEMO_DATA: List[Dict[str, Any]] = [
    {
        "Company": "Apex Horizon Technologies (High Risk Flag)",
        "DSRI": 1.88,
        "GMI": 1.42,
        "AQI": 1.65,
        "SGI": 1.45,
        "DEPI": 1.22,
        "SGAI": 1.18,
        "LVGI": 1.38,
        "TATA": 0.19,
        "Receivables-Net(t-1)": 2450.0,
        "Cash-Gen(t)": -410.0,
        "SalesGenAdmExpen - R&Dexpense(t-1)": 1180.0,
        "Sales(t)": 8900.0,
    },
    {
        "Company": "Vanguard Industrial Holdings (Clean Enterprise)",
        "DSRI": 0.98,
        "GMI": 0.94,
        "AQI": 0.91,
        "SGI": 1.04,
        "DEPI": 0.99,
        "SGAI": 0.96,
        "LVGI": 0.95,
        "TATA": -0.04,
        "Receivables-Net(t-1)": 1100.0,
        "Cash-Gen(t)": 2350.0,
        "SalesGenAdmExpen - R&Dexpense(t-1)": 840.0,
        "Sales(t)": 9850.0,
    },
    {
        "Company": "NovaBio Pharma Systems (Borderline Caution)",
        "DSRI": 1.24,
        "GMI": 1.15,
        "AQI": 1.21,
        "SGI": 1.32,
        "DEPI": 1.05,
        "SGAI": 1.25,
        "LVGI": 1.29,
        "TATA": 0.06,
        "Receivables-Net(t-1)": 820.0,
        "Cash-Gen(t)": 95.0,
        "SalesGenAdmExpen - R&Dexpense(t-1)": 690.0,
        "Sales(t)": 3600.0,
    },
]


# ==============================================================================
# 2. CALCULATION & INFERENCE ENGINE HELPERS
# ==============================================================================

def calculate_beneish_m_score(row: Dict[str, Any]) -> float:
    """
    Computes the 8-variable Beneish M-Score:
    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
        + 0.115*DEPI - 0.172*SGAI + 4.037*TATA + 0.0327*LVGI
    Standard threshold: M > -1.78 indicates high likelihood of manipulation.
    """
    dsri = float(row.get("DSRI", 1.0))
    gmi = float(row.get("GMI", 1.0))
    aqi = float(row.get("AQI", 1.0))
    sgi = float(row.get("SGI", 1.0))
    depi = float(row.get("DEPI", 1.0))
    sgai = float(row.get("SGAI", 1.0))
    tata = float(row.get("TATA", 0.0))
    lvgi = float(row.get("LVGI", 1.0))

    m_score = (
        -4.84
        + (0.920 * dsri)
        + (0.528 * gmi)
        + (0.404 * aqi)
        + (0.892 * sgi)
        + (0.115 * depi)
        - (0.172 * sgai)
        + (4.037 * tata)
        + (0.0327 * lvgi)
    )
    return round(m_score, 4)


def map_and_validate_columns(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Validates that all required columns are present in df, applying alias mapping
    for minor differences in casing, underscores, or formatting.
    """
    df_clean = df.copy()
    # Normalize headers for matching
    existing_cols_map = {str(col).strip(): col for col in df_clean.columns}
    normalized_cols_lower = {str(col).strip().lower(): col for col in df_clean.columns}

    missing_cols: List[str] = []
    renaming: Dict[str, str] = {}

    for req in EXPECTED_COLUMNS:
        if req in existing_cols_map:
            continue
        req_lower = req.lower()
        if req_lower in normalized_cols_lower:
            renaming[normalized_cols_lower[req_lower]] = req
            continue

        # Check aliases
        matched = False
        for alias in COLUMN_ALIASES.get(req, []):
            if alias.lower() in normalized_cols_lower:
                renaming[normalized_cols_lower[alias.lower()]] = req
                matched = True
                break

        if not matched:
            missing_cols.append(req)

    if renaming:
        df_clean = df_clean.rename(columns=renaming)

    if missing_cols:
        return None, missing_cols

    return df_clean, []


def call_fastapi_predict(
    api_base_url: str,
    payload: Dict[str, float],
    bypass_ssl: bool = True,
    timeout: int = 10,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Sends the feature dictionary to the FastAPI /predict endpoint.
    Handles network errors, HTTP errors, SSL exceptions, and timeouts.
    """
    url = api_base_url.rstrip("/") + "/predict"
    try:
        response = requests.post(
            url,
            json=payload,
            verify=not bypass_ssl,
            timeout=timeout,
            headers={"Content-Type": "application/json", "accept": "application/json"},
        )
        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API HTTP Error {response.status_code}: {response.text}"
    except requests.exceptions.SSLError as ssl_err:
        return (
            None,
            f"SSL Certificate Error: {ssl_err}. Tip: Enable 'Bypass SSL Verification' in sidebar to connect with self-signed VPS certificates.",
        )
    except requests.exceptions.ConnectionError:
        return (
            None,
            f"Connection refused at `{url}`. Verify that the FastAPI backend is running and reachable.",
        )
    except requests.exceptions.Timeout:
        return None, f"Request to `{url}` timed out after {timeout} seconds."
    except Exception as exc:
        return None, f"Unexpected error during API call: {str(exc)}"


def call_fastapi_health(
    api_base_url: str, bypass_ssl: bool = True, timeout: int = 5
) -> Tuple[bool, str, float]:
    """Pings /health endpoint to check server availability."""
    url = api_base_url.rstrip("/") + "/health"
    t0 = time.perf_counter()
    try:
        resp = requests.get(url, verify=not bypass_ssl, timeout=timeout)
        latency = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            return True, "Healthy", round(latency, 1)
        return False, f"Status Code {resp.status_code}", round(latency, 1)
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        return False, str(e), round(latency, 1)


# ==============================================================================
# 3. AGENTIC AI (GOOGLE GEMINI) INTEGRATION
# ==============================================================================

def generate_gemini_forensic_analysis(
    api_key: str,
    model_name: str,
    company_name: str,
    ml_probability: float,
    is_fraudulent: bool,
    m_score: float,
    metrics_dict: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """
    Calls Google Gemini using the google-generativeai library to synthesize
    the quantitative outputs into an authoritative forensic audit report.
    """
    if not GEMINI_AVAILABLE:
        return (
            None,
            "The `google-generativeai` library is not installed. Please run `pip install google-generativeai`.",
        )

    if not api_key or not api_key.strip():
        return (
            None,
            "Google Gemini API Key is required. Please provide it in the sidebar settings or set the `GEMINI_API_KEY` environment variable.",
        )

    try:
        genai.configure(api_key=api_key.strip())

        system_instruction = (
            "You are an elite Senior Forensic Accountant and Certified Fraud Examiner (CFE). "
            "You specialize in evaluating financial statement manipulation risks using the Beneish M-Score model "
            "and machine learning classification systems. "
            "Your objective is to provide a concise, sharp, and authoritative forensic commentary "
            "that explains WHY a company was flagged or cleared. "
            "Focus on accounting mechanics (e.g. channel stuffing, bogus receivables, unrecorded liabilities, "
            "capitalization of operating expenses, non-cash accrual divergence from operating cash flows) "
            "and suggest 2-3 specific substantive audit testing procedures."
        )

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
        )

        # Identify anomalous red flag ratios
        anomalies = []
        if metrics_dict.get("DSRI", 0) > 1.20:
            anomalies.append(f"DSRI = {metrics_dict['DSRI']:.2f} (Receivables surging significantly faster than sales)")
        if metrics_dict.get("AQI", 0) > 1.25:
            anomalies.append(f"AQI = {metrics_dict['AQI']:.2f} (Suspicious increase in capitalized/deferred assets)")
        if metrics_dict.get("GMI", 0) > 1.20:
            anomalies.append(f"GMI = {metrics_dict['GMI']:.2f} (Deteriorating gross profit margin)")
        if metrics_dict.get("SGI", 0) > 1.30:
            anomalies.append(f"SGI = {metrics_dict['SGI']:.2f} (High top-line revenue growth)")
        if metrics_dict.get("DEPI", 0) > 1.10:
            anomalies.append(f"DEPI = {metrics_dict['DEPI']:.2f} (Depreciation rate decelerating, extending useful lives)")
        if metrics_dict.get("SGAI", 0) > 1.20:
            anomalies.append(f"SGAI = {metrics_dict['SGAI']:.2f} (SG&A expenses rising faster than sales)")
        if metrics_dict.get("LVGI", 0) > 1.25:
            anomalies.append(f"LVGI = {metrics_dict['LVGI']:.2f} (Leverage index increasing)")
        if metrics_dict.get("TATA", 0) > 0.05:
            anomalies.append(f"TATA = {metrics_dict['TATA']:.4f} (Net income substantially higher than operating cash flow)")
        if metrics_dict.get("Cash-Gen(t)", 0) < 0:
            anomalies.append(f"Cash-Gen(t) = ${metrics_dict['Cash-Gen(t)']:,.2f} (Negative operating cash generation)")

        prompt = f"""
### FINANCIAL STATEMENT FORENSIC AUDIT REQUEST

Company Entity: {company_name}
ML Fraud Risk Probability: {ml_probability * 100:.2f}% (Threshold: {FRAUD_THRESHOLD * 100:.0f}%)
ML Classification Verdict: {"FLAGGED FOR FRAUD REVIEW" if is_fraudulent else "LOW RISK / NORMAL"}
Beneish M-Score: {m_score:.4f} (Benchmark Threshold: {BENEISH_THRESHOLD:.2f} — {"RED FLAG: High Manipulation Probability" if m_score > BENEISH_THRESHOLD else "NORMAL: Low Manipulation Probability"})

### Input Metrics Breakdown:
- Days Sales in Receivables Index (DSRI): {metrics_dict.get('DSRI', 'N/A')}
- Gross Margin Index (GMI): {metrics_dict.get('GMI', 'N/A')}
- Asset Quality Index (AQI): {metrics_dict.get('AQI', 'N/A')}
- Sales Growth Index (SGI): {metrics_dict.get('SGI', 'N/A')}
- Depreciation Index (DEPI): {metrics_dict.get('DEPI', 'N/A')}
- SG&A Expense Index (SGAI): {metrics_dict.get('SGAI', 'N/A')}
- Leverage Index (LVGI): {metrics_dict.get('LVGI', 'N/A')}
- Total Accruals to Total Assets (TATA): {metrics_dict.get('TATA', 'N/A')}
- Cash Generation (t): ${metrics_dict.get('Cash-Gen(t)', 0):,.2f}
- Receivables Net (t-1): ${metrics_dict.get('Receivables-Net(t-1)', 0):,.2f}
- Sales (t): ${metrics_dict.get('Sales(t)', 0):,.2f}
- SG&A minus R&D (t-1): ${metrics_dict.get('SalesGenAdmExpen - R&Dexpense(t-1)', 0):,.2f}

Detected Ratio Anomalies:
{chr(10).join(['- ' + a for a in anomalies]) if anomalies else '- None detected within critical breach thresholds.'}

### Instructions:
Provide a 3-part forensic audit memo formatted cleanly in Markdown:
1. **Executive Verdict & Summary**: 2 sentences summarizing the risk classification and composite score.
2. **Forensic Accounting Analysis**: Dissect the specific ratios that triggered the risk score. Explain the underlying accounting vulnerability (e.g. premature revenue recognition, deferred expense capitalization, or cash flow divergence).
3. **Targeted Substantive Audit Procedures**: 2-3 concrete forensic actions the external audit team or regulatory investigator must perform to corroborate or dismiss the red flags.
"""
        response = model.generate_content(prompt)
        return response.text, None

    except Exception as exc:
        return None, f"Gemini API Error: {str(exc)}"


# ==============================================================================
# 4. PLOTLY VISUALIZATION BUILDERS
# ==============================================================================

def build_probability_gauge(probability: float, threshold: float = 0.60) -> go.Figure:
    """Creates an enterprise-grade Plotly gauge chart for the fraud probability score."""
    pct_val = round(probability * 100, 2)
    
    # Dynamic bar color based on risk level
    if probability >= threshold:
        bar_color = "#E53935"  # Red
    elif probability >= 0.30:
        bar_color = "#FB8C00"  # Amber
    else:
        bar_color = "#43A047"  # Green

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct_val,
            number={"suffix": "%", "font": {"size": 42, "family": "Inter, Roboto, sans-serif"}},
            title={
                "text": "<b>Fraud Probability Score</b><br><span style='font-size:12px;color:gray'>L1-Penalized Logistic Regression Model</span>",
                "font": {"size": 18},
            },
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9E9E9E"},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "rgba(240, 242, 246, 0.5)",
                "borderwidth": 1,
                "bordercolor": "#B0BEC5",
                "steps": [
                    {"range": [0, 30], "color": "rgba(76, 175, 80, 0.20)"},
                    {"range": [30, 60], "color": "rgba(255, 193, 7, 0.20)"},
                    {"range": [60, 100], "color": "rgba(244, 67, 54, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#D32F2F", "width": 4},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )

    fig.update_layout(
        height=280,
        margin={"l": 25, "r": 25, "t": 50, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Roboto, sans-serif"},
    )
    return fig


def build_beneish_gauge(m_score: float, threshold: float = -1.78) -> go.Figure:
    """Builds a gauge chart for the Beneish M-Score centered around the -1.78 cutoff."""
    # Beneish typically spans from -5.0 to +2.0
    clamped_val = max(min(m_score, 2.0), -5.0)
    is_manipulator = m_score > threshold
    bar_color = "#E53935" if is_manipulator else "#43A047"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=m_score,
            number={"valueformat": ".2f", "font": {"size": 42, "family": "Inter, Roboto, sans-serif"}},
            title={
                "text": f"<b>Beneish M-Score: {m_score:.2f}</b><br><span style='font-size:12px;color:gray'>Cutoff: > {threshold:.2f} flags Manipulation</span>",
                "font": {"size": 18},
            },
            gauge={
                "axis": {"range": [-5.0, 2.0], "tickwidth": 1, "tickcolor": "#9E9E9E"},
                "bar": {"color": bar_color, "thickness": 0.3},
                "bgcolor": "rgba(240, 242, 246, 0.5)",
                "borderwidth": 1,
                "bordercolor": "#B0BEC5",
                "steps": [
                    {"range": [-5.0, threshold], "color": "rgba(76, 175, 80, 0.20)"},
                    {"range": [threshold, 2.0], "color": "rgba(244, 67, 54, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#D32F2F", "width": 4},
                    "thickness": 0.8,
                    "value": threshold,
                },
            },
        )
    )

    fig.update_layout(
        height=280,
        margin={"l": 25, "r": 25, "t": 50, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Roboto, sans-serif"},
    )
    return fig


def build_ratio_comparison_chart(metrics_dict: Dict[str, Any]) -> go.Figure:
    """Horizontal bar chart comparing the 8 Beneish indices against benchmark baselines."""
    beneish_keys = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]
    labels = [f"{k} ({METRIC_METADATA[k]['name']})" for k in beneish_keys]
    values = [float(metrics_dict.get(k, 1.0)) for k in beneish_keys]
    
    # Assign bar colors according to whether metric breaches red flag threshold
    colors = []
    for k, val in zip(beneish_keys, values):
        meta = METRIC_METADATA[k]
        thresh = meta["threshold"]
        if thresh is not None and val > thresh:
            colors.append("#E53935")  # Red Flag
        elif val > 1.05 and k != "TATA":
            colors.append("#FB8C00")  # Moderate
        else:
            colors.append("#1E88E5")  # Normal Blue

    fig = go.Figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#37474F", width=1)),
            text=[f"{v:.3f}" for v in values],
            textposition="auto",
        )
    )

    # Add vertical line for neutral baseline = 1.0
    fig.add_vline(x=1.0, line_width=1.5, line_dash="dash", line_color="#78909C", annotation_text="Baseline (1.0)")

    fig.update_layout(
        title="<b>Beneish 8-Index Breakdown vs Baseline</b>",
        xaxis_title="Index Value",
        height=380,
        margin={"l": 20, "r": 20, "t": 40, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(240, 242, 246, 0.2)",
        yaxis=dict(autorange="reversed"),
    )
    return fig


# ==============================================================================
# 5. STREAMLIT APPLICATION CORE
# ==============================================================================

def main():
    st.set_page_config(
        page_title="FraudLens AI | Financial Forensic SaaS",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom styling
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E293B;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748B;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: #FFFFFF;
            border-radius: 10px;
            padding: 1.1rem;
            border: 1px solid #E2E8F0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .status-badge-red {
            background-color: #FEE2E2;
            color: #991B1B;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        .status-badge-green {
            background-color: #DCFCE7;
            color: #166534;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        .status-badge-yellow {
            background-color: #FEF9C3;
            color: #854D0E;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            display: inline-block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Title & Subtitle Banner
    col_t1, col_t2 = st.columns([4, 1])
    with col_t1:
        st.markdown('<div class="main-header">🛡️ FraudLens AI | Financial Statement Forensic Dashboard</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sub-header">Dual-Engine Forensic Auditing: L1-Penalized Logistic Regression Inference + Beneish 8-Variable M-Score + Google Gemini AI Commentary</div>',
            unsafe_allow_html=True,
        )
    with col_t2:
        st.caption("Engine Version: **v1.2.0-SaaS**")
        st.caption("Threshold: **P ≥ 0.60 | M > -1.78**")

    # ==========================================================================
    # SIDEBAR CONFIGURATION
    # ==========================================================================
    with st.sidebar:
        st.header("⚙️ System Configuration")

        # 1. API Connection
        st.subheader("1. Backend API Endpoint")
        api_preset = st.selectbox(
            "Target Environment",
            ["Live Production (api.gloryatk.com)", "Local FastAPI (localhost:8000)", "Custom Endpoint"],
            index=0,
            help="Select the inference server endpoint.",
        )

        if api_preset == "Live Production (api.gloryatk.com)":
            default_url = DEFAULT_PRODUCTION_API
        elif api_preset == "Local FastAPI (localhost:8000)":
            default_url = DEFAULT_LOCAL_API
        else:
            default_url = ""

        api_base_url = st.text_input(
            "API Base URL",
            value=default_url,
            placeholder="https://api.gloryatk.com",
            help="Base URL of the Corporate Fraud Inference API.",
        )

        bypass_ssl = st.checkbox(
            "Bypass SSL Verification",
            value=True,
            help="Recommended for production VPS using temporary self-signed SSL certificate.",
        )

        # Quick Health Check Button
        if st.button("🔌 Ping API Health", use_container_width=True):
            if not api_base_url:
                st.warning("Please provide a valid API Base URL.")
            else:
                with st.spinner("Pinging API health endpoint..."):
                    healthy, msg, lat = call_fastapi_health(api_base_url, bypass_ssl=bypass_ssl)
                    if healthy:
                        st.success(f"🟢 API Online ({lat} ms) — {msg}")
                    else:
                        st.error(f"🔴 Connection Failed ({lat} ms): {msg}")

        st.divider()

        # 2. Gemini AI Integration Settings
        st.subheader("2. Agentic AI (Google Gemini)")
        env_gemini_key = os.getenv("GEMINI_API_KEY", "")
        gemini_api_key = st.text_input(
            "Gemini API Key",
            value=env_gemini_key,
            type="password",
            placeholder="AIzaSy...",
            help="Enter your Google AI Studio API key for natural language forensic insights.",
        )

        gemini_model = st.selectbox(
            "Gemini Model",
            ["gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.8-flash"],
            index=0,
            help="gemini-3.6-flash delivers fast, low-latency, and cost-efficient analysis.",
        )

        auto_run_ai = st.checkbox("Auto-generate AI Commentary", value=True)

        st.divider()

        # 3. Sample Data Presets
        st.subheader("3. Demo Datasets & Templates")
        if st.button("📥 Load Fictitious Demo Companies", use_container_width=True):
            st.session_state["df_input"] = pd.DataFrame(DEMO_DATA)
            st.session_state["data_source"] = "demo"
            st.success("Loaded 3 demo company records!")

        # Download Template CSV
        template_df = pd.DataFrame(DEMO_DATA)
        csv_buffer = io.StringIO()
        template_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📄 Download CSV Template",
            data=csv_buffer.getvalue(),
            file_name="financial_metrics_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Download Template Excel
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            template_df.to_excel(writer, index=False, sheet_name="FinancialMetrics")
        st.download_button(
            label="📊 Download Excel Template (.xlsx)",
            data=excel_buffer.getvalue(),
            file_name="financial_metrics_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # ==========================================================================
    # FILE UPLOAD & INGESTION WIDGET
    # ==========================================================================
    st.markdown("### 📁 1. Financial Statement Ingestion")
    
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel (.xlsx / .xls) file containing company financial metrics:",
        type=["csv", "xlsx", "xls"],
        help="Upload files formatted with the Beneish ratios and raw metrics. Download templates from sidebar if needed.",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
            st.session_state["df_input"] = df_raw
            st.session_state["data_source"] = f"upload: {uploaded_file.name}"
        except Exception as e:
            st.error(f"❌ Failed to parse uploaded file: {str(e)}")
            return

    # Check if data exists in session state
    if "df_input" not in st.session_state or st.session_state["df_input"] is None:
        st.info("👋 Upload a CSV/Excel file or click **'Load Fictitious Demo Companies'** in the sidebar to begin analysis.")
        return

    df_active = st.session_state["df_input"]

    # Validate Schema
    df_validated, missing_cols = map_and_validate_columns(df_active)
    if df_validated is None:
        st.error(f"❌ **Schema Validation Error**: Missing required columns: `{', '.join(missing_cols)}`")
        st.info("The model requires these 12 columns:\n" + ", ".join([f"`{c}`" for c in EXPECTED_COLUMNS]))
        return

    # Company selection widget if multiple rows exist
    total_records = len(df_validated)
    company_identifier_col = None
    for id_col in ["Company", "company", "Company_Name", "Ticker", "Entity", "Firm"]:
        if id_col in df_validated.columns:
            company_identifier_col = id_col
            break

    if total_records > 1:
        c_sel_col1, c_sel_col2 = st.columns([3, 1])
        with c_sel_col1:
            options = [
                f"Row {idx + 1}: {df_validated.iloc[idx][company_identifier_col] if company_identifier_col else f'Record {idx + 1}'}"
                for idx in range(total_records)
            ]
            selected_idx = st.selectbox("Select Company / Statement Record to Analyze:", range(total_records), format_func=lambda i: options[i])
        with c_sel_col2:
            st.metric("Total Records Ingestion", f"{total_records} Companies")
    else:
        selected_idx = 0

    selected_row = df_validated.iloc[selected_idx]
    company_name = str(selected_row[company_identifier_col]) if company_identifier_col else f"Record #{selected_idx + 1}"

    # Extract single company payload
    try:
        payload = {col: float(selected_row[col]) for col in EXPECTED_COLUMNS}
    except ValueError as ve:
        st.error(f"❌ Non-numeric value detected in required columns for {company_name}: {ve}")
        return

    # Calculate Beneish M-Score locally
    computed_m_score = calculate_beneish_m_score(payload)

    # ==========================================================================
    # BACKEND API INFERENCE
    # ==========================================================================
    st.divider()
    st.markdown(f"### 🔬 2. Forensic Analysis & Model Inference: **{company_name}**")

    if not api_base_url:
        st.warning("⚠️ Please provide an API Base URL in the sidebar.")
        return

    # Call FastAPI
    with st.spinner(f"Querying Inference API at {api_base_url}..."):
        api_result, api_error = call_fastapi_predict(
            api_base_url=api_base_url,
            payload=payload,
            bypass_ssl=bypass_ssl,
        )

    if api_error:
        st.error(f"❌ **API Inference Failed**: {api_error}")
        st.info("💡 You can run FastAPI locally using: `uvicorn main:app --reload --port 8000` or use `https://api.gloryatk.com`.")
        return

    # Extract API responses
    fraud_prob = float(api_result.get("fraud_probability", 0.0))
    is_fraud = bool(api_result.get("is_fraudulent", fraud_prob >= FRAUD_THRESHOLD))
    latency_ms = float(api_result.get("inference_latency_ms", 0.0))
    # If API provides M-score use it, otherwise use computed formula
    m_score = float(api_result.get("beneish_m_score", api_result.get("m_score", computed_m_score)))
    is_beneish_manipulator = m_score > BENEISH_THRESHOLD

    # ==========================================================================
    # VISUAL OUTPUT: TOP KPIS & GAUGES
    # ==========================================================================
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(
            f"""
            <div class="metric-card">
                <span style="font-size:0.85rem; color:#64748B; font-weight:600;">ML FRAUD PROBABILITY</span>
                <div style="font-size:1.85rem; font-weight:700; color:{'#DC2626' if is_fraud else '#16A34A'};">
                    {fraud_prob * 100:.2f}%
                </div>
                <div style="margin-top:4px;">
                    <span class="{'status-badge-red' if is_fraud else 'status-badge-green'}">
                        {'HIGH RISK ALERT' if is_fraud else 'NORMAL / LOW RISK'}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi2:
        st.markdown(
            f"""
            <div class="metric-card">
                <span style="font-size:0.85rem; color:#64748B; font-weight:600;">BENEISH M-SCORE</span>
                <div style="font-size:1.85rem; font-weight:700; color:{'#DC2626' if is_beneish_manipulator else '#16A34A'};">
                    {m_score:.4f}
                </div>
                <div style="margin-top:4px;">
                    <span class="{'status-badge-red' if is_beneish_manipulator else 'status-badge-green'}">
                        {'MANIPULATOR' if is_beneish_manipulator else 'NON-MANIPULATOR'}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi3:
        # Dual-Engine Consensus
        if is_fraud and is_beneish_manipulator:
            verdict_text = "CRITICAL: HIGH FRAUD RISK"
            verdict_badge = "status-badge-red"
            verdict_desc = "Both ML & Beneish engines trigger alerts."
        elif is_fraud or is_beneish_manipulator:
            verdict_text = "CAUTION: ELEVATED RISK"
            verdict_badge = "status-badge-yellow"
            verdict_desc = "Partial indicator breach detected."
        else:
            verdict_text = "CLEAN: LOW SUSPICION"
            verdict_badge = "status-badge-green"
            verdict_desc = "Within standard healthy operating norms."

        st.markdown(
            f"""
            <div class="metric-card">
                <span style="font-size:0.85rem; color:#64748B; font-weight:600;">CONSENSUS VERDICT</span>
                <div style="font-size:1.2rem; font-weight:700; margin-top:4px;">
                    <span class="{verdict_badge}">{verdict_text}</span>
                </div>
                <div style="font-size:0.8rem; color:#64748B; margin-top:6px;">{verdict_desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with kpi4:
        st.markdown(
            f"""
            <div class="metric-card">
                <span style="font-size:0.85rem; color:#64748B; font-weight:600;">INFERENCE LATENCY</span>
                <div style="font-size:1.85rem; font-weight:700; color:#2563EB;">
                    {latency_ms:.1f} <span style="font-size:1rem; font-weight:400;">ms</span>
                </div>
                <div style="font-size:0.8rem; color:#64748B; margin-top:4px;">
                    Server: <code>{api_base_url.split('://')[-1]}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Visual Plotly Gauges
    st.write("")
    gauge_col1, gauge_col2 = st.columns(2)
    with gauge_col1:
        fig_prob = build_probability_gauge(fraud_prob, threshold=FRAUD_THRESHOLD)
        st.plotly_chart(fig_prob, use_container_width=True)

    with gauge_col2:
        fig_beneish = build_beneish_gauge(m_score, threshold=BENEISH_THRESHOLD)
        st.plotly_chart(fig_beneish, use_container_width=True)

    # Ratio Breakdown Bar Chart
    fig_ratios = build_ratio_comparison_chart(payload)
    st.plotly_chart(fig_ratios, use_container_width=True)

    # ==========================================================================
    # DETAILED METRICS BREAKDOWN TABLE
    # ==========================================================================
    st.markdown("### 📊 3. Forensic Financial Metrics Breakdown")

    breakdown_rows = []
    for col_name in EXPECTED_COLUMNS:
        meta = METRIC_METADATA.get(col_name, {})
        val = payload[col_name]
        cat = meta.get("category", "General")
        thresh = meta.get("threshold")
        direction = meta.get("direction", "neutral")

        # Determine visual status flag
        if thresh is not None:
            if direction == "higher_is_worse" and val > thresh:
                status = "🔴 Red Flag (Breach)"
            elif direction == "lower_is_worse" and val < thresh:
                status = "🔴 Red Flag (Burn)"
            elif direction == "higher_is_worse" and val > 1.05 and col_name != "TATA":
                status = "🟡 Caution (Elevated)"
            else:
                status = "🟢 Normal"
        else:
            status = "⚪ Informational"

        breakdown_rows.append({
            "Metric Identifier": col_name,
            "Metric Full Name": meta.get("name", col_name),
            "Category": cat,
            "Value": f"{val:,.4f}" if abs(val) < 100 else f"${val:,.2f}",
            "Benchmark": meta.get("benchmark", "N/A"),
            "Red Flag Trigger": meta.get("red_flag", "N/A"),
            "Forensic Status": status,
            "Description": meta.get("desc", ""),
        })

    df_breakdown = pd.DataFrame(breakdown_rows)

    tab_table, tab_cards = st.tabs(["📋 Color-Coded Dataframe", "🗂️ Interactive Metric Cards"])

    with tab_table:
        def highlight_status(row):
            status = row["Forensic Status"]
            if "🔴" in status:
                return ["background-color: rgba(239, 68, 68, 0.15)"] * len(row)
            elif "🟡" in status:
                return ["background-color: rgba(245, 158, 11, 0.12)"] * len(row)
            elif "🟢" in status:
                return ["background-color: rgba(34, 197, 94, 0.08)"] * len(row)
            return [""] * len(row)

        styled_df = df_breakdown.style.apply(highlight_status, axis=1)
        st.dataframe(styled_df, use_container_width=True, height=450)

    with tab_cards:
        card_cols = st.columns(4)
        for i, col_name in enumerate(["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]):
            col_target = card_cols[i % 4]
            meta = METRIC_METADATA[col_name]
            val = payload[col_name]
            thresh = meta["threshold"]
            is_breach = thresh is not None and val > thresh
            with col_target:
                st.markdown(
                    f"""
                    <div class="metric-card" style="margin-bottom: 12px;">
                        <div style="font-weight:700; font-size:1.1rem; color:{'#DC2626' if is_breach else '#1E293B'};">
                            {col_name}: {val:.3f}
                        </div>
                        <div style="font-size:0.75rem; color:#64748B; margin-top:2px;">
                            {meta['name']}
                        </div>
                        <div style="margin-top:6px;">
                            <span class="{'status-badge-red' if is_breach else 'status-badge-green'}">
                                {'FLAGGED' if is_breach else 'NORMAL'}
                            </span>
                        </div>
                        <div style="font-size:0.75rem; color:#475569; margin-top:6px; line-height:1.2;">
                            {meta['desc']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ==========================================================================
    # AGENTIC AI (GOOGLE GEMINI) FORENSIC MEMO
    # ==========================================================================
    st.divider()
    st.markdown("### 🤖 4. Agentic AI Forensic Insights (Google Gemini)")

    with st.container(border=True):
        col_ai_head, col_ai_btn = st.columns([4, 1])
        with col_ai_head:
            st.markdown("#### 🧠 Automated Forensic Auditor Memo")
            st.caption("Synthesizes econometric models, operational cash dynamics, and anomaly patterns into natural language commentary.")
        with col_ai_btn:
            trigger_gemini = st.button("⚡ Re-analyze with Gemini", use_container_width=True)

        # Execute Gemini analysis if key is provided and auto-run or clicked
        if gemini_api_key and (auto_run_ai or trigger_gemini):
            with st.spinner("Gemini is conducting forensic statement analysis..."):
                ai_text, ai_err = generate_gemini_forensic_analysis(
                    api_key=gemini_api_key,
                    model_name=gemini_model,
                    company_name=company_name,
                    ml_probability=fraud_prob,
                    is_fraudulent=is_fraud,
                    m_score=m_score,
                    metrics_dict=payload,
                )

            if ai_err:
                st.warning(f"⚠️ {ai_err}")
            elif ai_text:
                st.markdown(ai_text)
        elif not gemini_api_key:
            st.info(
                "💡 **Google Gemini API Key is not set.** "
                "Enter your API key in the sidebar under **'2. Agentic AI (Google Gemini)'** "
                "or set the `GEMINI_API_KEY` environment variable to generate automated forensic narratives."
            )

    # ==========================================================================
    # BATCH INSPECTION OVERVIEW TABLE
    # ==========================================================================
    if total_records > 1:
        st.divider()
        st.markdown(f"### 📑 5. Portfolio Batch Summary ({total_records} Records)")
        if st.checkbox("Show Batch Audit Summary for All Uploaded Companies", value=False):
            with st.spinner("Processing batch inference across all records..."):
                batch_rows = []
                for idx in range(total_records):
                    r = df_validated.iloc[idx]
                    r_payload = {c: float(r[c]) for c in EXPECTED_COLUMNS}
                    r_m_score = calculate_beneish_m_score(r_payload)
                    r_name = str(r[company_identifier_col]) if company_identifier_col else f"Record {idx + 1}"

                    batch_rows.append({
                        "Company / Record": r_name,
                        "Beneish M-Score": r_m_score,
                        "Beneish Status": "Manipulator (Red Flag)" if r_m_score > BENEISH_THRESHOLD else "Non-Manipulator",
                        "DSRI": r_payload["DSRI"],
                        "AQI": r_payload["AQI"],
                        "GMI": r_payload["GMI"],
                        "TATA": r_payload["TATA"],
                        "Cash-Gen(t)": r_payload["Cash-Gen(t)"],
                    })

                df_batch = pd.DataFrame(batch_rows)
                st.dataframe(df_batch, use_container_width=True)


if __name__ == "__main__":
    main()
