"""
chart_generator.py
Validates / normalizes the chart spec JSON produced by the chart-selector
LLM, and prepares the data payload the frontend Plotly code needs.
"""
import json
import math
import decimal
import datetime
import numpy as np
import pandas as pd

ALLOWED_TYPES = {"bar", "line", "pie", "scatter", "heatmap", "table"}


def _safe(val):
    """
    Convert any non-JSON-serialisable value to a safe Python primitive.
    Timestamp / datetime -> ISO string, NaT -> None,
    numpy numerics -> native Python, NaN/Inf -> None.
    """
    if val is None:
        return None
    if val is pd.NaT:
        return None
    if isinstance(val, (pd.Timestamp, datetime.datetime)):
        return val.isoformat()
    if isinstance(val, datetime.date):
        return val.isoformat()
    if isinstance(val, datetime.time):
        return val.isoformat()
    if isinstance(val, datetime.timedelta):
        return str(val)
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(val, decimal.Decimal):
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, (np.ndarray, list, tuple)):
        return [_safe(v) for v in val]
    if isinstance(val, dict):
        return {k: _safe(v) for k, v in val.items()}
    if not isinstance(val, (str, int, bool)):
        return str(val)
    return val


def _safe_list(series) -> list:
    """Convert a pandas Series to a JSON-safe Python list."""
    return [_safe(v) for v in series.tolist()]


def _safe_records(df: pd.DataFrame) -> list:
    """Convert a DataFrame to a list of dicts that are fully JSON-safe."""
    return [
        {col: _safe(val) for col, val in row.items()}
        for row in df.to_dict(orient="records")
    ]


def parse_chart_spec(raw: str) -> dict:
    """Parses the LLM's JSON response, with a safe fallback."""
    try:
        cleaned = raw.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        spec = json.loads(cleaned)
    except Exception:
        spec = {}

    chart_type = spec.get("type", "table")
    if chart_type not in ALLOWED_TYPES:
        chart_type = "table"

    return {
        "type": chart_type,
        "x": spec.get("x"),
        "y": spec.get("y"),
        "title": spec.get("title", ""),
    }


def build_chart_payload(df: pd.DataFrame, spec: dict, max_points: int = 500) -> dict:
    """Builds the data arrays the frontend Plotly call needs."""
    if df is None or df.empty:
        return {"type": "table", "x": [], "y": [], "title": spec.get("title", ""), "data": []}

    trimmed = df.head(max_points)
    payload = {"type": spec["type"], "title": spec.get("title", "")}

    x_col, y_col = spec.get("x"), spec.get("y")
    if x_col in trimmed.columns:
        payload["x"] = _safe_list(trimmed[x_col])
    else:
        payload["x"] = _safe_list(trimmed.iloc[:, 0]) if trimmed.shape[1] > 0 else []

    if y_col in trimmed.columns:
        payload["y"] = _safe_list(trimmed[y_col])
    elif trimmed.shape[1] > 1:
        payload["y"] = _safe_list(trimmed.iloc[:, 1])
    else:
        payload["y"] = []

    payload["data"] = _safe_records(trimmed)
    return payload