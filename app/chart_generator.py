"""
chart_generator.py
Validates / normalizes the chart spec JSON produced by the chart-selector
LLM, and prepares the data payload the frontend Plotly code needs.
"""
import json
import pandas as pd

ALLOWED_TYPES = {"bar", "line", "pie", "scatter", "heatmap", "table"}


def parse_chart_spec(raw: str) -> dict:
    """Parses the LLM's JSON response, with a safe fallback."""
    try:
        # strip markdown fences if the model added them
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
        payload["x"] = trimmed[x_col].tolist()
    else:
        payload["x"] = trimmed.iloc[:, 0].tolist() if trimmed.shape[1] > 0 else []

    if y_col in trimmed.columns:
        payload["y"] = trimmed[y_col].tolist()
    elif trimmed.shape[1] > 1:
        payload["y"] = trimmed.iloc[:, 1].tolist()
    else:
        payload["y"] = []

    payload["data"] = trimmed.to_dict(orient="records")
    return payload
