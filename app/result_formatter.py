"""
result_formatter.py
Turns a raw Pandas DataFrame (potentially huge) into a compact summary
string that's cheap to feed to the answer LLM, instead of dumping
thousands of raw rows into the prompt.
"""
import math
import decimal
import datetime
import numpy as np
import pandas as pd


def _safe(val):
    """
    Recursively convert any value that is not natively JSON-serialisable:
      - pandas / numpy Timestamps  -> ISO-8601 string
      - datetime / date / time     -> ISO-8601 string
      - pandas NaT                 -> None
      - numpy integers / floats    -> native Python int / float
      - numpy booleans             -> native Python bool
      - decimal.Decimal            -> float
      - NaN / Inf                  -> None  (JSON has no NaN/Inf)
      - everything else            -> str() as last resort
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
    # Anything else: try str so the response never crashes
    if not isinstance(val, (str, int, bool)):
        return str(val)
    return val


def _safe_records(df: pd.DataFrame) -> list:
    """Convert a DataFrame to a list of dicts that are fully JSON-safe."""
    return [
        {col: _safe(val) for col, val in row.items()}
        for row in df.to_dict(orient="records")
    ]


def summarize_dataframe(df: pd.DataFrame, max_rows: int = 20, top_categorical: int = 5) -> dict:
    if df is None or df.empty:
        return {"row_count": 0, "columns": [], "summary_text": "No rows returned.", "preview_rows": []}

    row_count = len(df)
    col_info = []
    lines = [f"Row count: {row_count}"]

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            stats = df[col].describe()
            lines.append(
                f"- {col} (numeric): min={stats.get('min'):.2f}, "
                f"max={stats.get('max'):.2f}, mean={stats.get('mean'):.2f}"
            )
            col_info.append({"name": col, "type": "numeric"})
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null):
                lines.append(
                    f"- {col} (datetime): from {non_null.min().isoformat()} to {non_null.max().isoformat()}"
                )
            else:
                lines.append(f"- {col} (datetime): all null")
            col_info.append({"name": col, "type": "datetime"})
        else:
            top_vals = df[col].value_counts().head(top_categorical)
            top_str = ", ".join(f"{idx} ({cnt})" for idx, cnt in top_vals.items())
            lines.append(f"- {col} (categorical): top values -> {top_str}")
            col_info.append({"name": col, "type": "categorical"})

    preview = df.head(max_rows)

    return {
        "row_count": row_count,
        "columns": col_info,
        "summary_text": "\n".join(lines),
        "preview_rows": _safe_records(preview),
        "preview_csv": preview.to_csv(index=False),
    }