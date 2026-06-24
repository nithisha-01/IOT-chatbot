"""
result_formatter.py
Turns a raw Pandas DataFrame (potentially huge) into a compact summary
string that's cheap to feed to the answer LLM, instead of dumping
thousands of raw rows into the prompt.
"""
import pandas as pd


def summarize_dataframe(df: pd.DataFrame, max_rows: int = 20, top_categorical: int = 5) -> dict:
    if df is None or df.empty:
        return {"row_count": 0, "columns": [], "summary_text": "No rows returned.", "preview_rows": []}

    row_count = len(df)
    col_info = []
    lines = [f"Row count: {row_count}"]

    for col in df.columns:
        dtype = str(df[col].dtype)
        if pd.api.types.is_numeric_dtype(df[col]):
            stats = df[col].describe()
            lines.append(
                f"- {col} (numeric): min={stats.get('min'):.2f}, "
                f"max={stats.get('max'):.2f}, mean={stats.get('mean'):.2f}"
            )
            col_info.append({"name": col, "type": "numeric"})
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
        "preview_rows": preview.to_dict(orient="records"),
        "preview_csv": preview.to_csv(index=False),
    }
