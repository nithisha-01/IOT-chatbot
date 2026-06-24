"""
schema_loader.py
Auto-discovers DB schema via information_schema and builds per-table
text chunks. Uses TF-IDF similarity (no heavy ML deps needed) to retrieve
the most relevant tables for a given question - same role pgvector/FAISS
would play, but dependency-light so the POC runs anywhere out of the box.

To upgrade to real embeddings later: swap `TableIndex` internals for
FAISS + sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) -
the public interface (build(), search()) stays identical.
"""
from sqlalchemy import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class TableIndex:
    def __init__(self):
        self.chunks = []          # list of dicts: {table, text}
        self.vectorizer = None
        self.matrix = None

    def build(self, chunks: list):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        if not texts:
            self.vectorizer = None
            self.matrix = None
            return
        self.vectorizer = TfidfVectorizer().fit(texts)
        self.matrix = self.vectorizer.transform(texts)

    def search(self, query: str, top_k: int = 4):
        if not self.chunks or self.vectorizer is None:
            return self.chunks
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        ranked = sorted(zip(self.chunks, sims), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:top_k]]


def discover_schema(engine) -> list:
    """
    Introspects the connected database (PostgreSQL) and returns a list of
    per-table chunks: {table, columns, text} where `text` is a natural
    language description used both as the LLM prompt context and as the
    retrieval document.
    """
    chunks = []
    with engine.connect() as conn:
        tables = conn.execute(text("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        """)).fetchall()

        for (table_name,) in tables:
            cols = conn.execute(text("""
                SELECT column_name, data_type FROM information_schema.columns
                WHERE table_schema='public' AND table_name=:t
                ORDER BY ordinal_position
            """), {"t": table_name}).fetchall()

            col_lines = []
            sample_vals = {}
            for col_name, data_type in cols:
                try:
                    samples = conn.execute(
                        text(f'SELECT DISTINCT "{col_name}" FROM "{table_name}" '
                             f'WHERE "{col_name}" IS NOT NULL LIMIT 3')
                    ).fetchall()
                    sample_vals[col_name] = [str(s[0]) for s in samples]
                except Exception:
                    sample_vals[col_name] = []
                col_lines.append(f"  - {col_name} ({data_type}) e.g. {sample_vals[col_name]}")

            description = f"Table: {table_name}\nColumns:\n" + "\n".join(col_lines)
            chunks.append({
                "table": table_name,
                "columns": [c[0] for c in cols],
                "text": description,
            })
    return chunks


def build_schema_context(chunks: list) -> str:
    """Joins selected table chunks into the prompt context block."""
    return "\n\n".join(c["text"] for c in chunks)
