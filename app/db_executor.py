"""
db_executor.py
Unified database executor for PostgreSQL and SQL Server.
Enforces SELECT-only queries using sqlparse.
"""
import sqlparse
from sqlalchemy import create_engine, text
import pandas as pd


class QueryNotAllowedError(Exception):
    pass


class DBExecutor:
    """Wraps a SQLAlchemy engine. Swappable at runtime via connect()."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine = create_engine(connection_string, pool_pre_ping=True)

    def reconnect(self, connection_string: str):
        """Swap the underlying DB connection without restarting the app."""
        self.connection_string = connection_string
        self.engine = create_engine(connection_string, pool_pre_ping=True)
        # quick test
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    @staticmethod
    def is_select_only(sql: str) -> bool:
        """Guardrail: only allow a single SELECT / WITH statement."""
        statements = sqlparse.parse(sql)
        if len(statements) != 1:
            return False
        stmt = statements[0]
        stmt_type = stmt.get_type()
        if stmt_type not in ("SELECT", "UNKNOWN"):
            return False
        # UNKNOWN can happen for WITH (CTE); double check first token
        first_token = stmt.token_first(skip_cm=True)
        first_word = first_token.value.upper() if first_token else ""
        if first_word not in ("SELECT", "WITH"):
            return False
        # Block dangerous keywords just in case
        upper_sql = sql.upper()
        for banned in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE", "GRANT", "REVOKE"):
            if f" {banned} " in f" {upper_sql} " or upper_sql.strip().startswith(banned):
                return False
        return True

    def run_query(self, sql: str) -> pd.DataFrame:
        if not self.is_select_only(sql):
            raise QueryNotAllowedError("Only SELECT statements are allowed.")
        with self.engine.connect() as conn:
            df = pd.read_sql(text(sql), conn)
        return df

    def health(self) -> dict:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {"connected": True}
        except Exception as e:
            return {"connected": False, "error": str(e)}
