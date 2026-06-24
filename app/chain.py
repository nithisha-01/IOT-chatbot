"""
chain.py
Orchestrates: language detect -> (translate if AR) -> schema retrieval ->
SQL generation -> guardrail check -> DB execution -> result summarization ->
answer generation (in original language) -> chart spec -> follow-up chips.

Uses Groq via langchain-groq for all LLM calls, with different models for
heavy reasoning (SQL/answer) vs lightweight fast decisions (chart/suggestions).
"""
import os
import json
import sqlite3
import hashlib
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

from db_executor import DBExecutor, QueryNotAllowedError
from schema_loader import TableIndex, build_schema_context
from language_handler import detect_language
from result_formatter import summarize_dataframe
from chart_generator import parse_chart_spec, build_chart_payload

BASE_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"
CACHE_DB = BASE_DIR / "query_cache.sqlite"

SQL_PROMPT = (PROMPTS_DIR / "sql_gen_prompt.txt").read_text()
ANSWER_PROMPT = (PROMPTS_DIR / "answer_prompt.txt").read_text()

LANGUAGE_NAMES = {"en": "English", "ar": "Arabic"}


def _init_cache():
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS cache (
        key TEXT PRIMARY KEY, response TEXT NOT NULL
    )""")
    conn.commit()
    conn.close()


def _cache_get(key: str):
    conn = sqlite3.connect(CACHE_DB)
    row = conn.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
    conn.close()
    return json.loads(row[0]) if row else None


def _cache_set(key: str, value: dict):
    conn = sqlite3.connect(CACHE_DB)
    conn.execute("INSERT OR REPLACE INTO cache (key, response) VALUES (?, ?)",
                 (key, json.dumps(value)))
    conn.commit()
    conn.close()


class ChatWithDataChain:
    def __init__(self, db_executor: DBExecutor):
        self.db = db_executor
        self.index = TableIndex()
        self.memory = ConversationBufferWindowMemory(k=5, return_messages=False)

        self.sql_llm = ChatOpenAI(model=os.getenv("SQL_MODEL", "gpt-5.4-mini"), temperature=0)
        self.answer_llm = ChatOpenAI(model=os.getenv("ANSWER_MODEL", "gpt-5.4-mini"), temperature=0.3)
        self.chart_llm = ChatOpenAI(model=os.getenv("CHART_MODEL", "gpt-4.1-nano"), temperature=0)
        self.suggest_llm = ChatOpenAI(model=os.getenv("SUGGEST_MODEL", "gpt-4.1-nano"), temperature=0.5)

        _init_cache()
        self.refresh_schema()

    def refresh_schema(self):
        from schema_loader import discover_schema
        chunks = discover_schema(self.db.engine)
        self.index.build(chunks)
        return len(chunks)

    def _translate(self, text: str, target: str) -> str:
        msg = f"Translate the following text to {target}. Output ONLY the translation, nothing else.\n\nText: {text}"
        resp = self.sql_llm.invoke(msg)
        return resp.content.strip()

    def ask(self, question: str, language_override: str = None) -> dict:
        cache_key = hashlib.sha256(f"{question}|{language_override}".encode()).hexdigest()
        cached = _cache_get(cache_key)
        if cached:
            cached["from_cache"] = True
            return cached

        lang = detect_language(question, override=language_override)
        question_en = question if lang == "en" else self._translate(question, "English")

        relevant_chunks = self.index.search(question_en, top_k=4)
        schema_context = build_schema_context(relevant_chunks)
        memory_context = self.memory.load_memory_variables({}).get("history", "")

        sql_prompt = SQL_PROMPT.format(
            schema_context=schema_context,
            memory_context=memory_context,
            question=question_en,
        )
        sql = self.sql_llm.invoke(sql_prompt).content.strip().strip("`")
        if sql.lower().startswith("sql"):
            sql = sql[3:].strip()

        try:
            if not self.db.is_select_only(sql):
                raise QueryNotAllowedError("Generated SQL failed the SELECT-only guardrail.")
            df = self.db.run_query(sql)
        except Exception as e:
            # one retry with the error fed back in
            retry_prompt = sql_prompt + f"\n\nThe previous attempt failed with error: {e}\nFix it and return only corrected SQL:\n"
            sql = self.sql_llm.invoke(retry_prompt).content.strip().strip("`")
            if sql.lower().startswith("sql"):
                sql = sql[3:].strip()
            if not self.db.is_select_only(sql):
                raise QueryNotAllowedError("Generated SQL failed the SELECT-only guardrail after retry.")
            df = self.db.run_query(sql)

        summary = summarize_dataframe(df)

        answer_prompt = ANSWER_PROMPT.format(
            question=question,
            sql=sql,
            result_summary=summary["summary_text"],
            language_name=LANGUAGE_NAMES.get(lang, "English"),
        )
        answer = self.answer_llm.invoke(answer_prompt).content.strip()

        chart_prompt = (
            f"Given this question: '{question_en}' and these result columns: {summary['columns']} "
            f"with sample rows: {summary['preview_rows'][:5]}, decide the best chart type. "
            f'Respond ONLY with JSON like {{"type":"bar","x":"col","y":"col","title":"..."}}. '
            f'type must be one of: bar, line, pie, scatter, heatmap, table.'
        )
        chart_raw = self.chart_llm.invoke(chart_prompt).content
        chart_spec = parse_chart_spec(chart_raw)
        chart_payload = build_chart_payload(df, chart_spec)

        target_language = LANGUAGE_NAMES.get(lang, "English")
        suggest_prompt = (
            f"The user asked in {target_language}: '{question}'. "
            f"Suggest 3 short, natural follow-up questions they might ask next about this smart-building "
            f"sensor database. Respond ONLY as a JSON array of 3 strings. "
            f"Write every suggestion in {target_language} only, using the same language as the user's question."
        )
        try:
            raw_suggestions = self.suggest_llm.invoke(suggest_prompt).content.strip().strip("`")
            suggestions = json.loads(raw_suggestions)
            if not isinstance(suggestions, list):
                suggestions = []
            else:
                suggestions = [str(s).strip() for s in suggestions if str(s).strip()]
                if lang != "en":
                    normalized = []
                    for suggestion in suggestions:
                        try:
                            normalized.append(self._translate(suggestion, target_language))
                        except Exception:
                            normalized.append(suggestion)
                    suggestions = normalized
        except Exception:
            suggestions = []

        self.memory.save_context({"input": question}, {"output": answer})

        result = {
            "answer": answer,
            "sql": sql,
            "language": lang,
            "chart": chart_payload,
            "suggestions": suggestions[:3],
            "row_count": summary["row_count"],
            "preview_csv": summary.get("preview_csv", ""),
            "from_cache": False,
        }
        _cache_set(cache_key, result)
        return result
