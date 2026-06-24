"""
main.py
FastAPI backend exposing:
  POST /query    -> run a question through the chain
  POST /connect  -> swap DB connection string at runtime, re-index schema
  GET  /health   -> DB status + indexed table count + model names
Serves the single HTML UI at /.
"""
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from db_executor import DBExecutor, QueryNotAllowedError
from chain import ChatWithDataChain

BASE_DIR = Path(__file__).resolve().parent.parent
UI_FILE = BASE_DIR / "ui" / "index.html"

app = FastAPI(title="Chat With Data")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/smartbuilding")

db_executor = DBExecutor(DATABASE_URL)
chain = None
try:
    chain = ChatWithDataChain(db_executor)
except Exception as e:
    print(f"[startup warning] chain not ready yet: {e}")


class QueryRequest(BaseModel):
    question: str
    language: str | None = None  # "en" / "ar" / None for auto


class ConnectRequest(BaseModel):
    connection_string: str


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return FileResponse(UI_FILE)


@app.get("/health")
def health():
    db_status = db_executor.health()
    table_count = len(chain.index.chunks) if chain else 0
    return {
        "db": db_status,
        "indexed_tables": table_count,
        "models": {
            "sql": os.getenv("SQL_MODEL", "gpt-5.4-mini"),
            "answer": os.getenv("ANSWER_MODEL", "gpt-5.4-mini"),
            "chart": os.getenv("CHART_MODEL", "gpt-4.1-nano"),
            "suggest": os.getenv("SUGGEST_MODEL", "gpt-4.1-nano"),
        },
    }


@app.post("/connect")
def connect(req: ConnectRequest):
    global chain
    try:
        db_executor.reconnect(req.connection_string)
        chain = ChatWithDataChain(db_executor)
        return {"status": "connected", "indexed_tables": len(chain.index.chunks)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/query")
def query(req: QueryRequest):
    if chain is None:
        raise HTTPException(status_code=503, detail="Database not connected yet. Use /connect.")
    try:
        result = chain.ask(req.question, language_override=req.language)
        return result
    except QueryNotAllowedError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
