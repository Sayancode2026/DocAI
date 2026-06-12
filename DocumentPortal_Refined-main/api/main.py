import os
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from src.document_ingestion.data_ingestion import (
    DocHandler,
    DocumentComparator,
    ChatIngestor,
)
from src.document_analyzer.data_analysis import DocumentAnalyzer
from src.document_compare.document_comparator import DocumentComparatorLLM
from src.document_chat.retrieval import ConversationalRAG
from utils.document_ops import FastAPIFileAdapter, read_pdf_via_handler

# ── NEW: import the singleton memory store ─────────────────────────────────
from memory_store import MEMORY_STORE

from logger import GLOBAL_LOGGER as log

FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")

app = FastAPI(title="Document Portal API", version="0.1")

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── UI / Health ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    log.info("Serving UI homepage.")
    resp = templates.TemplateResponse("index.html", {"request": request})
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/health")
def health() -> Dict[str, str]:
    log.info("Health check passed.")
    return {"status": "ok", "service": "document-portal"}


# ── ANALYZE ─────────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)) -> Any:
    try:
        log.info(f"Received file for analysis: {file.filename}")
        dh = DocHandler()
        saved_path = dh.save_pdf(FastAPIFileAdapter(file))
        text = read_pdf_via_handler(dh, saved_path)
        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)
        log.info("Document analysis complete.")
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Error during document analysis")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


# ── COMPARE ─────────────────────────────────────────────────────────────────

@app.post("/compare")
async def compare_documents(
    reference: UploadFile = File(...), actual: UploadFile = File(...)
) -> Any:
    try:
        log.info(f"Comparing files: {reference.filename} vs {actual.filename}")
        dc = DocumentComparator()
        ref_path, act_path = dc.save_uploaded_files(
            FastAPIFileAdapter(reference), FastAPIFileAdapter(actual)
        )
        _ = ref_path, act_path
        combined_text = dc.combine_documents()
        comp = DocumentComparatorLLM()
        df = comp.compare_documents(combined_text)
        log.info("Document comparison completed.")
        return {"rows": df.to_dict(orient="records"), "session_id": dc.session_id}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Comparison failed")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")


# ── CHAT: INDEX ─────────────────────────────────────────────────────────────

@app.post("/chat/index")
async def chat_build_index(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5),
) -> Any:
    try:
        log.info(
            f"Indexing chat session. Session ID: {session_id}, "
            f"Files: {[f.filename for f in files]}"
        )
        wrapped = [FastAPIFileAdapter(f) for f in files]
        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs=use_session_dirs,
            session_id=session_id or None,
        )
        ci.built_retriver(
            wrapped, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k
        )

        # ── Clear any stale memory for this session when re-indexing ──────
        if ci.session_id:
            MEMORY_STORE.clear(ci.session_id)
            log.info(
                "Chat memory cleared for re-indexed session",
                session_id=ci.session_id,
            )

        log.info(f"Index created successfully for session: {ci.session_id}")
        return {"session_id": ci.session_id, "k": k, "use_session_dirs": use_session_dirs}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat index building failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


# ── CHAT: QUERY ─────────────────────────────────────────────────────────────

@app.post("/chat/query")
async def chat_query(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
) -> Any:
    """
    Ask a question against indexed documents.
    Chat history is managed automatically in-memory per session_id.
    Each call picks up where the last one left off — no extra params needed.
    """
    try:
        log.info(f"Received chat query: '{question}' | session: {session_id}")

        if use_session_dirs and not session_id:
            raise HTTPException(
                status_code=400,
                detail="session_id is required when use_session_dirs=True",
            )

        index_dir = (
            os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE
        )
        if not os.path.isdir(index_dir):
            raise HTTPException(
                status_code=404,
                detail=f"FAISS index not found at: {index_dir}. "
                       f"Please call /chat/index first.",
            )

        # ── Build RAG and run ─────────────────────────────────────────────
        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)

        # invoke() now handles history automatically via MEMORY_STORE
        response = rag.invoke(question)

        log.info("Chat query handled successfully.")
        return {
            "answer": response,
            "session_id": session_id,
            "k": k,
            "engine": "LCEL-RAG",
            # ── Tell the caller how many turns have been stored so far ────
            "history_turns": MEMORY_STORE.get_turn_count(session_id or ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


# ── CHAT: MEMORY MANAGEMENT (bonus endpoints) ────────────────────────────────

@app.delete("/chat/memory/{session_id}")
async def clear_chat_memory(session_id: str) -> Dict[str, str]:
    """
    Wipe the in-memory chat history for a session.
    Useful when the user wants to start a fresh conversation
    without re-uploading/re-indexing documents.
    """
    MEMORY_STORE.clear(session_id)
    log.info("Chat memory cleared via API", session_id=session_id)
    return {"status": "cleared", "session_id": session_id}


@app.get("/chat/memory/{session_id}")
async def get_chat_memory_info(session_id: str) -> Dict:
    """
    Return memory stats for a session — useful for debugging.
    """
    turns = MEMORY_STORE.get_turn_count(session_id)
    messages = MEMORY_STORE.get_history(session_id)
    history_preview = [
        {"role": "human" if i % 2 == 0 else "ai", "content": str(m.content)[:200]}
        for i, m in enumerate(messages)
    ]
    return {
        "session_id": session_id,
        "total_turns": turns,
        "total_messages": len(messages),
        "history_preview": history_preview,
    }


# command for running:
# uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
