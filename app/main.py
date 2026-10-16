import os
from io import BytesIO

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

from . import config
from .schemas import ChatRequest, ChatResponse, MemoryToggleRequest, HistoryMessage, AddTextRequest
from .memory_store import memory_store
from .rag_store import rag_registry
import time

from .llm_provider import chat_with_fallback, get_available_models, get_default_model

app = FastAPI(title="Local Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    available = get_available_models()
    if not available:
        return {
            "models": [],
            "message": "No provider API keys found. For free usage, add GOOGLE_API_KEY "
            "(ai.google.dev) or GROQ_API_KEY (console.groq.com) to your .env file. "
            "OPENAI_API_KEY / ANTHROPIC_API_KEY require paid credits.",
        }
    return {"models": available, "default": get_default_model()}

# ---------------------------------------------------------------------------
# Memory controls
# ---------------------------------------------------------------------------


@app.post("/session/{session_id}/memory")
def set_memory(session_id: str, req: MemoryToggleRequest):
    memory_store.set_enabled(session_id, req.enabled)
    return {"session_id": session_id, "memory_enabled": req.enabled}


@app.get("/session/{session_id}/memory")
def get_memory_status(session_id: str):
    return {"session_id": session_id, "memory_enabled": memory_store.is_enabled(session_id)}


@app.delete("/session/{session_id}/memory")
def clear_memory(session_id: str):
    memory_store.clear(session_id)
    return {"session_id": session_id, "cleared": True}


@app.get("/session/{session_id}/history", response_model=list[HistoryMessage])
def get_history(session_id: str):
    return memory_store.get_history(session_id)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id
    use_memory = req.use_memory if req.use_memory is not None else memory_store.is_enabled(
        session_id)
    memory_store.set_enabled(session_id, use_memory)
    model = req.model or get_default_model()
    if not model:
        raise HTTPException(
            status_code=400, detail="No model available. Add a provider API key to your .env file.")
    rag_sources = []
    system_prompt = "You are a helpful assistant."
    if req.use_rag:
        collection = rag_registry.get(req.collection or "default")
        result = collection.search(req.message, top_k=int(config.RAG_TOP_K))
        # Keep only chunks that are actually related to the question.
        result = [r for r in result if r[2] >= collection.min_score]
        if result:
            context_block = "\n\n---\n\n".join(chunk for chunk,
                                               _source, _score in result)
            system_prompt += ("\n\nContext:\n" + context_block)
            rag_sources = sorted({source for _c, source, _s in result})
    messages = memory_store.get_history(session_id) if use_memory else []
    messages = messages + [{"role": "user", "content": req.message}]
    started = time.perf_counter()
    try:
        reply, used_model, errors, reasoning = chat_with_fallback(
            model, messages, system_prompt, thinking=req.thinking)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    latency_ms = int((time.perf_counter() - started) * 1000)
    fallback_note = None
    if used_model != model:
        fallback_note = f"{model} failed, answered by {used_model}"
    elif errors:
        fallback_note = f"{model} answered after a retry"
    if use_memory:
        memory_store.add_message(session_id, "user", req.message)
        memory_store.add_message(session_id, "assistant", reply)
    return ChatResponse(reply=reply, model=used_model, used_memory=use_memory, used_rag=bool(rag_sources),
                        sources=rag_sources, latency_ms=latency_ms, fallback_note=fallback_note,
                        thinking=req.thinking, reasoning=reasoning)


@app.post("/rag/upload")
async def upload_document(collection:str=Form("default"), file: UploadFile = File(...)):
    raw = await file.read()
    filename = file.filename or "uploaded_file"
    if filename.lower().endswith(".pdf"):
        text = _extract_pdf_text(raw)
    else:
        try:
            text = raw.decode("utf-8", errors = "ignore")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode file: {str(e)}")
    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty or contains only whitespace.")
    col = rag_registry.get(collection)
    added = col.add_document(text, source = filename)
    return {"collection": collection, "chunks_added": added, **col.stats()}

@app.post("/rag/add_text")
def add_text(req: AddTextRequest):
    col = rag_registry.get(req.collection)
    added = col.add_document(req.text, source=req.source)
    return {"collection": req.collection, "chunks_added": added, **col.stats()}

@app.get("/rag/collections")
def get_collections():
    return {"collections": list(rag_registry.list_names())}

@app.get("/rag/collections/{name}")
def get_collection(name: str):
    return rag_registry.get(name).stats()

@app.delete("/rag/collections/{name}")
def delete_collection(name: str):
    path = os.path.join(config.DATA_DIR, f"{name}.pkl")
    existed = os.path.exists(path)
    if existed:
        os.remove(path)
    rag_registry._collections.pop(name, None)
    return {"collection": name, "deleted": existed}

@app.post("/rag/search")
def rag_search(collection:str,query:str="", top_k:int|None=None):
    col = rag_registry.get(collection)
    result = col.search(query, top_k=top_k or int(config.RAG_TOP_K))
    return {"results": [{"text":c,"source":s,"score":sc} for c, s, sc in result]}

def _extract_pdf_text(raw: bytes) -> str:
    reader = PdfReader(BytesIO(raw))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


if __name__ =="__main__":
    import uvicorn
    uvicorn.run("app.main:app", reload=True, host=config.HOST, port=int(config.PORT))