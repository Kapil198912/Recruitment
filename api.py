"""FastAPI wrapper for the fixed Part 2 CrewAI/RAG application."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field

from orchestration import (
    CrewResponse,
    RagResources,
    RequestBudgetExceeded,
    apply_input_guardrails,
    clear_response_cache,
    create_session_memory_chain,
    enforce_request_budget,
    run_request,
)
from rag_pipeline import Document, sentence_chunks


ROOT = Path(__file__).resolve().parent
LOG_FILE = ROOT / ".logs" / "api.jsonl"
DOCUMENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
_LOG_LOCK = threading.Lock()
_RESOURCE_LOCK = threading.Lock()
_resources: RagResources | None = None
_memory_chain: Any | None = None
_memory_resources_id: int | None = None


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


class AskResponse(BaseModel):
    trace_id: str
    session_id: str
    answer: str
    grounded: bool
    refused: bool
    sources: list[str]
    tool_names: list[str]
    review_approved: bool
    review_reason: str


class AddDocumentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=2, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]{1,63}$")
    text: str = Field(min_length=10, max_length=20_000)


class AddDocumentResponse(BaseModel):
    trace_id: str
    document_id: str
    chunks_added: int
    pii_values_masked: int


class WebSocketChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)


def _append_json_log(
    *,
    trace_id: str,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    request_text: str,
    event: str = "http_request",
    error_type: str | None = None,
) -> None:
    entry = {
        "@timestamp": datetime.now(timezone.utc).isoformat(),
        "trace_id": trace_id,
        "event": event,
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 3),
        "request_text": request_text,
    }
    if error_type is not None:
        entry["error_type"] = error_type
    encoded = json.dumps(entry, ensure_ascii=True, separators=(",", ":"))
    with _LOG_LOCK:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as log_file:
            log_file.write(encoded + "\n")


def _get_resources() -> RagResources:
    global _resources
    if _resources is None:
        with _RESOURCE_LOCK:
            if _resources is None:
                from orchestration import load_recommended_rag

                _resources = load_recommended_rag()
    return _resources


def _get_memory_chain(resources: RagResources) -> Any:
    global _memory_chain, _memory_resources_id
    if _memory_chain is None or _memory_resources_id != id(resources):
        with _RESOURCE_LOCK:
            if _memory_chain is None or _memory_resources_id != id(resources):
                _memory_chain = create_session_memory_chain(resources)
                _memory_resources_id = id(resources)
    return _memory_chain


def _validate_input(query: str, request: Request | None = None) -> str:
    result = apply_input_guardrails(query)
    if result.prompt_injection_detected:
        raise HTTPException(
            status_code=400,
            detail="Prompt-injection guardrail rejected the request.",
        )
    if request is not None:
        request.state.safe_request_text = result.sanitized_query
    try:
        enforce_request_budget(result.sanitized_query)
    except RequestBudgetExceeded as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    return result.sanitized_query


def _ask_in_session(
    query: str, resources: RagResources, session_id: str
) -> CrewResponse:
    chain = _get_memory_chain(resources)
    memory_result = chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}},
    )
    contextualized_query = memory_result.get("contextualized_query") or query
    return run_request(contextualized_query, resources)


app = FastAPI(
    title="Job Application Assistant",
    version="1.0.0",
    description="CrewAI application backed by the fixed sentence-based RAG collection.",
)


@app.middleware("http")
async def structured_request_logging(request: Request, call_next: Any) -> Any:
    trace_id = str(uuid4())
    request.state.trace_id = trace_id
    request.state.safe_request_text = ""
    started_at = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Trace-ID"] = trace_id
        return response
    finally:
        _append_json_log(
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            duration_ms=(time.perf_counter() - started_at) * 1000,
            request_text=request.state.safe_request_text,
        )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(request: Request, payload: AskRequest) -> AskResponse:
    safe_query = _validate_input(payload.query, request)
    session_id = payload.session_id or str(uuid4())
    request.state.safe_request_text = safe_query
    resources = await run_in_threadpool(_get_resources)
    crew_response = await run_in_threadpool(
        _ask_in_session, safe_query, resources, session_id
    )
    return AskResponse(
        trace_id=request.state.trace_id,
        session_id=session_id,
        **crew_response.model_dump(),
    )


@app.post("/add-document", response_model=AddDocumentResponse, status_code=201)
async def add_document(
    request: Request, payload: AddDocumentRequest
) -> AddDocumentResponse:
    guardrail = apply_input_guardrails(payload.text)
    safe_text = guardrail.sanitized_query.strip()
    request.state.safe_request_text = (
        f"document_id={payload.document_id}; text={safe_text}"
    )
    if guardrail.prompt_injection_detected:
        raise HTTPException(
            status_code=400,
            detail="Prompt-injection guardrail rejected the document.",
        )
    try:
        enforce_request_budget(safe_text)
    except RequestBudgetExceeded as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    resources = await run_in_threadpool(_get_resources)
    document = Document(doc_id=payload.document_id, text=safe_text)
    chunks = sentence_chunks([document])
    if not chunks:
        raise HTTPException(status_code=422, detail="Document produced no text chunks.")
    try:
        existing = resources.collection.get(
            where={"doc_id": payload.document_id},
            limit=1,
            include=["metadatas"],
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Could not check the indexed document collection."
        ) from exc
    if existing["ids"]:
        raise HTTPException(
            status_code=409, detail="A document with this ID is already indexed."
        )

    embeddings = await run_in_threadpool(
        resources.model.encode,
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    try:
        await run_in_threadpool(
            resources.collection.upsert,
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[{"doc_id": chunk.doc_id} for chunk in chunks],
            embeddings=embeddings.tolist(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Could not index the document."
        ) from exc
    clear_response_cache()
    return AddDocumentResponse(
        trace_id=request.state.trace_id,
        document_id=payload.document_id,
        chunks_added=len(chunks),
        pii_values_masked=guardrail.masked_pii_count,
    )


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    trace_id = str(uuid4())
    session_id = str(uuid4())
    await websocket.accept()
    resources: RagResources | None = None
    memory_chain: Any | None = None
    try:
        resources = await run_in_threadpool(_get_resources)
        memory_chain = await run_in_threadpool(_get_memory_chain, resources)
        while True:
            started_at = time.perf_counter()
            safe_message = ""
            status_code = 200
            error_name: str | None = None
            message_received = False
            try:
                raw_payload = await websocket.receive_json()
                message_received = True
                payload = WebSocketChatRequest.model_validate(raw_payload)
                guardrail = apply_input_guardrails(payload.message)
                safe_message = guardrail.sanitized_query
                if guardrail.prompt_injection_detected:
                    status_code = 400
                    await websocket.send_json(
                        {
                            "trace_id": trace_id,
                            "session_id": session_id,
                            "error": "Prompt-injection guardrail rejected the request.",
                        }
                    )
                    continue
                try:
                    enforce_request_budget(safe_message)
                except RequestBudgetExceeded as exc:
                    status_code = 413
                    await websocket.send_json(
                        {
                            "trace_id": trace_id,
                            "session_id": session_id,
                            "error": str(exc),
                        }
                    )
                    continue
                memory_result = await run_in_threadpool(
                    memory_chain.invoke,
                    {"input": safe_message},
                    config={"configurable": {"session_id": session_id}},
                )
                contextualized_query = (
                    memory_result.get("contextualized_query") or safe_message
                )
                result = await run_in_threadpool(
                    run_request, contextualized_query, resources
                )
                await websocket.send_json(
                    {
                        "trace_id": trace_id,
                        "session_id": session_id,
                        **result.model_dump(),
                    }
                )
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                status_code = 500
                await websocket.send_json(
                    {
                        "trace_id": trace_id,
                        "session_id": session_id,
                        "error": "The chat request could not be completed.",
                    }
                )
                error_name = type(exc).__name__
            else:
                error_name = None
            finally:
                if message_received:
                    _append_json_log(
                        trace_id=trace_id,
                        method="WEBSOCKET",
                        path="/ws/chat",
                        status_code=status_code,
                        duration_ms=(time.perf_counter() - started_at) * 1000,
                        request_text=safe_message,
                        event="websocket_turn",
                        error_type=error_name,
                    )
    except WebSocketDisconnect:
        return
    finally:
        _append_json_log(
            trace_id=trace_id,
            method="WEBSOCKET",
            path="/ws/chat",
            status_code=200,
            duration_ms=0,
            request_text="",
            event="websocket_closed",
        )
