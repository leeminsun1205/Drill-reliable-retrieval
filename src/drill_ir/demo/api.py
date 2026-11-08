"""Single-process API and static frontend for the interactive Kaggle demo."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Callable

import anyio

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from drill_ir.demo.backend import load_backend, normalize_evidence

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10000)

    @field_validator('question')
    @classmethod
    def trim_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('Câu hỏi không được để trống')
        return value


def create_app(*, backend_factory: Callable = load_backend, answer_stream=None,
               static_dir: Path | None = None) -> FastAPI:
    """Injection points allow contract tests without GPUs or external API calls."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.backend = None
        try:
            # One model/index instance per process. Run uvicorn with one worker.
            from starlette.concurrency import run_in_threadpool
            app.state.backend = await run_in_threadpool(backend_factory)
        except Exception:
            logger.exception('Demo backend initialization failed')
        yield
        app.state.backend = None

    app = FastAPI(title='DRiLL Legal QA', lifespan=lifespan)
    inference_lock = Lock()

    @app.get('/api/health')
    def health():
        ready = app.state.backend is not None
        return {'status': 'ready' if ready else 'unavailable', 'ready': ready}

    @app.post('/api/chat')
    def chat(payload: ChatRequest):
        if app.state.backend is None:
            raise HTTPException(503, 'Pipeline chưa sẵn sàng. Kiểm tra log máy chủ.')
        # The shared GPU pipeline is not reentrant. Reject overlap instead of
        # creating an unbounded queue of expensive work on a demo server.
        if not inference_lock.acquire(blocking=False):
            raise HTTPException(429, 'Hệ thống đang xử lý câu hỏi khác. Vui lòng thử lại.')

        def events():
            def event(kind, **data):
                return json.dumps({'type': kind, **data}, ensure_ascii=False) + '\n'
            try:
                yield event('status', message='Đang truy xuất thông tin…')
                evidence = [normalize_evidence(item) for item in app.state.backend.search(payload.question)]
                yield event('evidence', evidence=[asdict(item) for item in evidence])
                yield event('status', message='Đang soạn câu trả lời…')
                stream = answer_stream
                if stream is None:
                    from drill_ir.demo.answering import stream_answer
                    stream = stream_answer
                for piece in stream(payload.question, evidence):
                    yield event('delta', text=piece)
                yield event('done')
            except Exception:
                logger.exception('Demo request failed')
                yield event('error', message='Chưa xử lý được câu hỏi. Kiểm tra log máy chủ và thử lại.')

        iterator = events()

        class ChatResponse(StreamingResponse):
            async def __call__(self, scope, receive, send):
                try:
                    await super().__call__(scope, receive, send)
                finally:
                    # Close the generator even when the browser disconnects.
                    # Shield cleanup until an in-flight synchronous step ends.
                    with anyio.CancelScope(shield=True):
                        try:
                            await anyio.to_thread.run_sync(iterator.close)
                        finally:
                            inference_lock.release()

        return ChatResponse(iterator, media_type='application/x-ndjson', headers={
            'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no',
        })

    frontend = static_dir or Path(os.environ.get('DRILL_WEB_DIST', 'demo/web/dist'))
    if (frontend / 'index.html').is_file():
        app.mount('/', StaticFiles(directory=frontend, html=True), name='frontend')
    return app


app = create_app()
