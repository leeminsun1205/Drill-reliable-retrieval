"""Single-process API and static frontend for the interactive Kaggle demo."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Callable

import anyio

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from drill_ir.demo.backend import load_backend, normalize_evidence
from drill_ir.demo.scope import DIRECT_REPLIES, Intent, classify_question

from drill_ir.demo.memory import HistoryTurn, MAX_HISTORY_TURNS, resolve_question

from drill_ir.demo.interaction import clarify_question, related_questions, error_message

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10000)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)

    @field_validator('question')
    @classmethod
    def trim_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError('Câu hỏi không được để trống')
        return value


def create_app(*, backend_factory: Callable = load_backend, answer_stream=None,
               static_dir: Path | None = None, scope_classifier: Callable = classify_question,
               question_resolver: Callable = resolve_question,
               clarification_checker: Callable = clarify_question,
               suggestion_generator: Callable = related_questions) -> FastAPI:
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
        try:
            yield
        finally:
            backend = app.state.backend
            app.state.backend = None
            close = getattr(backend, 'close', None)
            if callable(close):
                await run_in_threadpool(close)

    app = FastAPI(title='DRiLL Legal QA', lifespan=lifespan)
    inference_lock = Lock()

    @app.get('/api/health')
    def health():
        ready = app.state.backend is not None
        return {'status': 'ready' if ready else 'unavailable', 'ready': ready}

    @app.post('/api/chat')
    def chat(payload: ChatRequest):
        try:
            intent = Intent(scope_classifier(payload.question, payload.history))
        except Exception as error:
            logger.exception('Demo scope classification failed')
            raise HTTPException(503, error_message(error, 'scope')) from None
        if intent in DIRECT_REPLIES:
            events = [
                {'type': 'delta', 'text': DIRECT_REPLIES[intent]},
                {'type': 'done'},
            ]
            return StreamingResponse(
                iter(json.dumps(event, ensure_ascii=False) + '\n' for event in events),
                media_type='application/x-ndjson',
                headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
            )
        if app.state.backend is None:
            raise HTTPException(503, 'Pipeline chưa sẵn sàng. Kiểm tra log máy chủ.')
        # The shared GPU pipeline is not reentrant. Reject overlap instead of
        # creating an unbounded queue of expensive work on a demo server.
        if not inference_lock.acquire(blocking=False):
            raise HTTPException(429, 'Hệ thống đang xử lý câu hỏi khác. Vui lòng thử lại.')

        def events():
            def event(event_type, **data):
                return json.dumps({'type': event_type, **data}, ensure_ascii=False) + '\n'
            started = monotonic()
            stage = 'context'
            try:
                if payload.history:
                    yield event('status', message='Đang làm rõ câu hỏi theo hội thoại…')
                search_question = question_resolver(payload.question, payload.history)
                yield event('status', message='Đang kiểm tra tình huống…', stage='understanding')
                clarification = clarification_checker(search_question, payload.history)
                if clarification is not None:
                    yield event('delta', text=clarification.question)
                    yield event('suggestions', questions=clarification.options)
                    yield event('done', kind='clarification', elapsed_ms=round((monotonic() - started) * 1000))
                    return
                stage = 'retrieval'
                yield event('status', message='Đang tìm điều luật…', stage='retrieval')
                evidence = [normalize_evidence(item) for item in app.state.backend.search(search_question)]
                yield event('evidence', evidence=[asdict(item) for item in evidence])
                stage = 'answer'
                yield event('status', message='Đang soạn câu trả lời…', stage='answer')
                stream = answer_stream
                if stream is None:
                    from drill_ir.demo.answering import stream_answer
                    stream = stream_answer
                answer_parts = []
                for piece in stream(payload.question, evidence, payload.history):
                    answer_parts.append(piece)
                    yield event('delta', text=piece)
                if evidence:
                    yield event('status', message='Đang gợi ý câu hỏi liên quan…', stage='suggestions')
                    try:
                        suggestions = suggestion_generator(payload.question, ''.join(answer_parts), evidence)
                        yield event('suggestions', questions=suggestions)
                    except Exception:
                        logger.exception('Related question generation failed')
                        yield event('warning', message='Chưa tạo được câu hỏi gợi ý; bạn vẫn có thể hỏi tiếp.')
                yield event('done', kind='answer', elapsed_ms=round((monotonic() - started) * 1000))
            except Exception as error:
                logger.exception('Demo request failed at %s', stage)
                yield event('error', message=error_message(error, stage))

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
