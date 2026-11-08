# FastAPI + React legal QA demo

The default demo uses React/TypeScript/Vite for the interface and FastAPI for
retrieval and streamed answers. Reusable Python code lives in
`src/drill_ir/demo/`; the former Streamlit modules are compatibility imports.
The frozen retrieval adapter and saved artifact contract are preserved,
including the existing 57/legacy-61-column LightGBM compatibility.

## Kaggle

Use `notebooks/demo/kaggle_fastapi_legal_qa.ipynb`. Attach the same corpus,
Harrier/E5 models, checkpoint-1234, complete-results archive and optional
retrieval caches as the former notebook. Enable Internet and T4 GPUs. Configure
`GITHUB_TOKEN`, `OPENAI_API_KEY`, `NGROK_AUTH_TOKEN` in Kaggle Secrets, and
`OPENAI_MODEL`/optional `OPENAI_BASE_URL` in notebook configuration.

The notebook installs Python dependencies, runs `npm ci` and `npm run build`,
checks artifacts, starts one Uvicorn worker on port 8501, waits for backend
readiness, then opens one Ngrok tunnel. Node.js >=20.19 or >=22.12 with npm is
used when available; otherwise the notebook downloads pinned Node.js 22.16.0
for Kaggle Linux x86_64 and verifies its archive against the official SHA-256
manifest before extraction. Internet access to nodejs.org is then required.
No separate Node server or frontend host is needed. Logs are written to
`/kaggle/working/legal_qa_api.log`. Startup can take many minutes without caches;
the launcher allows up to 30 minutes. Rerunning the launch cell stops its previous
process. Stop with the final notebook cell.

The session and tunnel must stay alive. This is a shared demo, not permanent
hosting. It has no login or persistent chat database. Anyone with the tunnel
URL can submit questions that use the configured answer API. Keep the URL for
the intended demo audience. One question runs at a time across clients; overlap
returns HTTP 429 to avoid concurrent inference on the shared GPU models.

## Local development

From the repository root, with Python environment and model paths configured:

```bash
python -m pip install -e '.[dense,vietnamese,ltr,demo]'
export DRILL_RETRIEVER_FACTORY=drill_ir.demo.pipeline_adapter:create_backend
# Set OPENAI_API_KEY, OPENAI_MODEL and optional OPENAI_BASE_URL in your shell.
# Set DRILL_CORPUS_PATH, DRILL_CHECKPOINT_PATH, DRILL_HARRIER_PATH,
# DRILL_E5_PATH and DRILL_COMPLETE_RESULTS_ARCHIVE for local attached artifacts.
npm --prefix demo/web ci
npm --prefix demo/web run build
python -m uvicorn drill_ir.demo.api:app --host 127.0.0.1 --port 8501 --workers 1
```

Open `http://127.0.0.1:8501`. For frontend development, run
`npm --prefix demo/web run dev` in a second terminal; Vite proxies `/api` to
port 8501. Do not use multiple Uvicorn workers: each would load its own models.
`DRILL_WEB_DIST` optionally selects the built frontend directory; the default
`demo/web/dist` is relative to the server working directory (repository root).
The retrieval adapter retains its existing Kaggle working/cache paths.

## API and retained behavior

- `GET /api/health`: `ready` and `status`; startup failure leaves the UI available.
- `POST /api/chat`: JSON `{"question":"..."}` (trimmed, 1–10,000 characters).
  Returns UTF-8 newline-delimited JSON events: `status`, `evidence`, `delta`,
  `done`; failures after streaming starts emit `error` instead of `done`.
  HTTP 422/429/503 describe invalid input, a busy server, or unavailable backend.
- Each evidence record preserves `aid`, `law_id`, `title`, `text`, `score`.
  Evidence is sent before answer generation so it remains visible on LLM errors.
- Markdown answers stream into the UI. Raw HTML is not rendered. Previous turns
  and their evidence remain in the current browser tab until refresh; they are
  not sent as conversational context to the model, matching the original demo.
- Only the question and selected evidence reach the OpenAI-compatible endpoint;
  keys stay in the backend. Internal errors are logged on the server.

No retrieval training, tuning, evaluation or performance measurement is part of
this UI migration. Contract tests inject a fake retriever and answer stream:

```bash
python -m pip install -e '.[demo,dev]' httpx
python -m pytest tests/test_demo_api.py
```

Local migration checks passed: four API contract tests, TypeScript checking,
Vite production build, and a built-static/API smoke check using an injected
backend. Full Kaggle GPU startup, Node bootstrap download, Ngrok exposure, and
real answer API streaming have not been executed in this local verification.
