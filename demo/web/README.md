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
- Retrieved sources are collapsed by default; click the source heading to
  expand them so the chat answer remains the focus.
- Markdown answers stream into the UI. Raw HTML is not rendered. Previous turns
  and their evidence are saved locally on this browser origin. Only the five
  latest successful question/answer pairs are sent as model context.
- Only the question, bounded conversation context and selected evidence reach the OpenAI-compatible endpoint;
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

## Scope of questions

The FastAPI demo only answers Vietnamese legal questions. Greetings and
questions such as “Bạn là ai?” receive a fixed introduction. Other questions
are classified using the same configured answer API/model before retrieval:
legal questions proceed; unrelated questions receive a fixed scope refusal;
unclear messages receive a request for clarification. Direct replies do not
retrieve evidence or call answer generation. This adds one API call to all
questions except exact common small-talk phrases. Failed/invalid
classification returns HTTP 503, without falling through to retrieval.

The routing tests use injected responses, not real classification accuracy
measurements. Semantic errors and prompt attacks remain possible; validate on
representative questions before presenting this as a reliable scope filter.
No retrieval evaluation pipeline is filtered by this demo-only router.

## Five-turn conversation memory (2026-11-08)

The React app sends the five most recent successful question/answer pairs in
`POST /api/chat` as `history: [{"question": "...", "answer": "..."}]`. Failed or
incomplete responses are excluded. Older turns remain visible but are not sent
as context. Each stored answer is truncated to 6,000 characters for context;
the full answer remains visible. The API rejects more than five pairs, questions
over 10,000 characters, and historical answers over 6,000 characters.

Scope classification sees the bounded history so it can interpret follow-ups.
For legal questions with history, one additional non-streaming call to the same
configured model rewrites the latest question into a standalone retrieval query.
For example, “Còn ô tô?” can inherit the prior red-light violation topic. The
rewritten query is used for retrieval and reranking. Answer generation sees
the original question, recent conversation and newly retrieved evidence.
History is explicitly not legal evidence: old claims/citations must not replace
sources returned for the current question. No-history queries keep their original
retrieval text. This changes only interactive demo behavior, not research runs.

Conversation history is saved in browser localStorage (up to 30 conversations),
not a shared server database. Refresh restores saved conversations; selecting
“Cuộc trò chuyện mới” starts a separate conversation with empty model context.
Each request is self-contained, so different browser sessions do not inherit
one another's history. Existing clients may omit `history` (default empty).

Context increases input tokens; legal questions with history add one model call
for query resolution. Live rewriting accuracy, provider behavior and retrieval
quality are not measured by the mocked contract tests. The legacy Streamlit UI
continues to call the answer layer without conversation history.

## Interactive workspace (2026-11-08)

- Sidebar: create, switch, rename, and delete conversations. Delete requires a
  local confirmation; export first if needed. Each conversation has its own
  five-turn model context. Renaming does not change that context.
- Browser storage: full messages/evidence are saved on this origin, not uploaded
  as a server-side session. Interrupted streams restore as failed turns and are
  excluded from context. Storage failure shows a notice and leaves chat usable.
  Changing the Ngrok hostname creates a different origin with separate storage.
  Shared-device users should delete private chats when finished.
- Welcome cards: military service, labor, land and family examples, without a
  corpus-coverage claim.
- Answer controls: copy, export the entire active conversation with original
  evidence as UTF-8 Markdown, retry a failed turn, or edit an earlier question.
  Retrying/editing replaces that turn and drops later turns before constructing
  context; canceled editing changes nothing.
- Source headings show the ranking score beside `aid`, before the article text.
  Opening the source list starts at the top; selecting a citation starts at that
  article heading. Source scrolling is independent of chat and is not reset
  during answer updates.
- Citation buttons: only retrieved `aid` values become interactive. Selecting a
  citation opens and highlights its source in a side panel (overlay on smaller
  screens). Sources remain closed by default and also open via the source button.
- Progress: contextual understanding, retrieval, answer and suggestions, with
  elapsed time. Scrolling up disables automatic scrolling; a jump button restores
  it. Source text includes no automatic claim of current legal validity.

### Clarification and three related questions

After scope classification and optional query resolution, an independent model
call checks whether a missing detail materially changes the answer. It can ask a
short clarification with up to three full-question choices, without running
retrieval or making a legal conclusion. Known context must not be asked again;
users requesting general information should be allowed a general answer. Choices
are sent as ordinary new questions with the same conversation context.

After a legal answer with evidence, another call requests exactly three distinct
related questions as JSON, validated for count, type, uniqueness and length (up
to 300 characters each). They are displayed below the answer and clicking one
submits it immediately. Suggestions are not legal facts and may need their own
retrieval. Invalid/unavailable suggestions emit a warning and do not discard the
answer. Invalid clarification formatting now produces a fixed request for
more detail; authentication/network failures still emit an error. Neither
case silently proceeds to retrieval or guesses missing details.

Both helpers use the existing configured OPENAI_MODEL/endpoint (no substitution),
with 20-second timeouts and no retries. Set `DRILL_DEMO_CLARIFICATION=0` or
`DRILL_DEMO_SUGGESTIONS=0` in the server environment to disable them independently.
By default both are on. Legal turns add one clarification call; completed answers
with sources add one suggestion call. History-based query resolution still adds
its separate call. This adds latency and provider cost. The API sends `status`,
`evidence`, `delta`, `suggestions` (`questions` array), optional `warning`, and
`done` events; `done.kind` identifies `answer` or `clarification`. Existing
`error` events and HTTP errors now distinguish recognized model, auth, quota and
timeout failures without sending provider bodies or internal paths to the UI.

### Verification

```bash
python -m pytest tests/test_demo_api.py tests/test_demo_scope.py tests/test_demo_memory.py tests/test_demo_interaction.py
npm --prefix demo/web test
npm --prefix demo/web run build
```

Optional browser integration checks use the pinned Playwright dev dependency:

```bash
cd demo/web
npx playwright install chromium --only-shell
npm run test:browser
```

The browser checks start a temporary localhost static server and mock API replies;
they do not call a model or external legal service. Screenshots go to ignored
`results/demo_ui_checks/`. Semantic quality of related questions/clarifications,
real provider JSON compatibility and live Kaggle inference remain unmeasured.

Verified locally: 40 Python demo tests, five frontend state tests, production
build, and Chromium desktop/mobile interaction checks all passed with mocked
API/model responses. The existing Starlette/httpx test deprecation warning
remains. No live model/provider or Kaggle GPU run was performed for this update.

### Provider JSON compatibility fix (2026-11-10)

Interaction JSON accepts a single valid object with an optional BOM, code fence
or prose wrapper. Empty, malformed, ambiguous/multiple objects and invalid
schemas remain unusable; the parser never repairs JSON syntax or logs raw
provider content. For clarification only, invalid format/schema returns a
fixed request for details with no retrieval instead of crashing the turn.
This is a deliberate refinement of the earlier fail-with-error behavior.
Provider auth/network failures are not turned into requests for user details.
Related-question format failure still retains the answer and emits a warning.
No extra model call, retry or provider-specific JSON-mode option was added.

### Responsive controls during streaming (2026-11-10)

A real local HTTP regression reproduced controls remaining locked when a stream
sent `done` but did not close the HTTP body. The frontend now completes at the
protocol `done` event, cancels its reader/connection and releases request state
without waiting for EOF. A 90-second no-data watchdog covers both response
headers and streaming reads; progress/data reset that timer. It is an idle
limit, not a total answer-duration limit.

The composer exposes “Dừng” while a request runs. Browsing/creating/renaming
conversations, opening/closing sources and opening delete confirmation remain
available during requests. There is still at most one submitted chat request
per browser tab. Deleting the running conversation aborts its client request;
responses update the originating conversation by ID even after navigation.
Stopped/timeout turns retain partial text but are excluded from model context.
Closing a client connection cannot immediately interrupt an already-running
server GPU operation; a new request can still receive 429 until it finishes.

Modal dialogs support their close button, Escape and outside click. Citation
renderers keep a stable React component identity during streaming. History is
saved after a 750-ms quiet period while streaming and flushed at completion or
pagehide, instead of serializing the entire history on each chunk.

`tests/interactions.mjs` uses a local HTTP server, not an AI provider. It checks
an open body after done, stalled responses, navigation/rename/delete/source
controls while busy, stop, deleting a running conversation, bounded storage
writes and the idle timeout. It is included in `npm run test:browser`. Both
this suite and the existing desktop/mobile browser suite passed locally,
along with the TypeScript/Vite build. No live Kaggle timing was measured.

### GPU memory placement (2026-11-11)

Dense encoders and all FAISS search indexes are explicitly placed on CPU. After
first retrieval, before the reranker is loaded, devices are chosen from a free
VRAM snapshot: at least 6 GiB free per participating device, prefer two eligible
GPUs with the freer one as primary, otherwise one GPU, otherwise CPU. The
threshold is operational headroom, not a measured guarantee. The model,
checkpoint, candidate pool and batch size 8 are preserved. CPU fallback can be
slower and CPU indexes require RAM. No live Kaggle memory/latency validation
has been performed. Other processes allocating memory after selection can
still cause OOM.

After pulling the backend fix, stop the previous API process and rerun the
launch cell (browser refresh alone cannot release its GPU allocations). The
first legal query logs `demo_reranker_placement_after_retrieval` with free GiB
and selected_device. For [3.52,14.46] GiB free it chooses cuda:1. No frontend
rebuild is needed for this backend-only change. FastAPI invokes backend.close
on shutdown. The local test suite is:

```bash
python -m pytest tests/test_demo_runtime.py tests/test_demo_api.py tests/test_demo_scope.py tests/test_demo_memory.py tests/test_demo_interaction.py
```

78 tests passed with mocked models/devices. A tiny real FAISS CPU index also
passed a local allocation check; full-corpus CPU RAM usage remains unmeasured.
