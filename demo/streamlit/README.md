> Legacy UI: the default demo is now [FastAPI + React](../web/README.md).
> Install the `streamlit` extra to run this interface. Shared Python modules
> live in `src/drill_ir/demo/`; local imports here preserve compatibility.

# Streamlit legal QA demo

This folder contains the user interface and OpenAI answer-generation layer for
the Kaggle legal-retrieval demo. `pipeline_adapter.py` implements the
`search(question)` adapter using the repository's retrieval, reranking,
LightGBM, and cardinality-selection code. The Kaggle launcher selects it with
`DRILL_RETRIEVER_FACTORY=pipeline_adapter:create_backend`.

The factory must return an object with `search(question)` and return the final
ranked evidence as `Evidence` records or dictionaries containing `aid` and
`text` (optionally `law_id`, `title`, and `score`). The adapter preserves fixed
windows 448/64, reference-off WRRF10 BM25/Harrier/E5 weights
0.15/0.65/0.20, pool-50 checkpoint-1234 reranking, the trained LightGBM model (57 reference-free features for new runs; legacy
61-column artifacts remain supported with their exact saved feature order), and the fitted Multiclass–Ridge policy. It requires the
complete-results archive from the selected Kaggle submission notebook; it does
not retrain or retune models at startup.

For the interactive process, dense encoders and FAISS indexes run on CPU.
After first retrieval, the shared adapter chooses reranking devices by free
VRAM (at least 6 GiB per device): two eligible GPUs, otherwise one, otherwise
CPU. Batch size remains 8. Device choice and CUDA snapshots are logged; the
threshold is operational headroom, not a measured peak-memory guarantee.
See `../web/README.md` for the current runtime behavior and limitations.

The app reads `OPENAI_API_KEY`, `OPENAI_MODEL`, and optional `OPENAI_BASE_URL`
from the process environment and streams Chat Completions text into the chat as
it arrives. Keep API keys in Kaggle Secrets; never place them in the notebook
source. The configured OpenAI-compatible endpoint must support streaming.
The chat input remains available for consecutive questions, and the current
session keeps the previous turns and their retrieved evidence visible.
The Kaggle notebook should populate the API key from Kaggle Secrets and start
Streamlit in a subprocess. Ngrok is started by the notebook, not by this app.

Run from this directory after setting the retrieval factory and OpenAI
environment variables:

```bash
streamlit run app.py --server.port 8501 --server.headless true
```

The OpenAI response is grounded in retrieved article text; selected article IDs
and text are displayed separately as evidence. The API receives only the
question and selected evidence, never the full corpus.
