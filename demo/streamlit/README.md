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
0.15/0.65/0.20, pool-50 checkpoint-1234 reranking, the trained 61-feature
LightGBM model, and the fitted Multiclass–Ridge policy. It requires the
complete-results archive from the selected Kaggle submission notebook; it does
not retrain or retune models at startup.

The app reads `OPENAI_API_KEY` and `OPENAI_MODEL` from the process environment.
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
