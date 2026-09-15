"""Document ingestion into Qdrant."""

import os

# Quiet, deterministic CLI output. Must run before sentence_transformers
# (imported by rag.embedding/rag.ingest here, and by agent.retrieval's
# retrieve/rerank modules via `import rag` in agent/retrieval/__init__.py)
# pulls in transformers/huggingface_hub, since
# transformers reads HF_HUB_DISABLE_PROGRESS_BARS once at import time to
# decide whether tqdm is active at all — setting it later has no effect.
# HF_HUB_DISABLE_XET avoids the "unauthenticated requests" notice, which
# the compiled hf_xet fast-download backend prints straight to stderr,
# bypassing Python logging entirely (so no logger/warnings config can
# catch it). setdefault() so a caller who wants the real thing back only
# needs to export the env var themselves before running.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
