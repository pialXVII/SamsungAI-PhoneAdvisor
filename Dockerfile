# syntax=docker/dockerfile:1
#
# Samsung Phone Advisor: CPU image for local Docker and Hugging Face Spaces.
#
#   docker build -t phone-advisor .
#   docker run --rm -p 8000:7860 phone-advisor        # http://127.0.0.1:8000
#
# The defaults suit a small free-tier host: SQLite seeded from the committed
# snapshot, template answers (USE_LLM=false) and the native agent orchestrator.
# To enable the generative model at run time:
#
#   docker run --rm -p 8000:7860 -e USE_LLM=true phone-advisor
#
# The first start then downloads Qwen2.5-1.5B (~3 GB) into the HF cache and
# generates on CPU: expect seconds per chat answer and minutes per review.
# For CrewAI crews as well, build with --build-arg WITH_CREWAI=true and run with
# -e USE_LLM=true -e AGENT_FRAMEWORK=auto.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU-only torch goes in first. The default PyPI wheel bundles the CUDA
# runtime, which adds gigabytes that a CPU host such as a free Space cannot use.
# Installing it before the requirements means sentence-transformers sees its
# torch dependency already satisfied and does not pull the CUDA build.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --index-url "${TORCH_INDEX_URL}" torch

COPY requirements-ci.txt /tmp/requirements-ci.txt
RUN pip install -r /tmp/requirements-ci.txt

# CrewAI is opt-in. It adds ~70 packages and only runs when an LLM is loaded,
# which the default USE_LLM=false never does (see requirements-ci.txt).
ARG WITH_CREWAI=false
RUN if [ "${WITH_CREWAI}" = "true" ]; then pip install "crewai>=1.0.0"; fi

# Hugging Face Spaces runs containers as uid 1000. Building as that same user
# means file ownership and cache paths are identical locally and on Spaces.
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# Bake the embedding model into the image so a cold start does not wait on a
# download. This layer sits above the source COPY, so it survives code edits.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}', device='cpu')"

# Generated state (SQLite file, FAISS index) lives outside the source tree, so
# it can be a volume. Created here so a fresh named volume inherits uid 1000
# ownership instead of root's.
RUN mkdir -p /home/user/state

COPY --chown=user . /home/user/app

ENV DB_BACKEND=sqlite \
    SQLITE_PATH=/home/user/state/samsung_phones.db \
    VECTOR_INDEX_PATH=/home/user/state/vector_index \
    USE_LLM=false \
    AGENT_FRAMEWORK=native \
    PORT=7860

EXPOSE 7860

# "healthy" means the database has phones and the vector index is loaded. A 200
# alone is not enough, because /health also answers 200 while degraded. The
# start period covers seeding plus the embedding warm-up on a slow CPU.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "import json,os,urllib.request; r=urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','7860'), timeout=8); raise SystemExit(0 if json.load(r).get('status')=='healthy' else 1)"

# Seeds the database from data/scraped_phones.json when empty, builds the
# index, then runs uvicorn on 0.0.0.0:$PORT in the same process.
CMD ["python", "scripts/bootstrap.py", "--serve"]
