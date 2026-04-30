FROM python:3.11-slim

WORKDIR /app

# Install surriti from local source (mounted via build context = repo root).
# In production this would be `pip install surriti` from PyPI; the local
# install here is what makes the dev loop fast.
COPY surriti/ /surriti/
RUN pip install --no-cache-dir /surriti

# App deps -- LLM lives in the vllm sidecar; embeddings run in-process via
# fastembed so we keep one network hop and avoid a second model server.
COPY myapp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the fastembed model at build time so the first request
# doesn't pay the cold-start cost.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('nomic-ai/nomic-embed-text-v1.5')"

COPY myapp/service.py .

ENV PORT=3000
EXPOSE 3000
CMD ["python", "service.py"]
