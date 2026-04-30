FROM python:3.11-slim

WORKDIR /app

# Install surriti from local source (mounted via build context = repo root)
COPY surriti/ /surriti/
RUN pip install --no-cache-dir /surriti

# App deps (lightweight: no torch/transformers -- LLM lives in ollama service)
COPY myapp/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY myapp/service.py .

ENV PORT=3000
EXPOSE 3000
CMD ["python", "service.py"]
