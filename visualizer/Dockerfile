FROM python:3.12-slim

WORKDIR /app

COPY visualizer/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY visualizer/server.py /app/server.py
COPY visualizer/static /app/static

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
