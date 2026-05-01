# Surriti Visualizer

Interactive D3 graph visualizer for the SurrealDB-backed Surriti graph.

## Run

From the repository root:

```bash
python3 -m pip install -r visualizer/requirements.txt
python3 visualizer/server.py
```

Open http://localhost:8080.

## Configuration

The server reads the same SurrealDB environment variables as Surriti:

```ini
SURRITI_SURREAL_URL=ws://localhost:8000/rpc
SURRITI_SURREAL_NS=myapp
SURRITI_SURREAL_DB=myapp
SURRITI_SURREAL_USER=root
SURRITI_SURREAL_PASS=root
VISUALIZER_PORT=8080
```

If you use the top-level `docker-compose.yml`, the likely local values are:

```ini
SURRITI_SURREAL_URL=ws://localhost:8000/rpc
SURRITI_SURREAL_NS=myapp
SURRITI_SURREAL_DB=myapp
SURRITI_SURREAL_USER=root
SURRITI_SURREAL_PASS=root
```
