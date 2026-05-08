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
SURRITI_SURREAL_USER=root        # ⚠ dev/local default — change in production
SURRITI_SURREAL_PASS=root        # ⚠ dev/local default — change in production
VISUALIZER_PORT=8080
SURRITI_API_KEY=change-me        # required for non-loopback API access
SURRITI_ALLOW_INSECURE_LOCAL=1   # default; local loopback can skip API key
```

> **Warning:** `root`/`root` credentials are provided as a convenience for local development only.
> Set `SURRITI_SURREAL_USER` and `SURRITI_SURREAL_PASS` to strong credentials before any non-local deployment.
>
> Visualizer API routes (`/api/*`) are now protected. For non-loopback access, provide
> `X-API-Key: $SURRITI_API_KEY` (or `Authorization: Bearer $SURRITI_API_KEY`).

If you use the top-level `docker-compose.yml`, the likely local values are:

```ini
SURRITI_SURREAL_URL=ws://localhost:8000/rpc
SURRITI_SURREAL_NS=myapp
SURRITI_SURREAL_DB=myapp
SURRITI_SURREAL_USER=root        # ⚠ dev/local default — change in production
SURRITI_SURREAL_PASS=root        # ⚠ dev/local default — change in production
```
