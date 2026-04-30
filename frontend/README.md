# LEGBA CLI

The frontend folder only contains the CLI. Connect directly to the BentoML service:
`ws://<host>:7100/ws/{session_id}`.

## Run outside Docker (talking to local compose ports)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r frontend/requirements.txt
```

Then in another terminal:
```bash
python frontend/cli.py --http http://localhost:7100 --ws ws://localhost:7100 --session dev
```
Type messages; you should see streamed chunks followed by a `[done]` line.

## Notes
- If you want per-turn correlation, store the last `turn_id` from the ack and ignore chunks/done that don’t match.
