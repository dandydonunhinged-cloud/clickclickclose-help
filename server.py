"""
RPCCP Platform — API Server
============================
FastAPI wrapper around rpccp.py engine.
Streams pass results via WebSocket. Serves frontend.

Run: uvicorn server:app --host 0.0.0.0 --port 8080 --reload
"""

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Add parent dir so we can import rpccp
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rpccp import RPCCP, OllamaProModel, OllamaModel, OpenRouterModel, PassResult

app = FastAPI(title="RPCCP Platform", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Runs storage — in-memory for MVP, JSON files on disk for persistence
RUNS_DIR = Path(os.environ.get("RPCCP_RUNS_DIR", "D:/DandyDon/rpccp_runs"))
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Active runs being streamed
active_runs: dict = {}


def load_env():
    """Load .env file into os.environ."""
    # Try local .env first, then DandyDon path
    for env_path in [Path(".env"), Path("C:/DandyDon/.env")]:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip()
                        if key and val and not val.startswith("<"):
                            os.environ.setdefault(key, val)
            break


load_env()


def get_cloud_models():
    """Build the cloud model pool from available providers."""
    return {
        1: OllamaProModel("nemotron-3-super"),
        2: OllamaProModel("kimi-k2-thinking"),
        3: OllamaProModel("qwen3.5:397b"),
        4: OllamaProModel("cogito-2.1:671b"),
        5: OllamaProModel("deepseek-v3.2"),
        6: OllamaProModel("nemotron-3-super"),
        7: OllamaProModel("kimi-k2-thinking"),
    }


# ── REST Endpoints ──────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "engine": "RPCCP", "version": "0.1.0"}


@app.get("/api/runs")
async def list_runs():
    """List all past RPCCP runs."""
    runs = []
    for f in sorted(RUNS_DIR.glob("rpccp_*.json"), reverse=True):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
                runs.append({
                    "id": f.stem,
                    "query": data.get("query", ""),
                    "timestamp": data.get("timestamp", ""),
                    "duration": data.get("total_duration", 0),
                    "type2": data.get("type2_detected", False),
                    "real_question": data.get("real_question", "")[:200],
                })
        except Exception:
            continue
    return runs


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """Get full results of a specific run."""
    filepath = RUNS_DIR / f"{run_id}.json"
    if not filepath.exists():
        # Try finding by partial match
        matches = list(RUNS_DIR.glob(f"*{run_id}*.json"))
        if matches:
            filepath = matches[0]
        else:
            return JSONResponse({"error": "Run not found"}, status_code=404)

    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/run")
async def start_run(payload: dict):
    """Start a new RPCCP run. Returns run_id for WebSocket streaming."""
    query = payload.get("query", "").strip()
    if not query:
        return JSONResponse({"error": "Query required"}, status_code=400)

    run_id = str(uuid.uuid4())[:8]
    mode = payload.get("mode", "cloud")

    active_runs[run_id] = {
        "query": query,
        "mode": mode,
        "status": "queued",
        "started": datetime.now(timezone.utc).isoformat(),
    }

    return {"run_id": run_id, "query": query, "mode": mode}


# ── WebSocket for streaming passes ─────────────────────────

@app.websocket("/api/ws/{run_id}")
async def ws_run(websocket: WebSocket, run_id: str):
    """Stream RPCCP passes in real-time via WebSocket."""
    await websocket.accept()

    run_info = active_runs.get(run_id)
    if not run_info:
        await websocket.send_json({"type": "error", "message": "Run not found"})
        await websocket.close()
        return

    query = run_info["query"]
    mode = run_info["mode"]

    try:
        await websocket.send_json({"type": "status", "message": "Starting RPCCP engine..."})

        # Build engine
        if mode == "cloud":
            models = get_cloud_models()
            engine = RPCCP(models=models, max_passes=6, verbose=False)
        else:
            engine = RPCCP(verbose=False)

        # Run passes with streaming
        pass_types = ["naive", "critique", "unconstrained", "expand", "paradigm", "push"]
        history = []

        for pass_num in range(1, 7):
            model = engine._get_model(pass_num)
            system = engine.__class__.__dict__  # Access PASS_SYSTEMS
            from rpccp import PASS_SYSTEMS

            await websocket.send_json({
                "type": "pass_start",
                "pass": pass_num,
                "pass_type": pass_types[pass_num - 1],
                "model": model.name,
            })

            # Run the pass in a thread to not block
            prompt = engine._build_pass_prompt(pass_num, query, history)
            system_prompt = PASS_SYSTEMS[pass_num]

            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None, lambda: model.generate(prompt, system=system_prompt)
            )

            objective, dimensions, analysis = engine._parse_response(raw)

            result = PassResult(
                pass_number=pass_num,
                pass_type=pass_types[pass_num - 1],
                prompt_sent=prompt,
                response=raw,
                model_used=model.name,
                objective_function=objective or f"Pass {pass_num} objective",
                dimensions_considered=dimensions or [],
                duration_seconds=0,
            )
            history.append(result)

            await websocket.send_json({
                "type": "pass_complete",
                "pass": pass_num,
                "pass_type": pass_types[pass_num - 1],
                "model": model.name,
                "objective": result.objective_function,
                "dimensions": result.dimensions_considered[:5],
                "preview": raw[:500],
                "full": raw,
            })

        # Collision
        await websocket.send_json({
            "type": "collision_start",
            "model": engine._get_model(7).name,
        })

        collision_model = engine._get_model(7)
        collision_prompt = engine._build_collision_prompt(query, history)

        loop = asyncio.get_event_loop()
        collision_raw = await loop.run_in_executor(
            None, lambda: collision_model.generate(collision_prompt, system=PASS_SYSTEMS[7])
        )

        collision = engine._parse_collision(collision_raw, history)

        await websocket.send_json({
            "type": "collision_complete",
            "real_question": collision.real_question,
            "type2_detected": collision.type2_detected,
            "objective_evolution": collision.objective_evolution,
            "full": collision_raw,
        })

        # Save run
        from rpccp import RPCCPResult
        rpccp_result = RPCCPResult(
            query=query,
            passes=history,
            collision=collision,
            final_synthesis=collision_raw,
            total_duration=0,
            models_used=list(set(p.model_used for p in history)),
        )
        engine._save_run(rpccp_result)

        await websocket.send_json({"type": "complete", "message": "RPCCP run complete."})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        active_runs.pop(run_id, None)


# ── Static files ────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
