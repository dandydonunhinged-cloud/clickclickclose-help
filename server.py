"""
RPCCP Platform — API Server
============================
FastAPI wrapper around rpccp.py engine.
Streams pass results via WebSocket. Serves frontend.

Tiers:
  Free: 1 run/day (no auth required, IP-based tracking)
  Pro: $49/mo — unlimited runs (API key auth)
  Teams: $199/mo — unlimited runs + shared history (API key auth)

Run: uvicorn server:app --host 0.0.0.0 --port 8080 --reload
"""

import asyncio
import hashlib
import json
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Add parent dir so we can import rpccp
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rpccp import RPCCP, OllamaProModel, OllamaModel, OpenRouterModel, PassResult

app = FastAPI(title="RPCCP Platform", version="0.2.0")

# CORS — restrict to our domains in production
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "https://clickclickclose.help,http://localhost:8080").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# Runs storage — JSON files on disk, or DATABASE_URL for Postgres
RUNS_DIR = Path(os.environ.get("RPCCP_RUNS_DIR", "./runs"))
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Active runs being streamed
active_runs: dict = {}

# Rate limiting — in-memory (per-process)
# Key: IP or API key hash → list of timestamps
rate_tracker: dict = defaultdict(list)

# Max query length
MAX_QUERY_LENGTH = 2000


# ── Config ─────────────────────────────────────────────────

def load_env():
    """Load .env file into os.environ."""
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

# API keys for Pro/Teams users — loaded from env or a keys file
# Format: RPCCP_API_KEYS=key1:pro,key2:teams,key3:pro
API_KEYS: dict = {}  # key_hash -> tier


def load_api_keys():
    raw = os.environ.get("RPCCP_API_KEYS", "")
    if not raw:
        return
    for entry in raw.split(","):
        entry = entry.strip()
        if ":" in entry:
            key, tier = entry.rsplit(":", 1)
            key_hash = hashlib.sha256(key.strip().encode()).hexdigest()
            API_KEYS[key_hash] = tier.strip()


load_api_keys()


# ── Auth & Rate Limiting ───────────────────────────────────

def get_client_id(request: Request, authorization: Optional[str] = Header(None)) -> tuple:
    """Returns (client_id, tier). Checks API key first, falls back to IP."""
    if authorization and authorization.startswith("Bearer "):
        key = authorization[7:]
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        tier = API_KEYS.get(key_hash)
        if tier:
            return key_hash, tier
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Free tier — identify by IP
    client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}", "free"


def check_rate_limit(client_id: str, tier: str):
    """Enforce rate limits by tier."""
    now = time.time()

    if tier in ("pro", "teams"):
        # Pro/Teams: 60 runs/hour (generous)
        window = 3600
        max_runs = 60
    else:
        # Free: 1 run/day
        window = 86400
        max_runs = 1

    # Clean old entries
    rate_tracker[client_id] = [t for t in rate_tracker[client_id] if now - t < window]

    if len(rate_tracker[client_id]) >= max_runs:
        if tier == "free":
            raise HTTPException(
                status_code=429,
                detail="Free tier: 1 run per day. Upgrade to Pro at clickclickclose.help for unlimited runs."
            )
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")

    rate_tracker[client_id].append(now)


# ── Model Pool ─────────────────────────────────────────────

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


# ── Input Validation ───────────────────────────────────────

def validate_run_id(run_id: str) -> str:
    """Sanitize run_id to prevent path traversal."""
    # Only allow alphanumeric, hyphens, underscores
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "", run_id)
    if not cleaned or len(cleaned) > 64:
        raise HTTPException(status_code=400, detail="Invalid run ID")
    return cleaned


def validate_query(query: str) -> str:
    """Validate and sanitize query input."""
    query = query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query required")
    if len(query) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Query too long ({len(query)} chars). Max: {MAX_QUERY_LENGTH}"
        )
    return query


# ── REST Endpoints ──────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "engine": "RPCCP", "version": "0.2.0"}


@app.get("/api/runs")
async def list_runs(request: Request, authorization: Optional[str] = Header(None)):
    """List past RPCCP runs. Pro/Teams see their own runs."""
    client_id, tier = get_client_id(request, authorization)

    runs = []
    for f in sorted(RUNS_DIR.glob("rpccp_*.json"), reverse=True):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
                # Free users see public runs only; Pro/Teams see their own
                run_owner = data.get("client_id", "")
                if tier == "free" and run_owner and run_owner != client_id:
                    continue
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
    return runs[:50]  # Cap at 50


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """Get full results of a specific run."""
    run_id = validate_run_id(run_id)
    filepath = RUNS_DIR / f"{run_id}.json"
    if not filepath.exists():
        return JSONResponse({"error": "Run not found"}, status_code=404)

    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/run")
async def start_run(payload: dict, request: Request, authorization: Optional[str] = Header(None)):
    """Start a new RPCCP run. Returns run_id for WebSocket streaming."""
    client_id, tier = get_client_id(request, authorization)
    query = validate_query(payload.get("query", ""))
    check_rate_limit(client_id, tier)

    run_id = str(uuid.uuid4())[:8]
    mode = payload.get("mode", "cloud")

    active_runs[run_id] = {
        "query": query,
        "mode": mode,
        "tier": tier,
        "client_id": client_id,
        "status": "queued",
        "started": datetime.now(timezone.utc).isoformat(),
    }

    return {"run_id": run_id, "query": query, "mode": mode, "tier": tier}


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

            t0 = time.time()
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None, lambda p=prompt, s=system_prompt: model.generate(p, system=s)
            )
            duration = time.time() - t0

            objective, dimensions, analysis = engine._parse_response(raw)

            result = PassResult(
                pass_number=pass_num,
                pass_type=pass_types[pass_num - 1],
                prompt_sent=prompt,
                response=raw,
                model_used=model.name,
                objective_function=objective or f"Pass {pass_num} objective",
                dimensions_considered=dimensions or [],
                duration_seconds=round(duration, 1),
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
                "duration": round(duration, 1),
            })

        # Collision
        await websocket.send_json({
            "type": "collision_start",
            "model": engine._get_model(7).name,
        })

        collision_model = engine._get_model(7)
        collision_prompt = engine._build_collision_prompt(query, history)

        t0 = time.time()
        loop = asyncio.get_event_loop()
        collision_raw = await loop.run_in_executor(
            None, lambda: collision_model.generate(collision_prompt, system=PASS_SYSTEMS[7])
        )
        collision_duration = time.time() - t0

        collision = engine._parse_collision(collision_raw, history)

        await websocket.send_json({
            "type": "collision_complete",
            "real_question": collision.real_question,
            "type2_detected": collision.type2_detected,
            "objective_evolution": collision.objective_evolution,
            "full": collision_raw,
            "duration": round(collision_duration, 1),
        })

        # Save run
        from rpccp import RPCCPResult
        total_duration = sum(p.duration_seconds for p in history) + collision_duration
        rpccp_result = RPCCPResult(
            query=query,
            passes=history,
            collision=collision,
            final_synthesis=collision_raw,
            total_duration=round(total_duration, 1),
            models_used=list(set(p.model_used for p in history)),
        )
        engine._save_run(rpccp_result)

        await websocket.send_json({
            "type": "complete",
            "message": "RPCCP run complete.",
            "total_duration": round(total_duration, 1),
        })

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
