"""
RPCCP Platform — API Server
============================
FastAPI wrapper around rpccp.py engine.
Streams pass results via WebSocket. Serves frontend.

Tiers:
  Free: 1 run/day (no auth required, IP-based tracking)
  Pro: $49/mo — unlimited runs (API key auth)
  Teams: $199/mo — unlimited runs + shared history (API key auth)
  Campaign (State House): $199/mo
  Campaign (State Senate): $299/mo
  Campaign (US House): $499/mo
  Campaign (US Senate): $999/mo

Run: uvicorn server:app --host 0.0.0.0 --port 8080
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sqlite3
import stripe
import bcrypt
import jwt as pyjwt
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# Add parent dir so we can import rpccp
# Prefer canonical C:/DandyDon/rpccp.py (local dev). Falls back to local copy (Render).
_canonical=Path(__file__).resolve().parent.parent
_local=Path(__file__).resolve().parent
sys.path.insert(0,str(_canonical if (_canonical/"rpccp.py").exists() else _local))
from rpccp import RPCCP, OllamaProModel, OllamaModel, OpenRouterModel, PassResult

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RPCCP Platform", version="0.3.0")

# ── Config — environment variables only, no hardcoded .env paths ──────────

def load_env():
    """Load .env file from current directory into os.environ (local dev only)."""
    env_path = Path(".env")
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


load_env()

# ── CORS — production only by default ────────────────────────────────────

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "https://clickclickclose.help"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── Security Headers Middleware ───────────────────────────────────────────

@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Add security headers to all HTTP responses."""
    response = await call_next(request)
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "connect-src 'self' wss://*.clickclickclose.help wss://clickclickclose.help"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    return response


# ── HTTPS Redirect Middleware ─────────────────────────────────────────────

@app.middleware("http")
async def https_redirect(request: Request, call_next):
    """Redirect HTTP to HTTPS in production (behind reverse proxy)."""
    proto = request.headers.get("x-forwarded-proto", "")
    if proto == "http":
        url = request.url.replace(scheme="https")
        return RedirectResponse(url=str(url), status_code=301)
    return await call_next(request)


# ── CSRF Protection Middleware ────────────────────────────────────────────

@app.middleware("http")
async def csrf_protection(request: Request, call_next):
    """Validate Origin header on state-changing requests."""
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        origin = request.headers.get("origin", "")
        if origin and origin not in ALLOWED_ORIGINS:
            return JSONResponse(
                {"error": "Origin not allowed"},
                status_code=403,
            )
    return await call_next(request)


# ── Runs storage ─────────────────────────────────────────────────────────

RUNS_DIR = Path(os.environ.get("RPCCP_RUNS_DIR", "./runs"))
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Active runs being streamed
active_runs: dict = {}

# Active WebSocket connections per IP
ws_connections: dict = defaultdict(int)  # ip -> count
MAX_WS_PER_IP = 5

# Concurrent run limits
MAX_GLOBAL_RUNS = 10
MAX_RUNS_PER_CLIENT = 2
active_run_clients: dict = defaultdict(int)  # client_id -> count

# Total run timeout (seconds)
RUN_TIMEOUT = 600  # 10 minutes

# Rate limiting — persisted to JSON file
RATE_FILE = Path(os.environ.get("RPCCP_RATE_FILE", "./rate_limits.json"))
rate_tracker: dict = defaultdict(list)

# Max query length
MAX_QUERY_LENGTH = 2000

# Valid modes
VALID_MODES = {"cloud", "local"}


def _load_rate_limits():
    """Load persisted rate limits from disk."""
    global rate_tracker
    try:
        if RATE_FILE.exists():
            with open(RATE_FILE, encoding="utf-8") as f:
                data = json.load(f)
                rate_tracker = defaultdict(list, {k: v for k, v in data.items()})
    except Exception:
        logger.exception("Failed to load rate limits")
        rate_tracker = defaultdict(list)


def _save_rate_limits():
    """Persist rate limits to disk."""
    try:
        # Clean expired entries before saving
        now = time.time()
        cleaned = {}
        for k, timestamps in rate_tracker.items():
            valid = [t for t in timestamps if now - t < 86400]
            if valid:
                cleaned[k] = valid
        with open(RATE_FILE, "w", encoding="utf-8") as f:
            json.dump(cleaned, f)
    except Exception:
        logger.exception("Failed to save rate limits")


_load_rate_limits()


# ── Stripe Config ─────────────────────────────────────────────────────────

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

# Subscriber storage (JSON file — upgrade to DB later)
SUBSCRIBERS_FILE = Path(os.environ.get("RPCCP_SUBSCRIBERS_FILE", "./subscribers.json"))


def _load_subscribers() -> dict:
    """Load subscriber data from disk."""
    try:
        if SUBSCRIBERS_FILE.exists():
            with open(SUBSCRIBERS_FILE, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        logger.exception("Failed to load subscribers")
    return {}


def _save_subscribers(data: dict):
    """Save subscriber data to disk."""
    try:
        with open(SUBSCRIBERS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        try:
            os.chmod(SUBSCRIBERS_FILE, 0o600)
        except OSError:
            pass
    except Exception:
        logger.exception("Failed to save subscribers")


# Stripe price mapping — tier name -> Stripe Price ID (set via env vars)
STRIPE_PRICES = {
    "pro": os.environ.get("STRIPE_PRICE_PRO", ""),
    "teams": os.environ.get("STRIPE_PRICE_TEAMS", ""),
    "campaign_state_house": os.environ.get("STRIPE_PRICE_CAMPAIGN_STATE_HOUSE", ""),
    "campaign_state_senate": os.environ.get("STRIPE_PRICE_CAMPAIGN_STATE_SENATE", ""),
    "campaign_us_house": os.environ.get("STRIPE_PRICE_US_HOUSE", ""),
    "campaign_us_senate": os.environ.get("STRIPE_PRICE_US_SENATE", ""),
}

# Tier display info
TIER_INFO = {
    "pro": {"name": "Pro", "price": 49, "features": "Unlimited runs, API key access"},
    "teams": {"name": "Teams", "price": 199, "features": "Unlimited runs, shared history, multiple API keys"},
    "campaign_state_house": {"name": "Campaign (State House)", "price": 199, "features": "RPCCP-powered campaign analysis"},
    "campaign_state_senate": {"name": "Campaign (State Senate)", "price": 299, "features": "RPCCP-powered campaign analysis"},
    "campaign_us_house": {"name": "Campaign (US House)", "price": 499, "features": "RPCCP-powered campaign analysis"},
    "campaign_us_senate": {"name": "Campaign (US Senate)", "price": 999, "features": "RPCCP-powered campaign analysis"},
}


# ── API keys for Pro/Teams users ─────────────────────────────────────────
# Format: RPCCP_API_KEYS=key1:pro,key2:teams,key3:pro
API_KEYS: dict = {}  # key_hash -> tier


def load_api_keys():
    """Load API keys from env and subscribers file."""
    raw = os.environ.get("RPCCP_API_KEYS", "")
    if raw:
        for entry in raw.split(","):
            entry = entry.strip()
            if ":" in entry:
                key, tier = entry.rsplit(":", 1)
                key_hash = hashlib.sha256(key.strip().encode()).hexdigest()
                API_KEYS[key_hash] = tier.strip()

    # Also load subscriber-generated keys
    subscribers = _load_subscribers()
    for email, sub_data in subscribers.items():
        api_key = sub_data.get("api_key", "")
        tier = sub_data.get("tier", "")
        status = sub_data.get("status", "")
        if api_key and tier and status == "active":
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            API_KEYS[key_hash] = tier


load_api_keys()


# ── Auth & Rate Limiting ─────────────────────────────────────────────────

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

    if tier in ("pro", "teams", "campaign_state_house", "campaign_state_senate", "campaign_us_house", "campaign_us_senate"):
        # Paid tiers: 60 runs/hour
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
                detail="Free tier: 1 run per day. Upgrade to Pro for unlimited runs."
            )
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")

    rate_tracker[client_id].append(now)
    _save_rate_limits()


def check_concurrent_limits(client_id: str):
    """Enforce concurrent run limits."""
    if len(active_runs) >= MAX_GLOBAL_RUNS:
        raise HTTPException(
            status_code=429,
            detail="Server is at capacity. Please try again in a few minutes."
        )
    if active_run_clients[client_id] >= MAX_RUNS_PER_CLIENT:
        raise HTTPException(
            status_code=429,
            detail="Maximum concurrent runs reached. Please wait for current run to complete."
        )


# ── Model Pool ───────────────────────────────────────────────────────────

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


# ── Input Validation ─────────────────────────────────────────────────────

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


# ── REST Endpoints ───────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "engine": "RPCCP", "version": "0.3.0"}


@app.get("/api/runs")
async def list_runs(request: Request, authorization: Optional[str] = Header(None)):
    """List past RPCCP runs. Scoped to authenticated user's own runs only."""
    client_id, tier = get_client_id(request, authorization)

    runs = []
    for f in sorted(RUNS_DIR.glob("rpccp_*.json"), reverse=True):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
                # Only show runs owned by this client
                run_owner = data.get("client_id", "")
                if not run_owner or run_owner != client_id:
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
async def get_run(run_id: str, request: Request, authorization: Optional[str] = Header(None)):
    """Get full results of a specific run. Requires ownership."""
    client_id, tier = get_client_id(request, authorization)
    run_id = validate_run_id(run_id)
    filepath = RUNS_DIR / f"{run_id}.json"
    if not filepath.exists():
        return JSONResponse({"error": "Run not found"}, status_code=404)

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    # Verify ownership
    run_owner = data.get("client_id", "")
    if not run_owner or run_owner != client_id:
        return JSONResponse({"error": "Run not found"}, status_code=404)

    return data


@app.post("/api/run")
async def start_run(payload: dict, request: Request, authorization: Optional[str] = Header(None)):
    """Start a new RPCCP run. Returns run_id and run_token for WebSocket auth."""
    client_id, tier = get_client_id(request, authorization)
    query = validate_query(payload.get("query", ""))

    # Validate mode
    mode = payload.get("mode", "cloud")
    if mode not in VALID_MODES:
        raise HTTPException(status_code=400, detail="Invalid mode")

    check_rate_limit(client_id, tier)
    check_concurrent_limits(client_id)

    # Full UUID for run IDs — not truncated
    run_id = str(uuid.uuid4())
    # Per-run secret token for WebSocket auth
    run_token = secrets.token_urlsafe(32)

    active_runs[run_id] = {
        "query": query,
        "mode": mode,
        "tier": tier,
        "client_id": client_id,
        "run_token": run_token,
        "status": "queued",
        "started": datetime.now(timezone.utc).isoformat(),
    }
    active_run_clients[client_id] += 1

    return {"run_id": run_id, "run_token": run_token, "query": query, "mode": mode, "tier": tier}


# ── Stripe Endpoints ─────────────────────────────────────────────────────

@app.post("/api/checkout")
async def create_checkout(payload: dict, request: Request):
    """Create a Stripe Checkout Session. Returns the checkout URL."""
    tier = payload.get("tier", "")
    if tier not in STRIPE_PRICES:
        raise HTTPException(status_code=400, detail="Invalid tier")

    price_id = STRIPE_PRICES[tier]
    if not price_id:
        raise HTTPException(status_code=500, detail="Pricing not configured for this tier")

    if not stripe.api_key:
        raise HTTPException(status_code=500, detail="Payment system not configured")

    success_url = payload.get("success_url", "https://clickclickclose.help/?payment=success")
    cancel_url = payload.get("cancel_url", "https://clickclickclose.help/?payment=cancel")

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=success_url + "&session_id={CHECKOUT_SESSION_ID}",
            cancel_url=cancel_url,
            metadata={"tier": tier},
        )
        return {"url": session.url, "session_id": session.id}
    except Exception:
        logger.exception("Stripe checkout session creation failed")
        raise HTTPException(status_code=500, detail="Failed to create checkout session")


@app.post("/api/webhook")
async def stripe_webhook(request: Request):
    """Stripe webhook handler. Provisions API key on successful payment."""
    payload_bytes = await request.body()

    if not STRIPE_WEBHOOK_SECRET:
        logger.error("STRIPE_WEBHOOK_SECRET not configured")
        raise HTTPException(status_code=500, detail="Webhook not configured")

    sig_header = request.headers.get("stripe-signature", "")
    try:
        event = stripe.Webhook.construct_event(
            payload_bytes, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception:
        logger.exception("Stripe webhook verification failed")
        raise HTTPException(status_code=400, detail="Webhook verification failed")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        customer_email = session.get("customer_details", {}).get("email", "")
        tier = session.get("metadata", {}).get("tier", "pro")
        stripe_customer_id = session.get("customer", "")
        subscription_id = session.get("subscription", "")

        # Generate a unique API key for this subscriber
        new_api_key = f"rpccp_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(new_api_key.encode()).hexdigest()

        # Store subscriber data
        subscribers = _load_subscribers()
        subscribers[customer_email] = {
            "api_key": new_api_key,
            "tier": tier,
            "status": "active",
            "stripe_customer_id": stripe_customer_id,
            "subscription_id": subscription_id,
            "created": datetime.now(timezone.utc).isoformat(),
        }
        _save_subscribers(subscribers)

        # Add to active API keys
        API_KEYS[key_hash] = tier
        logger.info(f"Provisioned {tier} API key for {customer_email}")

    elif event["type"] == "customer.subscription.deleted":
        subscription = event["data"]["object"]
        subscription_id = subscription.get("id", "")

        # Deactivate the subscriber
        subscribers = _load_subscribers()
        for email, sub_data in subscribers.items():
            if sub_data.get("subscription_id") == subscription_id:
                sub_data["status"] = "cancelled"
                # Remove from active keys
                old_key = sub_data.get("api_key", "")
                if old_key:
                    old_hash = hashlib.sha256(old_key.encode()).hexdigest()
                    API_KEYS.pop(old_hash, None)
                logger.info(f"Deactivated subscription for {email}")
                break
        _save_subscribers(subscribers)

    return {"status": "ok"}


@app.get("/api/billing")
async def get_billing(request: Request, authorization: Optional[str] = Header(None)):
    """Return current subscription status for authenticated user."""
    client_id, tier = get_client_id(request, authorization)

    if tier == "free":
        # Check remaining free runs
        now = time.time()
        timestamps = [t for t in rate_tracker.get(client_id, []) if now - t < 86400]
        remaining = max(0, 1 - len(timestamps))
        return {
            "tier": "free",
            "status": "active",
            "runs_remaining_today": remaining,
            "runs_used_today": len(timestamps),
        }

    # Paid tier — find subscriber info
    subscribers = _load_subscribers()
    for email, sub_data in subscribers.items():
        if sub_data.get("api_key"):
            key_hash = hashlib.sha256(sub_data["api_key"].encode()).hexdigest()
            if key_hash == client_id:
                return {
                    "tier": tier,
                    "status": sub_data.get("status", "active"),
                    "email": email,
                    "created": sub_data.get("created", ""),
                }

    return {"tier": tier, "status": "active"}


# ── WebSocket for streaming passes ───────────────────────────────────────

@app.websocket("/api/ws/{run_id}")
async def ws_run(websocket: WebSocket, run_id: str):
    """Stream RPCCP passes in real-time via WebSocket."""
    # Get client IP for connection limiting
    client_ip = websocket.client.host if websocket.client else "unknown"

    # Check WebSocket connection limit per IP
    if ws_connections[client_ip] >= MAX_WS_PER_IP:
        await websocket.close(code=1008, reason="Too many connections")
        return

    await websocket.accept()
    ws_connections[client_ip] += 1

    run_info = active_runs.get(run_id)
    if not run_info:
        await websocket.send_json({"type": "error", "message": "Run not found"})
        await websocket.close()
        ws_connections[client_ip] -= 1
        return

    # Authenticate WebSocket — require run_token as first message
    try:
        first_msg = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
        token = first_msg.get("run_token", "")
        if not secrets.compare_digest(token, run_info.get("run_token", "")):
            await websocket.send_json({"type": "error", "message": "Authentication failed"})
            await websocket.close()
            ws_connections[client_ip] -= 1
            return
    except (asyncio.TimeoutError, Exception):
        await websocket.send_json({"type": "error", "message": "Authentication timeout"})
        await websocket.close()
        ws_connections[client_ip] -= 1
        return

    query = run_info["query"]
    mode = run_info["mode"]
    client_id = run_info["client_id"]

    try:
        # Wrap entire run in a timeout
        await asyncio.wait_for(
            _execute_run(websocket, run_id, run_info, query, mode, client_id),
            timeout=RUN_TIMEOUT,
        )
    except asyncio.TimeoutError:
        try:
            await websocket.send_json({
                "type": "error",
                "message": "Run timed out. Please try a more specific query."
            })
        except Exception:
            pass
        logger.warning(f"Run {run_id} timed out after {RUN_TIMEOUT}s")
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception(f"WebSocket run error for {run_id}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": "Internal error. Please try again later."
            })
        except Exception:
            pass
    finally:
        active_runs.pop(run_id, None)
        ws_connections[client_ip] = max(0, ws_connections[client_ip] - 1)
        active_run_clients[client_id] = max(0, active_run_clients[client_id] - 1)


async def _execute_run(websocket: WebSocket, run_id: str, run_info: dict, query: str, mode: str, client_id: str):
    """Execute the RPCCP run and stream results. Separated for timeout wrapping."""
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

    # Save run — include client_id for ownership
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

    # Save with client_id for ownership tracking
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    slug = query[:40].replace(" ", "_").replace("?", "").replace("/", "_")
    slug = re.sub(r"[^a-zA-Z0-9_\-]", "", slug)
    filename = f"rpccp_{ts}_{slug}.json"

    data = {
        "query": rpccp_result.query,
        "client_id": client_id,
        "run_id": run_id,
        "timestamp": rpccp_result.timestamp,
        "total_duration": rpccp_result.total_duration,
        "models_used": rpccp_result.models_used,
        "type2_detected": rpccp_result.collision.type2_detected if rpccp_result.collision else False,
        "real_question": rpccp_result.collision.real_question if rpccp_result.collision else "",
        "objective_evolution": rpccp_result.collision.objective_evolution if rpccp_result.collision else [],
        "final_synthesis": rpccp_result.final_synthesis,
        "passes": [
            {
                "pass_number": p.pass_number,
                "pass_type": p.pass_type,
                "model_used": p.model_used,
                "objective_function": p.objective_function,
                "dimensions_considered": p.dimensions_considered,
                "response": p.response,
                "duration_seconds": p.duration_seconds,
            }
            for p in rpccp_result.passes
        ],
    }

    filepath = RUNS_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    try:
        os.chmod(filepath, 0o600)
    except OSError:
        pass

    await websocket.send_json({
        "type": "complete",
        "message": "RPCCP run complete.",
        "total_duration": round(total_duration, 1),
    })


# ── CCC Underwriting API ────────────────────────────────────────────────

from underwriting_db import get_db

@app.get("/api/ccc/stats")
async def ccc_stats():
    """Live lender/product counts for the website."""
    db = get_db()
    return db.get_stats()

@app.post("/api/ccc/match")
async def ccc_match(request: Request):
    """Match a deal to qualifying lender products."""
    data = await request.json()
    db = get_db()
    matches = db.match_deal(
        loan_type=data.get("loanType", "dscr"),
        property_type=data.get("propType"),
        txn_type=data.get("txn"),
        credit_score=int(data["credit"]) if data.get("credit") else None,
        ltv=float(data["ltv"]) if data.get("ltv") else None,
        dscr_ratio=float(data["dscr"]) if data.get("dscr") else None,
        state=data.get("state")
    )
    return {"matches": matches, "count": len(matches)}

@app.post("/api/ccc/submit")
async def ccc_submit(request: Request):
    """Save a deal submission and return matching products."""
    data = await request.json()
    db = get_db()

    # Match first
    matches = db.match_deal(
        loan_type=data.get("loanType", "dscr"),
        property_type=data.get("propType"),
        txn_type=data.get("txn"),
        credit_score=int(data["credit"]) if data.get("credit") else None,
        ltv=float(data["ltv"]) if data.get("ltv") else None,
        dscr_ratio=float(data["dscr"]) if data.get("dscr") else None,
        state=data.get("state")
    )

    # Save submission with matches
    data["matched_products"] = [{"lender": m["lender"], "product": m["product"]} for m in matches[:20]]
    sub_id = db.save_submission(data)

    return {
        "submission_id": sub_id,
        "matches": matches[:20],
        "count": len(matches),
        "message": f"Found {len(matches)} qualifying products from {len(set(m['lender'] for m in matches))} lenders"
    }

@app.get("/api/ccc/lenders")
async def ccc_lenders():
    """List all active lenders."""
    db = get_db()
    conn = db._conn()
    rows = conn.execute("""
        SELECT l.*, COUNT(p.id) as product_count
        FROM lenders l
        LEFT JOIN products p ON p.lender_id = l.id AND p.active = 1
        WHERE l.active = 1
        GROUP BY l.id
        ORDER BY l.name
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/api/ccc/lenders/{slug}/products")
async def ccc_lender_products(slug: str):
    """List all products for a specific lender."""
    db = get_db()
    conn = db._conn()
    lender = conn.execute("SELECT * FROM lenders WHERE slug=?", (slug,)).fetchone()
    if not lender:
        raise HTTPException(404, "Lender not found")
    products = conn.execute("""
        SELECT * FROM products WHERE lender_id=? AND active=1 ORDER BY name
    """, (lender["id"],)).fetchall()
    conn.close()
    return {"lender": dict(lender), "products": [dict(p) for p in products]}

# ── Borrower auto-populate ──────────────────────────────────────────────

@app.get("/api/ccc/borrower/{email}")
async def ccc_borrower_profile(email: str):
    """Return saved borrower profile for auto-populate."""
    db = get_db()
    conn = db._conn()
    borrower = conn.execute("SELECT * FROM borrowers WHERE email=?", (email,)).fetchone()
    if not borrower:
        conn.close()
        return {"found": False}
    deals = conn.execute("""
        SELECT id, loan_type, property_type, status, property_value, created_at
        FROM deals WHERE borrower_id=? ORDER BY created_at DESC
    """, (borrower["id"],)).fetchall()
    conn.close()
    return {"found": True, "borrower": dict(borrower), "deals": [dict(d) for d in deals]}

@app.post("/api/ccc/deal")
async def ccc_create_deal(request: Request):
    """Create a deal with borrower auto-create/update."""
    data = await request.json()
    db = get_db()
    conn = db._conn()

    # Upsert borrower
    email = data.get("email", "").strip().lower()
    if not email:
        raise HTTPException(400, "Email required")

    conn.execute("""
        INSERT INTO borrowers (email, name, phone, credit_score, experience, entity_type, state)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(email) DO UPDATE SET
            name=COALESCE(excluded.name, borrowers.name),
            phone=COALESCE(excluded.phone, borrowers.phone),
            credit_score=COALESCE(excluded.credit_score, borrowers.credit_score),
            experience=COALESCE(excluded.experience, borrowers.experience),
            entity_type=COALESCE(excluded.entity_type, borrowers.entity_type),
            state=COALESCE(excluded.state, borrowers.state),
            deal_count=borrowers.deal_count + 1,
            updated_at=datetime('now')
    """, (email, data.get("name"), data.get("phone"),
          int(data["credit"]) if data.get("credit") else None,
          data.get("experience"), data.get("entity"), data.get("state")))
    conn.commit()

    borrower = conn.execute("SELECT id FROM borrowers WHERE email=?", (email,)).fetchone()
    borrower_id = borrower["id"]

    # Calculate LTV
    value = float(data["value"]) if data.get("value") else None
    loan_amt = float(data["loan_amount"]) if data.get("loan_amount") else None
    down_pct = float(data["down"]) if data.get("down") else None
    ltv = None
    if value and loan_amt:
        ltv = (loan_amt / value) * 100
    elif value and down_pct:
        ltv = 100 - down_pct

    # Match deal
    credit = int(data["credit"]) if data.get("credit") else None
    dscr = float(data["dscr"]) if data.get("dscr") else None
    matches = db.match_deal(
        loan_type=data.get("loanType", "dscr"),
        property_type=data.get("propType"),
        txn_type=data.get("txn"),
        credit_score=credit,
        ltv=ltv,
        dscr_ratio=dscr,
        state=data.get("state")
    )

    # Create deal
    conn.execute("""
        INSERT INTO deals (borrower_id, loan_type, txn_type, property_type, property_address,
            state, city, property_value, loan_amount, down_pct, ltv, credit_score,
            monthly_rent, dscr_ratio, arv, rehab_budget, build_cost, noi,
            experience, entity_type, entity_name, denied_elsewhere, notes, source,
            matched_lender, matched_product, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'matched')
    """, (
        borrower_id, data.get("loanType"), data.get("txn"), data.get("propType"),
        data.get("address"), data.get("state"), data.get("city"),
        value, loan_amt, down_pct, ltv, credit,
        float(data["rent"]) if data.get("rent") else None,
        dscr,
        float(data["arv"]) if data.get("arv") else None,
        float(data["rehab"]) if data.get("rehab") else None,
        float(data["buildCost"]) if data.get("buildCost") else None,
        float(data["noi"]) if data.get("noi") else None,
        data.get("experience"), data.get("entity"), data.get("entityName"),
        1 if data.get("denied") == "yes" else 0,
        data.get("notes"), data.get("source"),
        matches[0]["lender"] if matches else None,
        matches[0]["product"] if matches else None
    ))
    conn.commit()
    deal_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Log status
    conn.execute("""
        INSERT INTO deal_status_log (deal_id, new_status, changed_by, notes)
        VALUES (?, 'matched', 'system', ?)
    """, (deal_id, f"Matched {len(matches)} products from {len(set(m['lender'] for m in matches))} lenders"))
    conn.commit()

    # Handle referral code
    ref_code = data.get("ref") or data.get("referral_code")
    if ref_code:
        partner = conn.execute("SELECT id FROM referral_partners WHERE referral_code=?",
                               (ref_code,)).fetchone()
        if partner:
            conn.execute("""
                INSERT INTO referrals (partner_id, deal_id, borrower_name, status)
                VALUES (?, ?, ?, 'referred')
            """, (partner["id"], deal_id, data.get("name")))
            conn.execute("""
                UPDATE referral_partners SET total_referrals = total_referrals + 1,
                    updated_at = datetime('now') WHERE id = ?
            """, (partner["id"],))
            conn.commit()

    conn.close()

    return {
        "deal_id": deal_id,
        "borrower_id": borrower_id,
        "matches": matches[:20],
        "total_matches": len(matches),
        "top_lender": matches[0]["lender"] if matches else None,
        "top_product": matches[0]["product"] if matches else None
    }

# ── Referral Partner Portal ─────────────────────────────────────────────

@app.post("/api/ccc/partner/login")
async def ccc_partner_login(request: Request):
    """Referral partner login."""
    data = await request.json()
    db = get_db()
    conn = db._conn()
    partner = conn.execute("SELECT * FROM referral_partners WHERE email=?",
                           (data.get("email", "").strip().lower(),)).fetchone()
    conn.close()
    if not partner or not partner["portal_password_hash"]:
        raise HTTPException(401, "Invalid credentials")
    if not bcrypt.checkpw(data.get("password", "").encode(), partner["portal_password_hash"].encode()):
        raise HTTPException(401, "Invalid credentials")
    token = pyjwt.encode(
        {"partner_id": partner["id"], "email": partner["email"], "exp": time.time() + 86400*30},
        os.environ.get("JWT_SECRET", "ccc-dev-secret"), algorithm="HS256"
    )
    return {"token": token, "partner": {"name": partner["name"], "company": partner["company"]}}

@app.get("/api/ccc/partner/dashboard")
async def ccc_partner_dashboard(authorization: str = Header(None)):
    """Referral partner dashboard — their deals, comp, pipeline."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Not authenticated")
    try:
        payload = pyjwt.decode(authorization[7:],
                               os.environ.get("JWT_SECRET", "ccc-dev-secret"), algorithms=["HS256"])
    except Exception:
        raise HTTPException(401, "Invalid token")

    partner_id = payload["partner_id"]
    db = get_db()
    conn = db._conn()

    partner = conn.execute("SELECT * FROM referral_partners WHERE id=?", (partner_id,)).fetchone()
    referrals = conn.execute("""
        SELECT r.*, d.loan_type, d.property_type, d.property_value, d.loan_amount,
               d.status as deal_status, d.matched_lender, d.funded_amount, d.broker_comp,
               d.created_at as deal_date
        FROM referrals r
        JOIN deals d ON r.deal_id = d.id
        WHERE r.partner_id = ?
        ORDER BY r.created_at DESC
    """, (partner_id,)).fetchall()
    conn.close()

    # Calculate pipeline stats
    pipeline = {"referred": 0, "in_progress": 0, "funded": 0, "total_value": 0, "total_comp": 0}
    for ref in referrals:
        if ref["deal_status"] == "funded":
            pipeline["funded"] += 1
            pipeline["total_comp"] += ref["comp_amount"] or 0
        elif ref["deal_status"] in ("new", "matched"):
            pipeline["referred"] += 1
        else:
            pipeline["in_progress"] += 1
        pipeline["total_value"] += ref["loan_amount"] or 0

    return {
        "partner": {"name": partner["name"], "company": partner["company"],
                     "referral_code": partner["referral_code"],
                     "total_referrals": partner["total_referrals"],
                     "total_funded": partner["total_funded"],
                     "total_comp_earned": partner["total_comp_earned"]},
        "pipeline": pipeline,
        "deals": [dict(r) for r in referrals]
    }

@app.get("/api/ccc/partner/deal/{deal_id}")
async def ccc_partner_deal_detail(deal_id: int, authorization: str = Header(None)):
    """Referral partner view of a specific deal — status, timeline, comp."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Not authenticated")
    try:
        payload = pyjwt.decode(authorization[7:],
                               os.environ.get("JWT_SECRET", "ccc-dev-secret"), algorithms=["HS256"])
    except Exception:
        raise HTTPException(401, "Invalid token")

    partner_id = payload["partner_id"]
    db = get_db()
    conn = db._conn()

    # Verify this deal belongs to this partner
    ref = conn.execute("""
        SELECT r.*, d.loan_type, d.txn_type, d.property_type, d.property_value,
               d.loan_amount, d.status, d.matched_lender, d.matched_product,
               d.submitted_to_lender_at, d.approved_at, d.closing_date,
               d.funded_at, d.funded_amount, d.broker_comp, d.created_at
        FROM referrals r
        JOIN deals d ON r.deal_id = d.id
        WHERE r.partner_id = ? AND r.deal_id = ?
    """, (partner_id, deal_id)).fetchone()

    if not ref:
        raise HTTPException(404, "Deal not found or not your referral")

    status_log = conn.execute("""
        SELECT new_status, notes, created_at FROM deal_status_log
        WHERE deal_id = ? ORDER BY created_at
    """, (deal_id,)).fetchall()
    conn.close()

    return {
        "deal": dict(ref),
        "timeline": [dict(s) for s in status_log],
        "comp": {
            "amount": ref["comp_amount"],
            "paid": bool(ref["comp_paid"]),
            "paid_at": ref["comp_paid_at"]
        }
    }


# ── Static files ─────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
