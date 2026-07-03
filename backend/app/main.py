from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .schemas import AlgorithmsResponse, EnvStateResponse, ResetRequest, StepRequest
from .simulation import ALGORITHMS, make_env, policy_cache, reset_env, step_env, arena_step, arena_reset, make_arena_envs

logger = logging.getLogger("vacuum-rl.api")


def _allowed_origins() -> list[str]:
    configured = os.getenv("CORS_ORIGINS", "")
    origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://*.vercel.app",
    ]
    if configured.strip():
        origins.extend(origin.strip() for origin in configured.split(",") if origin.strip())
    return origins


app = FastAPI(title="vacuum-rl API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_rest_env = make_env(seed=12345)
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


@app.on_event("startup")
def load_models_on_startup() -> None:
    import time
    start = time.perf_counter()
    policy_cache.load_all()
    elapsed = time.perf_counter() - start
    logger.info("Total startup model loading: %.2fs", elapsed)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "models_loaded": list(policy_cache._models.keys()),
        "models_failed": list(policy_cache._load_errors.keys()),
        "load_times_s": policy_cache.load_times,
        "startup_duration_s": policy_cache._startup_duration,
        "load_errors": {k: str(v) for k, v in policy_cache._load_errors.items()},
    }



@app.get("/algorithms", response_model=AlgorithmsResponse)
def algorithms() -> dict:
    load_errors = policy_cache.load_errors
    return {
        "algorithms": [
            {
                "id": spec.id,
                "name": spec.name,
                "source": spec.source,
                "model_path": str(spec.model_path.relative_to(spec.model_path.parents[3])),
                "available": spec.model_path.exists() and spec.id not in load_errors,
                "continuous_policy": spec.continuous_policy,
                "load_error": load_errors.get(spec.id),
            }
            for spec in ALGORITHMS.values()
        ],
        "actions": list(_rest_env.ACTIONS),
    }


@app.post("/reset", response_model=EnvStateResponse)
def reset(payload: ResetRequest = ResetRequest()) -> dict:
    return reset_env(_rest_env, algorithm=payload.algorithm, seed=payload.seed)


@app.websocket("/ws/step")
async def websocket_step(websocket: WebSocket) -> None:
    await websocket.accept()
    env = make_env(seed=12345)
    algorithm = "ppo"
    try:
        await websocket.send_json(reset_env(env, algorithm=algorithm, seed=12345))
    except Exception:
        env.close()
        return

    try:
        while True:
            try:
                payload = await websocket.receive_json()
                request_type = str(payload.get("type", "step")).lower()
                algorithm = str(payload.get("algorithm", algorithm)).lower()
                if request_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue
                if request_type == "reset":
                    seed = payload.get("seed")
                    await websocket.send_json(reset_env(env, algorithm=algorithm, seed=seed))
                    continue

                request = StepRequest(**payload)
                action = None if request.mode == "auto" else request.action
                await websocket.send_json(step_env(env, algorithm=request.algorithm, action=action))
            except ValueError as exc:
                await websocket.send_json({"error": str(exc), "recoverable": True})
            except RuntimeError as exc:
                await websocket.send_json({"error": str(exc), "recoverable": True})
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                logger.warning("ws/step inner error: %s", exc)
                try:
                    await websocket.send_json({"error": f"Internal error: {exc}", "recoverable": True})
                except Exception:
                    raise WebSocketDisconnect() from exc
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("ws/step fatal error: %s", exc)
    finally:
        env.close()


@app.websocket("/ws/arena")
async def websocket_arena(websocket: WebSocket) -> None:
    await websocket.accept()
    envs: dict[str, Any] = {}
    seed = 12345

    try:
        while True:
            try:
                payload = await websocket.receive_json()
                msg_type = str(payload.get("type", "")).lower()

                if msg_type == "init":
                    algorithms = payload.get("algorithms", [])
                    seed = payload.get("seed", seed)
                    for env in envs.values():
                        env.close()
                    envs = make_arena_envs(algorithms, seed=seed)
                    states = arena_reset(envs, seed=seed)
                    await websocket.send_json({"type": "arena_states", "states": states})

                elif msg_type == "step":
                    if not envs:
                        await websocket.send_json({"error": "No arena running. Send init first."})
                        continue
                    states = arena_step(envs)
                    await websocket.send_json({"type": "arena_states", "states": states})

                elif msg_type == "reset":
                    if not envs:
                        await websocket.send_json({"error": "No arena running."})
                        continue
                    seed = payload.get("seed", seed)
                    states = arena_reset(envs, seed=seed)
                    await websocket.send_json({"type": "arena_states", "states": states})

                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

                else:
                    await websocket.send_json({"error": f"Unknown message type: {msg_type}"})

            except ValueError as exc:
                await websocket.send_json({"type": "arena_error", "error": str(exc), "recoverable": True})
            except RuntimeError as exc:
                await websocket.send_json({"type": "arena_error", "error": str(exc), "recoverable": True})
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                logger.warning("ws/arena inner error: %s", exc)
                try:
                    await websocket.send_json({"type": "arena_error", "error": f"Internal error: {exc}", "recoverable": True})
                except Exception:
                    raise WebSocketDisconnect() from exc
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.error("ws/arena fatal error: %s", exc)
    finally:
        for env in envs.values():
            env.close()


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
