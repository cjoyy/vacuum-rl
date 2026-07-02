from __future__ import annotations

import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .schemas import AlgorithmsResponse, EnvStateResponse, ResetRequest, StepRequest
from .simulation import ALGORITHMS, make_env, policy_cache, reset_env, step_env


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


@app.on_event("startup")
def load_models_on_startup() -> None:
    policy_cache.load_all()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "vacuum-rl backend",
        "health": "/health",
        "algorithms": "/algorithms",
        "websocket": "/ws/step",
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
    await websocket.send_json(reset_env(env, algorithm=algorithm, seed=12345))

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
        env.close()
    except Exception as exc:
        await websocket.send_json({"error": str(exc)})
        env.close()
