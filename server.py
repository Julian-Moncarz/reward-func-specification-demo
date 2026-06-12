"""Reward Lab server: serves the demo UI, turns NL specs into reward code via
the Claude CLI, trains tabular Q-learning, and streams progress + rollouts."""

import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from envs import ENVS
from reward_gen import GenError, generate_reward
from trainer import RewardError, compile_reward, rollout, train, validate_reward

ROOT = Path(__file__).parent
app = FastAPI(title="Reward Lab")


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/api/envs")
def list_envs():
    return [
        {
            "id": e.id, "name": e.name, "icon": e.icon, "tagline": e.tagline,
            "intended": e.intended, "max_steps": e.max_steps,
            "state_docs": e.STATE_DOCS, "layout": e.layout(),
            "has_shift": e.id == "coins",
        }
        for e in ENVS.values()
    ]


class GenerateReq(BaseModel):
    env_id: str
    spec: str


@app.post("/api/generate")
def api_generate(req: GenerateReq):
    if req.env_id not in ENVS:
        return {"error": f"unknown env {req.env_id}"}
    try:
        res = generate_reward(req.env_id, req.spec)
        fn = compile_reward(res["code"])
        validate_reward(req.env_id, fn)
        return res
    except (GenError, RewardError) as e:
        return {"error": str(e)}


class TrainReq(BaseModel):
    env_id: str
    code: str
    shift: bool = False


@app.post("/api/train")
def api_train(req: TrainReq):
    """NDJSON stream: {stage:'training', ep, total} ... {stage:'done', rollout, shifted_rollout?}"""
    if req.env_id not in ENVS:
        return {"error": f"unknown env {req.env_id}"}

    def stream():
        try:
            fn = compile_reward(req.code)
            validate_reward(req.env_id, fn)
        except RewardError as e:
            yield json.dumps({"stage": "error", "error": str(e)}) + "\n"
            return

        q = queue.Queue()

        def progress(ep, total):
            q.put({"stage": "training", "ep": ep, "total": total})

        result = {}

        def work():
            try:
                Q = train(req.env_id, fn, progress=progress)
                result["rollout"] = rollout(req.env_id, Q, fn, shift=False)
                if req.shift:
                    result["shifted_rollout"] = rollout(req.env_id, Q, fn, shift=True)
            except Exception as e:
                result["error"] = str(e)
            q.put(None)

        t = threading.Thread(target=work, daemon=True)
        t.start()
        while True:
            msg = q.get()
            if msg is None:
                break
            yield json.dumps(msg) + "\n"
        if "error" in result:
            yield json.dumps({"stage": "error", "error": result["error"]}) + "\n"
        else:
            yield json.dumps({"stage": "done", "rollout": result["rollout"],
                              "shifted_rollout": result.get("shifted_rollout")}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
