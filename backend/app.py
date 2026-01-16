from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from datetime import date
from typing import List, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# -------------------------
# Config
# -------------------------
MODEL_NAME = os.getenv("ORBIT_MODEL", "all-MiniLM-L6-v2")
WIN_SIMILARITY = float(os.getenv("ORBIT_WIN_SIM", "0.79"))  # tune 0.75–0.82
MIN_SIM_FOR_VISIBLE = float(os.getenv("ORBIT_MIN_SIM", "0.05"))

# canvas "world" units (frontend draws these as pixels)
WORLD_SIZE = 900  # virtual square: -WORLD_SIZE/2 .. +WORLD_SIZE/2
CORE_RADIUS = 60  # if inside, you win (visual)
MAX_RADIUS = 390  # outer orbit radius


# -------------------------
# Helpers
# -------------------------
def load_targets(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        items = [ln.strip() for ln in f.readlines()]
    items = [w for w in items if w and not w.startswith("#")]
    if not items:
        raise RuntimeError("targets.txt is empty")
    return items


def today_seed(d: Optional[date] = None) -> int:
    # stable seed per day (UTC-ish; fine for prototype)
    d = d or date.today()
    s = d.isoformat().encode("utf-8")
    h = hashlib.sha256(s).hexdigest()
    return int(h[:8], 16)


def pick_daily_target(targets: List[str], seed: int) -> str:
    rng = np.random.default_rng(seed)
    return targets[int(rng.integers(0, len(targets)))]


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize(a)
    b = normalize(b)
    return float(np.dot(a, b))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def similarity_to_radius(sim: float) -> float:
    """
    Map cosine similarity to radius.
    Higher sim -> smaller radius (closer to core).
    Nonlinear mapping gives better "feel".
    """
    sim = clamp(sim, -1.0, 1.0)

    # keep things sane: below MIN_SIM_FOR_VISIBLE just slam near outer edge
    if sim < MIN_SIM_FOR_VISIBLE:
        return MAX_RADIUS

    # remap [MIN_SIM_FOR_VISIBLE .. 1] -> [1 .. 0]
    t = (sim - MIN_SIM_FOR_VISIBLE) / (1.0 - MIN_SIM_FOR_VISIBLE)
    t = clamp(t, 0.0, 1.0)

    # ease curve: makes near-correct guesses move sharply inward
    eased = 1.0 - (t ** 1.8)

    r = CORE_RADIUS + eased * (MAX_RADIUS - CORE_RADIUS)
    return float(r)


def seeded_projection_matrix(dim: int, seed: int) -> np.ndarray:
    """
    Stable 2D projection matrix, seeded per day so the 'orbit map'
    is consistent for everyone that day.
    """
    rng = np.random.default_rng(seed)
    # random gaussian projection to 2D
    M = rng.normal(0, 1, size=(dim, 2)).astype(np.float32)
    # normalize columns a bit
    M[:, 0] = M[:, 0] / (np.linalg.norm(M[:, 0]) + 1e-9)
    M[:, 1] = M[:, 1] / (np.linalg.norm(M[:, 1]) + 1e-9)
    return M


def embedding_to_angle(vec: np.ndarray, proj2d: np.ndarray) -> float:
    """
    Project embedding to 2D and compute angle.
    Angle gives the "semantic direction cluster" vibe.
    """
    xy = vec @ proj2d  # (2,)
    x, y = float(xy[0]), float(xy[1])
    ang = math.atan2(y, x)
    return ang


def orbit_motion_kind(sim: float) -> str:
    if sim >= 0.78:
        return "tight"
    if sim >= 0.68:
        return "pulled"
    if sim >= 0.55:
        return "drift"
    if sim >= 0.35:
        return "wobble"
    return "pushed"



# -------------------------
# API Models
# -------------------------
class GuessIn(BaseModel):
    guess: str


class NodeOut(BaseModel):
    word: str
    x: float
    y: float
    motion: str
    win: bool


class GuessOut(BaseModel):
    node: NodeOut
    solved: bool


# -------------------------
# App setup
# -------------------------
app = FastAPI(title="Semantic Orbit API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down later if you host it
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

targets = load_targets(os.path.join(os.path.dirname(__file__), "targets.txt"))
model = SentenceTransformer(MODEL_NAME)

# Cached daily state
@dataclass
class DailyState:
    seed: int
    target: str
    target_vec: np.ndarray
    proj2d: np.ndarray
    dim: int


_state: Optional[DailyState] = None


def get_state() -> DailyState:
    global _state
    seed = today_seed()
    if _state is None or _state.seed != seed:
        target = pick_daily_target(targets, seed)
        target_vec = model.encode([target], normalize_embeddings=True)[0].astype(np.float32)
        dim = target_vec.shape[0]
        proj2d = seeded_projection_matrix(dim, seed)
        _state = DailyState(seed=seed, target=target, target_vec=target_vec, proj2d=proj2d, dim=dim)
        print(f"[Daily] seed={seed} target={target}")
    return _state


@app.get("/api/daily")
def daily_info():
    st = get_state()
    # Do NOT reveal target. Just provide a share-safe puzzle id.
    return {"puzzle_id": f"{date.today().isoformat()}-{st.seed}"}


@app.post("/api/guess", response_model=GuessOut)
def submit_guess(payload: GuessIn):
    st = get_state()

    word = payload.guess.strip().lower()
    # Basic guardrails (avoid empty and huge strings)
    if not word or len(word) > 40:
        return GuessOut(
            node=NodeOut(word=word, x=0, y=0, motion="wobble", win=False),
            solved=False,
        )

    guess_vec = model.encode([word], normalize_embeddings=True)[0].astype(np.float32)

    sim = cosine_sim(guess_vec, st.target_vec)
    r = similarity_to_radius(sim)
    ang = embedding_to_angle(guess_vec, st.proj2d)

    # Turn polar -> cartesian
    x = r * math.cos(ang)
    y = r * math.sin(ang)

    win = sim >= WIN_SIMILARITY or r <= CORE_RADIUS + 4
    motion = orbit_motion_kind(sim)

    node = NodeOut(word=word, x=float(x), y=float(y), motion=motion, win=win)
    return GuessOut(node=node, solved=win)


# Optional: reveal endpoint for debugging (DON'T SHIP)
@app.get("/api/_debug_target")
def debug_target():
    st = get_state()
    return {"target": st.target, "seed": st.seed}
