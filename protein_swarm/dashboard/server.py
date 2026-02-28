"""
FastAPI server for the Protein Swarm dashboard.

Serves the static frontend and provides:
- POST /api/run — start a design run with JSON body (sequence, objective, options).
- GET /api/events — Server-Sent Events stream of progress (run_start, iteration_start,
  agents_started, agents_completed, sequence_update, run_complete).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from queue import Empty, Queue

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from protein_swarm.config import FoldingConfig, LLMConfig, MemoryConfig, SwarmConfig
from protein_swarm.orchestrator.engine import DesignEngine
from protein_swarm.utils.constants import AMINO_ACIDS

logger = logging.getLogger(__name__)

# In-memory event queue for the single active run (one run at a time).
_event_queue: Queue[dict] = Queue()
_run_lock = threading.Lock()
_run_active = False

app = FastAPI(title="Protein Swarm Dashboard", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory containing static files (index.html, etc.)
STATIC_DIR = Path(__file__).resolve().parent / "static"


class RunRequest(BaseModel):
    """Request body for POST /api/run. Mirrors CLI options."""

    sequence: str = Field(..., min_length=2, description="Initial amino acid sequence")
    objective: str = Field(..., min_length=1, description="Design objective / prompt")
    # Run params
    max_iterations: int = Field(default=50, ge=1, le=500)
    mutation_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    plateau_window: int = Field(default=5, ge=2)
    output_dir: str = Field(default="outputs")
    # LLM
    use_llm: bool = Field(default=False)
    llm_provider: str = Field(default="openai")
    llm_model: str = Field(default="gpt-4o-mini")
    # Folding
    modal_fold: bool = Field(default=False)
    modal_parallel: bool = Field(default=True)
    fold_backend: str = Field(default="dummy")
    remote_fold_backend: str = Field(default="esmfold")
    # Rosetta
    use_rosetta: bool = Field(default=True)
    rosetta_relax: bool = Field(default=False)
    rosetta_relax_cycles: int = Field(default=1, ge=1, le=20)
    rosetta_norm_target: float = Field(default=-200.0)
    rosetta_norm_scale: float = Field(default=50.0, gt=0.0)
    # Scoring weights
    w_physics: float = Field(default=0.55, ge=0.0)
    w_objective: float = Field(default=0.35, ge=0.0)
    w_confidence: float = Field(default=0.10, ge=0.0)
    # Debug
    debug: bool = Field(default=False)
    dump_prompts: bool = Field(default=False)


def _validate_sequence(sequence: str) -> str:
    seq = sequence.upper().strip()
    invalid = [c for c in seq if c not in AMINO_ACIDS]
    if invalid:
        raise ValueError(f"Invalid amino acid(s): {set(invalid)}. Allowed: {''.join(AMINO_ACIDS)}")
    return seq


def _run_engine(req: RunRequest) -> None:
    """Run the design engine in the current thread; events are pushed to _event_queue."""
    global _run_active
    sequence = _validate_sequence(req.sequence)

    llm_cfg = LLMConfig(
        provider=req.llm_provider,
        model=req.llm_model,
        api_key=None,
        temperature=0.7,
    )
    folding_cfg = FoldingConfig(
        use_rosetta=req.use_rosetta,
        rosetta_relax=req.rosetta_relax,
        rosetta_relax_cycles=req.rosetta_relax_cycles,
        rosetta_norm_target=req.rosetta_norm_target,
        rosetta_norm_scale=req.rosetta_norm_scale,
        w_physics=req.w_physics,
        w_objective=req.w_objective,
        w_confidence=req.w_confidence,
    )
    swarm_cfg = SwarmConfig(
        use_llm_agents=req.use_llm,
        llm=llm_cfg,
        max_iterations=req.max_iterations,
        mutation_rate=req.mutation_rate,
        modal_parallel=req.modal_parallel,
        modal_fold=req.modal_fold,
        fold_backend=req.fold_backend,
        remote_fold_backend=req.remote_fold_backend,
        output_dir=req.output_dir,
        confidence_threshold=req.confidence_threshold,
        plateau_window=req.plateau_window,
        debug=req.debug,
    )

    def on_progress(event: dict) -> None:
        _event_queue.put(event)

    engine = DesignEngine(
        swarm_config=swarm_cfg,
        folding_config=folding_cfg,
        memory_config=MemoryConfig(),
        dump_prompts=req.dump_prompts and req.debug,
    )
    try:
        engine.run(sequence, req.objective, progress_callback=on_progress)
    except Exception as e:
        logger.exception("Run failed")
        _event_queue.put({"type": "run_error", "message": str(e)})
    finally:
        with _run_lock:
            _run_active = False


@app.post("/api/run")
def api_run(req: RunRequest) -> dict:
    """Start a design run. Events stream via GET /api/events."""
    global _run_active
    try:
        _validate_sequence(req.sequence)
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    with _run_lock:
        if _run_active:
            return {"ok": False, "error": "A run is already in progress."}
        _run_active = True

    # Clear any stale events from a previous run
    while True:
        try:
            _event_queue.get_nowait()
        except Empty:
            break

    thread = threading.Thread(target=_run_engine, args=(req,), daemon=True)
    thread.start()
    return {"ok": True, "message": "Run started. Connect to GET /api/events for progress."}


@app.get("/api/events")
async def api_events():
    """Stream progress events as Server-Sent Events."""
    loop = asyncio.get_event_loop()

    async def event_generator():
        while True:
            try:
                event = await loop.run_in_executor(
                    None, lambda: _event_queue.get(timeout=30)
                )
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("run_complete", "run_error"):
                    break
            except Empty:
                yield ": keepalive\n\n"
            except Exception:
                break  # connection closed or stream ended

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/")
def index() -> FileResponse:
    """Serve the dashboard single-page app."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise FileNotFoundError(f"Dashboard static files not found at {STATIC_DIR}")
    return FileResponse(index_path)


@app.get("/static/{path:path}")
def static_file(path: str) -> FileResponse:
    """Serve static assets."""
    full = STATIC_DIR / path
    if not full.is_file():
        raise FileNotFoundError(path)
    return FileResponse(full)


def create_app() -> FastAPI:
    """Return the FastAPI app (for uvicorn)."""
    return app
