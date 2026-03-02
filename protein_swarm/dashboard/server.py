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
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from protein_swarm.config import FoldingConfig, LLMConfig, MemoryConfig, SwarmConfig
from protein_swarm.orchestrator.engine import DesignEngine
from protein_swarm.utils.constants import AMINO_ACIDS

logger = logging.getLogger(__name__)

# In-memory event queue for the single active run (one run at a time).
_event_queue: Queue[dict] = Queue()
_run_lock = threading.Lock()
_run_active = False
# Output dir of the last completed run (for View Protein).
_last_output_dir: Path | None = None

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
# SQLite database for run history and total-proteins counter
DATA_DIR = Path(__file__).resolve().parent / "data"
DB_PATH = DATA_DIR / "runs.db"
_db_lock = threading.Lock()


def _init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    initial_sequence TEXT NOT NULL,
                    final_sequence TEXT NOT NULL,
                    use_llm INTEGER NOT NULL,
                    iterations INTEGER NOT NULL,
                    protein_length INTEGER NOT NULL,
                    image_path TEXT
                )
                """
            )
            conn.commit()
            cur = conn.execute("PRAGMA table_info(runs)")
            columns = [row[1] for row in cur.fetchall()]
            if "image_path" not in columns:
                conn.execute("ALTER TABLE runs ADD COLUMN image_path TEXT")
                conn.commit()
        finally:
            conn.close()


def _record_run(
    initial_sequence: str,
    final_sequence: str,
    use_llm: bool,
    iterations: int,
    protein_length: int,
) -> int:
    """Record a run and return its id."""
    _init_db()
    created = datetime.now(timezone.utc).isoformat()
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute(
                """
                INSERT INTO runs (created_at, initial_sequence, final_sequence, use_llm, iterations, protein_length)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (created, initial_sequence, final_sequence, 1 if use_llm else 0, iterations, protein_length),
            )
            conn.commit()
            run_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            return run_id
        finally:
            conn.close()


def _get_history(limit: int = 100) -> list[dict]:
    _init_db()
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT id, created_at, initial_sequence, final_sequence, use_llm, iterations, protein_length, image_path FROM runs ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def _update_run_image_path(run_id: int, image_path: str) -> None:
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        try:
            conn.execute("UPDATE runs SET image_path = ? WHERE id = ?", (image_path, run_id))
            conn.commit()
        finally:
            conn.close()


def _pdb_to_png(pdb_path: Path, out_path: Path) -> bool:
    """Render PDB CA trace to a PNG image. Returns True on success."""
    try:
        from Bio.PDB import PDBParser
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("s", str(pdb_path))
        model = structure[0]
        coords = []
        for chain in model:
            for res in chain.get_residues():
                if res.id[0] != " ":
                    continue
                if "CA" in res:
                    coords.append(res["CA"].get_coord())
        if len(coords) < 2:
            return False
        import numpy as np
        xyz = np.array(coords)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z, c=z, cmap="viridis", s=8, alpha=0.9)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=15, azim=45)
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        plt.close()
        return True
    except Exception as e:
        logger.warning("PDB to image failed: %s", e)
        return False


def _generate_run_image_background(pdb_path: Path, run_id: int) -> None:
    """Run in background: generate PNG from PDB and update DB."""
    images_dir = DATA_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    out_path = images_dir / f"{run_id}.png"
    if _pdb_to_png(pdb_path, out_path):
        _update_run_image_path(run_id, f"images/{run_id}.png")
    else:
        logger.warning("Could not generate image for run %s", run_id)


def _get_total_proteins() -> int:
    _init_db()
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        try:
            cur = conn.execute("SELECT COUNT(*) FROM runs")
            return cur.fetchone()[0]
        finally:
            conn.close()


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
    # LLM (API key from OPENAI_API_KEY env only)
    use_llm: bool = Field(default=False)
    llm_provider: str = Field(default="openai")
    llm_model: str = Field(default="gpt-4o-mini")
    # Folding
    modal_fold: bool = Field(default=False)
    modal_parallel: bool = Field(default=True)
    fold_backend: str = Field(default="dummy")
    remote_fold_backend: str = Field(default="esmfold")
    # Rosetta (norm target/scale and scoring weights are hardcoded server-side)
    use_rosetta: bool = Field(default=True)
    rosetta_relax: bool = Field(default=False)
    rosetta_relax_cycles: int = Field(default=0, ge=0, le=20)
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
    global _run_active, _last_output_dir
    _last_output_dir = (Path.cwd() / req.output_dir).resolve()
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
        rosetta_norm_target=-200.0,
        rosetta_norm_scale=50.0,
        w_physics=0.55,
        w_objective=0.35,
        w_confidence=0.10,
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
        # Record run in SQLite for history and total-proteins counter
        metrics_path = _last_output_dir / "metrics.json"
        final_pdb = _last_output_dir / "final_structure.pdb"
        if metrics_path.is_file():
            try:
                data = json.loads(metrics_path.read_text())
                initial = data.get("initial_sequence") or sequence
                final = data.get("final_sequence") or sequence
                iters = data.get("total_iterations")
                if iters is None:
                    iters = req.max_iterations
                run_id = _record_run(
                    initial_sequence=initial,
                    final_sequence=final,
                    use_llm=req.use_llm,
                    iterations=iters,
                    protein_length=len(final),
                )
                if final_pdb.is_file() and run_id:
                    threading.Thread(
                        target=_generate_run_image_background,
                        args=(final_pdb, run_id),
                        daemon=True,
                    ).start()
            except Exception as db_err:
                logger.warning("Failed to record run to history: %s", db_err)
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

    if req.use_llm:
        llm_cfg = LLMConfig(
            provider=req.llm_provider,
            model=req.llm_model,
            api_key=None,
            temperature=0.7,
        )
        try:
            llm_cfg.resolve_api_key()
        except ValueError as e:
            return {"ok": False, "error": str(e) + " Export it before starting the dashboard, e.g. export OPENAI_API_KEY=sk-..."}

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


@app.get("/api/final-pdb", response_model=None)
def api_final_pdb() -> PlainTextResponse:
    """Return the final_structure.pdb content from the last run (for modal viewer)."""
    global _last_output_dir
    if _last_output_dir is None:
        return PlainTextResponse(content="No run completed yet.", status_code=404)
    pdb_path = _last_output_dir / "final_structure.pdb"
    if not pdb_path.is_file():
        return PlainTextResponse(content="Final structure file not found.", status_code=404)
    return PlainTextResponse(content=pdb_path.read_text())


@app.get("/api/history")
def api_history(limit: int = 100) -> dict:
    """Return run history for the History tab (start/end sequence, LLM, iterations, length)."""
    if limit < 1 or limit > 500:
        limit = 100
    rows = _get_history(limit=limit)
    # Serialize for JSON (sqlite3.Row may have int for use_llm)
    return {
        "runs": [
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "initial_sequence": r["initial_sequence"],
                "final_sequence": r["final_sequence"],
                "use_llm": bool(r["use_llm"]),
                "iterations": r["iterations"],
                "protein_length": r["protein_length"],
                "has_image": bool(r.get("image_path")),
            }
            for r in rows
        ],
    }


@app.get("/api/run-image/{run_id:int}", response_model=None)
def api_run_image(run_id: int):
    """Return the stored 3D image PNG for a run (for History tab eye button)."""
    rows = _get_history(limit=10000)
    run = next((r for r in rows if r["id"] == run_id), None)
    if not run or not run.get("image_path"):
        raise HTTPException(status_code=404, detail="No image for this run")
    image_path = DATA_DIR / run["image_path"]
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image file not found")
    return FileResponse(image_path, media_type="image/png")


@app.get("/api/stats")
def api_stats() -> dict:
    """Return total number of proteins (runs) for the live counter."""
    return {"total_proteins": _get_total_proteins()}


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
        raise HTTPException(status_code=404, detail=f"Not found: {path}")
    return FileResponse(full)


def create_app() -> FastAPI:
    """Return the FastAPI app (for uvicorn)."""
    return app
