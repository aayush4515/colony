"""
CLI entry point for the Protein Swarm Design Engine.

Usage:
    python -m protein_swarm.main design \
        --sequence "ACDEFGHIKLMNPQRSTVWY" \
        --objective "Design a stable helix-rich protein" \
        --max-iterations 20

    python -m protein_swarm.main design --help
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console

from protein_swarm.config import SwarmConfig, FoldingConfig, MemoryConfig, LLMConfig
from protein_swarm.orchestrator.engine import DesignEngine
from protein_swarm.utils.constants import AMINO_ACIDS

app = typer.Typer(
    name="protein-swarm",
    help="Swarm-Based Protein Design Engine — distributed multi-agent sequence optimisation.",
    add_completion=False,
)
console = Console()


def _validate_sequence(sequence: str) -> str:
    sequence = sequence.upper().strip()
    invalid = [ch for ch in sequence if ch not in AMINO_ACIDS]
    if invalid:
        raise typer.BadParameter(
            f"Invalid amino acid(s): {set(invalid)}. "
            f"Allowed: {''.join(AMINO_ACIDS)}"
        )
    if len(sequence) < 2:
        raise typer.BadParameter("Sequence must have at least 2 residues.")
    return sequence


@app.command()
def design(
    sequence: str = typer.Option(..., "--sequence", "-s", help="Initial amino acid sequence"),
    objective: str = typer.Option(..., "--objective", "-o", help="Natural language design goal"),
    max_iterations: int = typer.Option(50, "--max-iterations", "-n", help="Maximum design iterations"),
    mutation_rate: float = typer.Option(0.3, "--mutation-rate", "-m", help="Base mutation probability [0-1]"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducibility"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable verbose debug output"),
    no_modal: bool = typer.Option(False, "--no-modal", help="Run locally without Modal"),
    fold_backend: str = typer.Option(
        "dummy",
        "--fold-backend",
        help="Local folding backend: dummy | esmfold-local",
    ),
    modal_fold: bool = typer.Option(False, "--modal-fold", help="Run folding engine on Modal (remote)"),
    remote_fold_backend: str = typer.Option(
        "esmfold",
        "--remote-fold-backend",
        help="Remote (Modal) folding backend when --modal-fold: dummy | esmfold",
    ),
    modal_fold_gpu: bool = typer.Option(False, "--modal-fold-gpu", help="Use GPU worker for Modal folding"),
    output_dir: str = typer.Option("outputs", "--output-dir", help="Directory for output artefacts"),
    confidence_threshold: float = typer.Option(0.5, "--confidence-threshold", help="Minimum agent confidence"),
    plateau_window: int = typer.Option(5, "--plateau-window", help="Iterations for plateau detection"),
    # ── LLM flags ────────────────────────────────────────────────────────
    use_llm: bool = typer.Option(False, "--use-llm", help="Use LLM-backed agents instead of heuristics"),
    llm_provider: str = typer.Option("openai", "--llm-provider", help="LLM provider: openai | anthropic | together"),
    llm_model: str = typer.Option("gpt-4o-mini", "--llm-model", help="LLM model identifier"),
    llm_api_key: Optional[str] = typer.Option(None, "--llm-api-key", help="LLM API key (or set OPENAI_API_KEY env var)"),
    llm_temperature: float = typer.Option(0.7, "--llm-temperature", help="LLM sampling temperature"),
    # ── Rosetta / scoring flags ──────────────────────────────────────────
    use_rosetta: bool = typer.Option(True, "--use-rosetta/--no-rosetta", help="Enable local PyRosetta scoring"),
    rosetta_relax: bool = typer.Option(False, "--rosetta-relax", help="Run FastRelax before scoring"),
    rosetta_relax_cycles: int = typer.Option(1, "--rosetta-relax-cycles", help="FastRelax cycles"),
    w_physics: float = typer.Option(0.55, "--w-physics", help="Scoring weight: physics (Rosetta)"),
    w_objective: float = typer.Option(0.35, "--w-objective", help="Scoring weight: objective heuristic"),
    w_confidence: float = typer.Option(0.10, "--w-confidence", help="Scoring weight: pLDDT confidence"),
    rosetta_norm_target: float = typer.Option(-200.0, "--rosetta-norm-target", help="Sigmoid centre for Rosetta normalisation"),
    rosetta_norm_scale: float = typer.Option(50.0, "--rosetta-norm-scale", help="Sigmoid scale for Rosetta normalisation"),
    # ── Prompt debug flags ───────────────────────────────────────────────
    dump_prompts: bool = typer.Option(False, "--dump-prompts", help="Dump LLM prompts to outputs/debug/prompts/ (requires --debug)"),
) -> None:
    """Run the swarm-based protein design loop.

    Modal flags require a running Modal app. Deploy first with
    'modal deploy protein_swarm/modal_app/functions.py'.
    Use --no-modal for fully local execution.
    """
    sequence = _validate_sequence(sequence)

    if modal_fold_gpu and not modal_fold:
        modal_fold = True

    allowed_local = {"dummy", "esmfold-local"}
    if fold_backend not in allowed_local:
        raise typer.BadParameter(f"--fold-backend must be one of {sorted(allowed_local)}")

    allowed_remote = {"dummy", "esmfold"}
    if remote_fold_backend not in allowed_remote:
        raise typer.BadParameter(f"--remote-fold-backend must be one of {sorted(allowed_remote)}")

    llm_cfg = LLMConfig(
        provider=llm_provider,
        model=llm_model,
        api_key=llm_api_key,
        temperature=llm_temperature,
    )

    folding_cfg = FoldingConfig(
        use_rosetta=use_rosetta,
        rosetta_relax=rosetta_relax,
        rosetta_relax_cycles=rosetta_relax_cycles,
        w_physics=w_physics,
        w_objective=w_objective,
        w_confidence=w_confidence,
        rosetta_norm_target=rosetta_norm_target,
        rosetta_norm_scale=rosetta_norm_scale,
    )

    console.print(f"\n[bold]Protein Swarm Design Engine[/bold]")
    console.print(f"  Sequence length : {len(sequence)}")
    console.print(f"  Objective       : {objective}")
    console.print(f"  Max iterations  : {max_iterations}")
    console.print(f"  Mutation rate   : {mutation_rate}")
    console.print(f"  Modal parallel  : {not no_modal}")
    console.print(
        f"  Modal fold      : {modal_fold}"
        + (f" (GPU)" if modal_fold_gpu else " (CPU)" if modal_fold else "")
    )
    console.print(
        f"  Fold backend    : {fold_backend}"
        if not modal_fold
        else f"  Fold backend    : Modal/{remote_fold_backend}"
    )
    console.print(
        f"  Rosetta scoring : {'ON' if use_rosetta else 'OFF'}"
        + (f"  relax={rosetta_relax} cycles={rosetta_relax_cycles}" if use_rosetta else "")
    )
    console.print(
        f"  Score weights   : physics={folding_cfg.w_physics:.2f}  "
        f"objective={folding_cfg.w_objective:.2f}  "
        f"confidence={folding_cfg.w_confidence:.2f}"
    )
    console.print(
        f"  LLM agents      : {use_llm} ({llm_provider}/{llm_model})"
        if use_llm
        else f"  LLM agents      : {use_llm}"
    )
    console.print(f"  Output dir      : {output_dir}")
    console.print()

    swarm_cfg = SwarmConfig(
        use_llm_agents=use_llm,
        llm=llm_cfg,
        max_iterations=max_iterations,
        mutation_rate=mutation_rate,
        random_seed=seed,
        debug=debug,
        modal_parallel=not no_modal,
        modal_fold=modal_fold,
        modal_fold_gpu=modal_fold_gpu,
        fold_backend=fold_backend,
        remote_fold_backend=remote_fold_backend,
        output_dir=output_dir,
        confidence_threshold=confidence_threshold,
        plateau_window=plateau_window,
    )

    if dump_prompts:
        console.print(f"  Prompt dump     : ON (outputs/debug/prompts/)")

    engine = DesignEngine(
        swarm_config=swarm_cfg,
        folding_config=folding_cfg,
        memory_config=MemoryConfig(),
        dump_prompts=dump_prompts and debug,
    )

    try:
        result = engine.run(sequence, objective)
    except RuntimeError as e:
        if "Modal" in str(e):
            console.print(f"\n[bold red]Error:[/bold red] {e}")
            raise typer.Exit(code=1)
        raise

    console.print(f"\n[bold green]Done.[/bold green]  Artefacts saved to [cyan]{output_dir}/[/cyan]")
    console.print(f"  final_sequence.txt   — optimised sequence")
    console.print(f"  final_structure.pdb  — PDB structure file")
    console.print(f"  metrics.json         — run metrics")
    console.print(f"  history.json         — full iteration log")


@app.command()
def dashboard(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
) -> None:
    """Start the Protein Swarm dashboard (web UI)."""
    import uvicorn
    uvicorn.run(
        "protein_swarm.dashboard.server:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def validate(
    sequence: str = typer.Option(..., "--sequence", "-s", help="Amino acid sequence to validate"),
) -> None:
    """Validate an amino acid sequence without running the design loop."""
    try:
        seq = _validate_sequence(sequence)
        console.print(f"[green]Valid sequence[/green] ({len(seq)} residues): {seq}")
    except typer.BadParameter as e:
        console.print(f"[red]Invalid:[/red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
