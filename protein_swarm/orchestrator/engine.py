"""
Main orchestration engine — runs the design loop locally and dispatches
agent work to Modal (or runs locally in non-Modal mode).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Callable

import numpy as np

from protein_swarm.agents.constraint_guard import validate_proposals
from protein_swarm.agents.memory_curator import curate_memory
from protein_swarm.agents.objective_compiler import compile_objective
from protein_swarm.agents.residue_agent import run_residue_agent_local
from protein_swarm.config import SwarmConfig, FoldingConfig, MemoryConfig
from protein_swarm.folding.fold_engine import DummyFoldEngine, FoldEngine
from protein_swarm.folding.goal_eval import evaluate_design_goal
from protein_swarm.folding.structure_analysis import (
    build_structure_context,
    compute_distance_matrix_from_pdb,
    dssp_secondary_structure,
)
from protein_swarm.memory.memory_store import MemoryStore
from protein_swarm.orchestrator.decision import should_accept, should_stop
from protein_swarm.orchestrator.mutation_merge import merge_mutations
from protein_swarm.schemas import (
    AgentInput,
    DesignResult,
    FoldResult,
    GoalEvaluation,
    GlobalMemoryStats,
    IterationRecord,
    MutationProposal,
    ObjectiveSpec,
    StructureContext,
)
from protein_swarm.utils.logging import (
    log_debug,
    log_early_stop,
    log_final_result,
    log_iteration_header,
    log_mutation_summary,
    log_proposals_table,
    log_score_delta,
)

logger = logging.getLogger(__name__)


def _rosetta_to_physics_score(
    rosetta_total: float,
    norm_target: float,
    norm_scale: float,
) -> float:
    """Convert Rosetta energy (lower=better) to a 0-1 score (higher=better)."""
    return 1.0 / (1.0 + math.exp((rosetta_total - norm_target) / norm_scale))


class DesignEngine:
    """Orchestrates the full multi-agent design loop."""

    def __init__(
        self,
        swarm_config: SwarmConfig | None = None,
        folding_config: FoldingConfig | None = None,
        memory_config: MemoryConfig | None = None,
        fold_engine: FoldEngine | None = None,
        dump_prompts: bool = False,
    ) -> None:
        self._cfg = swarm_config or SwarmConfig()
        self._fold_cfg = folding_config or FoldingConfig()
        self._mem_cfg = memory_config or MemoryConfig()
        self._fold_engine: FoldEngine = fold_engine or self._make_fold_engine()
        self._dump_prompts = dump_prompts

    def _make_fold_engine(self) -> FoldEngine:
        backend = self._cfg.fold_backend

        if backend == "dummy":
            return DummyFoldEngine(self._fold_cfg)

        if backend == "esmfold-local":
            from protein_swarm.folding.fold_engine import ESMFoldEngine
            return ESMFoldEngine(self._fold_cfg)

        raise ValueError(f"Unknown fold_backend='{backend}'")

    # ── throttling helpers ───────────────────────────────────────────────

    def _effective_mutation_rate(self, consecutive_rejects: int) -> float:
        cfg = self._cfg
        if consecutive_rejects < cfg.reject_throttle_after:
            return cfg.mutation_rate
        rate = cfg.mutation_rate * (cfg.reject_mutation_decay ** consecutive_rejects)
        return max(rate, cfg.min_mutation_rate)

    def _effective_conf_threshold(self, consecutive_rejects: int) -> float:
        cfg = self._cfg
        if consecutive_rejects < cfg.reject_throttle_after:
            return cfg.confidence_threshold
        threshold = cfg.confidence_threshold + cfg.reject_conf_bump * consecutive_rejects
        return min(threshold, 0.99)

    # ── structure context computation (once per iteration) ───────────────

    def _compute_iteration_context(
        self,
        sequence: str,
        objective: ObjectiveSpec,
        last_pdb_path: str | None,
    ) -> tuple[
        list[StructureContext],
        dict[int, str] | None,
        np.ndarray | None,
        list[tuple[int, str]] | None,
        GoalEvaluation,
    ]:
        """Pre-compute structural context + goal evaluation for all positions.

        Returns per-position StructureContext list, dssp_map, dist_matrix,
        ca_positions, and GoalEvaluation.
        """
        n = len(sequence)
        dist_matrix: np.ndarray | None = None
        ca_positions: list[tuple[int, str]] | None = None
        dssp_map: dict[int, str] | None = None

        if last_pdb_path and Path(last_pdb_path).exists():
            try:
                ca_positions, dist_matrix = compute_distance_matrix_from_pdb(last_pdb_path)
            except Exception as e:
                logger.warning("Distance matrix computation failed: %s", e)
                dist_matrix = None
                ca_positions = None

            try:
                dssp_map = dssp_secondary_structure(last_pdb_path)
                if not dssp_map:
                    dssp_map = None
            except Exception as e:
                logger.warning("DSSP computation failed: %s", e)
                dssp_map = None

        contexts: list[StructureContext] = []
        for pos in range(n):
            ctx = build_structure_context(
                sequence, pos, last_pdb_path,
                dist_matrix=dist_matrix,
                ca_positions=ca_positions,
                dssp_map=dssp_map,
            )
            contexts.append(ctx)

        goal_eval = evaluate_design_goal(sequence, objective, dssp_map)

        return contexts, dssp_map, dist_matrix, ca_positions, goal_eval

    # ── main loop ────────────────────────────────────────────────────────

    def run(
        self,
        sequence: str,
        objective_text: str,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> DesignResult:
        """Execute the full design loop and return the result.

        If progress_callback is provided, it is called with JSON-serializable
        event dicts (run_start, iteration_start, agents_started, agents_completed,
        sequence_update, run_complete) for dashboard streaming.
        """
        def emit(event: dict) -> None:
            if progress_callback:
                progress_callback(event)

        objective = compile_objective(
            objective_text,
            use_llm=self._cfg.use_llm_agents,
            llm_config=self._cfg.llm if self._cfg.use_llm_agents else None,
        )
        memory = MemoryStore(len(sequence), self._mem_cfg)
        output_dir = Path(self._cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        current_seq = sequence
        best_score = float("-inf")
        score_history: list[float] = []
        history: list[IterationRecord] = []
        consecutive_rejects = 0
        last_pdb_path: str | None = None
        last_accepted_fold: FoldResult | None = None

        emit({"type": "run_start", "sequence_length": len(sequence), "max_iterations": self._cfg.max_iterations})

        if self._cfg.modal_fold:
            emit({"type": "log", "message": "Warming up modal container for esmfold", "icon": "spinner"})
        initial_fold = self._fold(current_seq, objective, output_dir, -1)
        if self._cfg.modal_fold:
            emit({"type": "log", "message": "Warming up modal container for esmfold", "icon": "check"})
        best_score = initial_fold.combined_score
        score_history.append(best_score)
        last_pdb_path = initial_fold.pdb_path
        last_accepted_fold = initial_fold

        def _fold_to_dict(fr: FoldResult) -> dict:
            d = fr.model_dump()
            out: dict = {}
            for k, v in d.items():
                if v is not None and hasattr(v, "item"):
                    out[k] = float(v)
                else:
                    out[k] = v
            return out

        emit({
            "type": "sequence_update",
            "current_sequence": current_seq,
            "best_score": float(best_score),
            "rosetta_energy": float(initial_fold.rosetta_total_score) if initial_fold.rosetta_total_score is not None else None,
            "energy_trend": "unknown",
            "mean_plddt": float(initial_fold.energy * 100.0) if initial_fold.energy is not None else None,
        })

        early_stop_reason: str = ""
        for iteration in range(self._cfg.max_iterations):
            emit({"type": "iteration_start", "iteration": iteration, "max_iterations": self._cfg.max_iterations})
            log_iteration_header(iteration, self._cfg.max_iterations)

            eff_mutation_rate = self._effective_mutation_rate(consecutive_rejects)
            eff_conf_threshold = self._effective_conf_threshold(consecutive_rejects)

            if self._cfg.debug:
                log_debug(
                    f"Throttle: consecutive_rejects={consecutive_rejects}  "
                    f"eff_mutation_rate={eff_mutation_rate:.4f}  "
                    f"eff_conf_threshold={eff_conf_threshold:.4f}"
                )

            # pre-compute structure + goal context once per iteration
            structure_contexts, dssp_map, _, _, goal_eval = (
                self._compute_iteration_context(current_seq, objective, last_pdb_path)
            )
            global_stats = memory.get_global_stats(last_k=5)

            if self._cfg.debug:
                log_debug(
                    f"Goal: {goal_eval.goal_score:.1f}/100 ({goal_eval.rating})  "
                    f"Energy trend: {global_stats.energy_trend}"
                )

            num_agents = len(current_seq)
            if iteration == 0 and self._cfg.modal_parallel and self._cfg.use_llm_agents:
                emit({"type": "log", "message": "Warming up modal container for llm agents", "icon": "spinner"})
            if iteration == 0 and self._cfg.modal_parallel and self._cfg.use_llm_agents:
                emit({"type": "log", "message": "Warming up modal container for llm agents", "icon": "check"})
            spawn_msg = f"Spawning {num_agents} agents for iteration {iteration + 1}"
            emit({"type": "agents_started", "iteration": iteration, "num_agents": num_agents})
            emit({"type": "log", "message": spawn_msg, "icon": "spinner"})
            proposals = self._run_agents(
                current_seq, memory, objective, iteration,
                mutation_rate_override=eff_mutation_rate,
                structure_contexts=structure_contexts,
                global_stats=global_stats,
                goal_eval=goal_eval,
            )
            emit({"type": "log", "message": spawn_msg, "icon": "check"})
            proposals = validate_proposals(proposals, objective, len(current_seq))

            if self._cfg.debug:
                log_proposals_table([p.model_dump() for p in proposals])

            candidate_seq, applied = merge_mutations(
                current_seq, proposals, eff_conf_threshold,
            )

            log_mutation_summary(len(current_seq), len(applied))

            fold_result = self._fold(candidate_seq, objective, output_dir, iteration)

            accepted = should_accept(fold_result.combined_score, best_score)
            score_delta = fold_result.combined_score - best_score
            log_score_delta(fold_result.combined_score, best_score, accepted)

            new_best = best_score if not accepted else float(fold_result.combined_score)
            emit({
                "type": "agents_completed",
                "iteration": iteration,
                "proposals": [p.model_dump() for p in proposals],
                "applied": [m.model_dump() for m in applied],
                "candidate_sequence": candidate_seq,
                "fold_result": _fold_to_dict(fold_result),
                "accepted": accepted,
                "best_score": float(new_best),
            })
            if accepted:
                emit({"type": "log", "message": "sequence accepted", "icon": "check", "variant": "success"})
            else:
                emit({"type": "log", "message": f"sequence rejected (Δ score {score_delta:+.4f})", "icon": "check", "variant": "error"})

            record = IterationRecord(
                iteration=iteration,
                sequence_before=current_seq,
                sequence_after=candidate_seq,
                mutations=applied,
                fold_result=fold_result,
                accepted=accepted,
                score_delta=score_delta,
            )
            history.append(record)

            num_mutations = len(applied)
            reason_str = f"{'ACCEPTED' if accepted else 'REJECTED'} delta={score_delta:+.4f}"

            memory.record_iteration_result(
                iteration=iteration,
                accepted=accepted,
                combined_score=fold_result.combined_score,
                objective_score=fold_result.objective_score,
                physics_score=fold_result.energy,
                rosetta_total_score=fold_result.rosetta_total_score,
                design_goal_score=goal_eval.goal_score if goal_eval else 0.0,
                num_mutations=num_mutations,
                sequence=candidate_seq,
                reason=reason_str,
            )

            if accepted:
                current_seq = candidate_seq
                best_score = fold_result.combined_score
                last_pdb_path = fold_result.pdb_path
                last_accepted_fold = fold_result
                memory.record_success(
                    applied, iteration=iteration,
                    combined_score=fold_result.combined_score,
                    objective_score=fold_result.objective_score,
                    physics_score=fold_result.energy,
                    rosetta_total_score=fold_result.rosetta_total_score,
                    design_goal_score=goal_eval.goal_score if goal_eval else None,
                )
                consecutive_rejects = 0
            else:
                memory.record_failure(
                    applied, iteration=iteration,
                    combined_score=fold_result.combined_score,
                    objective_score=fold_result.objective_score,
                    physics_score=fold_result.energy,
                    rosetta_total_score=fold_result.rosetta_total_score,
                    design_goal_score=goal_eval.goal_score if goal_eval else None,
                )
                consecutive_rejects += 1

            score_history.append(best_score)

            if last_accepted_fold is not None:
                emit({
                    "type": "sequence_update",
                    "current_sequence": current_seq,
                    "best_score": float(best_score),
                    "rosetta_energy": float(last_accepted_fold.rosetta_total_score) if last_accepted_fold.rosetta_total_score is not None else None,
                    "energy_trend": global_stats.energy_trend,
                    "mean_plddt": float(last_accepted_fold.energy * 100.0) if last_accepted_fold.energy is not None else None,
                })

            curate_memory(memory, iteration)

            stop, reason = should_stop(iteration, score_history, self._cfg)
            if stop:
                early_stop_reason = reason
                emit({"type": "log", "message": f"Terminated early: {reason}", "icon": "check"})
                log_early_stop(reason)
                break

        final_fold = self._fold(current_seq, objective, output_dir, self._cfg.max_iterations)
        log_final_result(current_seq, best_score, final_fold.pdb_path)

        result = DesignResult(
            initial_sequence=sequence,
            final_sequence=current_seq,
            objective=objective_text,
            best_score=best_score,
            total_iterations=len(history),
            history=history,
            final_pdb_path=final_fold.pdb_path,
        )

        self._save_artefacts(result, output_dir)
        emit({"type": "log", "message": f"Final Protein Design: {current_seq}", "icon": "check"})
        emit({
            "type": "run_complete",
            "final_sequence": current_seq,
            "best_score": float(best_score),
            "total_iterations": len(history),
            "early_stop_reason": early_stop_reason if early_stop_reason else None,
        })
        return result

    # ── agent dispatch ───────────────────────────────────────────────────

    def _run_agents(
        self,
        sequence: str,
        memory: MemoryStore,
        objective: ObjectiveSpec,
        iteration: int,
        *,
        mutation_rate_override: float | None = None,
        structure_contexts: list[StructureContext] | None = None,
        global_stats: GlobalMemoryStats | None = None,
        goal_eval: GoalEvaluation | None = None,
    ) -> list[MutationProposal]:
        if self._cfg.modal_parallel:
            return self._run_agents_modal(
                sequence, memory, objective, iteration,
                mutation_rate_override=mutation_rate_override,
                structure_contexts=structure_contexts,
                global_stats=global_stats,
                goal_eval=goal_eval,
            )
        return self._run_agents_local(
            sequence, memory, objective, iteration,
            mutation_rate_override=mutation_rate_override,
            structure_contexts=structure_contexts,
            global_stats=global_stats,
            goal_eval=goal_eval,
        )

    def _run_agents_local(
        self,
        sequence: str,
        memory: MemoryStore,
        objective: ObjectiveSpec,
        iteration: int,
        *,
        mutation_rate_override: float | None = None,
        structure_contexts: list[StructureContext] | None = None,
        global_stats: GlobalMemoryStats | None = None,
        goal_eval: GoalEvaluation | None = None,
    ) -> list[MutationProposal]:
        proposals: list[MutationProposal] = []
        for pos in range(len(sequence)):
            agent_input = self._build_agent_input(
                sequence, pos, memory, objective, iteration,
                mutation_rate_override=mutation_rate_override,
                structure_context=structure_contexts[pos] if structure_contexts else None,
                global_stats=global_stats,
                goal_eval=goal_eval,
            )
            proposal = run_residue_agent_local(agent_input)
            proposals.append(proposal)
        return proposals

    def _run_agents_modal(
        self,
        sequence: str,
        memory: MemoryStore,
        objective: ObjectiveSpec,
        iteration: int,
        *,
        mutation_rate_override: float | None = None,
        structure_contexts: list[StructureContext] | None = None,
        global_stats: GlobalMemoryStats | None = None,
        goal_eval: GoalEvaluation | None = None,
    ) -> list[MutationProposal]:
        import modal

        agent_fn = modal.Function.from_name("protein-swarm", "run_residue_agent_remote")

        inputs: list[AgentInput] = []
        for pos in range(len(sequence)):
            inputs.append(self._build_agent_input(
                sequence, pos, memory, objective, iteration,
                mutation_rate_override=mutation_rate_override,
                structure_context=structure_contexts[pos] if structure_contexts else None,
                global_stats=global_stats,
                goal_eval=goal_eval,
            ))

        input_dicts = [inp.model_dump() for inp in inputs]

        if self._cfg.debug:
            log_debug(f"Dispatching {len(inputs)} agents to Modal (iteration {iteration})")

        try:
            results = list(agent_fn.map(input_dicts))
        except Exception as e:
            raise RuntimeError(
                "Failed to call Modal agents. Make sure the app is deployed:\n"
                "  modal deploy protein_swarm/modal_app/functions.py\n"
                "Or run fully local with --no-modal"
            ) from e
        return [MutationProposal(**r) for r in results]

    def _build_agent_input(
        self,
        sequence: str,
        position: int,
        memory: MemoryStore,
        objective: ObjectiveSpec,
        iteration: int,
        *,
        mutation_rate_override: float | None = None,
        structure_context: StructureContext | None = None,
        global_stats: GlobalMemoryStats | None = None,
        goal_eval: GoalEvaluation | None = None,
    ) -> AgentInput:
        llm = self._cfg.llm

        pos_history = memory.get_position_history(position, last_k=10)
        nbr_history = memory.get_neighborhood_history(
            position, radius=self._cfg.neighbourhood_window, last_k=10,
        )

        return AgentInput(
            sequence=sequence,
            position=position,
            neighbourhood_window=self._cfg.neighbourhood_window,
            memory_summary=memory.get_summary_for_position(position),
            objective=objective,
            mutation_rate=mutation_rate_override if mutation_rate_override is not None else self._cfg.mutation_rate,
            random_seed=self._cfg.random_seed,
            use_llm=self._cfg.use_llm_agents,
            llm_provider=llm.provider,
            llm_model=llm.model,
            llm_api_key=llm.resolve_api_key() if self._cfg.use_llm_agents else None,
            llm_temperature=llm.temperature,
            llm_max_tokens=llm.max_tokens,
            llm_max_retries=llm.max_retries,
            structure_context=structure_context,
            global_memory_stats=global_stats,
            position_history=pos_history,
            neighborhood_history=nbr_history,
            goal_evaluation=goal_eval,
            iteration=iteration,
            dump_prompt=self._dump_prompts and self._cfg.use_llm_agents,
        )

    # ── folding / scoring ────────────────────────────────────────────────

    def _fold(
        self,
        sequence: str,
        objective: ObjectiveSpec,
        output_dir: Path,
        iteration: int,
    ) -> FoldResult:
        if self._cfg.modal_fold:
            return self._fold_modal(sequence, objective, iteration)
        return self._fold_local(sequence, objective, output_dir, iteration)

    def _fold_local(
        self,
        sequence: str,
        objective: ObjectiveSpec,
        output_dir: Path,
        iteration: int,
    ) -> FoldResult:
        result = self._fold_engine.fold_and_score(sequence, objective, output_dir, iteration)

        if not self._fold_cfg.use_rosetta:
            return result

        return self._rescore_with_rosetta(
            pdb_path=result.pdb_path,
            sequence=sequence,
            objective=objective,
            mean_plddt=result.energy * 100.0,
        )

    def _fold_modal(
        self,
        sequence: str,
        objective: ObjectiveSpec,
        iteration: int,
    ) -> FoldResult:
        import modal

        from protein_swarm.folding.scoring import compute_objective_score
        from protein_swarm.folding.structure_utils import sanitize_sequence, write_pdb_text

        sequence = sanitize_sequence(sequence)

        remote_backend = getattr(self._cfg, "remote_fold_backend", "esmfold")

        if remote_backend == "esmfold":
            fn_name = "run_esmfold"
        elif remote_backend == "dummy":
            fn_name = "run_fold_dummy_remote"
        else:
            raise ValueError(f"Unknown remote_fold_backend='{remote_backend}'")

        fold_fn = modal.Function.from_name("protein-swarm", fn_name)

        if self._cfg.debug:
            log_debug(f"Folding on Modal backend='{remote_backend}' (iteration {iteration})")

        try:
            if remote_backend == "esmfold":
                result = fold_fn.remote(sequence)
                pdb_text = result["pdb"]
                mean_plddt = float(result["mean_plddt"])

                output_dir = Path(self._cfg.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                pdb_path = write_pdb_text(pdb_text, output_dir / f"iter_{iteration}.pdb")

                return self._rescore_with_rosetta(
                    pdb_path=pdb_path,
                    sequence=sequence,
                    objective=objective,
                    mean_plddt=mean_plddt,
                )

            objective_dict = objective.model_dump()
            result_dict = fold_fn.remote(sequence, objective_dict, iteration)
            return FoldResult(**result_dict)

        except Exception as e:
            raise RuntimeError(
                f"Failed to call Modal fold function '{fn_name}' ({type(e).__name__}: {e}).\n"
                "Make sure the app is deployed: modal deploy protein_swarm/modal_app/functions.py\n"
                "Or remove --modal-fold to fold locally."
            ) from e

    def _rescore_with_rosetta(
        self,
        pdb_path: str,
        sequence: str,
        objective: ObjectiveSpec,
        mean_plddt: float,
    ) -> FoldResult:
        from protein_swarm.folding.scoring import compute_objective_score

        cfg = self._fold_cfg
        confidence_score = mean_plddt / 100.0
        obj_score = compute_objective_score(sequence, objective)

        rosetta_total: float | None = None

        if cfg.use_rosetta:
            from protein_swarm.folding.rosetta_energy import score_pdb_with_pyrosetta

            rosetta_result = score_pdb_with_pyrosetta(
                pdb_path,
                relax=cfg.rosetta_relax,
                relax_cycles=cfg.rosetta_relax_cycles,
            )
            rosetta_total = rosetta_result["rosetta_total_score"]
            physics_score = _rosetta_to_physics_score(
                rosetta_total, cfg.rosetta_norm_target, cfg.rosetta_norm_scale,
            )

            if rosetta_result["relaxed_pdb_path"]:
                pdb_path = rosetta_result["relaxed_pdb_path"]

            if self._cfg.debug:
                log_debug(
                    f"  pLDDT={mean_plddt:.1f}  confidence={confidence_score:.4f}  "
                    f"rosetta={rosetta_total:.1f}  physics={physics_score:.4f}  "
                    f"objective={obj_score:.4f}"
                )
        else:
            physics_score = confidence_score
            if self._cfg.debug:
                log_debug(
                    f"  pLDDT={mean_plddt:.1f}  confidence={confidence_score:.4f}  "
                    f"objective={obj_score:.4f}  (rosetta disabled, physics=confidence)"
                )

        combined = (
            cfg.w_physics * physics_score
            + cfg.w_objective * obj_score
            + cfg.w_confidence * confidence_score
        )

        if self._cfg.debug:
            log_debug(
                f"  combined = {cfg.w_physics:.2f}*{physics_score:.4f} "
                f"+ {cfg.w_objective:.2f}*{obj_score:.4f} "
                f"+ {cfg.w_confidence:.2f}*{confidence_score:.4f} "
                f"= {combined:.4f}"
            )

        return FoldResult(
            pdb_path=pdb_path,
            energy=round(physics_score, 6),
            objective_score=round(obj_score, 6),
            combined_score=round(combined, 6),
            rosetta_total_score=round(rosetta_total, 2) if rosetta_total is not None else None,
        )

    # ── artefact persistence ─────────────────────────────────────────────

    @staticmethod
    def _save_artefacts(result: DesignResult, output_dir: Path) -> None:
        (output_dir / "final_sequence.txt").write_text(result.final_sequence)

        import shutil
        final_pdb_dest = output_dir / "final_structure.pdb"
        if Path(result.final_pdb_path).exists():
            shutil.copy2(result.final_pdb_path, final_pdb_dest)

        metrics = {
            "initial_sequence": result.initial_sequence,
            "final_sequence": result.final_sequence,
            "objective": result.objective,
            "best_score": result.best_score,
            "total_iterations": result.total_iterations,
            "final_pdb_path": str(final_pdb_dest),
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        history_data = [rec.model_dump() for rec in result.history]
        (output_dir / "history.json").write_text(json.dumps(history_data, indent=2, default=str))
