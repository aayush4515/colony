from __future__ import annotations

import modal
from protein_swarm.modal_app.app import app, agent_image, fold_image

GPU_TYPE = "H100"

_model = None
_tokenizer = None


@app.function(image=agent_image, timeout=120)
def run_residue_agent_remote(agent_input_dict: dict) -> dict:
    """Execute a single residue agent on Modal infrastructure."""
    from protein_swarm.schemas import AgentInput
    from protein_swarm.agents.residue_agent import run_residue_agent_local

    agent_input = AgentInput(**agent_input_dict)
    proposal = run_residue_agent_local(agent_input)
    return proposal.model_dump()


@app.function(
    image=fold_image,
    gpu=GPU_TYPE,
    timeout=60 * 10,
    retries=1,
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_esmfold(sequence: str) -> dict:
    """Run ESMFold on GPU. Returns {"pdb": str, "mean_plddt": float}."""
    global _model, _tokenizer

    import os
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    sequence = "".join(ch for ch in sequence.upper().strip() if ch.isalpha())

    # Use HF_TOKEN from Modal secret for authenticated Hub requests (avoids rate-limit warning)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if _model is None:
        model_name = "facebook/esmfold_v1"
        _tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        _model = EsmForProteinFolding.from_pretrained(model_name, token=hf_token)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = _model.to(device)
        _model.eval()

    device = next(_model.parameters()).device

    with torch.no_grad():
        if hasattr(_model, "infer_pdb"):
            pdb_text = _model.infer_pdb(sequence)
            # Get pLDDT by running forward pass
            inputs = _tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = _model(**inputs)
            plddt = getattr(outputs, "plddt", None)
            mean_plddt = float(plddt.detach().float().mean().item()) if plddt is not None else 50.0
        else:
            inputs = _tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = _model(**inputs)

            plddt = getattr(outputs, "plddt", None)
            mean_plddt = float(plddt.detach().float().mean().item()) if plddt is not None else 50.0

            pdbs = _model.output_to_pdb(outputs)
            if not pdbs:
                raise RuntimeError("ESMFold output_to_pdb returned empty list")
            pdb_text = pdbs[0] if isinstance(pdbs, list) else str(pdbs)

    return {"pdb": pdb_text, "mean_plddt": mean_plddt}
