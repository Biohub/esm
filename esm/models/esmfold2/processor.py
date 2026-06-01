import random
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from esm.models.esmfold2.conformers import load_ccd
from esm.models.esmfold2.output import build_molecular_complex_from_features
from esm.models.esmfold2.prepare_input import ChainInfo, prepare_esmfold2_input
from esm.models.esmfold2.types import (
    MSA,
    Modification,
    ProteinInput,
    StructurePredictionInput,
)
from esm.utils.structure.molecular_complex import MolecularComplexResult


@contextmanager
def _seed_context(seed: int | None):
    if seed is None:
        yield
        return
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def clean_esmfold2_input(input: StructurePredictionInput) -> StructurePredictionInput:
    """Group identical protein sequences into the same ProteinInput with multiple ids.

    Example: Passing a tetramer like [ProteinInput(id=["0"], seq="AAA|AAA|BBB|BBB")]
    gets converted into [ProteinInput(id=["0_0", "0_1"], seq="AAA"),
                         ProteinInput(id=["0_2", "0_3"], seq="BBB")]

    Preserves the original order of unique sequences. Also converts "|" chainbreak
    tokens to ":" in the sequence.
    """
    cleaned_sequences: list = []
    chain_to_ids: dict[str, list[str]] = {}
    chain_to_modifications: dict[str, list] = {}
    chain_to_msa: dict[str, MSA | None] = {}

    for item in input.sequences:
        if isinstance(item, ProteinInput):
            sequence = ":".join(item.sequence.split("|"))
            if ":" not in sequence:
                cleaned_sequences.append(item)
                continue

            if ":" in sequence and input.covalent_bonds is not None:
                raise ValueError(
                    "Covalent bonds are not supported when using chainbreaks. "
                    "Chains must be separated into multiple ProteinInput objects."
                )

            base_id = item.id[0] if isinstance(item.id, list) else item.id
            chain_to_ids = {}
            chain_to_modifications = {}
            chain_to_msa = {}
            chains = sequence.split(":")

            chain_start_positions = []
            pos = 0
            for chain in chains:
                chain_start_positions.append(pos)
                pos += len(chain) + 1

            if item.modifications is not None:
                for chain_idx, chain in enumerate(chains):
                    chain_start = chain_start_positions[chain_idx]
                    chain_end = chain_start + len(chain)
                    chain_modifications = []
                    for mod in item.modifications:
                        if chain_start <= mod.position < chain_end:
                            adjusted_mod = Modification(
                                position=mod.position - chain_start, ccd=mod.ccd
                            )
                            chain_modifications.append(adjusted_mod)
                    if chain not in chain_to_modifications:
                        chain_to_modifications[chain] = chain_modifications
                    else:
                        chain_to_modifications[chain].extend(chain_modifications)

            if item.msa is not None:
                for chain_idx, chain in enumerate(chains):
                    if chain not in chain_to_msa:
                        chain_start = chain_start_positions[chain_idx]
                        chain_end = chain_start + len(chain)
                        chain_msa = item.msa.select_positions(  # type: ignore
                            np.arange(chain_start, chain_end)
                        )
                        chain_to_msa[chain] = chain_msa

            for i, chain in enumerate(chains):
                chain_id = base_id + "_" + str(i)
                if chain in chain_to_ids:
                    chain_to_ids[chain].append(chain_id)
                else:
                    chain_to_ids[chain] = [chain_id]
                    cleaned_sequences.append((item, chain))
        else:
            cleaned_sequences.append(item)

    for i in range(len(cleaned_sequences)):
        if isinstance(cleaned_sequences[i], tuple):
            item, chain = cleaned_sequences[i]
            chain_ids = chain_to_ids[chain]
            chain_modifications = (
                chain_to_modifications.get(chain) if item.modifications else None
            )
            chain_msa = chain_to_msa.get(chain) if item.msa else None
            cleaned_sequences[i] = ProteinInput(
                id=chain_ids,
                sequence=chain,
                msa=chain_msa,
                modifications=chain_modifications,
            )

    return StructurePredictionInput(
        sequences=cleaned_sequences,
        distogram_conditioning=input.distogram_conditioning,
        covalent_bonds=input.covalent_bonds,
    )


class ESMFold2InputBuilder:
    def __init__(self, ccd_cache: Path | None = None):
        load_ccd(ccd_cache)

    def prepare_input(
        self,
        input: StructurePredictionInput,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[dict, list[ChainInfo]]:
        """Prepare raw input for the folding model.

        Converts user-provided StructurePredictionInput into batched tensors
        ready for model inference.

        Parameters
        ----------
        input : StructurePredictionInput
            Input specification (sequences, structures, constraints, etc.).
        seed : int, optional
            Random seed for reproducibility.
        device : torch.device or str, optional
            Target device for the returned tensors. Defaults to CPU; pass
            ``model.device`` to skip a separate ``.to(...)`` step. ``fold()``
            forwards ``model.device`` automatically.

        Returns
        -------
        tuple[dict, list[ChainInfo]]
            Batched input tensors and chain metadata for output processing.
        """
        structure_prediction_input = clean_esmfold2_input(input)
        with _seed_context(seed) if seed is not None else nullcontext():
            features, chain_infos = prepare_esmfold2_input(
                structure_prediction_input, seed=seed
            )
            features = {
                k: (v[None].to(device) if device is not None else v[None])
                if isinstance(v, torch.Tensor)
                else v
                for k, v in features.items()
            }

        return features, chain_infos

    def __call__(
        self,
        input: StructurePredictionInput,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[dict, list[ChainInfo]]:
        return self.prepare_input(input, seed=seed, device=device)

    def decode(
        self,
        output: dict[str, torch.Tensor],
        features: dict[str, torch.Tensor],
        chain_infos: list[ChainInfo],
        *,
        num_diffusion_samples: int = 1,
        complex_id: str = "pred",
    ) -> MolecularComplexResult | list[MolecularComplexResult]:
        """Convert raw model outputs into one MolecularComplexResult per sample.

        Parameters
        ----------
        output : dict[str, Tensor]
            Output dict returned by ESMFold2Model.forward.
        features : dict[str, Tensor]
            Feature dict from :meth:`prepare_input` (batched, on the model device).
        chain_infos : list[ChainInfo]
            Chain metadata returned alongside `features`.
        num_diffusion_samples : int
            Number of diffusion samples present in the output (Bm = B * num_diffusion_samples).
        complex_id : str
            Identifier assigned to each MolecularComplex.

        Returns
        -------
        MolecularComplexResult or list[MolecularComplexResult]
            A single result when num_diffusion_samples == 1, otherwise a list of length Bm.
        """
        atom_mask = features["atom_attention_mask"][0]
        ref_element = features["ref_element"][0]
        ref_atom_name_chars = features["ref_atom_name_chars"][0]

        sample_coords = output["sample_atom_coords"]
        plddts = output["plddt"]
        Bm = sample_coords.shape[0]

        ptm_t = output.get("ptm")
        iptm_t = output.get("iptm")
        pae_t = output.get("pae")
        distogram_t = output.get("distogram_logits")
        pair_chains_t = output.get("pair_chains_iptm")
        residue_index_t = output.get("residue_index")
        entity_id_t = output.get("entity_id")

        results: list[MolecularComplexResult] = []
        for i in range(Bm):
            mc = build_molecular_complex_from_features(
                coords=sample_coords[i],
                plddt=plddts[i],
                atom_mask=atom_mask,
                ref_element=ref_element,
                ref_atom_name_chars=ref_atom_name_chars,
                chain_infos=chain_infos,
                complex_id=complex_id,
            )
            results.append(
                MolecularComplexResult(
                    complex=mc,
                    plddt=plddts[i].detach().cpu(),
                    ptm=float(ptm_t[i].item()) if ptm_t is not None else None,
                    iptm=float(iptm_t[i].item()) if iptm_t is not None else None,
                    pae=pae_t[i].detach().cpu() if pae_t is not None else None,
                    distogram=(
                        distogram_t[0].detach().cpu()
                        if distogram_t is not None
                        else None
                    ),
                    pair_chains_iptm=(
                        pair_chains_t[i].detach().cpu()
                        if pair_chains_t is not None
                        else None
                    ),
                    residue_index=(
                        residue_index_t[0].detach().cpu()
                        if residue_index_t is not None
                        else None
                    ),
                    entity_id=(
                        entity_id_t[0].detach().cpu()
                        if entity_id_t is not None
                        else None
                    ),
                )
            )

        if num_diffusion_samples == 1 and len(results) == 1:
            return results[0]
        return results

    def fold(
        self,
        model: Any,
        input: StructurePredictionInput,
        *,
        num_loops: int = 20,
        num_sampling_steps: int = 200,
        num_diffusion_samples: int = 1,
        seed: int | None = None,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        max_inference_sigma: float | None = None,
        early_exit: bool = False,
        complex_id: str = "pred",
    ) -> MolecularComplexResult | list[MolecularComplexResult]:
        """Fold a structure end-to-end: encode → model → decode.

        Parameters
        ----------
        model : ESMFold2Model
            The folding model. Must already be on the target device and in eval mode.
        input : StructurePredictionInput
            User-facing input specification.
        num_loops, num_sampling_steps, num_diffusion_samples : int
            Inference knobs forwarded to the model.
        seed : int, optional
            Seeds both input prep (SMILES conformer generation) and diffusion sampling.
        noise_scale, step_scale, max_inference_sigma, early_exit
            Optional sampler overrides forwarded to the model when not None.
        complex_id : str
            Identifier assigned to the predicted MolecularComplex(es).

        Returns
        -------
        MolecularComplexResult or list[MolecularComplexResult]
            A single result when num_diffusion_samples == 1, otherwise a list.
        """
        features, chain_infos = self.prepare_input(
            input, seed=seed, device=model.device
        )

        sampler_kwargs: dict[str, Any] = {}
        if noise_scale is not None:
            sampler_kwargs["noise_scale"] = noise_scale
        if step_scale is not None:
            sampler_kwargs["step_scale"] = step_scale
        if max_inference_sigma is not None:
            sampler_kwargs["max_inference_sigma"] = max_inference_sigma

        with torch.no_grad():
            with _seed_context(seed) if seed is not None else nullcontext():
                output = model(
                    **features,
                    num_loops=num_loops,
                    num_sampling_steps=num_sampling_steps,
                    num_diffusion_samples=num_diffusion_samples,
                    early_exit=early_exit,
                    **sampler_kwargs,
                )

        return self.decode(
            output,
            features,
            chain_infos,
            num_diffusion_samples=num_diffusion_samples,
            complex_id=complex_id,
        )


__all__ = ["ESMFold2InputBuilder", "clean_esmfold2_input"]
