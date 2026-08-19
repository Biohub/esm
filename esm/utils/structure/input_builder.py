from dataclasses import dataclass
from typing import Any, Literal, Sequence, TypeAlias, Union

import numpy as np

from esm.utils.msa import MSA

# fmt: off
MSAInput: TypeAlias = Union[
    MSA,
    None,
]
# fmt: on


@dataclass
class Modification:
    position: int  # zero-indexed
    ccd: str
    smiles: str | None = None  # TODO(mlee): add smiles support


@dataclass
class ProteinInput:
    id: str | list[str]
    sequence: str
    modifications: list[Modification] | None = None
    msa: MSAInput = None


@dataclass
class RNAInput:
    id: str | list[str]
    sequence: str
    modifications: list[Modification] | None = None
    msa: MSAInput = None


@dataclass
class DNAInput:
    id: str | list[str]
    sequence: str
    modifications: list[Modification] | None = None


@dataclass
class LigandInput:
    id: str | list[str]
    smiles: str | None = None
    ccd: list[str] | None = None


@dataclass
class DistogramConditioning:
    chain_id: str
    distogram: np.ndarray


@dataclass
class PocketConditioning:
    binder_chain_id: str
    contacts: list[tuple[str, int]]


@dataclass
class CovalentBond:
    chain_id1: str
    res_idx1: int
    atom_idx1: int
    chain_id2: str
    res_idx2: int
    atom_idx2: int


@dataclass
class StructurePredictionInput:
    sequences: Sequence[ProteinInput | RNAInput | DNAInput | LigandInput]
    pocket: PocketConditioning | None = None
    distogram_conditioning: list[DistogramConditioning] | None = None
    covalent_bonds: list[CovalentBond] | None = None


def _entity_key(item: ProteinInput | RNAInput | DNAInput | LigandInput) -> tuple:
    """Canonical identity of a chain's chemical entity, ignoring chain ids.

    Chains collapse to one entity only if they are chemically identical, so
    modifications are part of the key: a modified chain and its unmodified twin
    are separate entities, and merging them would make them ``sym_id`` copies of
    each other.

    Entity identity is chemical, not per-instance. Two copies of one entity can
    therefore still differ in token count -- a covalently bonded copy of a CCD
    ligand drops its leaving atoms while a free copy keeps them -- so "same
    entity" does not imply "same length".
    """
    if isinstance(item, LigandInput):
        # ccd wins over smiles at tokenization, so a redundant smiles alongside a
        # ccd must not split one entity into two.
        if item.ccd:
            return ("NONPOLYMER", None, tuple(item.ccd))
        return ("NONPOLYMER", item.smiles, ())
    if isinstance(item, ProteinInput):
        entity_type = "PROTEIN"
    elif isinstance(item, RNAInput):
        entity_type = "RNA"
    elif isinstance(item, DNAInput):
        entity_type = "DNA"
    else:  # pragma: no cover - the annotation excludes unsupported inputs
        raise TypeError(f"Unsupported sequence input type: {type(item)}")
    # `msa` is excluded on purpose: it is stored per chain name, so two copies of
    # one entity may legitimately carry different MSAs.
    return (
        entity_type,
        item.sequence,
        frozenset(
            (mod.position, mod.ccd, mod.smiles) for mod in item.modifications or []
        ),
    )


def serialize_structure_prediction_input(all_atom_input: StructurePredictionInput):
    def create_chain_data(seq_input, chain_type: str) -> dict[str, Any]:
        chain_data: dict[str, Any] = {
            "sequence": seq_input.sequence,
            "id": seq_input.id,
            "type": chain_type,
        }
        if hasattr(seq_input, "modifications") and seq_input.modifications:
            mods = [
                {"position": mod.position, "ccd": mod.ccd}
                for mod in seq_input.modifications
            ]
            chain_data["modifications"] = mods
        if not hasattr(seq_input, "msa"):
            pass
        elif seq_input.msa is None:
            chain_data["msa"] = None
        elif isinstance(seq_input.msa, MSA):
            chain_data["msa"] = seq_input.msa.state_dict(json_serializable=True)
        else:
            error_msg = f"MSA must be None or MSA. Got {seq_input.msa} instead."
            raise AttributeError(error_msg)
        return chain_data

    sequences = []
    for seq_input in all_atom_input.sequences:
        if isinstance(seq_input, ProteinInput):
            sequences.append(create_chain_data(seq_input, "protein"))
        elif isinstance(seq_input, RNAInput):
            sequences.append(create_chain_data(seq_input, "rna"))
        elif isinstance(seq_input, DNAInput):
            sequences.append(create_chain_data(seq_input, "dna"))
        elif isinstance(seq_input, LigandInput):
            sequences.append(
                {
                    "smiles": seq_input.smiles,
                    "id": seq_input.id,
                    "ccd": seq_input.ccd,
                    "type": "ligand",
                }
            )
        else:
            raise ValueError(f"Unsupported sequence input type: {type(seq_input)}")

    result: dict[str, Any] = {"sequences": sequences}

    if all_atom_input.covalent_bonds is not None:
        result["covalent_bonds"] = [
            {
                "chain_id1": bond.chain_id1,
                "res_idx1": bond.res_idx1,
                "atom_idx1": bond.atom_idx1,
                "chain_id2": bond.chain_id2,
                "res_idx2": bond.res_idx2,
                "atom_idx2": bond.atom_idx2,
            }
            for bond in all_atom_input.covalent_bonds
        ]

    if all_atom_input.pocket is not None:
        result["pocket"] = {
            "binder_chain_id": all_atom_input.pocket.binder_chain_id,
            "contacts": all_atom_input.pocket.contacts,
        }

    if all_atom_input.distogram_conditioning is not None:
        result["distogram_conditioning"] = [
            {"chain_id": disto.chain_id, "distogram": disto.distogram.tolist()}
            for disto in all_atom_input.distogram_conditioning
        ]

    return result


def deserialize_structure_prediction_input(
    data: dict[str, Any],
) -> StructurePredictionInput:
    """Inverse of :func:`serialize_structure_prediction_input`.

    Reconstructs a :class:`StructurePredictionInput` from the JSON-safe dict
    produced by ``serialize_structure_prediction_input``. Values round-trip;
    ``DistogramConditioning.distogram`` dtype follows from JSON (``int64``
    for integer entries, ``float64`` for floats) — cast back to the original
    dtype if downstream code requires a specific one.
    """

    def _mods(chain: dict[str, Any]) -> list[Modification] | None:
        raw = chain.get("modifications")
        if not raw:
            return None
        return [Modification(position=m["position"], ccd=m["ccd"]) for m in raw]

    def _msa(chain: dict[str, Any]) -> MSAInput:
        if "msa" not in chain or chain["msa"] is None:
            return None
        msa_blk = chain["msa"]
        if isinstance(msa_blk, str):
            raise ValueError(f"Unexpected MSA string value: {msa_blk!r}")
        return MSA.from_sequences(msa_blk["sequences"])

    sequences: list[ProteinInput | RNAInput | DNAInput | LigandInput] = []
    for chain in data["sequences"]:
        t = chain["type"]
        if t == "protein":
            sequences.append(
                ProteinInput(
                    id=chain["id"],
                    sequence=chain["sequence"],
                    modifications=_mods(chain),
                    msa=_msa(chain),
                )
            )
        elif t == "rna":
            sequences.append(
                RNAInput(
                    id=chain["id"],
                    sequence=chain["sequence"],
                    modifications=_mods(chain),
                    msa=_msa(chain),
                )
            )
        elif t == "dna":
            sequences.append(
                DNAInput(
                    id=chain["id"],
                    sequence=chain["sequence"],
                    modifications=_mods(chain),
                )
            )
        elif t == "ligand":
            sequences.append(
                LigandInput(
                    id=chain["id"], smiles=chain.get("smiles"), ccd=chain.get("ccd")
                )
            )
        else:
            raise ValueError(f"Unsupported sequence type: {t!r}")

    pocket: PocketConditioning | None = None
    if (pocket_blk := data.get("pocket")) is not None:
        pocket = PocketConditioning(
            binder_chain_id=pocket_blk["binder_chain_id"],
            contacts=[tuple(c) for c in pocket_blk["contacts"]],
        )

    distogram_conditioning: list[DistogramConditioning] | None = None
    if (disto_blk := data.get("distogram_conditioning")) is not None:
        distogram_conditioning = [
            DistogramConditioning(
                chain_id=d["chain_id"], distogram=np.asarray(d["distogram"])
            )
            for d in disto_blk
        ]

    covalent_bonds: list[CovalentBond] | None = None
    if (bonds_blk := data.get("covalent_bonds")) is not None:
        covalent_bonds = [
            CovalentBond(
                chain_id1=b["chain_id1"],
                res_idx1=b["res_idx1"],
                atom_idx1=b["atom_idx1"],
                chain_id2=b["chain_id2"],
                res_idx2=b["res_idx2"],
                atom_idx2=b["atom_idx2"],
            )
            for b in bonds_blk
        ]

    return StructurePredictionInput(
        sequences=sequences,
        pocket=pocket,
        distogram_conditioning=distogram_conditioning,
        covalent_bonds=covalent_bonds,
    )
