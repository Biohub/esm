from __future__ import annotations

import io
import itertools
import random
import re
import warnings
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from pathlib import Path
from subprocess import check_output
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Sequence

import biotite.structure as bs
import brotli
import msgpack
import msgpack_numpy
import numpy as np
import torch
from biotite.database import rcsb
from biotite.file import InvalidFileError
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFCategory, CIFColumn, CIFData, CIFFile
from biotite.structure.io.pdbx import set_structure as set_structure_pdbx
from biotite.structure.io.pdbx.convert import _get_transformations, get_structure
from biotite.structure.util import matrix_rotate
from scipy.spatial import KDTree

from esm.utils import residue_constants
from esm.utils.misc import slice_python_object_as_numpy
from esm.utils.structure.affine3d import Affine3D
from esm.utils.structure.aligner import Aligner
from esm.utils.structure.atom_indexer import AtomIndexer
from esm.utils.structure.metrics import compute_gdt_ts, compute_lddt_ca
from esm.utils.structure.mmcif_parsing import MmcifWrapper, NoProteinError
from esm.utils.structure.protein_chain import (
    ProteinChain,
    _str_key_to_int_key,
    chain_to_ndarray,
    index_by_atom_name,
    infer_CB,
)
from esm.utils.types import PathOrBuffer

msgpack_numpy.patch()

SINGLE_LETTER_CHAIN_IDS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
)


def _parse_operation_expression(expression):
    """
    Get successive operation steps (IDs) for the given
    ``oper_expression``.
    Form the cartesian product, if necessary.
    Copied from biotite and fixed a bug
    """
    # Split groups by parentheses:
    # use the opening parenthesis as delimiter
    # and just remove the closing parenthesis
    expressions_per_step = expression.replace(")", "").split("(")
    expressions_per_step = [e for e in expressions_per_step if len(e) > 0]
    # Important: Operations are applied from right to left
    expressions_per_step.reverse()

    operations = []
    for expr in expressions_per_step:
        cur_expr = expr.split(",")
        cur_op = []
        # Deal with e='1-10,20-30,40-50' type expressions
        for e in cur_expr:
            if "-" in e:
                first, last = e.split("-")
                cur_op.extend(str(id) for id in range(int(first), int(last) + 1))
            else:
                cur_op.append(e)
        operations.append(cur_op)

    # Cartesian product of operations
    return list(itertools.product(*operations))


def _apply_transformations_fast(chains, transformation_dict, operations):
    """
    Get subassembly by applying the given operations to the input
    structure containing affected asym IDs.
    """
    # Additional first dimesion for 'structure.repeat()'
    results = []

    # Apply corresponding transformation for each copy in the assembly
    for c in chains:
        for operation in operations:
            coord = c.atom37_positions.copy()
            # Execute for each transformation step
            # in the operation expression
            for op_step in operation:
                T = transformation_dict[op_step]
                # Rotate
                coord = matrix_rotate(coord, T.rotation)
                # Translate
                coord += T.target_translation
            new_chain = replace(c, atom37_positions=coord)
            results.append(new_chain)

    return results


@dataclass
class ProteinComplexMetadata:
    entity_lookup: dict[int, int]
    chain_lookup: dict[int, str]
    mmcif: MmcifWrapper | None = None
    # This is a dictionary that maps assembly ids to the list of unique chains
    # in that assembly. Allows for usage of `switch_assembly`.
    assembly_composition: dict[str, list[str]] | None = None


@dataclass
class DockQSingleScore:
    native_chains: tuple[str, str]
    DockQ: float
    interface_rms: float
    ligand_rms: float
    fnat: float
    fnonnat: float
    clashes: float
    F1: float
    DockQ_F1: float


@dataclass
class DockQResult:
    total_dockq: float
    native_interfaces: int
    chain_mapping: dict[str, str]
    interfaces: dict[tuple[str, str], DockQSingleScore]
    # zip(aligned.chain_iter(), native.chain_iter()) gives you the pairing
    # aligned.rmsd(native) should give you a low rmsd irrespective of shuffling
    aligned: ProteinComplex
    aligned_rmsd: float


@dataclass(frozen=True)
class ProteinComplex:
    """Dataclass with atom37 representation of an entire protein complex."""

    id: str
    sequence: str
    entity_id: np.ndarray  # entities map to unique sequences
    chain_id: np.ndarray  # multiple chains might share an entity id
    sym_id: np.ndarray  # complexes might be copies of the same chain
    residue_index: np.ndarray
    insertion_code: np.ndarray
    atom37_positions: np.ndarray
    atom37_mask: np.ndarray
    confidence: np.ndarray
    # This metadata is parsed from the MMCIF file. For synthetic data, we do a best effort.
    metadata: ProteinComplexMetadata

    def __post_init__(self):
        l = len(self.sequence)
        assert self.atom37_positions.shape[0] == l, (self.atom37_positions.shape, l)
        assert self.atom37_mask.shape[0] == l, (self.atom37_mask.shape, l)
        assert self.residue_index.shape[0] == l, (self.residue_index.shape, l)
        assert self.insertion_code.shape[0] == l, (self.insertion_code.shape, l)
        assert self.confidence.shape[0] == l, (self.confidence.shape, l)
        assert self.entity_id.shape[0] == l, (self.entity_id.shape, l)
        assert self.chain_id.shape[0] == l, (self.chain_id.shape, l)
        assert self.sym_id.shape[0] == l, (self.sym_id.shape, l)

    def __getitem__(self, idx: int | list[int] | slice | np.ndarray):
        """This function slices protein complexes without consideration of chain breaks
        NOTE: When slicing with a boolean mask, it's possible that the output array won't
        be the expected length. This is because we do our best to preserve chainbreak tokens.
        """

        if isinstance(idx, int):
            idx = [idx]
        if isinstance(idx, list):
            raise ValueError(
                "ProteinComplex doesn't supports indexing with lists of indices"
            )

        if isinstance(idx, np.ndarray):
            is_chainbreak = np.asarray([s == "|" for s in self.sequence])
            idx = idx.astype(bool) | is_chainbreak

        complex = self._unsafe_slice(idx)
        if len(complex) == 0:
            return complex

        # detect runs of chainbreaks by searching for instances of '||' in complex.sequence
        chainbreak_runs = np.asarray(
            [
                complex.sequence[i : i + 2] == "||"
                for i in range(len(complex.sequence) - 1)
            ]
            + [complex.sequence[-1] == "|"]
        )
        # We should remove as many chainbreaks as possible from the start of the sequence
        for i in range(len(chainbreak_runs)):
            if complex.sequence[i] == "|":
                chainbreak_runs[i] = True
            else:
                break
        complex = complex._unsafe_slice(~chainbreak_runs)
        return complex

    def _unsafe_slice(self, idx: int | list[int] | slice | np.ndarray):
        sequence = slice_python_object_as_numpy(self.sequence, idx)
        return replace(
            self,
            sequence=sequence,
            entity_id=self.entity_id[..., idx],
            chain_id=self.chain_id[..., idx],
            sym_id=self.sym_id[..., idx],
            residue_index=self.residue_index[..., idx],
            insertion_code=self.insertion_code[..., idx],
            atom37_positions=self.atom37_positions[..., idx, :, :],
            atom37_mask=self.atom37_mask[..., idx, :],
            confidence=self.confidence[..., idx],
        )

    def __len__(self):
        return len(self.sequence)

    @property
    def num_chains(self):
        return len(self.chain_boundaries)

    @cached_property
    def atoms(self) -> AtomIndexer:
        return AtomIndexer(self, property="atom37_positions", dim=-2)

    @cached_property
    def atom_mask(self) -> AtomIndexer:
        return AtomIndexer(self, property="atom37_mask", dim=-1)

    @cached_property
    def chain_lengths(self) -> np.ndarray:
        return np.diff(self.chain_boundaries, axis=1).flatten()

    @cached_property
    def chain_boundaries(self) -> list[tuple[int, int]]:
        cb = [-1]
        for i, s in enumerate(self.sequence):
            if s == "|":
                cb.append(i)
        cb.append(len(self))
        return [(cb[i] + 1, cb[i + 1]) for i in range(len(cb) - 1)]

    def get_chain_by_index(self, index: int) -> ProteinChain:
        try:
            start, end = self.chain_boundaries[index]
            return self[start:end].as_chain()
        except IndexError:
            raise IndexError(f"Chain index {index} out of bounds")

    def get_chain_by_id(
        self, chain_id: str, sample_chain_if_duplicate: bool = True
    ) -> ProteinChain:
        valid_indices = [
            index
            for index, id_of_index in self.metadata.chain_lookup.items()
            if id_of_index == chain_id
        ]
        if not valid_indices:
            raise KeyError(f"Chain ID {chain_id} not found")
        if sample_chain_if_duplicate:
            index_to_return = random.choice(valid_indices)
            return self.get_chain_by_index(index_to_return)
        else:
            if len(valid_indices) > 1:
                raise ValueError(f"Multiple chains with chain ID {chain_id} found")
            return self.get_chain_by_index(valid_indices[0])

    def chain_iter(self) -> Iterable[ProteinChain]:
        for start, end in self.chain_boundaries:
            c = self[start:end]
            yield c.as_chain()

    def as_chain(self, force_conversion: bool = False) -> ProteinChain:
        """Convert the ProteinComplex to a ProteinChain.

        Args:
            force_conversion (bool): Forces the conversion into a protein chain even if the complex has multiple chains.
                The purpose of this is to use ProteinChain specific functions (like cbeta_contacts).

        """
        if not force_conversion:
            assert len(np.unique(self.chain_id)) == 1, f"{self.id}"
            assert len(np.unique(self.entity_id)) == 1, f"{self.id}"
            if self.chain_id[0] not in self.metadata.chain_lookup:
                warnings.warn("Chain ID not found in metadata, using 'A' as default")
            if self.entity_id[0] not in self.metadata.entity_lookup:
                warnings.warn("Entity ID not found in metadata, using None as default")
            chain_id = self.metadata.chain_lookup.get(self.chain_id[0], "A")
            entity_id = self.metadata.entity_lookup.get(self.entity_id[0], None)
        else:
            chain_id = "A"
            entity_id = None

        return ProteinChain(
            id=self.id,
            sequence=self.sequence,
            chain_id=chain_id,
            entity_id=entity_id,
            atom37_positions=self.atom37_positions,
            atom37_mask=self.atom37_mask,
            residue_index=self.residue_index,
            insertion_code=self.insertion_code,
            confidence=self.confidence,
            mmcif=self.metadata.mmcif,
        )

    @classmethod
    def from_pdb(cls, path: PathOrBuffer, id: str | None = None) -> "ProteinComplex":
        atom_array = PDBFile.read(path).get_structure(
            model=1, extra_fields=["b_factor"]
        )

        chains = []
        for chain in bs.chain_iter(atom_array):
            chain = chain[~chain.hetero]
            if len(chain) == 0:
                continue
            chains.append(ProteinChain.from_atomarray(chain, id))
        return ProteinComplex.from_chains(chains)

    def to_pdb(self, path: PathOrBuffer, include_insertions: bool = True):
        atom_array = None
        for chain in self.chain_iter():
            carr = (
                chain.atom_array
                if include_insertions
                else chain.atom_array_no_insertions
            )
            atom_array = carr if atom_array is None else atom_array + carr
        f = PDBFile()
        f.set_structure(atom_array)
        f.write(path)

    def to_pdb_string(self, include_insertions: bool = True) -> str:
        buf = io.StringIO()
        self.to_pdb(buf, include_insertions=include_insertions)
        buf.seek(0)
        return buf.read()

    def normalize_chain_ids_for_pdb(self):
        # Since PDB files have 1-letter chain IDs and don't support the idea of a symmetric index,
        # we can normalize it instead which might be necessary for DockQ and to_pdb.
        ids = SINGLE_LETTER_CHAIN_IDS
        chains = []
        for i, chain in enumerate(self.chain_iter()):
            chain = replace(chain, chain_id=ids[i])
            if i > len(ids):
                raise RuntimeError("Too many chains to write to PDB file")
            chains.append(chain)

        return ProteinComplex.from_chains(chains)

    def find_assembly_ids_with_chain(self, id: str) -> list[str]:
        good_chains = []
        if (comp := self.metadata.assembly_composition) is not None:
            for assembly_id, chain_ids in comp.items():
                if id in chain_ids:
                    good_chains.append(assembly_id)
        else:
            raise ValueError(
                "Cannot switch assemblies on this ProteinComplex, you must create the assembly from mmcif to support this"
            )
        return good_chains

    def switch_assembly(self, id: str):
        assert self.metadata.mmcif is not None
        return get_assembly_fast(self.metadata.mmcif, assembly_id=id)

    def state_dict(self, backbone_only=False, json_serializable=False):
        """This state dict is optimized for storage, so it turns things to fp16 whenever
        possible. Note that we also only support int32 residue indices, I'm hoping we don't
        need more than 2**32 residues..."""
        dct = {k: v for k, v in vars(self).items()}
        if backbone_only:
            dct["atom37_mask"][:, 3:] = False
        dct["atom37_positions"] = dct["atom37_positions"][dct["atom37_mask"]]
        for k, v in dct.items():
            if isinstance(v, np.ndarray):
                match v.dtype:
                    case np.int64:
                        dct[k] = v.astype(np.int32)
                    case np.float64 | np.float32:
                        dct[k] = v.astype(np.float16)
                    case _:
                        pass
                if json_serializable:
                    dct[k] = v.tolist()
            elif isinstance(v, ProteinComplexMetadata):
                dct[k] = asdict(v)
        dct["metadata"]["mmcif"] = None
        # These can be populated with non-serializable objects and are not needed for reconstruction
        dct.pop("atoms", None)
        dct.pop("atom_mask", None)
        dct.pop("per_chain_kd_trees", None)
        return dct

    def to_blob(self, backbone_only=False) -> bytes:
        return brotli.compress(msgpack.dumps(self.state_dict(backbone_only)), quality=5)

    @classmethod
    def from_state_dict(cls, dct):
        # Note: assembly_composition is *supposed* to have string keys.
        dct = _str_key_to_int_key(dct, ignore_keys=["assembly_composition"])

        for k, v in dct.items():
            if isinstance(v, list):
                dct[k] = np.array(v)

        atom37 = np.full((*dct["atom37_mask"].shape, 3), np.nan)
        atom37[dct["atom37_mask"]] = dct["atom37_positions"]
        dct["atom37_positions"] = atom37
        dct = {
            k: (v.astype(np.float32) if k in ["atom37_positions", "confidence"] else v)
            for k, v in dct.items()
        }
        if "chain_boundaries" in dct:
            del dct["chain_boundaries"]
        if "chain_boundaries" in dct["metadata"]:
            del dct["metadata"]["chain_boundaries"]
        dct["metadata"] = ProteinComplexMetadata(**dct["metadata"])
        return cls(**dct)

    @classmethod
    def from_blob(cls, input: Path | str | io.BytesIO | bytes):
        """NOTE(@zlin): blob + sparse coding + brotli + fp16 reduces memory
        of chains from 52G/1M chains to 20G/1M chains, I think this is a good first
        shot at compressing and dumping chains to disk. I'm sure there's better ways."""
        match input:
            case Path() | str():
                bytes = Path(input).read_bytes()
            case io.BytesIO():
                bytes = input.getvalue()
            case _:
                bytes = input
        return cls.from_state_dict(
            msgpack.loads(brotli.decompress(bytes), strict_map_key=False)
        )

    @classmethod
    def from_rcsb(cls, pdb_id: str, keep_source: bool = False) -> ProteinComplex:
        f: io.StringIO = rcsb.fetch(pdb_id, "cif")  # type: ignore
        return cls.from_mmcif(f, id=pdb_id, keep_source=keep_source, is_predicted=False)

    @classmethod
    def from_mmcif(
        cls,
        path: PathOrBuffer,
        id: str | None = None,
        assembly_id: str | None = None,
        is_predicted: bool = False,
        keep_source: bool = False,
    ):
        """Return a ProteinComplex object from an mmcif file.
        TODO(@zeming): there's actually multiple complexes per file, but for ease of implementation,
        we only consider the first defined complex!

        Args:
            path (str | Path | io.TextIO): Path or buffer to read mmcif file from. Should be uncompressed.
            id (str, optional): String identifier to assign to structure. Will attempt to infer otherwise.
            is_predicted (bool): If True, reads b factor as the confidence readout. Default: False.
            chain_id (str, optional): Select a chain corresponding to (author) chain id.
        """
        mmcif = MmcifWrapper.read(path, id)
        return get_assembly_fast(mmcif, assembly_id=assembly_id)

    @classmethod
    def from_chains(
        cls,
        chains: Sequence[ProteinChain],
        mmcif: MmcifWrapper | None = None,
        all_assembly_metadata_dictionary: dict[str, list[str]] | None = None,
    ):
        if not chains:
            raise ValueError(
                "Cannot create a ProteinComplex from an empty list of chains"
            )

        # TODO(roshan): Make a proper protein complex class
        def join_arrays(arrays: Sequence[np.ndarray], sep: np.ndarray):
            full_array = []
            for array in arrays:
                full_array.append(array)
                full_array.append(sep)
            full_array = full_array[:-1]
            return np.concatenate(full_array, 0)

        sep_tokens = {
            "residue_index": np.array([-1]),
            "insertion_code": np.array([""]),
            "atom37_positions": np.full([1, 37, 3], np.nan),
            "atom37_mask": np.zeros([1, 37], dtype=bool),
            "confidence": np.array([0]),
        }

        array_args: dict[str, np.ndarray] = {
            name: join_arrays([getattr(chain, name) for chain in chains], sep)
            for name, sep in sep_tokens.items()
        }

        multimer_arrays = []
        chain2num_max = -1
        chain2num = {}
        ent2num_max = -1
        ent2num = {}
        total_index = 0
        for i, c in enumerate(chains):
            num_res = c.residue_index.shape[0]
            if c.chain_id not in chain2num:
                chain2num[c.chain_id] = (chain2num_max := chain2num_max + 1)
            chain_id_array = np.full([num_res], chain2num[c.chain_id], dtype=np.int64)

            if c.entity_id is None:
                entity_num = (ent2num_max := ent2num_max + 1)
            else:
                if c.entity_id not in ent2num:
                    ent2num[c.entity_id] = (ent2num_max := ent2num_max + 1)
                entity_num = ent2num[c.entity_id]
            entity_id_array = np.full([num_res], entity_num, dtype=np.int64)

            sym_id_array = np.full([num_res], i, dtype=np.int64)

            multimer_arrays.append(
                {
                    "chain_id": chain_id_array,
                    "entity_id": entity_id_array,
                    "sym_id": sym_id_array,
                }
            )

            total_index += num_res + 1

        sep = np.array([-1])
        update = {
            name: join_arrays([dct[name] for dct in multimer_arrays], sep=sep)
            for name in ["chain_id", "entity_id", "sym_id"]
        }
        array_args.update(update)

        metadata = ProteinComplexMetadata(
            mmcif=mmcif,
            chain_lookup={v: k for k, v in chain2num.items()},
            entity_lookup={v: k for k, v in ent2num.items()},
            assembly_composition=all_assembly_metadata_dictionary,
        )

        return cls(
            id=chains[0].id,
            sequence=residue_constants.CHAIN_BREAK_TOKEN.join(
                chain.sequence for chain in chains
            ),
            metadata=metadata,
            **array_args,
        )

    def infer_oxygen(self) -> ProteinComplex:
        """Oxygen position is fixed given N, CA, C atoms. Infer it if not provided."""
        O_missing_indices = np.argwhere(
            ~np.isfinite(self.atoms["O"]).all(axis=1)
        ).squeeze()

        O_vector = torch.tensor([0.6240, -1.0613, 0.0103], dtype=torch.float32)
        N, CA, C = torch.from_numpy(self.atoms[["N", "CA", "C"]]).float().unbind(dim=1)
        N = torch.roll(N, -3)
        N[..., -1, :] = torch.nan

        # Get the frame defined by the CA-C-N atom
        frames = Affine3D.from_graham_schmidt(CA, C, N)
        O = frames.apply(O_vector)
        atom37_positions = self.atom37_positions.copy()
        atom37_mask = self.atom37_mask.copy()

        atom37_positions[O_missing_indices, residue_constants.atom_order["O"]] = O[
            O_missing_indices
        ].numpy()
        atom37_mask[O_missing_indices, residue_constants.atom_order["O"]] = ~np.isnan(
            atom37_positions[O_missing_indices, residue_constants.atom_order["O"]]
        ).any(-1)
        new_chain = replace(
            self, atom37_positions=atom37_positions, atom37_mask=atom37_mask
        )
        return new_chain

    def infer_cbeta(self, infer_cbeta_for_glycine: bool = False) -> ProteinComplex:
        """Return a new chain with inferred CB atoms at all residues except GLY.

        Args:
            infer_cbeta_for_glycine (bool): If True, infers a beta carbon for glycine
                residues, even though that residue doesn't have one.  Default off.

                NOTE(rverkuil): The reason for having this switch in the first place
                is that sometimes we want a (inferred) CB coordinate for every residue,
                for example for making a pairwise distance matrix, or doing an RMSD
                calculation between two designs for a given structural template, w/
                CB atoms.
        """
        atom37_positions = self.atom37_positions.copy()
        atom37_mask = self.atom37_mask.copy()

        N, CA, C = np.moveaxis(self.atoms[["N", "CA", "C"]], 1, 0)
        # See usage in trDesign codebase.
        # https://github.com/gjoni/trDesign/blob/f2d5930b472e77bfacc2f437b3966e7a708a8d37/02-GD/utils.py#L140
        inferred_cbeta_positions = infer_CB(C, N, CA, 1.522, 1.927, -2.143)
        if not infer_cbeta_for_glycine:
            inferred_cbeta_positions[np.array(list(self.sequence)) == "G", :] = np.nan

        atom37_positions[:, residue_constants.atom_order["CB"]] = (
            inferred_cbeta_positions
        )
        atom37_mask[:, residue_constants.atom_order["CB"]] = ~np.isnan(
            atom37_positions[:, residue_constants.atom_order["CB"]]
        ).any(-1)
        new_chain = replace(
            self, atom37_positions=atom37_positions, atom37_mask=atom37_mask
        )
        return new_chain

    @classmethod
    def from_open_source(cls, pc: ProteinComplex):
        # TODO(@zeming): deprecated, should delete
        return pc

    @classmethod
    def concat(cls, objs: list[ProteinComplex]) -> ProteinComplex:
        pdb_ids = [obj.id for obj in objs]
        if len(set(pdb_ids)) > 1:
            raise RuntimeError(
                "Concatention of protein complexes across different PDB ids is unsupported"
            )
        return ProteinComplex.from_chains(
            list(itertools.chain.from_iterable(obj.chain_iter() for obj in objs))
        )

    def _sanity_check_complexes_are_comparable(self, other: ProteinComplex):
        assert len(self) == len(other), "Protein complexes must have the same length"
        assert len(list(self.chain_iter())) == len(
            list(other.chain_iter())
        ), "Protein complexes must have the same number of chains"

    def rmsd(
        self,
        target: ProteinComplex,
        also_check_reflection: bool = False,
        mobile_inds: list[int] | np.ndarray | None = None,
        target_inds: list[int] | np.ndarray | None = None,
        only_compute_backbone_rmsd: bool = False,
        compute_chain_assignment: bool = True,
    ):
        """
        Compute the RMSD between this protein chain and another.

        Args:
            target (ProteinComplex): The target (other) protein complex to compare to.
            also_check_reflection (bool, optional): If True, also check if the reflection of the mobile atoms has a lower RMSD.
            mobile_inds (list[int], optional): The indices of the mobile atoms to align. These are NOT residue indices
            target_inds (list[int], optional): The indices of the target atoms to align. These are NOT residue indices
            only_compute_backbone_rmsd (bool, optional): If True, only compute the RMSD of the backbone atoms.
        """
        if compute_chain_assignment:
            aligned = self.dockq(target).aligned
        else:
            aligned = self

        aligner = Aligner(
            aligned if mobile_inds is None else aligned[mobile_inds],
            target if target_inds is None else target[target_inds],
            only_compute_backbone_rmsd,
        )
        avg_rmsd = aligner.rmsd

        if not also_check_reflection:
            return avg_rmsd

        aligner = Aligner(
            aligned if mobile_inds is None else aligned[mobile_inds],
            target if target_inds is None else target[target_inds],
            only_compute_backbone_rmsd,
            use_reflection=True,
        )
        avg_rmsd_neg = aligner.rmsd

        return min(avg_rmsd, avg_rmsd_neg)

    def lddt_ca(
        self,
        target: ProteinComplex,
        mobile_inds: list[int] | np.ndarray | None = None,
        target_inds: list[int] | np.ndarray | None = None,
        compute_chain_assignment: bool = True,
        **kwargs,
    ) -> float | np.ndarray:
        """Compute the LDDT between this protein complex and another.

        Arguments:
            target (ProteinComplex): The other protein complex to compare to.
            mobile_inds (list[int], np.ndarray, optional): The indices of the mobile atoms to align. These are NOT residue indices
            target_inds (list[int], np.ndarray, optional): The indices of the target atoms to align. These are NOT residue indices

        Returns:
            float | np.ndarray: The LDDT score between the two protein chains, either
                a single float or per-residue LDDT scores if `per_residue` is True.
        """
        if compute_chain_assignment:
            aligned = self.dockq(target).aligned
        else:
            aligned = self
        lddt = compute_lddt_ca(
            torch.tensor(aligned.atom37_positions[mobile_inds]).unsqueeze(0),
            torch.tensor(target.atom37_positions[target_inds]).unsqueeze(0),
            torch.tensor(aligned.atom37_mask[mobile_inds]).unsqueeze(0),
            **kwargs,
        )
        return float(lddt) if lddt.numel() == 1 else lddt.numpy().flatten()

    def gdt_ts(
        self,
        target: ProteinComplex,
        mobile_inds: list[int] | np.ndarray | None = None,
        target_inds: list[int] | np.ndarray | None = None,
        compute_chain_assignment: bool = True,
        **kwargs,
    ) -> float | np.ndarray:
        """Compute the GDT_TS between this protein complex and another.

        Arguments:
            target (ProteinComplex): The other protein complex to compare to.
            mobile_inds (list[int], np.ndarray, optional): The indices of the mobile atoms to align. These are NOT residue indices
            target_inds (list[int], np.ndarray, optional): The indices of the target atoms to align. These are NOT residue indices

        Returns:
            float: The GDT_TS score between the two protein chains.
        """
        if compute_chain_assignment:
            aligned = self.dockq(target).aligned
        else:
            aligned = self
        gdt_ts = compute_gdt_ts(
            mobile=torch.tensor(
                index_by_atom_name(aligned.atom37_positions[mobile_inds], "CA"),
                dtype=torch.float32,
            ).unsqueeze(0),
            target=torch.tensor(
                index_by_atom_name(target.atom37_positions[target_inds], "CA"),
                dtype=torch.float32,
            ).unsqueeze(0),
            atom_exists_mask=torch.tensor(
                index_by_atom_name(aligned.atom37_mask[mobile_inds], "CA", dim=-1)
                & index_by_atom_name(target.atom37_mask[target_inds], "CA", dim=-1)
            ).unsqueeze(0),
            **kwargs,
        )
        return float(gdt_ts) if gdt_ts.numel() == 1 else gdt_ts.numpy().flatten()

    def dockq(self, native: ProteinComplex):
        # This function uses dockqv2 to compute the DockQ score. Because it does a mapping
        # over all possible chains, it's quite slow. Be careful not to use this in an inference loop
        # or something that requires fast scoring. It defaults to 8 CPUs.
        #
        # TODO(@zeming): Because we haven't properly implemented protein complexes for mmcif,
        # if your protein has multi-letter or repeated chain IDs, this will fail. Please call
        # pc = pc.normalize_chain_ids_for_pdb() before calling this function in that case (limit is 62 chains)

        try:
            pass
        except BaseException:
            raise RuntimeError(
                "DockQ is not installed. Please update your environment."
            )
        self._sanity_check_complexes_are_comparable(native)

        def sanity_check_chain_ids(pc: ProteinComplex):
            ids = []
            for i, chain in enumerate(pc.chain_iter()):
                if i > len(SINGLE_LETTER_CHAIN_IDS):
                    raise ValueError("Too many chains to write to PDB file")
                if len(chain.chain_id) > 1:
                    raise ValueError(
                        "We only supports single letter chain IDs for DockQ"
                    )
                ids.append(chain.chain_id)
            if len(set(ids)) != len(ids):
                raise ValueError(f"Duplicate chain IDs in protein complex: {ids}")
            return ids

        sanity_check_chain_ids(self)
        sanity_check_chain_ids(native)

        with TemporaryDirectory() as tdir:
            dir = Path(tdir)
            self.to_pdb(dir / "self.pdb")
            native.to_pdb(dir / "native.pdb")

            output = check_output(["DockQ", dir / "self.pdb", dir / "native.pdb"])
        lines = output.decode().split("\n")

        # Remove the header comments
        start_index = next(
            i for i, line in enumerate(lines) if line.startswith("Model")
        )
        lines = lines[start_index:]

        result = {}
        interfaces = []
        current_interface: dict = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("Model  :"):
                pass  # Tmp pdb file location, it's useless...
            elif line.startswith("Native :"):
                pass  # Tmp pdb file location, it's useless...
            elif line.startswith("Total DockQ"):
                total_dockq_match = re.search(
                    r"Total DockQ over (\d+) native interfaces: ([\d.]+) with (.*) model:native mapping",
                    line,
                )
                if total_dockq_match:
                    result["value"] = float(total_dockq_match.group(2))
                    result["native interfaces"] = int(total_dockq_match.group(1))
                    native_chains, self_chains = total_dockq_match.group(3).split(":")
                    result["mapping"] = dict(zip(native_chains, self_chains))
                else:
                    raise RuntimeError(
                        "Failed to parse DockQ output, maybe your DockQ version is wrong?"
                    )
            elif line.startswith("Native chains:"):
                if current_interface:
                    interfaces.append(current_interface)
                current_interface = {
                    "Native chains": line.split(":")[1].strip().split(", ")
                }
            elif line.startswith("Model chains:"):
                current_interface["Model chains"] = (
                    line.split(":")[1].strip().split(", ")
                )
            elif ":" in line:
                key, value = line.split(":", 1)
                current_interface[key.strip()] = float(value.strip())

        if current_interface:
            interfaces.append(current_interface)

        def parse_dict(d: dict[str, Any]) -> DockQSingleScore:
            return DockQSingleScore(
                native_chains=tuple(d["Native chains"]),  # type: ignore
                DockQ=float(d["DockQ"]),
                interface_rms=float(d["irms"]),
                ligand_rms=float(d["Lrms"]),  # Note the capitalization difference
                fnat=float(d["fnat"]),
                fnonnat=float(d["fnonnat"]),
                clashes=float(d["clashes"]),
                F1=float(d["F1"]),
                DockQ_F1=float(d["DockQ_F1"]),
            )

        inv_mapping = {v: k for k, v in result["mapping"].items()}

        self_chain_map = {c.chain_id: c for c in self.chain_iter()}
        realigned = []
        for chain in native.chain_iter():
            realigned.append(self_chain_map[inv_mapping[chain.chain_id]])

        realigned = ProteinComplex.from_chains(realigned)
        aligner = Aligner(realigned, native)
        realigned = aligner.apply(realigned)

        result = DockQResult(
            total_dockq=result["value"],
            native_interfaces=result["native interfaces"],
            chain_mapping=result["mapping"],
            interfaces={
                (i["Model chains"][0], i["Model chains"][1]): parse_dict(i)
                for i in interfaces
            },
            aligned=realigned,
            aligned_rmsd=aligner.rmsd,
        )

        return result

    @cached_property
    def per_chain_kd_trees(self):
        # Iterate over chains, build KDTree for each chain
        kdtrees = []

        CA = self.atoms["CA"]

        for start, end in self.chain_boundaries:
            chain_CA = CA[start:end]
            chain_CA = chain_CA[np.isfinite(chain_CA).all(axis=-1)]
            kdtrees.append(KDTree(chain_CA))

        return kdtrees

    def chain_adjacency(self, cutoff: float = 8.0) -> np.ndarray:
        # Compute adjacency matrix for protein complex
        num_chains = self.num_chains
        adjacency = np.zeros((num_chains, num_chains), dtype=bool)
        for (i, kdtree), (j, kdtree2) in itertools.combinations(
            enumerate(self.per_chain_kd_trees), 2
        ):
            adj = kdtree.query_ball_tree(kdtree2, cutoff)
            any_is_adjacent = any(len(a) > 0 for a in adj)
            adjacency[i, j] = any_is_adjacent
            adjacency[j, i] = any_is_adjacent
        return adjacency

    def chain_adjacency_by_index(self, index: int, cutoff: float = 8.0) -> np.ndarray:
        num_chains = len(self.chain_boundaries)
        adjacency = np.zeros(num_chains, dtype=bool)
        for i, kdtree in enumerate(self.per_chain_kd_trees):
            if i == index:
                continue
            adj = kdtree.query_ball_tree(self.per_chain_kd_trees[index], cutoff)
            adjacency[i] = any(len(a) > 0 for a in adj)
        return adjacency

    def add_prefix_to_chain_ids(self, prefix: str) -> ProteinComplex:
        """Rename all chains in the complex with a given prefix.

        Args:
            prefix (str): The prefix to use for the new chain IDs. Each chain will be
                named as "{prefix}_{chain_id}".

        Returns:
            ProteinComplex: A new protein complex with renamed chains.
        """
        new_chains = []
        for chain in self.chain_iter():
            # Create new chain with updated chain_id
            new_chain = replace(chain, chain_id=f"{prefix}_{chain.chain_id}")
            new_chains.append(new_chain)
        return ProteinComplex.from_chains(new_chains)

    def sasa(self, by_residue: bool = True):
        chain = self.as_chain(force_conversion=True)
        return chain.sasa(by_residue=by_residue)

    def to_mmcif_string(self) -> str:
        """Convert the ProteinComplex to mmCIF format.

        Returns:
            str: The mmCIF content as a string.
        """
        # Convert the ProteinComplex to a biotite AtomArray
        # Collect all atoms from all chains
        all_atoms = []
        for chain in self.chain_iter():
            chain_atom_array = chain.atom_array
            # Convert AtomArray to list of atoms and add to collection
            all_atoms.extend(chain_atom_array)

        # Create combined AtomArray from all atoms
        if not all_atoms:
            raise ValueError("No atoms found in protein complex")

        atom_array = bs.array(all_atoms)

        # Create CIF file
        f = CIFFile()
        set_structure_pdbx(f, atom_array, data_block=self.id)

        # Add entity information for proper mmCIF structure
        self._add_entity_information(f)

        # Write to string
        output = io.StringIO()
        f.write(output)
        return output.getvalue()

    def _add_entity_information(self, cif_file: CIFFile) -> None:
        """Add entity, entity_poly, and struct_asym sections to CIF file."""

        # Group chains by sequence to create unique entities
        entity_map = {}  # sequence -> entity_id
        chain_to_entity = {}  # chain_id -> entity_id
        entity_sequences = {}  # entity_id -> sequence
        entity_id_counter = 1

        for chain in self.chain_iter():
            sequence = chain.sequence
            if sequence not in entity_map:
                entity_map[sequence] = entity_id_counter
                entity_sequences[entity_id_counter] = sequence
                entity_id_counter += 1
            chain_to_entity[chain.chain_id] = entity_map[sequence]

        # Create _entity section
        entity_ids = []
        entity_types = []
        entity_descriptions = []

        for entity_id in sorted(entity_sequences.keys()):
            entity_ids.append(str(entity_id))
            entity_types.append("polymer")
            entity_descriptions.append(f"Protein chain (entity {entity_id})")

        cif_file.block["entity"] = CIFCategory(
            name="entity",
            columns={
                "id": CIFColumn(
                    data=CIFData(array=np.array(entity_ids), dtype=np.str_)
                ),
                "type": CIFColumn(
                    data=CIFData(array=np.array(entity_types), dtype=np.str_)
                ),
                "pdbx_description": CIFColumn(
                    data=CIFData(array=np.array(entity_descriptions), dtype=np.str_)
                ),
            },
        )

        # Create _entity_poly section
        poly_entity_ids = []
        poly_types = []
        poly_nstd_linkages = []
        poly_sequences = []

        for entity_id in sorted(entity_sequences.keys()):
            poly_entity_ids.append(str(entity_id))
            poly_types.append("polypeptide(L)")
            poly_nstd_linkages.append("no")
            poly_sequences.append(entity_sequences[entity_id])

        cif_file.block["entity_poly"] = CIFCategory(
            name="entity_poly",
            columns={
                "entity_id": CIFColumn(
                    data=CIFData(array=np.array(poly_entity_ids), dtype=np.str_)
                ),
                "type": CIFColumn(
                    data=CIFData(array=np.array(poly_types), dtype=np.str_)
                ),
                "nstd_linkage": CIFColumn(
                    data=CIFData(array=np.array(poly_nstd_linkages), dtype=np.str_)
                ),
                "pdbx_seq_one_letter_code": CIFColumn(
                    data=CIFData(array=np.array(poly_sequences), dtype=np.str_)
                ),
            },
        )

        # Create _struct_asym section
        asym_ids = []
        asym_entity_ids = []
        asym_details = []

        for chain in self.chain_iter():
            asym_ids.append(chain.chain_id)
            asym_entity_ids.append(str(chain_to_entity[chain.chain_id]))
            asym_details.append("")

        cif_file.block["struct_asym"] = CIFCategory(
            name="struct_asym",
            columns={
                "id": CIFColumn(data=CIFData(array=np.array(asym_ids), dtype=np.str_)),
                "entity_id": CIFColumn(
                    data=CIFData(array=np.array(asym_entity_ids), dtype=np.str_)
                ),
                "details": CIFColumn(
                    data=CIFData(array=np.array(asym_details), dtype=np.str_)
                ),
            },
        )


def get_assembly_fast(
    mmcif: MmcifWrapper,
    assembly_id=None,
    model=None,
    data_block=None,
    altloc="first",
    use_author_fields=True,
):
    pdbx_file = mmcif.raw
    if pdbx_file is None:
        raise InvalidFileError("No mmCIF data loaded")
    assembly_gen_category = pdbx_file.block["pdbx_struct_assembly_gen"]
    if assembly_gen_category is None:
        raise InvalidFileError("File has no 'pdbx_struct_assembly_gen' category")

    struct_oper_category = pdbx_file.block["pdbx_struct_oper_list"]
    if struct_oper_category is None:
        raise InvalidFileError("File has no 'pdbx_struct_oper_list' category")

    if assembly_id is None:
        assembly_id = assembly_gen_category["assembly_id"].data.array[0]
    elif assembly_id not in assembly_gen_category["assembly_id"].data.array:
        raise KeyError(f"File has no Assembly ID '{assembly_id}'")

    ### Calculate all possible transformations
    transformations = _get_transformations(struct_oper_category)

    ### Get structure according to additional parameters
    structure = get_structure(
        pdbx_file, model, data_block, altloc, ["label_asym_id"], use_author_fields
    )[0]  # type: ignore
    # TODO(@zeming) This line will remove all non-protein structural elements,
    # we should remove this when we want to parse these too.
    structure: bs.AtomArray = structure[
        bs.filter_amino_acids(structure) & ~structure.hetero  # type: ignore
    ]
    if len(structure) == 0:
        raise NoProteinError
    unique_asym_ids = np.unique(structure.label_asym_id)  # type: ignore
    asym2chain = {}
    asym2auth = {}
    for asym_id in unique_asym_ids:
        sub_structure: bs.AtomArray = structure[structure.label_asym_id == asym_id]  # type: ignore
        chain_id: str = sub_structure[0].chain_id  # type: ignore
        (
            sequence,
            atom_positions,
            atom_mask,
            residue_index,
            insertion_code,
            confidence,
            entity_id,
        ) = chain_to_ndarray(sub_structure, mmcif, chain_id, False)

        asym2chain[asym_id] = ProteinChain(
            id=mmcif.id or "unknown",
            sequence=sequence,
            chain_id=chain_id,
            entity_id=entity_id,
            atom37_positions=atom_positions,
            atom37_mask=atom_mask,
            residue_index=residue_index,
            insertion_code=insertion_code,
            confidence=confidence,
            mmcif=None,
        )
        asym2auth[asym_id] = chain_id

    ### Get transformations and apply them to the affected asym IDs
    assembly = []
    assembly_id_dict: dict[str, list[str]] = {}

    # Process the target assembly ID
    for aid, op_expr, asym_id_expr in zip(
        assembly_gen_category["assembly_id"].data.array,
        assembly_gen_category["oper_expression"].data.array,
        assembly_gen_category["asym_id_list"].data.array,
    ):
        if aid == assembly_id:
            # Parse operations and asym IDs for this specific entry
            operations = _parse_operation_expression(op_expr)
            asym_ids = asym_id_expr.split(",")

            # Filter affected asym IDs to only protein chains, preserving order
            sub_structures = [
                asym2chain[asym_id] for asym_id in asym_ids if asym_id in asym2chain
            ]

            # Apply transformations
            sub_assembly = _apply_transformations_fast(
                sub_structures, transformations, operations
            )
            assembly.extend(sub_assembly)

            # Build assembly_id_dict for this entry
            assembly_id_dict[aid] = assembly_id_dict.get(aid, []) + [
                asym2auth[id_] for id_ in asym_ids if id_ in asym2auth
            ]

    if len(assembly) == 0:
        raise NoProteinError
    return ProteinComplex.from_chains(assembly, mmcif, assembly_id_dict)


def protein_chain_to_protein_complex(chain: ProteinChain) -> ProteinComplex:
    if "|" not in chain.sequence:
        return ProteinComplex.from_chains([chain])
    chain_breaks = np.array(list(chain.sequence)) == "|"
    chain_break_inds = np.where(chain_breaks)[0]
    chain_break_inds = np.concatenate([[0], chain_break_inds, [len(chain)]])
    chain_break_inds = np.array(list(zip(chain_break_inds[:-1], chain_break_inds[1:])))
    complex_chains = []
    for start, end in chain_break_inds:
        if start != 0:
            start += 1
        complex_chains.append(chain[start:end])
    complex_chains = [
        ProteinChain.from_atom37(
            chain.atom37_positions,
            sequence=chain.sequence,
            chain_id=SINGLE_LETTER_CHAIN_IDS[i],
            entity_id=i,
        )
        for i, chain in enumerate(complex_chains)
    ]
    return ProteinComplex.from_chains(complex_chains)
