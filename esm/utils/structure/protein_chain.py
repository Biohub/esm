from __future__ import annotations

import io
import warnings
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping, Sequence

import biotite.structure as bs
import brotli
import msgpack
import msgpack_numpy
import numpy as np
import torch
from biotite.database import rcsb
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFCategory, CIFColumn, CIFData, CIFFile
from biotite.structure.io.pdbx import set_structure as set_structure_pdbx
from scipy.spatial import ConvexHull, KDTree
from scipy.spatial.distance import cdist, pdist, squareform

from esm.utils import residue_constants
from esm.utils.misc import slice_python_object_as_numpy
from esm.utils.structure.affine3d import Affine3D
from esm.utils.structure.aligner import Aligner
from esm.utils.structure.atom_indexer import AtomIndexer
from esm.utils.structure.metrics import compute_gdt_ts, compute_lddt_ca
from esm.utils.structure.mmcif_parsing import MmcifWrapper, Residue
from esm.utils.structure.normalize_coordinates import (
    apply_frame_to_coords,
    get_protein_normalization_frame,
)
from esm.utils.structure.protein_structure import index_by_atom_name
from esm.utils.types import PathOrBuffer

msgpack_numpy.patch()
CHAIN_ID_CONST = "A"


def _str_key_to_int_key(dct: dict, ignore_keys: list[str] | None = None) -> dict:
    new_dict = {}
    for k, v in dct.items():
        v_new = v
        if k not in ignore_keys and isinstance(v, dict):
            v_new = _str_key_to_int_key(v, ignore_keys=ignore_keys)
        # Note assembly_composition is *supposed* to have string keys.
        if isinstance(k, str) and k.isdigit():
            new_dict[int(k)] = v_new
        else:
            new_dict[k] = v_new
    return new_dict


def _num_non_null_residues(seqres_to_structure_chain: Mapping[int, Residue]) -> int:
    return sum(
        residue.residue_number is not None
        for residue in seqres_to_structure_chain.values()
    )


def infer_CB(C, N, Ca, L: float = 1.522, A: float = 1.927, D: float = -2.143):
    """
    Inspired by a util in trDesign:
    https://github.com/gjoni/trDesign/blob/f2d5930b472e77bfacc2f437b3966e7a708a8d37/02-GD/utils.py#L92

    input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
    output: 4th coord
    """
    norm = lambda x: x / np.sqrt(np.square(x).sum(-1, keepdims=True) + 1e-8)
    with np.errstate(invalid="ignore"):  # inf - inf = nan is ok here
        vec_bc = N - Ca
        vec_ba = N - C
    bc = norm(vec_bc)
    n = norm(np.cross(vec_ba, bc))
    m = [bc, np.cross(n, bc), n]
    d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]
    return Ca + sum([m * d for m, d in zip(m, d)])


def chain_to_ndarray(
    atom_array: bs.AtomArray, mmcif: MmcifWrapper, chain_id: str, is_predicted=False
):
    entity_id = None
    for entity, chains in mmcif.entities.items():
        if chain_id in chains:
            entity_id = entity
    num_res = len(mmcif.chain_to_seqres[chain_id])
    sequence = mmcif.chain_to_seqres[chain_id]

    atom_positions = np.full([num_res, residue_constants.atom_type_num, 3], np.nan)
    atom_mask = np.full([num_res, residue_constants.atom_type_num], False, dtype=bool)
    residue_index = np.full([num_res], -1, dtype=np.int64)
    insertion_code = np.full([num_res], "", dtype="<U4")

    confidence = np.ones([num_res], dtype=np.float32)

    for res_index in range(num_res):
        chain = atom_array[atom_array.chain_id == chain_id]
        assert isinstance(chain, bs.AtomArray)
        res_at_position = mmcif.seqres_to_structure[chain_id][res_index]

        if res_at_position.residue_number is None:
            continue

        residue_index[res_index] = res_at_position.residue_number
        insertion_code[res_index] = res_at_position.insertion_code
        res = chain[
            (chain.res_id == res_at_position.residue_number)
            & (chain.ins_code == res_at_position.insertion_code)
            & (chain.hetero == res_at_position.hetflag)
        ]
        assert isinstance(res, bs.AtomArray)

        # Atom level features
        for atom in res:
            atom_name = atom.atom_name
            if atom_name == "SE" and atom.res_name == "MSE":
                # Put the coords of the selenium atom in the sulphur column
                atom_name = "SD"

            if atom_name in residue_constants.atom_order:
                atom_positions[res_index, residue_constants.atom_order[atom_name]] = (
                    atom.coord
                )
                atom_mask[res_index, residue_constants.atom_order[atom_name]] = True
                if is_predicted and atom_name == "CA":
                    confidence[res_index] = atom.b_factor

    assert all(sequence), "Some residue name was not specified correctly"
    return (
        sequence,
        atom_positions,
        atom_mask,
        residue_index,
        insertion_code,
        confidence,
        entity_id,
    )


@dataclass(frozen=True)
class ProteinChain:
    """Dataclass with atom37 representation of a single protein chain."""

    id: str
    sequence: str
    chain_id: str  # author chain id - mutable
    entity_id: int | None
    residue_index: np.ndarray
    insertion_code: np.ndarray
    atom37_positions: np.ndarray
    atom37_mask: np.ndarray
    confidence: np.ndarray
    mmcif: MmcifWrapper | None = None

    def __post_init__(self):
        assert self.atom37_mask.dtype == bool, self.atom37_mask.dtype
        assert self.atom37_positions.shape[0] == len(self.sequence), (
            self.atom37_positions.shape,
            len(self.sequence),
        )
        assert self.atom37_mask.shape[0] == len(self.sequence), (
            self.atom37_mask.shape,
            len(self.sequence),
        )
        assert self.residue_index.shape[0] == len(self.sequence), (
            self.residue_index.shape,
            len(self.sequence),
        )
        assert self.insertion_code.shape[0] == len(self.sequence), (
            self.insertion_code.shape,
            len(self.sequence),
        )
        assert self.confidence.shape[0] == len(self.sequence), (
            self.confidence.shape,
            len(self.sequence),
        )

    @cached_property
    def atoms(self) -> AtomIndexer:
        return AtomIndexer(self, property="atom37_positions", dim=-2)

    @cached_property
    def atom_mask(self) -> AtomIndexer:
        return AtomIndexer(self, property="atom37_mask", dim=-1)

    @cached_property
    def atom_array(self) -> bs.AtomArray:
        atoms = []
        for res_name, res_idx, ins_code, positions, mask, conf in zip(
            self.sequence,
            self.residue_index,
            self.insertion_code,
            self.atom37_positions,
            self.atom37_mask.astype(bool),
            self.confidence,
        ):
            for i, pos in zip(np.where(mask)[0], positions[mask]):
                atom = bs.Atom(
                    coord=pos,
                    chain_id="A" if self.chain_id is None else self.chain_id,
                    res_id=res_idx,
                    ins_code=ins_code,
                    res_name=residue_constants.restype_1to3.get(res_name, "UNK"),
                    hetero=False,
                    atom_name=residue_constants.atom_types[i],
                    element=residue_constants.atom_types[i][0],
                    b_factor=conf,
                )
                atoms.append(atom)
        return bs.array(atoms)

    @cached_property
    def residue_index_no_insertions(self) -> np.ndarray:
        return self.residue_index + np.cumsum(self.insertion_code != "")

    @cached_property
    def atom_array_no_insertions(self) -> bs.AtomArray:
        atoms = []
        for res_idx, (res_name, positions, mask, conf) in enumerate(
            zip(
                self.sequence,
                self.atom37_positions,
                self.atom37_mask.astype(bool),
                self.confidence,
            )
        ):
            for i, pos in zip(np.where(mask)[0], positions[mask]):
                atom = bs.Atom(
                    coord=pos,
                    # hard coded to as we currently only support single chain structures
                    chain_id=CHAIN_ID_CONST,
                    res_id=res_idx + 1,
                    res_name=residue_constants.restype_1to3.get(res_name, "UNK"),
                    hetero=False,
                    atom_name=residue_constants.atom_types[i],
                    element=residue_constants.atom_types[i][0],
                    b_factor=conf,
                )
                atoms.append(atom)
        return bs.array(atoms)

    def __getitem__(self, idx: int | list[int] | slice | np.ndarray | torch.Tensor):
        if isinstance(idx, int):
            idx = [idx]
        if isinstance(idx, torch.Tensor):
            idx = idx.cpu().numpy()

        sequence = slice_python_object_as_numpy(self.sequence, idx)
        return replace(
            self,
            sequence=sequence,
            residue_index=self.residue_index[..., idx],
            insertion_code=self.insertion_code[..., idx],
            atom37_positions=self.atom37_positions[..., idx, :, :],
            atom37_mask=self.atom37_mask[..., idx, :],
            confidence=self.confidence[..., idx],
        )

    def __len__(self):
        return len(self.sequence)

    def cbeta_contacts(self, distance_threshold: float = 8.0) -> np.ndarray:
        distance = self.pdist_CB
        contacts = (distance < distance_threshold).astype(np.int64)
        contacts[np.isnan(distance)] = -1
        np.fill_diagonal(contacts, -1)
        return contacts

    def to_pdb(self, path: PathOrBuffer, include_insertions: bool = True):
        """Dssp works better w/o insertions."""
        f = PDBFile()
        if not include_insertions:
            f.set_structure(self.atom_array_no_insertions)
        else:
            f.set_structure(self.atom_array)
        f.write(path)

    def to_pdb_string(self, include_insertions: bool = True) -> str:
        buf = io.StringIO()
        self.to_pdb(buf, include_insertions=include_insertions)
        buf.seek(0)
        return buf.read()

    def to_mmcif(self, path: PathOrBuffer):
        f = CIFFile()
        set_structure_pdbx(f, self.atom_array, data_block=self.id)

        # incantations molstar needs to render pLDDT / confidence onto
        # the structure with "alphafold-view"
        f.block["ma_qa_metric"] = CIFCategory(
            name="ma_qa_metric",
            columns={
                "id": CIFColumn(data=CIFData(array=np.array([1, 2]), dtype=np.int64)),
                "mode": CIFColumn(
                    data=CIFData(array=np.array(["global", "local"]), dtype=np.str_)
                ),
                "name": CIFColumn(
                    data=CIFData(array=np.array(["pLDDT", "pLDDT"]), dtype=np.str_)
                ),
            },
        )

        # table is a duplicate of data already in the atom array, but
        # needed by molstar to render pLDDT / confidence
        resid_pldd_table = {
            # hard coded to as we currently only support single chain structures
            "label_asym_id": CIFColumn(
                data=CIFData(
                    array=[CHAIN_ID_CONST] * len(self.residue_index), dtype=np.str_
                )
            ),
            "label_comp_id": CIFColumn(
                data=CIFData(
                    array=[
                        residue_constants.restype_1to3.get(c, "UNK")
                        for c in self.sequence
                    ],
                    dtype=np.str_,
                )
            ),
            "label_seq_id": CIFColumn(
                data=CIFData(array=self.residue_index, dtype=np.int64)
            ),
            "ordinal_id": CIFColumn(
                data=CIFData(array=self.residue_index, dtype=np.int64)
            ),
            # hard coded to show these are all local plDDT values
            "metric_id": CIFColumn(
                data=CIFData(array=["2"] * len(self.residue_index), dtype=np.str_)
            ),
            "metric_value": CIFColumn(
                data=CIFData(array=self.confidence, dtype=np.float32)
            ),
            # hard coded to show there are the initial version, there are no revisions
            "model_id": CIFColumn(
                data=CIFData(array=["1"] * len(self.residue_index), dtype=np.str_)
            ),
        }
        f.block["ma_qa_metric_local"] = CIFCategory(
            name="ma_qa_metric_local", columns=resid_pldd_table
        )
        f.write(path)

    def to_mmcif_string(self) -> str:
        buf = io.StringIO()
        self.to_mmcif(buf)
        buf.seek(0)
        return buf.read()

    def state_dict(self, backbone_only=False, json_serializable=False):
        """This state dict is optimized for storage, so it turns things to fp16 whenever
        possible. Note that we also only support int32 residue indices, I'm hoping we don't
        need more than 2**32 residues..."""
        dct = {k: v for k, v in asdict(self).items() if k not in ["mmcif"]}
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
        return dct

    def to_blob(self, backbone_only=False) -> bytes:
        return brotli.compress(msgpack.dumps(self.state_dict(backbone_only)), quality=5)

    @classmethod
    def from_open_source(cls, pc: ProteinChain):
        return cls(**vars(pc))

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
        return cls(**dct, mmcif=None)

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
        return cls.from_state_dict(msgpack.loads(brotli.decompress(bytes)))

    def sasa(self, by_residue: bool = True):
        arr = self.atom_array_no_insertions
        sasa_per_atom = bs.sasa(arr)  # type: ignore
        if by_residue:
            # Sum per-atom SASA into residue "bins", with np.bincount.
            assert arr.res_id is not None
            # NOTE(rverkuil): arr.res_id is 1-indexed, but np.bincount returns a sum for bin 0, so we strip.
            # NOTE(aderry): We compute only for residues with coordinates, return NaN otherwise.
            num_trailing_residues = len(self) - arr.res_id.max()
            sasa_per_residue = np.concatenate(
                [
                    np.bincount(arr.res_id, weights=sasa_per_atom)[1:],
                    np.zeros(num_trailing_residues),
                ]
            )
            sasa_per_residue[~self.atom37_mask.any(-1)] = np.nan
            assert len(sasa_per_residue) == len(self)
            return sasa_per_residue
        return sasa_per_atom

    def sap_score(self, aggregation: str = "atom") -> np.ndarray:
        """Computes per-atom SAP score.
        Can optionally aggregate by residue (by averaging over atoms. NOTE: this returns values only for residues that have coordinates!)
        or full-protein (sum of SAP score for atoms with SAP > 0, as in Lauer et al. 2011)."""
        sap_radius = 5.0
        arr = self.atom_array_no_insertions

        # asserts to avoid type errors
        assert arr.res_id is not None
        assert arr.res_name is not None
        assert arr.atom_name is not None
        assert arr.coord is not None

        # compute SASA and residue-specific properties
        sasa_per_atom = self.sasa(by_residue=False)
        resid_to_resname = dict(zip(arr.res_id, arr.res_name))

        max_side_chain_asa = np.full(len(self), np.nan)
        res_hydrophobicity = np.full(len(self), np.nan)
        resolved_res_mask = self.atom37_mask.any(-1)
        num_trailing_residues = len(self) - arr.res_id.max()

        max_side_chain_asa[resolved_res_mask] = np.array(
            [
                residue_constants.side_chain_asa[resid_to_resname[i]]
                for i in np.unique(arr.res_id)
            ]
        )
        res_hydrophobicity[resolved_res_mask] = np.array(
            [
                residue_constants.hydrophobicity[resid_to_resname[i]]
                for i in np.unique(arr.res_id)
            ]
        )
        assert len(max_side_chain_asa) == len(self)
        assert len(res_hydrophobicity) == len(self)

        # compute SAP score
        is_side_chain = ~bs.filter_peptide_backbone(arr)
        sasa_per_atom[is_side_chain] = 0
        kdtree = KDTree(arr.coord)
        neighbors = kdtree.query_ball_tree(kdtree, sap_radius, p=2.0)
        sap_by_atom = np.zeros_like(sasa_per_atom)
        for i, nn_list in enumerate(neighbors):
            saa_nn = np.zeros_like(sasa_per_atom)
            saa_nn[nn_list] = sasa_per_atom[nn_list]
            sasa_within_r = np.concatenate(
                [
                    np.bincount(arr.res_id, weights=saa_nn)[1:],
                    np.zeros(num_trailing_residues),
                ]
            )
            sap = np.nansum((sasa_within_r / max_side_chain_asa) * res_hydrophobicity)
            sap_by_atom[i] = sap

        match aggregation:
            case "atom":
                return sap_by_atom
            case "residue":
                sap_by_residue = np.concatenate(
                    [
                        np.bincount(arr.res_id, weights=sap_by_atom)[1:],
                        np.zeros(num_trailing_residues),
                    ]
                ) / (
                    np.concatenate(
                        [np.bincount(arr.res_id)[1:], np.zeros(num_trailing_residues)]
                    )
                    + 1e-8
                )
                sap_by_residue[~resolved_res_mask] = np.nan
                assert len(sap_by_residue) == len(self)
                return sap_by_residue
            case "protein":
                return sum(sap_by_atom[sap_by_atom > 0])  # pyright: ignore[reportReturnType]
            case _:
                raise ValueError(
                    f"Invalid aggregation method: {aggregation}. Must be one of 'atom', 'residue', or 'protein'"
                )

    def globularity(self) -> float:
        # Computes globularity using total volumes divided by MVEE.
        # We make the simplifying approximation that atoms never overlap.
        # The globularity is only computed where structure exists.
        # Besides the approximation above, this is inspired by:

        # https://www.mdpi.com/2073-4352/11/12/1539
        # NOTE(@zeming): due to the approximation we make here, that atoms never overlap, you might get >1 globularity
        mask = self.atom37_mask.any(-1)
        points = self.atom37_positions[self.atom37_mask]
        sequence = [aa for aa, m in zip(self.sequence, mask) if m]  # type: ignore
        A, _ = self._mvee(points, tol=1e-3)
        mvee_volume = (4 * np.pi) / (3 * np.sqrt(np.linalg.det(A)))
        volume = sum(residue_constants.amino_acid_volumes[x] for x in sequence)
        ratio = volume / mvee_volume

        # The paper says you must compare the ellipsoidal profile with T, a measurement of
        # how elongated the ellipsoid is. We want a single number, so we multiply by 1/2T, so
        # that value is normalized between 0-1
        eigenvalues = np.linalg.eigvals(A)
        R = 1 / np.sqrt(eigenvalues)
        # ellipsoid radii length triangle inequality coefficient
        T = max(R[0] / (R[1] + R[2]), R[1] / (R[0] + R[2]), R[2] / (R[0] + R[1]))
        elongation_metric = 1 / max(T, 1)
        return ratio * elongation_metric

    @staticmethod
    def _mvee(P: np.ndarray, tol, max_iter=10000):
        # Finds minimum volume enclosing ellipsoid of a set of points.
        # Returns A, c where the ellipse is defined as:
        #    (x-c).T @ A @ (x-c) = 1
        hull = ConvexHull(P)
        P = P[hull.vertices]
        P = P.T

        # Data points
        d, N = P.shape
        Q = np.zeros((d + 1, N))
        Q[:d, :] = P[:d, :N]
        Q[d, :] = np.ones((1, N))

        # Initializations
        count = 1
        err = 1.0
        u = np.full((N, 1), 1 / N)  # 1st iteration

        # Khachiyan Algorithm
        for i in range(max_iter):
            X = Q.dot(np.diag(u.squeeze())) @ Q.T
            M = np.diag(Q.T @ np.linalg.inv(X) @ Q)
            maximum, j = np.max(M), np.argmax(M)
            step_size = (maximum - d - 1) / ((d + 1) * (maximum - 1))
            new_u = (1 - step_size) * u
            new_u[j] += step_size
            count += 1
            err = np.linalg.norm(new_u - u)
            u = new_u
            if err < tol:
                break
        else:
            raise ValueError("MVEE did not converge")

        d = P.shape[0]  # Fixed: use P.shape[0] instead of P.shape
        U = np.diag(u.squeeze())

        # The A matrix for the ellipse
        A = (1 / d) * np.linalg.inv(P @ U @ P.T - (P @ u) @ (P @ u).T)

        # Center of the ellipse
        c = P @ u

        return A, c

    def radius_of_gyration(self):
        arr = self.atom_array_no_insertions
        return bs.gyration_radius(arr)

    def align(
        self,
        target: ProteinChain,
        mobile_inds: list[int] | np.ndarray | None = None,
        target_inds: list[int] | np.ndarray | None = None,
        only_use_backbone: bool = False,
    ):
        """
        Aligns the current protein to the provided target.

        Args:
            target (ProteinChain): The target protein to align to.
            mobile_inds (list[int], np.ndarray, optional): The indices of the mobile atoms to align. These are NOT residue indices
            target_inds (list[int], np.ndarray, optional): The indices of the target atoms to align. These are NOT residue indices
            only_use_backbone (bool, optional): If True, only align the backbone atoms.
        """
        aligner = Aligner(
            self if mobile_inds is None else self[mobile_inds],
            target if target_inds is None else target[target_inds],
            only_use_backbone,
        )

        return aligner.apply(self)

    def rmsd(
        self,
        target: ProteinChain,
        also_check_reflection: bool = False,
        mobile_inds: list[int] | np.ndarray | None = None,
        target_inds: list[int] | np.ndarray | None = None,
        only_compute_backbone_rmsd: bool = False,
    ):
        """
        Compute the RMSD between this protein chain and another.

        Args:
            target (ProteinChain): The target (other) protein chain to compare to.
            also_check_reflection (bool, optional): If True, also check if the reflection of the mobile atoms has a lower RMSD.
            mobile_inds (list[int], optional): The indices of the mobile atoms to align. These are NOT residue indices
            target_inds (list[int], optional): The indices of the target atoms to align. These are NOT residue indices
            only_compute_backbone_rmsd (bool, optional): If True, only compute the RMSD of the backbone atoms.
        """
        if isinstance(target, bs.AtomArray):
            raise ValueError(
                "Support for bs.AtomArray removed, use "
                "ProteinChain.from_atomarry for ProteinChain."
            )
        aligner = Aligner(
            self if mobile_inds is None else self[mobile_inds],
            target if target_inds is None else target[target_inds],
            only_compute_backbone_rmsd,
        )
        avg_rmsd = aligner.rmsd

        if not also_check_reflection:
            return avg_rmsd

        aligner = Aligner(
            self if mobile_inds is None else self[mobile_inds],
            target if target_inds is None else target[target_inds],
            only_compute_backbone_rmsd,
            use_reflection=True,
        )
        avg_rmsd_neg = aligner.rmsd

        return min(avg_rmsd, avg_rmsd_neg)

    def lddt_ca(
        self,
        native: ProteinChain,
        mobile_inds: list[int] | np.ndarray | None = None,
        target_inds: list[int] | np.ndarray | None = None,
        **kwargs,
    ) -> float | np.ndarray:
        """Compute the LDDT between this protein chain and another. NOTE: LDDT IS NOT SYMMETRIC.
        The call should always be prediction.lddt_ca(native).

        Arguments:
            native (ProteinChain): The ground truth protein chain
            mobile_inds (list[int], np.ndarray, optional): The indices of the mobile atoms to align. These are NOT residue indices
            target_inds (list[int], np.ndarray, optional): The indices of the target atoms to align. These are NOT residue indices

        Returns:
            float | np.ndarray: The LDDT score between the two protein chains, either
                a single float or per-residue LDDT scores if `per_residue` is True.
        """
        lddt = compute_lddt_ca(
            torch.tensor(self.atom37_positions[mobile_inds]).unsqueeze(0),
            torch.tensor(native.atom37_positions[target_inds]).unsqueeze(0),
            torch.tensor(native.atom37_mask[mobile_inds]).unsqueeze(0),
            **kwargs,
        )
        return float(lddt) if lddt.numel() == 1 else lddt.numpy().flatten()

    def gdt_ts(
        self,
        target: ProteinChain,
        mobile_inds: list[int] | np.ndarray | None = None,
        target_inds: list[int] | np.ndarray | None = None,
        **kwargs,
    ) -> float | np.ndarray:
        """Compute the GDT_TS between this protein chain and another.

        Arguments:
            target (ProteinChain): The other protein chain to compare to.
            mobile_inds (list[int], np.ndarray, optional): The indices of the mobile atoms to align. These are NOT residue indices
            target_inds (list[int], np.ndarray, optional): The indices of the target atoms to align. These are NOT residue indices

        Returns:
            float: The GDT_TS score between the two protein chains.
        """
        gdt_ts = compute_gdt_ts(
            mobile=torch.tensor(
                index_by_atom_name(self.atom37_positions[mobile_inds], "CA"),
                dtype=torch.float32,
            ).unsqueeze(0),
            target=torch.tensor(
                index_by_atom_name(target.atom37_positions[target_inds], "CA"),
                dtype=torch.float32,
            ).unsqueeze(0),
            atom_exists_mask=torch.tensor(
                index_by_atom_name(self.atom37_mask[mobile_inds], "CA", dim=-1)
                & index_by_atom_name(target.atom37_mask[target_inds], "CA", dim=-1)
            ).unsqueeze(0),
            **kwargs,
        )
        return float(gdt_ts) if gdt_ts.numel() == 1 else gdt_ts.numpy().flatten()

    @classmethod
    def chain_iterable_from_mmcif(
        cls,
        path: PathOrBuffer | MmcifWrapper,
        id: str | None = None,
        is_predicted: bool = False,
        keep_source: bool = False,
    ):
        """Return a list[ProteinChain] object from an mmcif file, a iterable list of all protein chain
        from an mmcif file
        """
        if isinstance(path, MmcifWrapper):
            mmcif = path
        else:
            mmcif = MmcifWrapper.read(path, id)
        for chain in bs.chain_iter(mmcif.structure):
            chain = chain[bs.filter_amino_acids(chain) & ~chain.hetero]
            if len(chain) == 0:
                continue
            chain_id = chain.chain_id[0]
            entity_id = None
            for entity, chains in mmcif.entities.items():
                if chain_id in chains:
                    entity_id = entity
            assert entity_id is not None
            (
                sequence,
                atom_positions,
                atom_mask,
                residue_index,
                insertion_code,
                confidence,
                _,
            ) = chain_to_ndarray(chain, mmcif, chain_id, is_predicted)
            assert all(sequence), "Some residue name was not specified correctly"

            yield cls(
                id=mmcif.id,
                sequence=sequence,
                chain_id=chain_id,
                entity_id=entity_id,
                atom37_positions=atom_positions,
                atom37_mask=atom_mask,
                residue_index=residue_index,
                insertion_code=insertion_code,
                confidence=confidence,
                mmcif=mmcif if keep_source else None,
            )

    @classmethod
    def from_mmcif(
        cls,
        path: PathOrBuffer | MmcifWrapper,
        chain_id: str | None = None,
        entity_id: int | None = None,
        id: str | None = None,
        is_predicted: bool = False,
        keep_source: bool = False,
    ):
        """Return a ProteinChain object from an mmcif file.

        Args:
            path (str | Path | io.TextIO): Path or buffer to read mmcif file from. Should be uncompressed.
            id (str, optional): String identifier to assign to structure. Will attempt to infer otherwise.
            is_predicted (bool): If True, reads b factor as the confidence readout. Default: False.
            chain_id (str, optional): Select a chain corresponding to (author) chain id.
            entity_id (int, optional): Select a chain corresponding to a particular entity.

        If neither `chain_id` nor `entity_id` is specified, defaults to the first entity.
        """
        if isinstance(path, MmcifWrapper):
            mmcif = path
        else:
            mmcif = MmcifWrapper.read(path, id)

        # If neither chain_id nor entity_id is specified, default to the first entity
        if chain_id is None and entity_id is None:
            if not mmcif.entities:
                raise ValueError("Structure contains no entities")
            entity_id = min(mmcif.entities.keys())  # Pick the first entity by ID

        if entity_id is not None:
            assert chain_id is None
            if entity_id not in mmcif.entities:
                raise ValueError(
                    f"Structure does not contain entity `{entity_id}`. Valid entities: {mmcif.entities.keys()}"
                )
            chains = mmcif.entities[entity_id]

            # Select the chain id corresponding to the longest chain. If all are equal length, selects the first.
            chain_id = max(
                chains,
                key=lambda chain: _num_non_null_residues(
                    mmcif.seqres_to_structure[chain]
                ),
            )
        else:
            assert chain_id is not None
            for entity, chains in mmcif.entities.items():
                if chain_id in chains:
                    entity_id = entity
        if entity_id is None:
            warnings.warn(
                "Failed to detect entity_id from mmcif file, it may be malformed."
            )

        atom_array = mmcif.structure
        (
            sequence,
            atom_positions,
            atom_mask,
            residue_index,
            insertion_code,
            confidence,
            _,
        ) = chain_to_ndarray(atom_array, mmcif, chain_id, is_predicted)
        assert all(sequence), "Some residue name was not specified correctly"

        return cls(
            id=mmcif.id,
            sequence=sequence,
            chain_id=chain_id,
            entity_id=entity_id,
            atom37_positions=atom_positions,
            atom37_mask=atom_mask.astype(bool),
            residue_index=residue_index,
            insertion_code=insertion_code,
            confidence=confidence,
            mmcif=mmcif if keep_source else None,
        )

    @classmethod
    def from_atom37(
        cls,
        atom37_positions: np.ndarray | torch.Tensor,
        *,
        id: str | None = None,
        sequence: str | None = None,
        chain_id: str | None = None,
        entity_id: int | None = None,
        residue_index: np.ndarray | torch.Tensor | None = None,
        insertion_code: np.ndarray | None = None,
        confidence: np.ndarray | torch.Tensor | None = None,
    ):
        if isinstance(atom37_positions, torch.Tensor):
            atom37_positions = atom37_positions.cpu().numpy()
            if atom37_positions.ndim == 4:
                if atom37_positions.shape[0] != 1:
                    raise ValueError(
                        f"Cannot handle batched inputs, atom37_positions has shape {atom37_positions.shape}"
                    )
                atom37_positions = atom37_positions[0]

        assert isinstance(atom37_positions, np.ndarray)
        seqlen = atom37_positions.shape[0]

        atom_mask = np.isfinite(atom37_positions).all(-1)

        if id is None:
            id = ""

        if sequence is None:
            sequence = "A" * seqlen

        if chain_id is None:
            chain_id = "A"

        if residue_index is None:
            residue_index = np.arange(1, seqlen + 1)
        elif isinstance(residue_index, torch.Tensor):
            residue_index = residue_index.cpu().numpy()
            assert isinstance(residue_index, np.ndarray)
            if residue_index.ndim == 2:
                if residue_index.shape[0] != 1:
                    raise ValueError(
                        f"Cannot handle batched inputs, residue_index has shape {residue_index.shape}"
                    )
                residue_index = residue_index[0]
        assert isinstance(residue_index, np.ndarray)

        if insertion_code is None:
            insertion_code = np.array(["" for _ in range(seqlen)])

        if confidence is None:
            confidence = np.ones(seqlen, dtype=np.float32)
        elif isinstance(confidence, torch.Tensor):
            confidence = confidence.cpu().numpy()
            assert isinstance(confidence, np.ndarray)
            if confidence.ndim == 2:
                if confidence.shape[0] != 1:
                    raise ValueError(
                        f"Cannot handle batched inputs, confidence has shape {confidence.shape}"
                    )
                confidence = confidence[0]
        assert isinstance(confidence, np.ndarray)

        return cls(
            id=id,
            sequence=sequence,  # type: ignore
            chain_id=chain_id,
            entity_id=entity_id,
            atom37_positions=atom37_positions,
            atom37_mask=atom_mask.astype(bool),
            residue_index=residue_index,
            insertion_code=insertion_code,
            confidence=confidence,
        )

    @classmethod
    def from_backbone_atom_coordinates(
        cls, backbone_atom_coordinates: np.ndarray | torch.Tensor, **kwargs
    ):
        """Create a ProteinChain from a set of backbone atom coordinates.

        This function simply expands the seqlen x 3 x 3 array of backbone atom
        coordinates to a seqlen x 37 x 3 array of all atom coordinates, with the padded
        positions set to infinity. This allows us to use from_atom37 to create the
        appropriate ProteinChain object with the appropriate atom37_mask.

        This function passes all kwargs to from_atom37.
        """
        if isinstance(backbone_atom_coordinates, torch.Tensor):
            backbone_atom_coordinates = backbone_atom_coordinates.cpu().numpy()
            if backbone_atom_coordinates.ndim == 4:
                if backbone_atom_coordinates.shape[0] != 1:
                    raise ValueError(
                        f"Cannot handle batched inputs, backbone_atom_coordinates has "
                        f"shape {backbone_atom_coordinates.shape}"
                    )
                backbone_atom_coordinates = backbone_atom_coordinates[0]

        assert isinstance(backbone_atom_coordinates, np.ndarray)
        assert backbone_atom_coordinates.ndim == 3
        assert backbone_atom_coordinates.shape[-2] == 3
        assert backbone_atom_coordinates.shape[-1] == 3

        atom37_positions = np.full(
            (backbone_atom_coordinates.shape[0], 37, 3),
            np.inf,
            dtype=backbone_atom_coordinates.dtype,
        )
        atom37_positions[:, :3, :] = backbone_atom_coordinates

        return cls.from_atom37(atom37_positions=atom37_positions, **kwargs)

    @classmethod
    def from_pdb(
        cls,
        path: PathOrBuffer,
        chain_id: str = "detect",
        id: str | None = None,
        is_predicted: bool = False,
    ) -> "ProteinChain":
        """Return a ProteinChain object from an pdb file. NOTE: prefer mmcif for rcsb PDB files.
        This function is mostly to interface with old PDB files and predicted structures -
        it will not fill out the entity id correctly

        Args:
            path (str | Path | io.TextIO): Path or buffer to read mmcif file from. Should be uncompressed.
            id (str, optional): String identifier to assign to structure. Will attempt to infer otherwise.
            is_predicted (bool): If True, reads b factor as the confidence readout. Default: False.
            chain_id (str, optional): Select a chain corresponding to (author) chain id. "detect" uses the
                first detected chain
        """

        if id is not None:
            file_id = id
        else:
            match path:
                case Path() | str():
                    file_id = Path(path).with_suffix("").name
                case _:
                    file_id = "null"

        atom_array = PDBFile.read(path).get_structure(
            model=1, extra_fields=["b_factor"]
        )
        if chain_id == "detect":
            chain_id = atom_array.chain_id[0]
        atom_array = atom_array[
            bs.filter_amino_acids(atom_array)
            & ~atom_array.hetero
            & (atom_array.chain_id == chain_id)
        ]

        entity_id = 1  # Not supplied in PDBfiles

        sequence = "".join(
            residue_constants.restype_3to1.get(monomer[0].res_name, "X")
            for monomer in bs.residue_iter(atom_array)
        )
        num_res = len(sequence)

        atom_positions = np.full(
            [num_res, residue_constants.atom_type_num, 3], np.nan, dtype=np.float32
        )
        atom_mask = np.full(
            [num_res, residue_constants.atom_type_num], False, dtype=bool
        )
        residue_index = np.full([num_res], -1, dtype=np.int64)
        insertion_code = np.full([num_res], "", dtype="<U4")

        confidence = np.ones([num_res], dtype=np.float32)

        for i, res in enumerate(bs.residue_iter(atom_array)):
            chain = atom_array[atom_array.chain_id == chain_id]
            assert isinstance(chain, bs.AtomArray)

            res_index = res[0].res_id
            residue_index[i] = res_index
            insertion_code[i] = res[0].ins_code

            # Atom level features
            for atom in res:
                atom_name = atom.atom_name
                if atom_name == "SE" and atom.res_name == "MSE":
                    # Put the coords of the selenium atom in the sulphur column
                    atom_name = "SD"

                if atom_name in residue_constants.atom_order:
                    atom_positions[i, residue_constants.atom_order[atom_name]] = (
                        atom.coord
                    )
                    atom_mask[i, residue_constants.atom_order[atom_name]] = True
                    if is_predicted and atom_name == "CA":
                        confidence[i] = atom.b_factor

        assert all(sequence), "Some residue name was not specified correctly"

        return cls(
            id=file_id,
            sequence=sequence,
            chain_id=chain_id,
            entity_id=entity_id,
            atom37_positions=atom_positions,
            atom37_mask=atom_mask.astype(bool),
            residue_index=residue_index,
            insertion_code=insertion_code,
            confidence=confidence,
            mmcif=None,
        )

    @classmethod
    def from_mds(cls, data: dict[str, Any]) -> "ProteinChain":
        return cls(
            id=data["id"],
            chain_id=data["chain_id"],
            entity_id=data["entity_id"],
            sequence=data["sequence"],
            residue_index=data["residue_index"],
            insertion_code=np.asarray(data["insertion_code"]),
            atom37_positions=data["atom37_positions"],
            atom37_mask=data["atom37_mask"].astype(bool),
            confidence=data["confidence"],
            mmcif=None,
        )

    @classmethod
    def from_rcsb(
        cls,
        pdb_id: str,
        chain_id: str | None = None,
        entity_id: int | None = None,
        keep_source: bool = False,
    ) -> ProteinChain:
        f: io.StringIO = rcsb.fetch(pdb_id, "cif")  # type: ignore
        return cls.from_mmcif(
            f,
            id=pdb_id,
            chain_id=chain_id,
            entity_id=entity_id,
            keep_source=keep_source,
            is_predicted=False,
        )

    @classmethod
    def from_atomarray(
        cls, atom_array: bs.AtomArray, id: str | None = None, is_predicted: bool = False
    ) -> "ProteinChain":
        """A simple converter from bs.AtomArray -> ProteinChain.
        Uses PDB file format as intermediate."""
        atom_array = atom_array.copy()
        atom_array.box = None  # remove surrounding box, from_pdb won't handle this
        pdb_file = PDBFile()  # pyright: ignore
        pdb_file.set_structure(atom_array)

        buf = io.StringIO()
        pdb_file.write(buf)
        buf.seek(0)
        return cls.from_pdb(buf, id=id, is_predicted=is_predicted)

    def get_normalization_frame(self) -> Affine3D:
        """Given a set of coordinates, compute a single frame.
        Specifically, we compute the average position of the N, CA, and C atoms use those 3 points to construct a frame using the Gram-Schmidt algorithm. The average CA position is used as the origin of the frame.

        Returns:
            Affine3D: [] tensor of Affine3D frame
        """
        coords = torch.from_numpy(self.atom37_positions)
        frame = get_protein_normalization_frame(coords)

        return frame

    def apply_frame(self, frame: Affine3D) -> ProteinChain:
        """Given a frame, apply the frame to the protein's coordinates.

        Args:
            frame (Affine3D): [] tensor of Affine3D frame

        Returns:
            ProteinChain: Transformed protein chain
        """
        coords = torch.from_numpy(self.atom37_positions).to(frame.trans.dtype)
        coords = apply_frame_to_coords(coords, frame)
        atom37_positions = coords.numpy()
        return replace(self, atom37_positions=atom37_positions)

    def normalize_coordinates(self) -> ProteinChain:
        """Normalize the coordinates of the protein chain."""
        return self.apply_frame(self.get_normalization_frame())

    def infer_oxygen(self) -> ProteinChain:
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

    @cached_property
    def inferred_cbeta(self) -> np.ndarray:
        """Infer cbeta positions based on N, C, CA."""
        N, CA, C = np.moveaxis(self.atoms[["N", "CA", "C"]], 1, 0)
        # See usage in trDesign codebase.
        # https://github.com/gjoni/trDesign/blob/f2d5930b472e77bfacc2f437b3966e7a708a8d37/02-GD/utils.py#L140
        CB = infer_CB(C, N, CA, 1.522, 1.927, -2.143)
        return CB

    def infer_cbeta(self, infer_cbeta_for_glycine: bool = False) -> ProteinChain:
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

        inferred_cbeta_positions = self.inferred_cbeta
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

    @cached_property
    def pdist_CA(self) -> np.ndarray:
        CA = self.atoms["CA"]
        pdist_CA = squareform(pdist(CA))
        return pdist_CA

    @cached_property
    def pdist_CB(self) -> np.ndarray:
        pdist_CB = squareform(pdist(self.inferred_cbeta))
        return pdist_CB

    @classmethod
    def as_complex(cls, chains: Sequence[ProteinChain]):
        raise RuntimeError(
            ".as_complex() has been deprecated in favor of .concat(). "
            ".concat() will eventually be deprecated in favor of ProteinComplex..."
        )

    @classmethod
    def concat(cls, chains: Sequence[ProteinChain], use_chainbreak: bool = True):
        sep_tokens = {
            "residue_index": np.array([-1]),
            "insertion_code": np.array([""]),
            "atom37_positions": np.full([1, 37, 3], np.inf),
            "atom37_mask": np.zeros([1, 37], dtype=bool),
            "confidence": np.array([0]),
        }

        def join_arrays(arrays: Sequence[np.ndarray], sep: np.ndarray):
            if use_chainbreak:
                full_array = []
                for array in arrays:
                    full_array.append(array)
                    full_array.append(sep)
                full_array = full_array[:-1]
                return np.concatenate(full_array, 0)
            else:
                return np.concatenate(arrays, 0)

        array_args: dict[str, np.ndarray] = {
            name: join_arrays([getattr(chain, name) for chain in chains], sep)
            for name, sep in sep_tokens.items()
        }

        chain_break = residue_constants.CHAIN_BREAK_TOKEN if use_chainbreak else ""
        return cls(
            id=chains[0].id,
            sequence=chain_break.join(chain.sequence for chain in chains),
            chain_id="A",
            entity_id=None,
            mmcif=None,
            **array_args,
        )

    def find_nonpolymer_contacts(self):
        assert self.mmcif is not None
        nonpolymer_and_chain_id_to_array = self.mmcif.non_polymer_coords

        results = []
        for (
            nonpolymer,
            _,
        ), nonpolymer_array in nonpolymer_and_chain_id_to_array.items():
            assert nonpolymer_array.coord is not None
            chain_coords = self.atom37_positions[self.atom37_mask]
            distance = cdist(nonpolymer_array.coord, chain_coords)

            is_contact = distance < 5
            if not is_contact.any():
                continue
            contacting_atoms = np.where(is_contact.any(0))[0]
            chain_index = np.where(self.atom37_mask)[0]
            contacting_residues = np.unique(chain_index[contacting_atoms])

            result = {
                "ligand": nonpolymer.name,
                "ligand_id": nonpolymer.comp_id,
                "contacting_residues": contacting_residues.tolist(),
            }
            results.append(result)
        return results

    def select_residue_indices(
        self, indices: list[int | str], ignore_x_mismatch: bool = False
    ) -> ProteinChain:
        numeric_indices = [
            idx if isinstance(idx, int) else int(idx[1:]) for idx in indices
        ]
        mask = np.isin(self.residue_index, numeric_indices)
        new = self[mask]
        mismatches = []
        for aa, idx in zip(new.sequence, indices):
            if isinstance(idx, int):
                continue
            if aa == "X" and ignore_x_mismatch:
                continue
            if aa != idx[0]:
                mismatches.append((aa, idx))
        if mismatches:
            mismatch_str = "; ".join(
                f"Position {idx[1:]}, Expected: {idx[0]}, Received: {aa}"
                for aa, idx in mismatches
            )
            raise RuntimeError(mismatch_str)

        return new

    def to_structure_encoder_inputs(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert protein chain to structure encoder inputs.

        Returns:
            tuple: (coordinates, plddt, residue_index) where:
                - coordinates: (1, L, 37, 3) tensor of atom positions
                - plddt: (1, L) tensor of confidence scores
                - residue_index: (1, L) tensor of residue indices
        """
        # Convert to tensors and add batch dimension
        coordinates = (
            torch.from_numpy(self.atom37_positions).float().unsqueeze(0)
        )  # (1, L, 37, 3)
        plddt = torch.from_numpy(self.confidence).float().unsqueeze(0)  # (1, L)
        residue_index = (
            torch.from_numpy(self.residue_index).long().unsqueeze(0)
        )  # (1, L)

        return coordinates, plddt, residue_index
