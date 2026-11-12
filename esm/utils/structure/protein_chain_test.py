from dataclasses import replace

import biotite.structure as bs
import numpy as np
import pytest
import torch

from esm.utils import residue_constants
from esm.utils.structure.metrics import compute_gdt_ts, compute_rmsd
from esm.utils.structure.protein_chain import ProteinChain


@pytest.fixture
def sap_chain() -> ProteinChain:
    positions = np.full((5, 37, 3), np.nan)
    # Missing terminal/interior residues exercise residue aggregation locally.
    for residue, atoms in {
        1: {"N": -5, "CA": -4, "C": -3, "O": -2, "OXT": -1, "CB": 0},
        3: {"N": 5, "CA": 6, "C": 7, "O": 8, "OXT": 9, "CB": 10, "CG": 11},
    }.items():
        for name, x in atoms.items():
            positions[residue, residue_constants.atom_order[name]] = [x, 0, 0]
    return ProteinChain.from_atom37(positions, sequence="AAADA")


@pytest.mark.parametrize("atom_name", ["N", "CA", "C", "O", "OXT"])
def test_sap_ignores_backbone_sasa(
    sap_chain: ProteinChain, monkeypatch: pytest.MonkeyPatch, atom_name: str
) -> None:
    arr = sap_chain.atom_array_no_insertions
    sasa = np.where(arr.atom_name == atom_name, 100.0, 0.0)
    assert np.count_nonzero(sasa) == 2
    monkeypatch.setattr(bs, "sasa", lambda _: sasa.copy())

    np.testing.assert_array_equal(sap_chain.sap_score(), np.zeros(len(arr)))
    assert sap_chain.sap_score(aggregation="protein") == 0.0


@pytest.mark.parametrize("backbone_sasa", [0.0, 1000.0])
def test_sap_sidechain_neighborhoods_and_aggregation(
    sap_chain: ProteinChain, monkeypatch: pytest.MonkeyPatch, backbone_sasa: float
) -> None:
    arr = sap_chain.atom_array_no_insertions
    sasa = np.full(len(arr), backbone_sasa)
    sasa[(arr.res_name == "ALA") & (arr.atom_name == "CB")] = 32.39045
    sasa[(arr.res_name == "ASP") & (arr.atom_name == "CB")] = 55.1045
    sasa[(arr.res_name == "ASP") & (arr.atom_name == "CG")] = 22.0418
    monkeypatch.setattr(bs, "sasa", lambda _: sasa.copy())

    ala_score = 0.5 * 0.116
    asp_cb_score = 0.5 * -0.472
    asp_score = 0.7 * -0.472
    expected = np.where(arr.res_name == "ALA", ala_score, asp_score)
    # ASP N is exactly 5 A from both CB atoms, but outside the ASP CG neighborhood.
    expected[(arr.res_name == "ASP") & (arr.atom_name == "N")] = (
        ala_score + asp_cb_score
    )
    atom_scores = sap_chain.sap_score(aggregation="atom")
    np.testing.assert_allclose(atom_scores, expected)
    assert len(atom_scores) == len(arr)
    assert np.any(atom_scores > 0) and np.any(atom_scores < 0)

    np.testing.assert_allclose(
        sap_chain.sap_score(aggregation="residue"),
        [
            np.nan,
            ala_score,
            np.nan,
            (ala_score + asp_cb_score + 6 * asp_score) / 7,
            np.nan,
        ],
        equal_nan=True,
    )
    protein_score = sap_chain.sap_score(aggregation="protein")
    assert protein_score == pytest.approx(atom_scores[atom_scores > 0].sum())
    assert protein_score == pytest.approx(6 * ala_score)


def _helix_chain(num_res: int = 30, seed: int = 0) -> ProteinChain:
    """Backbone of a right-handed helix with a little noise, so it is chiral."""
    rng = np.random.default_rng(seed)
    t = np.arange(num_res * 3) * (100 / 3) * np.pi / 180
    backbone = np.stack([2.3 * np.cos(t), 2.3 * np.sin(t), 0.5 * np.arange(len(t))], -1)
    backbone = backbone + rng.normal(scale=0.2, size=backbone.shape)
    positions = np.full((num_res, 37, 3), np.nan)
    positions[:, :3] = backbone.reshape(num_res, 3, 3)
    return ProteinChain.from_atom37(positions, sequence="A" * num_res)


def _mirror(chain: ProteinChain) -> ProteinChain:
    return replace(chain, atom37_positions=chain.atom37_positions * [-1.0, 1.0, 1.0])


def _handedness(chain: ProteinChain) -> float:
    ca = chain.atoms["CA"]
    a, b, c, d = ca[:-3], ca[1:-2], ca[2:-1], ca[3:]
    return float(np.mean(np.sign(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a))))


def test_rmsd_does_not_superimpose_mirror_image() -> None:
    chain = _helix_chain()
    mirror = _mirror(chain)

    assert chain.rmsd(chain) == pytest.approx(0.0, abs=1e-5)
    assert mirror.rmsd(chain) > 1.0
    assert mirror.rmsd(chain, also_check_reflection=True) == pytest.approx(
        0.0, abs=1e-4
    )


def test_align_preserves_handedness() -> None:
    chain = _helix_chain()
    mirror = _mirror(chain)

    assert _handedness(chain) != 0
    assert _handedness(mirror) == -_handedness(chain)
    assert _handedness(mirror.align(chain)) == _handedness(mirror)


def test_metrics_rmsd_and_gdt_ts_reject_mirror_image() -> None:
    chain = _helix_chain()
    ca = torch.tensor(chain.atoms["CA"], dtype=torch.float32)[None]
    mirror_ca = ca * torch.tensor([-1.0, 1.0, 1.0])

    assert float(compute_rmsd(mirror_ca, ca)) > 1.0
    assert float(compute_gdt_ts(mirror_ca, ca)) < 1.0
