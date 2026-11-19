"""Multi-GPU ``"big"``: splitting pair rows over ranks must not change the fold.

The ranks here are CPU processes in a gloo group, launched the way ``torchrun``
launches them, so every collective of the row-sharded path runs for real.
"""

import os
import socket
import statistics

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from esm.models.esmfold2 import EsmFold2Config, EsmFold2Model, rowpair
from esm.models.esmfold2 import layers as _layers
from tests.conftest import _protein_of_length, esmfold2_complex_features

pytestmark = [
    pytest.mark.manual,
    pytest.mark.skipif(
        not rowpair.ROWPAIR_AVAILABLE, reason="opt_core is an optional install"
    ),
]

TINY_MSA_ENCODER = dict(
    enabled=True,
    hidden_size=32,
    outer_hidden_size=8,
    num_hidden_layers=2,
    num_attention_heads=2,
    head_width=8,
)
# 30 + 37 = 67 tokens: the last rank's rows are ragged at P = 2 (34 / 33).
CHAINS = (_protein_of_length(30), _protein_of_length(37))
MSA_DEPTH = 6
FORWARD = dict(num_loops=1, num_sampling_steps=2, num_diffusion_samples=2)
SEED = 1234
COMPARED = ("sample_atom_coords", "plddt", "pae", "pde", "ptm", "iptm")
# Pair-shaped outputs are assembled in rank 0's host memory only.
RANK0_ONLY = ("distogram_logits", "pae", "pde")


def _model(config: dict) -> EsmFold2Model:
    torch.manual_seed(0)
    model = EsmFold2Model(EsmFold2Config(**config)).eval()
    model.set_chunk_size(None)
    model.set_memory_mode("big")
    return model


def _inputs(model: EsmFold2Model) -> tuple[dict, torch.Tensor]:
    features, lm_hidden_states, _ = esmfold2_complex_features(model, CHAINS)
    length = features["res_type"].shape[-1]
    gen = torch.Generator().manual_seed(1)
    msa = torch.randint(2, _layers.NUM_RES_TYPES, (1, MSA_DEPTH, length), generator=gen)
    msa[0, 0] = features["res_type"][0]
    features.update(
        msa=msa,
        msa_attention_mask=torch.ones(1, MSA_DEPTH, length, dtype=torch.bool),
        has_deletion=torch.zeros(1, MSA_DEPTH, length, dtype=torch.bool),
        deletion_value=torch.zeros(1, MSA_DEPTH, length),
    )
    return features, lm_hidden_states


def _fold(model: EsmFold2Model, features: dict, lm_hidden_states: torch.Tensor) -> dict:
    torch.manual_seed(SEED)
    out = model(**features, lm_hidden_states=lm_hidden_states, **FORWARD)
    keys = (*COMPARED, "distogram_logits")
    return {k: out[k] for k in keys if k in out}


def _rank(rank: int, world: int, port: int, case: tuple, out_dir: str) -> None:
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    model = case[0]
    model.set_memory_mode("big", shard_rows=True)
    calls = [0]
    trimul_rows = rowpair.trimul_residual_rows_

    def counted(*args, **kwargs):
        calls[0] += 1
        return trimul_rows(*args, **kwargs)

    setattr(rowpair, "trimul_residual_rows_", counted)
    try:
        out = _fold(*case)
        out["trimul_row_calls"] = torch.tensor(calls[0])
        torch.save(out, os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def case(tiny_esmfold2_config) -> tuple:
    """Model, features and LM states, built once and inherited by forked ranks."""
    raw = tiny_esmfold2_config.to_dict()
    raw.pop("model_type", None)
    raw["msa_encoder"] = TINY_MSA_ENCODER
    model = _model(raw)
    return (model, *_inputs(model))


@pytest.mark.parametrize("route", ["whole", "rows"])
@pytest.mark.parametrize("world", [2])
def test_row_sharded_fold_matches_one_rank(case, world, route, tmp_path, monkeypatch):
    monkeypatch.setattr(_layers, "FLASH_ATTN_AVAILABLE", False)
    # Forked ranks inherit the sampler route from the environment.
    monkeypatch.setenv("ESMFOLD2_ROWPAIR_SAMPLER", route)
    ref = _fold(*case)

    # Fork, not spawn: ranks inherit the built model instead of re-importing.
    mp.start_processes(
        _rank,
        args=(world, _free_port(), case, str(tmp_path)),
        nprocs=world,
        join=True,
        start_method="fork",
    )
    ranks = [torch.load(tmp_path / f"rank{r}.pt") for r in range(world)]
    # The fold really was split: the row TriMul ran on every rank.
    assert all(int(got["trimul_row_calls"]) > 0 for got in ranks)

    assert all(key not in got for got in ranks[1:] for key in RANK0_ONLY)
    for key in (*COMPARED, "distogram_logits"):
        for r, got in enumerate(ranks[:1] if key in RANK0_ONLY else ranks):
            assert got[key].shape == ref[key].shape, (key, r)
            # Every rank returns the same result.
            torch.testing.assert_close(got[key], ranks[0][key], rtol=0, atol=0)
        # The TriMul ring and the row-blocked GEMMs only reorder fp32 sums.
        torch.testing.assert_close(
            ranks[0][key], ref[key], atol=2e-4, rtol=1e-4, msg=key
        )


# 600 x 600 x 384 elements: above the whole-draw cut, so the Philox replica runs.
REPLICA_N, REPLICA_C = 600, 384


@pytest.mark.gpu
@pytest.mark.parametrize("world", [2])
@pytest.mark.parametrize("draw", ["uniform", "dropout"])
def test_rows_of_a_draw_are_rows_of_the_whole_draw(world, draw):
    """Each rank's rows equal the single-device draw, and the generator ends level."""
    from opt_core.mem.rowpair.dist import Layout  # ty: ignore[unresolved-import]

    device = torch.device("cuda")
    N, C = REPLICA_N, REPLICA_C
    assert N * N * C > rowpair._WHOLE_DRAW_MAX_ELEMS
    x = torch.randn(1, N, N, C, device=device)

    def whole() -> torch.Tensor:
        if draw == "uniform":
            return torch.empty(1, N, N, C, device=device).uniform_(-0.9, 0.9)
        return torch.nn.functional.dropout(x, p=0.25, training=True)

    torch.manual_seed(7)
    ref = whole()
    ref_next = torch.rand(4, device=device)
    for rank in range(world):
        layout = Layout.auto(N, world, rank)
        torch.manual_seed(7)
        if draw == "uniform":
            got = rowpair.uniform_rows(layout, C, -0.9, 0.9, device)
        else:
            got = rowpair.dropout_rows(
                lambda g0, g1: x[:, g0:g1], layout, 0.25, C, torch.float32, device
            )
        assert torch.equal(got, ref[:, layout.r0 : layout.r1]), rank
        assert torch.equal(torch.rand(4, device=device), ref_next), rank


# ---------------------------------------------------------------------------
# Real checkpoint on two GPUs (manual: CI has no multi-GPU runners)
# ---------------------------------------------------------------------------

# P=2 vs P=1 differs by 0.4-1.3 A on this input; two seeds differ by ~25 A.
COORDS_RMSD_BOUND = 3.0

# The torchrun entry point that folds the real checkpoint once.
FOLD_SCRIPT = os.path.join(os.path.dirname(__file__), "esmfold2_rowpair_fold.py")


def _kabsch_rmsd(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.double() - a.double().mean(0)
    b = b.double() - b.double().mean(0)
    u, _, vt = torch.linalg.svd(a.T @ b)
    d = torch.sign(torch.linalg.det(u @ vt))
    r = u @ torch.diag(torch.tensor([1.0, 1.0, float(d)], dtype=a.dtype)) @ vt
    return float(((a @ r - b) ** 2).sum(-1).mean().sqrt())


def test_two_gpus_match_one_gpu_on_the_real_checkpoint(tmp_path):
    """P=2 vs P=1 on the real checkpoint: coordinates within a few angstrom."""
    import subprocess
    import sys

    def run(world: int) -> list[torch.Tensor]:
        out = tmp_path / f"p{world}.pt"
        cmd = [sys.executable, "-m", "torch.distributed.run",
               f"--nproc-per-node={world}", FOLD_SCRIPT, str(out)]  # fmt: skip
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        subprocess.run(cmd, check=True, env=env)
        return torch.load(out)

    one, two = run(1), run(2)
    rmsd = statistics.mean(map(_kabsch_rmsd, one, two))
    print(f"\nP=2 vs P=1 coords RMSD, mean over samples: {rmsd:.2f} A")
    assert rmsd <= COORDS_RMSD_BOUND, rmsd
