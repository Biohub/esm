import torch

from esm.utils.structure.protein_structure import compute_alignment_tensors

MIRROR = torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)


def _random_rotations(batch_size: int) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(batch_size, 3, 3, dtype=torch.float64))
    q[:, :, 0] *= torch.linalg.det(q).sign()[:, None]
    return q


def test_alignment_recovers_a_rigid_transform() -> None:
    torch.manual_seed(0)
    target = torch.randn(8, 20, 3, dtype=torch.float64)
    rotations = _random_rotations(8)
    shift = torch.randn(8, 1, 3, dtype=torch.float64)
    mobile = (target - shift) @ rotations.transpose(1, 2)

    rotation = compute_alignment_tensors(mobile, target)[4]

    # The function returns the rotation in the row-vector convention: mobile @ R.
    torch.testing.assert_close(rotation, rotations)


def test_alignment_never_returns_a_reflection() -> None:
    torch.manual_seed(0)
    target = torch.randn(8, 20, 3, dtype=torch.float64)
    mobile = target @ _random_rotations(8).transpose(1, 2)
    # Mirror every other structure, so one batch mixes both cases.
    mobile[::2] = mobile[::2] * MIRROR

    rotation = compute_alignment_tensors(mobile, target)[4]

    torch.testing.assert_close(
        torch.linalg.det(rotation), torch.ones(8, dtype=torch.float64)
    )
    # The mirrored structures can no longer be superimposed exactly.
    centered_mobile, _, centered_target, *_ = compute_alignment_tensors(mobile, target)
    error = (centered_mobile @ rotation - centered_target).norm(dim=-1).mean(dim=-1)
    assert (error[::2] > 0.1).all()
    assert (error[1::2] < 1e-9).all()
