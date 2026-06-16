import torch
import torch.nn.functional as F

from esm.utils.structure.affine3d import Affine3D


def masked_mean(
    mask: torch.Tensor,
    value: torch.Tensor,
    dim: int | None | tuple[int, ...] = None,
    eps=1e-10,
) -> torch.Tensor:
    """Compute the mean of `value` where only positions where `mask == true` are
    counted.
    """
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))


def _pae_bins(
    max_bin: float = 31, num_bins: int = 64, device: torch.device = torch.device("cpu")
):
    bins = torch.linspace(0, max_bin, steps=(num_bins - 1), device=device)
    step = max_bin / (num_bins - 2)
    bin_centers = bins + step / 2
    bin_centers = torch.cat(
        [bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0
    )
    return bin_centers


def _compute_pae_masks(mask: torch.Tensor):
    square_mask = (mask.unsqueeze(-1) * mask.unsqueeze(-2)).bool()
    return square_mask


def compute_predicted_aligned_error(
    logits: torch.Tensor,
    aa_mask: torch.Tensor,
    sequence_id: torch.Tensor | None = None,
    max_bin: float = 31,
) -> torch.Tensor:
    bins = _pae_bins(max_bin, logits.shape[-1], logits.device)
    square_mask = _compute_pae_masks(aa_mask)
    min_v = torch.finfo(logits.dtype).min
    probs = logits.masked_fill(~square_mask.unsqueeze(-1), min_v).softmax(dim=-1)

    return (probs * bins).sum(dim=-1)


@torch.no_grad  # ty:ignore[too-many-positional-arguments]
def compute_tm(logits: torch.Tensor, aa_mask: torch.Tensor, max_bin: float = 31.0):
    square_mask = _compute_pae_masks(aa_mask)
    seqlens = aa_mask.sum(-1, keepdim=True)
    bins = _pae_bins(max_bin, logits.shape[-1], logits.device)
    d0 = 1.24 * (seqlens.clamp_min(19) - 15) ** (1 / 3) - 1.8
    f_d = 1.0 / (1 + (bins / d0.unsqueeze(-1)) ** 2)

    min_v = torch.finfo(logits.dtype).min
    probs = logits.masked_fill(~square_mask.unsqueeze(-1), min_v).softmax(dim=-1)
    # This is the sum over bins
    ptm = (probs * f_d.unsqueeze(-2)).sum(dim=-1)
    # This is the mean over residues j
    ptm = masked_mean(square_mask, ptm, dim=-1)
    # The we do a max over residues i
    return ptm.max(dim=-1).values


def tm_loss(
    logits: torch.Tensor,
    pred_affine: torch.Tensor,
    targ_affine: torch.Tensor,
    targ_mask: torch.Tensor,
    tm_mask: torch.Tensor | None = None,
    sequence_id: torch.Tensor | None = None,
    max_bin: float = 31,
):
    pred = Affine3D.from_tensor(pred_affine)
    targ = Affine3D.from_tensor(targ_affine)

    def transform(affine: Affine3D):
        pts = affine.trans[..., None, :, :]
        return affine.invert()[..., None].apply(pts)

    with torch.no_grad():
        sq_diff = (transform(pred) - transform(targ)).square().sum(dim=-1)

        num_bins = logits.shape[-1]
        sq_bins = torch.linspace(
            0, max_bin, num_bins - 1, device=logits.device
        ).square()
        # Gets the bin id by using a sum.
        true_bins = (sq_diff[..., None] > sq_bins).sum(dim=-1).long()

    errors = F.cross_entropy(logits.movedim(3, 1), true_bins, reduction="none")
    square_mask = _compute_pae_masks(targ_mask)
    loss = masked_mean(square_mask, errors, dim=(-1, -2))

    if tm_mask is not None:
        loss = masked_mean(tm_mask, loss, dim=None)
    else:
        loss = loss.mean()

    return loss
