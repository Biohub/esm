import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.amp import autocast

from esm.utils import residue_constants
from esm.utils.misc import binpack, unbinpack
from esm.utils.structure.protein_structure import (
    compute_alignment_tensors,
    compute_gdt_ts_no_alignment,
    compute_rmsd_no_alignment,
)


def contact_precision(
    predictions: Tensor,
    targets: Tensor,
    src_lengths: Tensor | None = None,
    minsep: int = 6,
    maxsep: int | None = None,
    override_length: int | None = None,  # for casp
):
    """Computes contact precisions.

    For protein contact prediction, precision is measured for the top (L/K) highest confidence predictions,
    with L being the length of the protein sequence and K generally being equal to 1 or 5.

    K = 5 measures the predictions of the very highest confidence contacts, while K = 1 is a more general measure
    over all relatively high confidence predictions.

    Since there are roughly ~L true contacts in a protein, this is a reasonable cutoff.


    Args:
        predictions (Tensor): Tensor of probabilities of size (B, L, L)
        targets (Tensor): Tensor of true contacts of size (B, L, L)
        src_lengths (Tensor, optional): Lengths of each sample in the batch, if using variable lengths.
            If not provided, inferred from the size of the predictions.
        minsep (int): Minimum separation distance to consider. We often want to measure contacts at a
            certain range. Typical ranges are short [6, 12), medium [12, 24), and long [24, inf).
        maxsep (int, optional): Used in conjunction with minsep to specify a contact range. If not provided uses
            assumes no maximum range
        override_length (int, optional): Used for casp evaluation where sometimes the "true" length is not
            the same as the length of the input. Kept for posterity, we probably don't need this argument.
    """
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)

    # Check sizes
    if predictions.size() != targets.size():
        raise ValueError(
            f"Size mismatch. Received predictions of size {predictions.size()}, "
            f"targets of size {targets.size()}"
        )
    device = predictions.device

    batch_size, seqlen, _ = predictions.size()

    # Step 1) Construct a mask of size [B, L, L] to mask invalid contacts
    seqlen_range = torch.arange(seqlen, device=device)
    sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
    sep = sep.unsqueeze(0)
    # Mask contacts that are closer than minsep
    valid_mask = sep >= minsep
    # Mask contacts where target is negative (padding or unknown)
    valid_mask = valid_mask & (targets >= 0)  # negative targets are invalid

    # Mask contacts that are farther than maxsep, if provided
    if maxsep is not None:
        valid_mask &= sep < maxsep

    if src_lengths is not None:
        # If the lengths of the individual sequences are provided, mask positions
        # that are farther than the end of the sequence.
        valid = seqlen_range.unsqueeze(0) < src_lengths.unsqueeze(1)
        valid_mask &= valid.unsqueeze(1) & valid.unsqueeze(2)
    else:
        src_lengths = torch.full([batch_size], seqlen, device=device, dtype=torch.long)

    # Fill in the logit tensor with -inf for all invalid positions
    predictions = predictions.masked_fill(~valid_mask, float("-inf"))

    # Step 2) Select the top half of the prediction (should be symmetric)
    x_ind, y_ind = np.triu_indices(seqlen, minsep)
    predictions_upper = predictions[:, x_ind, y_ind]
    targets_upper = targets[:, x_ind, y_ind]

    # Step 3) Select the topk values in each batch where k = L (length of sequence)
    topk = seqlen if override_length is None else max(seqlen, override_length)
    # Indices are the indices into the predictions corresponding to the most confident predictions
    indices = predictions_upper.argsort(dim=-1, descending=True)[:, :topk]
    # topk_targets are the target values corresponding to the above indices
    topk_targets = targets_upper[torch.arange(batch_size).unsqueeze(1), indices]
    if topk_targets.size(1) < topk:
        # If there aren't enough targets, pad to the output.
        topk_targets = F.pad(topk_targets, [0, topk - topk_targets.size(1)])

    # Step 4) Sum the accuracy at of the top-i predictions for i in 1, L
    # topk_targets => 1/0 true vs. false contact, sorted by confidence of prediction
    # cmumulative sum => Number of correct answers for the top-i predictions.
    cumulative_dist = topk_targets.type_as(predictions).cumsum(-1)

    # Step 5) Find the gather indices. This should be P@(L / K) for varous values of K
    # The values will differ for each batch.
    gather_lengths = src_lengths.unsqueeze(1)
    if override_length is not None:
        gather_lengths = override_length * torch.ones_like(
            gather_lengths, device=device
        )

    # This gets you (0.1 * L, 0.2 * L, 0.3 * L, etc.)
    gather_indices = (
        (torch.arange(0.1, 1.1, 0.1, device=device).unsqueeze(0) * gather_lengths).type(
            torch.long
        )
        - 1
    ).clamp_min(0)

    # Step 6) Gather the results and divide by the number of guesses to get the precision.
    binned_cumulative_dist = cumulative_dist.gather(1, gather_indices)
    binned_precisions = binned_cumulative_dist / (gather_indices + 1).type_as(
        binned_cumulative_dist
    )

    # Select specific P@L/k. pl5 is index 1 b/c that corresponds to L * 0.2 in
    # gather_indices above
    pl5 = binned_precisions[:, 1]
    # pl2 = binned_precisions[:, 4]
    pl = binned_precisions[:, 9]
    # AUC is the integral wrt K of P@L/K for K in range(1, L)
    auc = binned_precisions.mean(-1)

    return {"AUC": auc, "P@L": pl, "P@L5": pl5}


def compute_lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    pairwise_all_atom_mask: torch.Tensor | None = None,
    cutoff: float | torch.Tensor = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
    sequence_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Computes LDDT for a protein. Tensor sizes below include some optional dimensions. Specifically:
        Nstates:
            all_atom_pred_pos can contain multiple states in the first dimension which corresponds to outputs from different layers of a model (e.g. each IPA block). The return size will be [Nstates x Batch size] if this is included.
        Natoms:
            LDDT can be computed for all atoms or some atoms. The second to last dimension should contain the *FLATTENED* representation of L x Natoms. If you want to calculate for atom37, e.g., this will be of size (L * 37). If you are only calculating CA LDDT, it will be of size L.

    Args:
        all_atom_pred_pos (Tensor[float], [(Nstates x) B x (L * Natoms x) 3]): Tensor of predicted positions
        all_atom_positions (Tensor[float], [B x (L * Natoms x) 3]): Tensor of true positions
        all_atom_mask (Tensor[float], [B x (L * Natoms)]): Tensor of masks, indicating whether an atom exists.
        pairwise_all_atom_mask (Tensor[float], [B x (L * Natoms x L * Natoms)], optional): Tensor of masks, indicating whether a pair of atoms should be considered in the LDDT calculation.
        cutoff (float): Max distance to score lddt over. This can either be a float, or a tensor of shape [B, L, L] to allow for per-residue cutoffs, e.g. if you want to use a different cutoff for nucleic acids.
        per_residue (bool): Whether to return per-residue or full-protein lddt.
        sequence_id (Tensor, optional): Sequence id tensor for binpacking. NOTE: only supported for lddt_ca calculations, not when Natoms is passed!

    Returns:
        LDDT Tensor:
            if per_residue:
                Tensor[float], [(Nstates x) B x (L * Natoms)]
            else:
                Tensor[float], [(Nstates x) B]
    """
    all_atom_mask = all_atom_mask[..., None]  # add a dimension for broadcasting
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_positions[..., None, :] - all_atom_positions[..., None, :, :])
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (all_atom_pred_pos[..., None, :] - all_atom_pred_pos[..., None, :, :]) ** 2,
            dim=-1,
        )
    )
    mask = all_atom_mask * rearrange(all_atom_mask, "... a b -> ... b a")
    if pairwise_all_atom_mask is not None:
        mask = mask * pairwise_all_atom_mask

    if sequence_id is not None:
        # TODO: This will work for lddt_ca, but not for regular lddt
        # Problem is that regular lddt has natoms * nres scores, so would need to repeat this mask by natoms
        # Leaving for now because it won't fail silently so should be ook.
        seqid_mask = sequence_id[..., None] == sequence_id[..., None, :]
        mask = mask * seqid_mask.type_as(mask)

    return compute_lddt_from_dmat(
        dmat_pred, dmat_true, mask, cutoff=cutoff, eps=eps, per_residue=per_residue
    )


def compute_lddt_from_dmat(
    dmat_pred: torch.Tensor,
    dmat_true: torch.Tensor,
    pairwise_mask: torch.Tensor,
    cutoff: float | torch.Tensor = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
):
    """
    Compute LDDT from pre-computed distance matrices.
    This is useful when you want to compute LDDT with multiple different masks or cutoffs, e.g. for different molecule types (protein, nucleic acid, etc.).

    Args:
        dmat_pred (Tensor[float], [B x L x L]): Predicted distance matrix
        dmat_true (Tensor[float], [B x L x L]): True distance matrix
        pairwise_mask (Tensor[float], [B x L x L]): Pairwise mask indicating which pairs of atoms to consider
        cutoff (float): Max distance to score lddt over. This can either be a float, or a tensor of shape [B, L, L] to allow for per-residue cutoffs, e.g. if you want to use a different cutoff for nucleic acids.
        per_residue (bool): Whether to return per-residue or full-protein lddt.

    Returns:
        LDDT Tensor:
            if per_residue:
                Tensor[float], [B x L]
            else:
                Tensor[float], [B]
    """
    n = dmat_true.size(-1)
    dists_to_score = (
        (dmat_true < cutoff)
        * pairwise_mask
        * (1.0 - torch.eye(n, device=dmat_true.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)
    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))
    return score


def compute_lddt_ca(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
    sequence_id: torch.Tensor | None = None,
) -> torch.Tensor:
    ca_pos = residue_constants.atom_order["CA"]
    if all_atom_pred_pos.dim() != 3:
        all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos]

    return compute_lddt(
        all_atom_pred_pos,
        all_atom_positions,
        all_atom_mask,
        cutoff=cutoff,
        eps=eps,
        per_residue=per_residue,
        sequence_id=sequence_id,
    )


# NOTE(roshan): no_grad required for stack_variable_length_tensors apparently... let's revisit if we want to backprop
@torch.no_grad()
@autocast("cuda", enabled=False)
def compute_rmsd(
    mobile: torch.Tensor,
    target: torch.Tensor,
    atom_exists_mask: torch.Tensor | None = None,
    sequence_id: torch.Tensor | None = None,
    reduction: str = "batch",
):
    """
    Compute RMSD between two batches of structures with support for masking invalid atoms using PyTorch.

    Args:
    - mobile (torch.Tensor): Batch of coordinates of structure to be superimposed in shape (B, N, 3)
    - target (torch.Tensor): Batch of coordinates of structure that is fixed in shape (B, N, 3)
    - atom_exists_mask (torch.Tensor, optional): Mask for Whether an atom exists of shape (B, N)
    - sequence_id (torch.Tensor, optional): Sequence id tensor for binpacking.
    - reduction (str): One of "batch", "per_sample", "per_residue".

    Returns:
    If reduction == "batch":
        (torch.Tensor): 0-dim, Average Root Mean Square Deviation between the structures for each batch
    If reduction == "per_sample":
        (torch.Tensor): (B,)-dim, Root Mean Square Deviation between the structures for each batch
    If reduction == "per_residue":
        (torch.Tensor): (B, N)-dim, Root Mean Square Deviation between the structures for residue in the batch
    """

    (centered_mobile, _, centered_target, _, rotation_matrix, num_valid_atoms) = (
        compute_alignment_tensors(
            mobile=mobile,
            target=target,
            atom_exists_mask=atom_exists_mask,
            sequence_id=sequence_id,
        )
    )

    # Apply transformation to centered structure
    rotated_mobile = torch.matmul(centered_mobile, rotation_matrix)

    # Compute rmsd for centered structures
    rmsd = compute_rmsd_no_alignment(
        rotated_mobile, centered_target, num_valid_atoms, reduction=reduction
    )
    if reduction == "per_residue" and sequence_id is not None:
        rmsd = binpack(rmsd, sequence_id, pad_value=0)
    return rmsd


def compute_gdt_ts(
    mobile: torch.Tensor,
    target: torch.Tensor,
    atom_exists_mask: torch.Tensor | None = None,
    sequence_id: torch.Tensor | None = None,
    reduction: str = "per_sample",
):
    """
    Compute GDT_TS between two batches of structures with support for masking invalid atoms using PyTorch.

    Args:
    - mobile (torch.Tensor): Batch of coordinates of structure to be superimposed in shape (B, N, 3)
    - target (torch.Tensor): Batch of coordinates of structure that is fixed in shape (B, N, 3)
    - atom_exists_mask (torch.Tensor, optional): Mask for Whether an atom exists of shape (B, N)
    - sequence_id (torch.Tensor, optional): Sequence id tensor for binpacking.
    - reduction (str): One of "batch", "per_sample", "per_residue".

    Returns:
    If reduction == "batch":
        (torch.Tensor): 0-dim, GDT_TS between the structures for each batch
    If reduction == "per_sample":
        (torch.Tensor): (B,)-dim, GDT_TS between the structures for each sample in the batch
    """
    if atom_exists_mask is None:
        atom_exists_mask = torch.isfinite(target).all(dim=-1)
    (centered_mobile, _, centered_target, _, rotation_matrix, _) = (
        compute_alignment_tensors(
            mobile=mobile,
            target=target,
            atom_exists_mask=atom_exists_mask,
            sequence_id=sequence_id,
        )
    )

    # Apply transformation to centered structure
    rotated_mobile = torch.matmul(centered_mobile, rotation_matrix)

    # the coordinate tensors returned by `compute_alignment_tensors` are unbinpacked and contain zeros for invalid positions
    # so `compute_gdt_ts_no_alignment` requires `atom_exists_mask` to be passed and be unbinpacked
    if sequence_id is not None:
        atom_exists_mask = unbinpack(atom_exists_mask, sequence_id, pad_value=False)
    return compute_gdt_ts_no_alignment(
        rotated_mobile, centered_target, atom_exists_mask, reduction
    )
