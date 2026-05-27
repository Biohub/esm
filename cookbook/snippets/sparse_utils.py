from typing import Iterable

import torch


def remove_indexes(
    sparse_coo_tensor: torch.Tensor, indexes_to_remove: Iterable[int]
) -> torch.Tensor:
    """Remove entries at specified position indexes from sparse features.

    This function removes positions and remaps the remaining indices to be contiguous.
    For example, if we remove position 1 from a tensor with positions [0, 1, 2, 3],
    the result will have positions [0, 1, 2] (where old position 2 becomes new position 1).

    For example, remove_indexes(x, [0, -1]) will return the equivalent of tensor.to_dense().numpy()[1:-1]

    Args:
        sparse_coo_tensor: A sparse COO tensor of shape (num_positions, num_features)
        indexes_to_remove: Iterable of position indexes to remove (supports negative indexing)

    Returns:
        A new sparse COO tensor with the specified positions removed and indices remapped
    """
    if not sparse_coo_tensor.is_sparse or sparse_coo_tensor.layout != torch.sparse_coo:
        raise TypeError("sparse_coo_tensor must be a torch.sparse_coo_tensor.")

    if sparse_coo_tensor.dim() != 2:
        raise ValueError(
            f"sparse tensors with more than 2 dimensions are not supported, got {sparse_coo_tensor.dim()} dimensions"
        )

    indices = sparse_coo_tensor.indices()
    values = sparse_coo_tensor.values()
    num_positions = sparse_coo_tensor.size(0)
    num_features = sparse_coo_tensor.size(1)

    # Convert negative indices to positive and create sorted list
    indexes_to_remove_list = []
    for idx in indexes_to_remove:
        if idx < 0:
            idx = num_positions + idx
        indexes_to_remove_list.append(idx)
    indexes_to_remove_set = set(indexes_to_remove_list)

    if max(indexes_to_remove_set) > num_positions - 1:
        raise ValueError(
            f"Index to remove {max(indexes_to_remove_set)} is out of bounds for tensor with size {num_positions}"
        )

    position_indices = indices[0]
    mask = ~torch.isin(
        position_indices,
        torch.tensor(list(indexes_to_remove_set), device=position_indices.device),
    )
    filtered_indices = indices[:, mask]
    new_values = values[mask]

    # Create mapping from old positions to new positions
    # new position = old position - count(removed positions < old position)
    old_positions = filtered_indices[0]
    sorted_removed = sorted(indexes_to_remove_set)
    position_mapping = torch.zeros(
        num_positions, dtype=torch.long, device=position_indices.device
    )
    removed_count = 0
    removed_idx = 0

    for pos in range(num_positions):
        while removed_idx < len(sorted_removed) and sorted_removed[removed_idx] < pos:
            removed_count += 1
            removed_idx += 1
        position_mapping[pos] = pos - removed_count

    # Apply mapping to position indices
    new_position_indices = position_mapping[old_positions]

    # Construct new indices with remapped positions
    new_indices = torch.stack([new_position_indices, filtered_indices[1]], dim=0)

    new_num_positions = num_positions - len(indexes_to_remove_set)
    return torch.sparse_coo_tensor(
        new_indices, new_values, size=(new_num_positions, num_features)
    ).coalesce()


def max_pool(sparse_coo_tensor: torch.Tensor, axis: int) -> torch.Tensor:
    """Max pool sparse features along the specified axis.

    Args:
        sparse_coo_tensor: A sparse COO tensor of shape (num_positions, num_features)
        axis: The axis to pool over (0 for positions, 1 for features)

    Returns:
        Max-pooled tensor.
    """
    if not sparse_coo_tensor.is_sparse or sparse_coo_tensor.layout != torch.sparse_coo:
        raise TypeError("sparse_coo_tensor must be a torch.sparse_coo_tensor.")

    if sparse_coo_tensor.dim() != 2:
        raise ValueError(
            f"sparse tensors with more than 2 dimensions are not supported, got {sparse_coo_tensor.dim()} dimensions"
        )

    if axis not in (0, 1):
        raise ValueError(f"axis must be 0 or 1, got {axis}")

    indices = sparse_coo_tensor.indices()
    values = sparse_coo_tensor.values()

    if axis == 0:
        # Pool over positions (axis 0), return max per feature
        output_size = sparse_coo_tensor.size(1)
        scatter_indices = indices[1]  # feature indices
    else:  # axis == 1
        # Pool over features (axis 1), return max per position
        output_size = sparse_coo_tensor.size(0)
        scatter_indices = indices[0]  # position indices

    result = torch.zeros(output_size, dtype=values.dtype, device=values.device)
    result.scatter_reduce_(
        0, scatter_indices, values, reduce="amax", include_self=False
    )
    return result
