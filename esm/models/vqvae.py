import torch
import torch.nn as nn

from esm.layers.blocks import UnifiedTransformerBlock
from esm.layers.codebook import EMACodebook
from esm.layers.structure_proj import Dim6RotStructureHead
from esm.layers.transformer_stack import TransformerStack
from esm.utils.constants import esm3 as C
from esm.utils.misc import knn_graph
from esm.utils.structure.affine3d import Affine3D, build_affine3d_from_coordinates
from esm.utils.structure.predicted_aligned_error import (
    compute_predicted_aligned_error,
    compute_tm,
)


class RelativePositionEmbedding(nn.Module):
    """
    Embedding layer for relative position embeddings. `bins` is the number of positions relative
    to the query position that are considered before clipping. For instance, if `bins=10`, then
    the relative position embedding will have 21 positions, [-10, 10].
    """

    def __init__(self, bins, embedding_dim, init_std=0.02):
        super().__init__()
        self.bins = bins

        self.embedding = torch.nn.Embedding(2 * bins + 2, embedding_dim)
        self.embedding.weight.data.normal_(0, init_std)

    def forward(self, query_residue_index, key_residue_index):
        """
        Input:
          query_residue_index: (B, ) tensor of source indices (dytpe=torch.long)
          key_residue_index: (B, L) tensor of target indices (dytpe=torch.long)
        Output:
          embeddings: B x L x embedding_dim tensor of embeddings
        """

        assert query_residue_index.dtype == torch.long
        assert key_residue_index.dtype == torch.long
        assert query_residue_index.ndim == 1
        assert key_residue_index.ndim == 2

        diff = key_residue_index - query_residue_index.unsqueeze(1)
        diff = diff.clamp(-self.bins, self.bins)
        diff = diff + self.bins + 1  # add 1 to adjust for padding index
        output = self.embedding(diff)
        return output


class PairwisePredictionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        downproject_dim: int,
        hidden_dim: int,
        n_bins: int,
        bias: bool = True,
        pairwise_state_dim: int = 0,
    ):
        super().__init__()
        self.downproject = nn.Linear(input_dim, downproject_dim, bias=bias)
        self.linear1 = nn.Linear(
            downproject_dim + pairwise_state_dim, hidden_dim, bias=bias
        )
        self.activation_fn = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_bins, bias=bias)

    def forward(self, x, pairwise: torch.Tensor | None = None):
        """
        Args:
            x: [B x L x D]

        Output:
            [B x L x L x K]
        """
        x = self.downproject(x)
        # Let x_i be a vector of size (B, D).
        # Input is {x_1, ..., x_L} of size (B, L, D)
        # Output is 2D where x_ij = cat([x_i * x_j, x_i - x_j])
        q, k = x.chunk(2, dim=-1)

        prod = q[:, None, :, :] * k[:, :, None, :]
        diff = q[:, None, :, :] - k[:, :, None, :]
        x_2d = [prod, diff]
        if pairwise is not None:
            x_2d.append(pairwise)
        x = torch.cat(x_2d, dim=-1)
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.norm(x)
        x = self.linear2(x)
        return x


class RegressionHead(nn.Module):
    def __init__(self, embed_dim: int, output_dim: int):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, output_dim)

    def forward(self, features):
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.norm(x)
        x = self.output(x)
        return x


class CategoricalMixture:
    def __init__(self, param, bins=50, start=0, end=1):
        # All tensors are of shape ..., bins.
        self.logits = param
        bins = torch.linspace(
            start, end, bins + 1, device=self.logits.device, dtype=torch.float32
        )
        self.v_bins = (bins[:-1] + bins[1:]) / 2

    def log_prob(self, true):
        # Shapes are:
        #     self.probs: ... x bins
        #     true      : ... (floating point # for target)
        true_index = (
            (true.unsqueeze(-1) - self.v_bins[[None] * true.ndim]).abs().argmin(-1)
        )
        nll = self.logits.log_softmax(-1)
        return torch.take_along_dim(nll, true_index.unsqueeze(-1), dim=-1).squeeze(-1)

    def mean(self):
        return (
            self.logits.to(self.v_bins.dtype).softmax(-1) @ self.v_bins.unsqueeze(1)
        ).squeeze(-1)

    def median(self):
        return self.v_bins[self.logits.max(-1).indices]


class GeometricEncoderStack(TransformerStack):
    def __init__(self, d_model, n_heads, v_heads, n_layers):
        super().__init__(d_model, n_heads, v_heads, 0)
        self.blocks = nn.ModuleList(
            [
                UnifiedTransformerBlock(
                    d_model,
                    n_heads,
                    v_heads=v_heads,
                    use_geom_attn=True,
                    use_plain_attn=False,
                    expansion_ratio=4,
                    bias=True,
                )
                for i in range(n_layers)
            ]
        )
        self.norm = nn.Identity()


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


def node_gather(s: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    return batched_gather(s.unsqueeze(-3), edges, -2, no_batch_dims=len(s.shape) - 1)


class StructureTokenEncoder(nn.Module):
    def __init__(self, d_model, n_heads, v_heads, n_layers, d_out, n_codes):
        super().__init__()
        # We only support fully-geometric structure token encoders for now...
        # setting n_layers_geom to something that's not n_layers won't work because
        # sequence ID isn't supported fully in this repo for plain-old transformers
        self.transformer = GeometricEncoderStack(d_model, n_heads, v_heads, n_layers)
        self.pre_vq_proj = nn.Linear(d_model, d_out)
        self.codebook = EMACodebook(n_codes, d_out)
        self.relative_positional_embedding = RelativePositionEmbedding(
            32, d_model, init_std=0.02
        )
        self.knn = 16

    def encode_local_structure(
        self,
        coords: torch.Tensor,
        affine: Affine3D,
        attention_mask: torch.Tensor,
        sequence_id: torch.Tensor | None,
        affine_mask: torch.Tensor,
        residue_index: torch.Tensor | None = None,
    ):
        """This function allows for a multi-layered encoder to encode tokens with a local receptive fields. The implementation is as follows:

        1. Starting with (B, L) frames, we find the KNN in structure space. This now gives us (B, L, K) where the last dimension is the local
        neighborhood of all (B, L) residues.
        2. We reshape these frames to (B*L, K) so now we have a large batch of a bunch of local neighborhoods.
        3. Pass the (B*L, K) local neighborhoods through a stack of geometric reasoning blocks, effectively getting all to all communication between
        all frames in the local neighborhood.
        4. This gives (B*L, K, d_model) embeddings, from which we need to get a single embedding per local neighborhood. We do this by simply
        taking the embedding corresponding to the query node. This gives us (B*L, d_model) embeddings.
        5. Reshape back to (B, L, d_model) embeddings
        """
        assert coords.size(-1) == 3 and coords.size(-2) == 3, "need N, CA, C"
        with torch.no_grad():
            knn_edges, _ = self.find_knn_edges(
                coords,
                ~attention_mask,
                coord_mask=affine_mask,
                sequence_id=sequence_id,
                knn=self.knn,
            )
            B, L, E = knn_edges.shape

            affine_tensor = affine.tensor  # for easier manipulation
            T_D = affine_tensor.size(-1)
            knn_affine_tensor = node_gather(affine_tensor, knn_edges)
            knn_affine_tensor = knn_affine_tensor.view(-1, E, T_D).contiguous()
            affine = Affine3D.from_tensor(knn_affine_tensor)
            knn_sequence_id = (
                node_gather(sequence_id.unsqueeze(-1), knn_edges).view(-1, E)
                if sequence_id is not None
                else torch.zeros(B * L, E, dtype=torch.int64, device=coords.device)
            )
            knn_affine_mask = node_gather(affine_mask.unsqueeze(-1), knn_edges).view(
                -1, E
            )
            knn_chain_id = torch.zeros(
                B * L, E, dtype=torch.int64, device=coords.device
            )

            if residue_index is None:
                res_idxs = knn_edges.view(-1, E)
            else:
                res_idxs = node_gather(residue_index.unsqueeze(-1), knn_edges).view(
                    -1, E
                )

        z = self.relative_positional_embedding(res_idxs[:, 0], res_idxs)

        z, _, _, _ = self.transformer.forward(
            x=z,
            sequence_id=knn_sequence_id,
            affine=affine,
            affine_mask=knn_affine_mask,
            chain_id=knn_chain_id,
        )

        # Unflatten the output and take the query node embedding, which will always be the first one because
        # a node has distance 0 with itself and the KNN are sorted.
        z = z.view(B, L, E, -1)
        z = z[:, :, 0, :]

        return z

    @staticmethod
    def find_knn_edges(
        coords,
        padding_mask,
        coord_mask,
        sequence_id: torch.Tensor | None = None,
        knn: int | None = None,
    ) -> tuple:
        assert knn is not None, "Must specify a non-null knn to find_knn_edges"
        # Coords are N, CA, C
        coords = coords.clone()
        coords[~coord_mask] = 0

        if sequence_id is None:
            sequence_id = torch.zeros(
                (coords.shape[0], coords.shape[1]), device=coords.device
            ).long()

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):  # type: ignore
            ca = coords[..., 1, :]
            edges, edge_mask = knn_graph(
                ca, coord_mask, padding_mask, sequence_id, no_knn=knn
            )

        return edges, edge_mask

    def encode(
        self,
        coords: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        residue_index: torch.Tensor | None = None,
    ):
        coords = coords[..., :3, :]
        affine, affine_mask = build_affine3d_from_coordinates(coords=coords)

        if attention_mask is None:
            attention_mask = torch.ones_like(affine_mask, dtype=torch.bool)
        attention_mask = attention_mask.bool()

        if sequence_id is None:
            sequence_id = torch.zeros_like(affine_mask, dtype=torch.int64)

        z = self.encode_local_structure(
            coords=coords,
            affine=affine,
            attention_mask=attention_mask,
            sequence_id=sequence_id,
            affine_mask=affine_mask,
            residue_index=residue_index,
        )

        z = z.masked_fill(~affine_mask.unsqueeze(2), 0)
        z = self.pre_vq_proj(z)

        z_q, min_encoding_indices, _ = self.codebook(z)

        return z_q, min_encoding_indices


class StructureTokenDecoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        super().__init__()
        self.decoder_channels = d_model

        self.vqvae_codebook_size = C.VQVAE_CODEBOOK_SIZE
        self.special_tokens = C.VQVAE_SPECIAL_TOKENS
        self.max_pae_bin = C.VQVAE_MAX_PAE_BIN

        self.embed = nn.Embedding(
            self.vqvae_codebook_size + len(self.special_tokens), d_model
        )
        self.decoder_stack = TransformerStack(
            d_model, n_heads, 1, n_layers, scale_residue=False, n_layers_geom=0
        )

        self.affine_output_projection = Dim6RotStructureHead(
            self.decoder_channels, 10, predict_torsion_angles=False
        )

        direction_loss_bins = C.VQVAE_DIRECTION_LOSS_BINS
        pae_bins = C.VQVAE_PAE_BINS
        self.pairwise_bins = [
            64,  # distogram
            direction_loss_bins * 6,  # direction bins
            pae_bins,  # predicted aligned error
        ]
        self.pairwise_classification_head = PairwisePredictionHead(
            self.decoder_channels,
            downproject_dim=128,
            hidden_dim=128,
            n_bins=sum(self.pairwise_bins),
            bias=False,
        )

        plddt_bins = C.VQVAE_PLDDT_BINS
        self.plddt_head = RegressionHead(
            embed_dim=self.decoder_channels, output_dim=plddt_bins
        )

    def decode(
        self,
        structure_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(structure_tokens, dtype=torch.bool)

        attention_mask = attention_mask.bool()
        if sequence_id is None:
            sequence_id = torch.zeros_like(structure_tokens, dtype=torch.int64)
        # not supported for now
        chain_id = torch.zeros_like(structure_tokens, dtype=torch.int64)

        # check that BOS and EOS are set correctly
        assert (
            structure_tokens[:, 0].eq(self.special_tokens["BOS"]).all()
        ), "First token in structure_tokens must be BOS token"
        assert (
            structure_tokens[
                torch.arange(structure_tokens.shape[0]), attention_mask.sum(1) - 1
            ]
            .eq(self.special_tokens["EOS"])
            .all()
        ), "Last token in structure_tokens must be EOS token"
        assert (
            (structure_tokens < 0).sum() == 0
        ), "All structure tokens set to -1 should be replaced with BOS, EOS, PAD, or MASK tokens by now, but that isn't the case!"

        x = self.embed(structure_tokens)
        # !!! NOTE: Attention mask is actually unused here so watch out
        x, _, _, _ = self.decoder_stack.forward(
            x, affine=None, affine_mask=None, sequence_id=sequence_id, chain_id=chain_id
        )

        tensor7_affine, bb_pred = self.affine_output_projection(
            x, affine=None, affine_mask=torch.zeros_like(attention_mask)
        )

        pae, ptm = None, None
        pairwise_logits = self.pairwise_classification_head(x)
        _, _, pae_logits = [
            (o if o.numel() > 0 else None)
            for o in pairwise_logits.split(self.pairwise_bins, dim=-1)
        ]

        special_tokens_mask = structure_tokens >= min(self.special_tokens.values())
        pae = compute_predicted_aligned_error(
            pae_logits,  # type: ignore
            aa_mask=~special_tokens_mask,
            sequence_id=sequence_id,
            max_bin=self.max_pae_bin,
        )
        # This might be broken for chainbreak tokens? We might align to the chainbreak
        ptm = compute_tm(
            pae_logits,  # type: ignore
            aa_mask=~special_tokens_mask,
            max_bin=self.max_pae_bin,
        )

        plddt_logits = self.plddt_head(x)
        plddt_value = CategoricalMixture(
            plddt_logits, bins=plddt_logits.shape[-1]
        ).mean()

        return dict(
            tensor7_affine=tensor7_affine,
            bb_pred=bb_pred,
            plddt=plddt_value,
            ptm=ptm,
            predicted_aligned_error=pae,
        )
