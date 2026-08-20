import os

from cookbook.snippets.sae import get_sae_features, get_sae_features_single
from cookbook.snippets.sparse_utils import remove_indexes
from esm.sdk.api import SAEConfig
from esm.sdk.forge import ESMCForgeInferenceClient

# Create ESMC 600M client
client = ESMCForgeInferenceClient(
    model="esmc-600m-2024-12", url="https://biohub.ai", token=os.environ["ESM_API_KEY"]
)

# normalize feature activations by TF-IDF. Upweights activations
# of more highly specific features
sae_config = SAEConfig(
    models=["esmc-600m-2024-12_k64_codebook16384_layer27"], normalize_features=True
)

# Create a protein
sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQATHVDQWDWEWAGIKATEAFLPDYPDLDA"
sequences = [sequence] * 10

# get unpooled features for a single sequence
unpooled_features = get_sae_features_single(client, sae_config, sequence, pool=False)
print(f"Got unpooled SAE features with shape {unpooled_features.shape}")
print(f"is_sparse: {unpooled_features.is_sparse}")
print(f"layout: {unpooled_features.layout}")

# To remove bos/eos tokens efficiently from sparse tensors, we use a utility
unpooled_features = remove_indexes(unpooled_features, {0, -1})
print(
    f"Unpooled SAE features after removing BOS/EOS have shape {unpooled_features.shape}"
)

# get pooled features for a batch
# this function pools by default to save memory.
features = get_sae_features(client, sae_config, sequences)
print(
    f"Got SAE features for {len(features)} sequences, each with shape {features[0].shape}"
)
