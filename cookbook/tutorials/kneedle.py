from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from Bio import SeqIO
from torch import linalg as la
import torch
import numpy as np
from kneed import KneeLocator
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()
HUGGING_FACE_API_KEY = os.getenv('HUGGING_FACE_API_KEY')
login(token=HUGGING_FACE_API_KEY)

project_root = Path(__file__).parent.parent.parent
data_file = project_root / "data" / "kneedle_seqs.fasta"

# Initialize the ESM3InferenceClient with your API key
model: ESM3InferenceClient = ESM3.from_pretrained("esm3-open").to("cuda")

NUM_LAYERS = 48

def mask(sequence: str) -> str:
    """Sample from a uniform distribution of percentage of tokens to mask, between 0 and 100."""
    percentage_to_mask = np.random.uniform(0, 1)
    num_tokens_to_mask = int(len(sequence) * percentage_to_mask)
    indices_to_mask = np.random.choice(len(sequence), num_tokens_to_mask, replace=False)
    masked_sequence = list(sequence)
    for idx in indices_to_mask:
        masked_sequence[idx] = "_"
    return "".join(masked_sequence)

def read_fasta(file_path):
    """Reads a FASTA file and returns a list of sequences."""
    sequences = []
    with open(file_path, "r") as f:
        for record in SeqIO.parse(f, "fasta"):
            sequences.append(str(record.seq))
    return sequences

def angular_distance(a, b):
    cos = torch.inner(a, b) / (la.vector_norm(a) * la.vector_norm(b))
    cos = torch.clamp(cos, -1.0, 1.0)
    return torch.acos(cos) / torch.pi

overall_distances = []
sequences = read_fasta(data_file)
for seq in sequences:
    masked_seq = mask(seq)
    protein = ESMProtein(sequence=masked_seq)
    protein = model.generate(
        protein,
        GenerationConfig(track="sequence", num_steps=1, temperature=0.7)
    )
    hiddens = model.transformer.get_hiddens()
    last_token_per_layer = torch.stack([h[-1][-1] for h in hiddens], dim=0)
    distances = [angular_distance(last_token_per_layer[i], last_token_per_layer[i + 1]) for i in range(NUM_LAYERS - 1)]
    distances = torch.tensor(distances)
    overall_distances.append(distances.detach().cpu())

overall_distances = torch.stack(overall_distances, dim=0)
final_distances = overall_distances.mean(dim=0)
print(final_distances.tolist())

# Use KneeLocator to find the "knee" in the curve
x = np.arange(len(final_distances))
y = final_distances.float().numpy()
rev_y = np.flip(y)
kneedle_encoder = KneeLocator(x, y, curve="convex", direction="increasing", interp_method="polynomial", polynomial_degree=2, online=True)
kneedle_decoder = KneeLocator(x, rev_y, curve="convex", direction="increasing", interp_method="polynomial", polynomial_degree=2, online=True)
print(f"Knee point found at layer: {kneedle_encoder.knee}")
print(f"Knee point found at layer (reverse) from the end: {NUM_LAYERS - kneedle_decoder.knee if kneedle_decoder.knee else None}")