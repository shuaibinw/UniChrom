from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from Bio import SeqIO
import numpy as np
import pandas as pd

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("500M_human_ref")
model = AutoModelForMaskedLM.from_pretrained("500M_human_ref")

# Choose the length to which the input sequences are padded. 
max_length = tokenizer.model_max_length

# Function to load sequences from a FASTA file
def load_fasta(fasta_file):
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
    return sequences

# Load your FASTA file (example: "your_sequences.fasta")
fasta_file = "GM12878_promoter.fasta"
sequences = load_fasta(fasta_file)

# Tokenize the sequences
tokens_ids = tokenizer.batch_encode_plus(sequences, return_tensors="pt", padding="max_length", max_length=max_length)["input_ids"]

# Compute the embeddings
attention_mask = tokens_ids != tokenizer.pad_token_id
torch_outs = model(
    tokens_ids,
    attention_mask=attention_mask,
    encoder_attention_mask=attention_mask,
    output_hidden_states=True
)

# Compute the embeddings for the sequences
embeddings = torch_outs['hidden_states'][-1].detach().numpy()
print(f"Embeddings shape: {embeddings.shape}")
print(f"Embeddings per token: {embeddings}")

# # Add embed dimension axis for mean calculation
# attention_mask = torch.unsqueeze(attention_mask, dim=-1)

# # Compute mean embeddings per sequence
# mean_sequence_embeddings = torch.sum(attention_mask * embeddings, axis=-2) / torch.sum(attention_mask, axis=1)
# print(f"Mean sequence embeddings: {mean_sequence_embeddings.shape}")

# 保存为 .npy 文件
np.save("GM12878_promoter.npy", embeddings)  # 保存为npy文件
print("Embeddings saved as sequence_embeddings.npy")

