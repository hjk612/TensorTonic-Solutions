import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """

    emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    nn.init.xavier_uniform_(emb.weight)
    return emb

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    """
    # Your code here
    return embedding(tokens) * math.sqrt(d_model)