import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # 1. Get the head dimension dynamically from the last dimension of K
    d_k = K.size(-1)
    
    # 2. Compute raw attention scores via matrix multiplication
    attn_scores = Q @ K.transpose(-2, -1)
    
    # 3. Scale the scores to prevent vanishing gradients in softmax
    scaled_attn_scores = attn_scores / math.sqrt(d_k)
    
    # 4. Apply softmax along the last dimension to get probabilities
    # (Dim -1 represents the columns/keys dimension)
    attn_probs = F.softmax(scaled_attn_scores, dim=-1)
    
    # 5. Multiply probabilities by V to get the final contextualized values
    output = attn_probs @ V
    
    return output