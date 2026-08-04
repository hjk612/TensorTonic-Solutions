import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    pe = np.zeros((seq_length, d_model))
    
    # 2. Reshape positions into a column vector: shape (seq_len, 1)
    position = np.arange(seq_length)[:, np.newaxis]
    
    # 3. Compute the exponential divisor term for even dimensions: shape (d_model/2,)
    # We step by 2 because each divisor maps to a pairs of sine and cosine functions.
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # 4. Assign sine values to even indices (0, 2, 4...)
    pe[:, 0::2] = np.sin(position * div_term)
    
    # 5. Assign cosine values to odd indices (1, 3, 5...)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe
            

    