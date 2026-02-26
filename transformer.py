"""
Transformer Architecture Implementation from Scratch
=====================================================
Based on "Attention Is All You Need" (Vaswani et al., 2017)

This module implements the complete Transformer architecture including:
- Scaled Dot-Product Attention
- Multi-Head Self-Attention
- Positional Encoding
- Position-wise Feed-Forward Networks
- Encoder and Decoder layers
- Complete Transformer model

Author: Auto-generated implementation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Scaled Dot-Product Attention
# =============================================================================
class ScaledDotProductAttention(nn.Module):
    """
    Computes attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

    The scaling factor sqrt(d_k) prevents the dot products from growing
    too large in magnitude, which would push softmax into regions with
    extremely small gradients.
    """

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, heads, seq_len, d_k)
            key:   (batch, heads, seq_len, d_k)
            value: (batch, heads, seq_len, d_v)
            mask:  optional mask to prevent attention to certain positions
        Returns:
            output: weighted sum of values
            attention_weights: the attention distribution
        """
        d_k = query.size(-1)

        # QK^T / sqrt(d_k) -> (batch, heads, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask: fill masked positions with -inf so softmax gives ~0
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax over the last dimension (key sequence length)
        attention_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values
        output = torch.matmul(attention_weights, value)
        return output, attention_weights


# =============================================================================
# 2. Multi-Head Self-Attention
# =============================================================================
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention allows the model to jointly attend to information
    from different representation subspaces at different positions.

    Instead of performing a single attention function with d_model dimensions,
    we project Q, K, V into h different d_k-dimensional spaces, perform
    attention in parallel, then concatenate and project the results.

    MultiHead(Q,K,V) = Concat(head_1, ..., head_h) * W_o
    where head_i = Attention(Q*W_q_i, K*W_k_i, V*W_v_i)
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Args:
            d_model:   total model dimension (e.g. 512)
            num_heads: number of parallel attention heads (e.g. 8)
            dropout:   dropout rate applied to attention weights
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        # Linear projections for Q, K, V and the output
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: (batch, seq_len, d_model)
            mask: optional attention mask
        """
        batch_size = query.size(0)

        # 1) Project and reshape: (batch, seq, d_model) -> (batch, heads, seq, d_k)
        query = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key   = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2) Apply attention on all heads in parallel
        attn_output, attn_weights = self.attention(query, key, value, mask)

        # 3) Concatenate heads: (batch, heads, seq, d_k) -> (batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 4) Final linear projection
        output = self.w_o(attn_output)
        output = self.dropout(output)
        return output


# =============================================================================
# 3. Positional Encoding
# =============================================================================
class PositionalEncoding(nn.Module):
    """
    Injects information about the position of tokens in the sequence.

    Since the Transformer contains no recurrence or convolution, positional
    encodings are added to give the model a sense of token order.

    Uses sine and cosine functions of different frequencies:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    This scheme allows the model to learn relative positions because for any
    fixed offset k, PE(pos+k) can be represented as a linear function of PE(pos).
    """

    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Compute the division term: 10000^(2i/d_model) using log space for stability
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension: (1, max_seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Register as buffer (not a parameter — no gradient needed)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model) — token embeddings
        Returns:
            embeddings + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# 4. Position-wise Feed-Forward Network
# =============================================================================
class FeedForward(nn.Module):
    """
    Two-layer fully connected network applied to each position independently.
    
    FFN(x) = ReLU(x * W_1 + b_1) * W_2 + b_2
    
    The inner layer expands the dimension (typically 4x), then projects back.
    """
    
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# =============================================================================
# 5. Encoder Layer
# =============================================================================
class EncoderLayer(nn.Module):
    """
    A single encoder layer consisting of:
    1. Multi-head self-attention
    2. Position-wise feed-forward network
    Each sublayer has residual connection and layer normalization.
    """
    
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


# =============================================================================
# 6. Decoder Layer
# =============================================================================
class DecoderLayer(nn.Module):
    """
    A single decoder layer consisting of:
    1. Masked multi-head self-attention
    2. Multi-head cross-attention (over encoder output)
    3. Position-wise feed-forward network
    Each sublayer has residual connection and layer normalization.
    """
    
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Cross-attention over encoder output
        attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x


# =============================================================================
# 7. Complete Transformer Model
# =============================================================================
class Transformer(nn.Module):
    """
    The complete Transformer model for sequence-to-sequence tasks.
    
    Architecture:
    - Input embedding + positional encoding
    - Stack of N encoder layers
    - Output embedding + positional encoding
    - Stack of N decoder layers
    - Final linear projection to vocabulary
    
    This implementation follows "Attention Is All You Need" (Vaswani et al., 2017)
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, dropout=0.1, max_seq_len=5000):
        super().__init__()
        
        # Embeddings
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.d_model = d_model
    
    def encode(self, src, src_mask=None):
        """Encode source sequence."""
        x = self.encoder_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """Decode target sequence given encoder output."""
        x = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Full forward pass.
        
        Args:
            src: source token indices (batch, src_len)
            tgt: target token indices (batch, tgt_len)
            src_mask: mask for source padding
            tgt_mask: mask for target (causal + padding)
        
        Returns:
            logits: (batch, tgt_len, tgt_vocab_size)
        """
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        logits = self.fc_out(decoder_output)
        
        return logits


def create_causal_mask(seq_len):
    """
    Create a causal mask for decoder self-attention.
    Prevents positions from attending to subsequent positions.
    
    Returns:
        mask: (seq_len, seq_len) with 0s in upper triangle, 1s elsewhere
    """
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask


def create_padding_mask(seq, pad_idx=0):
    """
    Create mask for padding tokens.
    
    Args:
        seq: (batch, seq_len) token indices
        pad_idx: index of padding token
    
    Returns:
        mask: (batch, 1, seq_len) with 0 at padding positions
    """
    mask = (seq != pad_idx).unsqueeze(1)
    return mask


# Example usage
if __name__ == "__main__":
    # Create a small transformer
    model = Transformer(
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1
    )
    
    # Example inputs
    batch_size = 2
    src_len = 10
    tgt_len = 8
    
    src = torch.randint(0, 10000, (batch_size, src_len))
    tgt = torch.randint(0, 10000, (batch_size, tgt_len))
    
    # Create masks
    src_mask = create_padding_mask(src)
    tgt_mask = create_causal_mask(tgt_len)
    
    # Forward pass
    output = model(src, tgt, src_mask, tgt_mask)
    print(f"Input shape: src={src.shape}, tgt={tgt.shape}")
    print(f"Output shape: {output.shape}")
