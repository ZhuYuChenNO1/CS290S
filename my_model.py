import csv
import os
import torch
import torch.nn as nn
import math
# Postional encoding layer for transformer model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout):
        super(TransformerEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.positional_encoding = PositionalEncoding(input_size, dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(input_size, hidden_size, num_heads, dropout) for _ in range(num_layers)])
        
    def forward(self, x):
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return x
    
class EncoderLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(input_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(input_size, hidden_size, dropout)
        self.layer_norm1 = nn.LayerNorm(input_size)
        self.layer_norm2 = nn.LayerNorm(input_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self attention
        x = self.layer_norm1(x)
        x = x + self.dropout1(self.self_attn(x, x, x))
        # Feed forward
        x = self.layer_norm2(x)
        x = x + self.dropout2(self.feed_forward(x))
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.q_linear = nn.Linear(input_size, input_size)
        self.k_linear = nn.Linear(input_size, input_size)
        self.v_linear = nn.Linear(input_size, input_size)
        self.output_linear = nn.Linear(input_size, input_size)
        
    def forward(self, q, k, v):
        # Linear projections
        batch_size = q.size(0)
        q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.input_size // self.num_heads).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.input_size // self.num_heads).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.input_size // self.num_heads).transpose(1, 2)
        
        # Attention
        attn = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.input_size)
        attn = nn.functional.softmax(attn, dim=-1)
        attn = nn.functional.dropout(attn, p=self.dropout)
        
        # Output projection
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.input_size)
        output = self.output_linear(output)
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.dropout(x, p=self.dropout)
        x = self.linear2(x)
        return x
    

class TransformerDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, dropout):
        super(TransformerDecoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.positional_encoding = PositionalEncoding(input_size, dropout)
        self.decoder_layers = nn.ModuleList([DecoderLayer(input_size, hidden_size, num_heads, dropout) for _ in range(num_layers)])
        
    def forward(self, x):
        x = self.positional_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(input_size, num_heads, dropout)
        self.src_attn = MultiHeadAttention(input_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(input_size, hidden_size, dropout)
        self.layer_norm1 = nn.LayerNorm(input_size)
        self.layer_norm2 = nn.LayerNorm(input_size)
        self.layer_norm3 = nn.LayerNorm(input_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self attention
        x = self.layer_norm1(x)
        x = x + self.dropout1(self.self_attn(x, x, x))
        # Source attention
        x = self.layer_norm2(x)
        x = x + self.dropout2(self.src_attn(x, x, x))
        # Feed forward
        x = self.layer_norm3(x)
        x = x + self.dropout3(self.feed_forward(x))
        return x
    

class MYTransformer(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, num_layers, num_heads, dropout):
        super(MYTransformer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.encoder = TransformerEncoder(input_size, hidden_size, num_layers, num_heads, dropout)
        self.decoder = TransformerDecoder(input_size, hidden_size, num_layers, num_heads, dropout)
        self.output_linear = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.reshape(x.shape[0], x.shape[-1])[None]
        # Encoder
        x = self.encoder(x)
        # Decoder
        x = self.decoder(x)
        # Output [1]
        x = self.output_linear(x)
        b, t, f = x.shape
        x = x.reshape(b, t*f)
        
        # pick laregst and minimum 1/3 values
        # x = x.flatten()
        # l = max(1,len(x)//3)
        # max_val,_ = x.topk(l)
        # min_val,_ = x.topk(l, largest=False)
        # x = torch.cat((max_val, min_val), dim=-1).mean()

        output = x.mean()
        
        output = self.sigmoid(output)
        
        # reshape to [1,1]
        output = output.reshape(1,1)
        
        # We don't need the batch dimension 
        return output
# =============================================================================