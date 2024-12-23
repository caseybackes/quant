# following along with video: https://www.youtube.com/watch?v=ISNdQcPhsts

import torch
import torch.nn as nn
import math 


class InputEmbedding(nn.Module):

    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x:torch.Tensor):
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len:int, dropout:0.20):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.d_model = d_model
        # create a matrix of shape seq_len x d_model
        pe = torch.zeros(seq_len, d_model) # pe = positional encoding matrix
        # create a vecor of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # using log space for numerical stability (was an update after the paper was published to my understanding)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        # apply sin to even indices in the vector; 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        # apply cos to odd indices in the vector; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)
        # add a batch dimension to the positional encoding matrix -> (1, seq_len, d_model)        
        pe = pe.unsqueeze(0) # tensor of shape (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x:torch.Tensor):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps:float=10e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplicative parameter
        self.bias = nn.Parameter(torch.zeros(1)) # additive parameter

    def forward(self, x:torch.Tensor):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and b1
        self.dropout = nn.Dropout(dropout) 
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and b2

    def forward(self, x:torch.Tensor):
        # batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) 
         

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # W_q
        self.w_k = nn.Linear(d_model, d_model) # W_k
        self.w_v = nn.Linear(d_model, d_model) # W_v from transformer paper. 
        
        self.w_o = nn.Linear(d_model, d_model) # W_o
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask=None, dropout=0.01):
        d_k = query.shape[-1]
        
        # batch, h, seq_len, d_k -> batch, seq_len, h, d_k
        attention_scores = (query @ key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # batch, h, seq_len, seq_len
        if dropout:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, mask:torch.Tensor):
        # batch, seq_len, d_model
        query = self.w_q(q) # batch, seq_len, d_model -> batch, seq_len, d_model
        key = self.w_k(k) # batch, seq_len, d_model -> batch, seq_len, d_model
        value = self.w_v(v) # batch, seq_len, d_model -> batch, seq_len, d_model
        # batch, seq_len, d_model -> batch, seq_len, h, d_k --> batch, h, seq_len, d_k
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # batch, h, seq_len, d_k -> batch, seq_len, h, d_k --> batch, seq_len, d_model
        x = x.transpose(1,2).contiguous().view(x.shape[0], x.shape[2], self.d_model)
        # batch, seq_len, d_model -> batch, seq_len, d_model
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, d_model:int, dropout:float=0.10):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        

    def forward(self, x:torch.Tensor, sublayer:nn.Module):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers 
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout:float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x:torch.Tensor):
        # batch, seq_len, d_model -> batch, seq_len, vocab_size
        # log softmax applied 
        return torch.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder:Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos:PositionalEncoding, tgt_pos:PositionalEncoding, projection_layer:ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src =self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
               
    def project(self, x):
        return self.projection_layer(x)

    @property  
    def device(self):
        return next(self.parameters()).device
    
    @property
    def mem_use(self):
        if self.device.type == 'cuda':
            mem_bytes = torch.cuda.memory_allocated(self.device)
            mem_mb = mem_bytes / 10**6
            mem_gb = mem_bytes / 10**9
            device = 'cuda'
            device_name = torch.cuda.get_device_name()
        else:
            import psutil
            process = psutil.Process()
            mem_bytes = process.memory_info().rss
            mem_mb = mem_bytes / 10**6
            mem_gb = mem_bytes / 10**9
            device = 'cpu'
            device_name = 'cpu'

        return {"mb": mem_mb, "gb": mem_gb, 'device': device, 'device_name': device_name}        
                
     
    def numel(self):
        return sum(p.numel() for p in self.parameters())


def build_transformer(
        src_vocab_size: int, 
        tgt_vocab_size: int,
        src_seq_len: int,
        tgt_seq_len: int,
        d_model: int = 512,
        N: int = 6,
        h: int = 8,
        dropout: float = 0.1, 
        d_ff: int = 2048,
        ):


    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block =  DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # init params

    for p in transformer.parameters():
        if p.dim() < 2:
            # Handle 1D tensors separately
            nn.init.uniform_(p)
        else:
            nn.init.xavier_uniform_(p)    
    return transformer

if __name__ == "__main__":
    transformer = build_transformer(50000, 50000, 1024*3, 1024*3).to('cuda')
    
    print("Transformer Successfully Built")
    print(transformer.mem_use)
    
    param_count = transformer.numel()
    if param_count >= 1e9:
        param_suffix = "B"
        param_count /= 1e9
    elif param_count >= 1e6:
        param_suffix = "M"
        param_count /= 1e6
    elif param_count >= 1e3:
        param_suffix = "K"
        param_count /= 1e3
    else:
        param_suffix = ""

    print(f"Number of parameters: {param_count:.0f}{param_suffix}")    
    
# next: training loop
# https://www.youtube.com/watch?v=ISNdQcPhsts time: 1:16:00
















































































