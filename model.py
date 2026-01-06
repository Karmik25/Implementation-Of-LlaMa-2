# from model import ModelArgs
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # number of heads for queries
    n_kv_heads:Optional[int] = None # number of heads for keys and values
    vocab_size: int = -1 # This will be set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Neede for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

def precomute_theta_pos_frequencies(head_dim:int, seq_len:int, device:str, theta: float = 10000.0):
    assert head_dim % 2 == 0, "Head dimension must be even for RoPE."
    
    #Building the theta parameters
    #According to the formula, theta_i = 10000 ^ (-2(i-1) / dim), where i = [1, 2, .... dim/2] with shape = (head_dim/2,)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    theta = 1 / (theta ** (theta_numerator / head_dim)).to(device)   # why did we to 1/? because in the formula the power is negative

    #construct the positions(the "m" paramter)
    #shape = (seq_len)
    m = torch.arange(seq_len, device = device)

    #multiply each theta by each position using the outer product
    freqs = torch.outer(m, theta).float()

    # we can compute complex number in polar form, c = R* exp(i * m * theta), where R = 1
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex  # shape = (seq_len, head_dim/2)

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device:str):
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_complex
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep:int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x
    else:
        return (
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        #the gamma paramter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        #rsqrt 1/sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)

class SelfAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        # indicates the number of heads for keys and values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        #indicates the number of heads for queries
        self.n_heads_q = args.n_heads
        # indicates how many times key and value heads are repeated to match the number of query heads
        self.n_rep = args.n_heads // self.n_kv_heads   
        # indicates the dimension of each head
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias = False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias = False)

        # self.cache_k = torch.zeros("cache_k",(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device = args.device)

        # self.cache_v = torch.zeros("cache_v",(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), device = args.device)

        self.cache_k = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim),
            device=args.device
        )

        self.cache_v = torch.zeros(
            (args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim),
            device=args.device
        )

    def forward(self, x: torch.Tensor, start_pos:int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape # (B, Seq_len = 1, Dim)
        
        # apply the wq, wk and wv matrices to quiries, keys and values
        # (B, 1, Dim) -> (B, 1, H_Q * head_dim)  here  H_Q * head_dim = dim
        xq = self.wq(x)
        # (B, 1, Dim) -> (B, 1, H_KV * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)
        
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)  # (B, 1, H_Q * head_dim) -> (B, 1, H_Q, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)  # (B, 1, H_KV * head_dim) -> (B, 1, H_KV, head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)  # (B, 1, H_KV * head_dim) -> (B, 1, H_KV, head_dim)

        # Apply rotary embeddings to queries and keys
        xq = apply_rotary_embeddings(xq, freqs_complex, x.device)  # (B, 1, H_Q, head_dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, x.device)  # (B, 1, H_KV, head_dim)

        # replace the entry  in the cache  for this token
        # we are basically appending the new key and value to the cache at the position start_pos so that during generation we can retrieve all the previous keys and values
        self.cache_k[:, start_pos : start_pos + seq_len, :, :] = xk
        self.cache_v[:, start_pos : start_pos + seq_len, :, :] = xv

        # retrieve all the keys and values from the cache
        keys = self.cache_k[:batch_size, 0:start_pos + seq_len]  # (B, Seq_len_total, H_KV, head_dim)
        values = self.cache_v[:batch_size, 0:start_pos + seq_len]  # (B, Seq_len_total, H_KV, head_dim)

        # repeat the keys and values to match the number of query heads
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (B, 1, H_Q, head_dim) ->  (B, H_Q, 1, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)  
        values = values.transpose(1, 2)  

        # (B, H_Q, 1, head_dim) x (B, H_Q, head_dim, Seq_len_KV) -> (B, H_Q, 1, Seq_len_total)
        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim = -1).type_as(xq)

        # Attention output: (B, H_Q, 1, Seq_len_total) x (B, H_Q, Seq_len_total, head_dim) -> (B, H_Q, 1, head_dim)
        output = torch.matmul(scores, values)  # (B, H_Q, 1, head_dim)

        output = (output.transpose(1,2).contiguous().view(batch_size, seq_len, self.n_heads_q * self.head_dim))  # (B, 1, H_Q * head_dim)
        return self.wo(output)  # (B, 1, Dim)


class FeedForward(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2* hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)

        # Round the hidden dimension to the nearest multiple
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        # hidden = 7, multiple_of = 5
        # (7 + 4) // 5 =  2
        # 2 * 5 = 10

        self.w1 = nn.Linear(args.dim, hidden_dim, bias = False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias = False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias = False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V =  self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads # 4096/n_heads = 4096/32 = 128 (check params.json)
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        
        #Normalization before self attention
        self.attention_norm = RMSNorm(args.dim, eps = args.norm_eps)
        #Normalization before feed forward
        self.ffn_norm = RMSNorm(args.dim, eps = args.norm_eps)
    
    def forward(self, x:torch.Tensor, start_pos:int, freqs_complex: torch.Tensor):
        # (B, Seq_len, Dim) + (B, Seq_len, Dim) -> (B, Seq_len, Dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)  # as in the diagram, x(input embedding) is normalized and then sent to skip xonnection and self attention

        out = h + self.feed_forward.forward(self.ffn_norm(h))  # the output of self attention is normalized and then sent to skip connection and feed forward network

        return out

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim) # convert the input tokens to embeddings  

        self.layers = nn.ModuleList()    # Embedding sent to layers
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        

        self.norm = RMSNorm(args.dim, eps = args.norm_eps)  # the layer is sent to normalization
        self.output = nn.Linear(args.dim, self.vocab_size, bias = False)

        self.freqs_complex = precomute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device = self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # tokens: (batch_size, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time will be processed during generation."

        # (batch_size, seq_len) -> (batch_size, seq_len, dim) we convert the tokens to embeddings
        h = self.tok_embeddings(tokens)

        # retrieve the pairs
        freqs_complex = self.freqs_complex[start_pos : start_pos + seq_len]

        #consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output
    





