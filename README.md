# Implementation of LLaMA-2 from Scratch (PyTorch)

![LLaMA-2 Architecture Diagram](https://drive.google.com/uc?export=view&id=1QSSuG-hmKc_f9qRi1gfaTKRcMQm0DlML)


## Overview
This repository contains a **from-scratch implementation of the LLaMA-2 architecture** using **Python and PyTorch**.  
The goal of this project is to understand and reproduce the internal working of modern large language models by implementing each component explicitly rather than relying on high-level libraries.

The implementation follows the **decoder-only Transformer design** used in LLaMA-2, including:
- Rotary Positional Embeddings (RoPE)
- RMSNorm
- Grouped Query Attention (GQA)
- SwiGLU feed-forward networks
- KV caching for efficient autoregressive inference

---

## LLaMA-2 Architecture (High Level)

LLaMA-2 is a **decoder-only Transformer** trained for autoregressive language modeling.  
At a high level, the architecture consists of:

1. **Token Embedding Layer**
2. **Stack of Transformer Decoder Blocks**
   - RMSNorm
   - Self-Attention with RoPE
   - Feed-Forward Network (SwiGLU)
   - Residual Connections
3. **Final RMSNorm**
4. **Linear Projection to Vocabulary**

During inference, tokens are generated **one at a time**, using cached keys and values to avoid recomputing attention for previous tokens.

---

## Key Architectural Components

### 1. Token Embeddings
Each input token is mapped to a dense vector representation before being passed into the Transformer layers.

- Implemented using `nn.Embedding`
- Shape: `(batch_size, seq_len, dim)`

---

### 2. Rotary Positional Embeddings (RoPE)
Instead of absolute positional embeddings, LLaMA-2 uses **rotary embeddings** to encode token positions directly into query and key vectors.

**Why RoPE?**
- Better extrapolation to longer sequences
- No learned positional embedding table
- Rotation applied directly in attention space

**Implementation highlights**
- Frequencies are precomputed once
- Queries and keys are rotated using complex arithmetic

---

### 3. Self-Attention with Grouped Query Attention (GQA)
LLaMA-2 uses **Grouped Query Attention**, where:
- Number of query heads > number of key/value heads
- Keys and values are shared across multiple query heads

**Benefits**
- Reduced memory usage
- Faster inference
- Maintains model quality

**KV Cache**
- Keys and values are cached across decoding steps
- Enables efficient autoregressive generation

---

### 4. RMSNorm
LLaMA-2 replaces LayerNorm with **RMSNorm**, which normalizes only by the root mean square.

**Advantages**
- Computationally cheaper
- Empirically more stable for large models

RMSNorm is applied:
- Before self-attention
- Before feed-forward network
- At the final output layer

---

### 5. Feed-Forward Network (SwiGLU)
The feed-forward block follows the **SwiGLU** design:

