#!/usr/bin/env python3
"""
Analyze the singular value distribution of GPT-2 attention weights
to understand why small rank reductions cause large quality drops.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel

def analyze_attention_singular_values():
    """Analyze singular values of Q, K, V projection matrices"""
    
    # Load GPT-2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    
    # Get first layer's attention weights
    layer0 = model.transformer.h[0]
    W_qkv = layer0.attn.c_attn.weight.data  # [768, 2304]
    
    # Split into Q, K, V weights
    D = 768
    H = 12
    dh = D // H  # 64
    
    W_q = W_qkv[:, :D].view(D, H, dh)      # [768, 12, 64]
    W_k = W_qkv[:, D:2*D].view(D, H, dh)   # [768, 12, 64]
    W_v = W_qkv[:, 2*D:3*D].view(D, H, dh) # [768, 12, 64]
    
    print("=== Singular Value Analysis ===")
    print(f"Model: GPT-2, Layer: 0")
    print(f"D={D}, H={H}, dh={dh}")
    print()
    
    # Analyze each head for Q, K, V
    for matrix_name, W in [("Query", W_q), ("Key", W_k), ("Value", W_v)]:
        print(f"--- {matrix_name} Matrix Analysis ---")
        
        all_singular_values = []
        for head in range(H):
            W_head = W[:, head, :]  # [768, 64]
            U, S, Vh = torch.linalg.svd(W_head, full_matrices=False)
            all_singular_values.append(S.numpy())
        
        # Average singular values across heads
        avg_sv = np.mean(all_singular_values, axis=0)
        
        # Calculate cumulative energy
        total_energy = np.sum(avg_sv**2)
        cumulative_energy = np.cumsum(avg_sv**2) / total_energy
        
        # Find rank needed for different energy thresholds
        rank_90 = np.argmax(cumulative_energy >= 0.90) + 1
        rank_95 = np.argmax(cumulative_energy >= 0.95) + 1
        rank_99 = np.argmax(cumulative_energy >= 0.99) + 1
        
        print(f"  Full rank: {len(avg_sv)}")
        print(f"  Rank for 90% energy: {rank_90}")
        print(f"  Rank for 95% energy: {rank_95}")
        print(f"  Rank for 99% energy: {rank_99}")
        
        # Your setting analysis
        your_rank = int(0.9 * dh)  # 57
        your_energy = cumulative_energy[your_rank-1] if your_rank <= len(avg_sv) else 1.0
        print(f"  Your rank ({your_rank}): {your_energy:.1%} energy")
        
        # Show the singular value decay
        print(f"  SV[0] = {avg_sv[0]:.4f} (largest)")
        print(f"  SV[{your_rank-1}] = {avg_sv[your_rank-1]:.4f} (your cutoff)")
        print(f"  SV[{len(avg_sv)-1}] = {avg_sv[-1]:.4f} (smallest)")
        print(f"  Ratio SV[{your_rank-1}]/SV[0] = {avg_sv[your_rank-1]/avg_sv[0]:.4f}")
        print()

if __name__ == "__main__":
    analyze_attention_singular_values()


