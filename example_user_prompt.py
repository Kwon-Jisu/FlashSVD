#!/usr/bin/env python3
"""
Example usage of the modified profile_svd_kv_infer.py with user prompt functionality

This demonstrates how to use the new --user-prompt flag to generate a 256-token 
prompt and perform prefilling + decoding inference.
"""

import subprocess
import os

def run_user_prompt_inference():
    """
    Example of running the modified script with user prompt functionality
    """
    
    print("=== Example Usage of Modified profile_svd_kv_infer.py ===\n")
    
    # Change to the correct directory
    os.chdir("/home/zs89/FlashSVD/decoders/gpt2")
    
    # Example 1: Basic user prompt inference with SVD model
    print("1. Basic user prompt inference (SVD model only):")
    print("   python profile_svd_kv_infer.py --user-prompt --generate-tokens 50")
    print()
    
    # Example 2: Compare SVD vs Dense model
    print("2. Compare SVD vs Dense model performance:")
    print("   python profile_svd_kv_infer.py --user-prompt --generate-tokens 100 --compare-dense")
    print()
    
    # Example 3: With validation and different rank ratios
    print("3. With validation and custom rank ratios:")
    print("   python profile_svd_kv_infer.py --user-prompt --generate-tokens 75 --validate --rank-ratio-attn 0.5 --rank-ratio-mlp 0.8")
    print()
    
    print("Key features added:")
    print("- --user-prompt: Enables user prompt mode")
    print("- --generate-tokens N: Number of tokens to generate after the prompt (default: 100)")
    print("- --compare-dense: Also run inference with the dense baseline model")
    print()
    
    print("What happens when you run with --user-prompt:")
    print("1. Generates a coherent 256-token prompt about AI")
    print("2. Saves the prompt to 'prompt_exp.txt' in the current directory")
    print("3. Performs prefilling: processes the entire prompt at once")
    print("4. Performs decoding: generates tokens one by one using KV cache")
    print("5. Shows detailed timing and performance metrics")
    print("6. Displays the generated continuation text")
    print()

if __name__ == "__main__":
    run_user_prompt_inference()
