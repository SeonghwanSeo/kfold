#!/usr/bin/env python3
"""
Memory leak test script for ECSI + PairMixer
Tests if system memory grows while process memory stays stable
"""

import gc
import os
import sys
import time
import psutil
import torch

def get_memory_stats():
    """Get current memory statistics"""
    process = psutil.Process()
    process_mem_mb = process.memory_info().rss / 1024**2
    system_mem_mb = psutil.virtual_memory().used / 1024**2
    return process_mem_mb, system_mem_mb

def test_tensor_lifecycle():
    """Test tensor lifecycle to identify leak patterns"""
    print("\n=== Testing Tensor Lifecycle ===")
    
    initial_proc, initial_sys = get_memory_stats()
    print(f"Initial - Process: {initial_proc:.1f} MB, System: {initial_sys:.1f} MB")
    
    # Simulate ECSI-like operations
    B, N, L = 4, 5, 2000
    
    for iteration in range(50):
        # Simulate x_t, x_apo pattern
        x_t = torch.randn(B, N, L, 3)
        x_apo = torch.randn(B, N, L, 3)
        x0_hat = torch.zeros(B, N, L, 3)
        
        # Simulate coefficient computations
        t = torch.rand(B, N, 1, 1)
        alpha_t = 1 - t
        beta_t = t
        gamma_t = 2 * torch.sqrt(t * (1 - t) + 1e-8)
        
        # Problematic pattern: z_hat references x_t, x0_hat, x_apo
        z_hat = (x_t - alpha_t * x0_hat - beta_t * x_apo) / (gamma_t + 1e-8)
        
        # Problematic pattern: drift references z_hat, x0_hat, x_apo
        alpha_dot = -torch.ones_like(t)
        beta_dot = torch.ones_like(t)
        gamma_dot = (1 - 2 * t) / (torch.sqrt(t * (1 - t) + 1e-8) + 1e-8)
        
        drift = alpha_dot * x0_hat + beta_dot * x_apo + gamma_dot * z_hat
        
        # Problematic pattern: self-referential update
        noise = torch.randn_like(x_t)
        x_t = x_t + drift * 0.01 + 0.1 * noise
        
        if iteration % 10 == 9:
            gc.collect()
            curr_proc, curr_sys = get_memory_stats()
            proc_growth = curr_proc - initial_proc
            sys_growth = curr_sys - initial_sys
            
            print(f"Iter {iteration+1:2d}: Process +{proc_growth:6.1f} MB, "
                  f"System +{sys_growth:6.1f} MB")
            
            if sys_growth > 200 and proc_growth < 50:
                print("  ⚠️  WARNING: Potential leak (system >> process growth)")
    
    final_proc, final_sys = get_memory_stats()
    print(f"\nFinal growth - Process: +{final_proc - initial_proc:.1f} MB, "
          f"System: +{final_sys - initial_sys:.1f} MB")

def test_partial_closure_leak():
    """Test if partial closures cause leaks"""
    print("\n=== Testing Partial Closure Leak ===")
    from functools import partial
    
    initial_proc, initial_sys = get_memory_stats()
    print(f"Initial - Process: {initial_proc:.1f} MB, System: {initial_sys:.1f} MB")
    
    closures = []
    B, L = 4, 512
    
    for iteration in range(50):
        # Create tensor
        mask = torch.randn(B, L, L)
        
        # Create partial closures (like PairMixer does)
        for _ in range(48):  # 48 blocks
            closure = partial(lambda x, m: x * m, m=mask.float())
            closures.append(closure)
        
        # Clear old closures
        if len(closures) > 480:  # Keep last 10 iterations
            closures = closures[-480:]
        
        if iteration % 10 == 9:
            gc.collect()
            curr_proc, curr_sys = get_memory_stats()
            proc_growth = curr_proc - initial_proc
            sys_growth = curr_sys - initial_sys
            
            print(f"Iter {iteration+1:2d}: Closures={len(closures)}, "
                  f"Process +{proc_growth:6.1f} MB, System +{sys_growth:6.1f} MB")
    
    final_proc, final_sys = get_memory_stats()
    print(f"\nFinal growth - Process: +{final_proc - initial_proc:.1f} MB, "
          f"System: +{final_sys - initial_sys:.1f} MB")

if __name__ == "__main__":
    print("Memory Leak Test for ECSI + PairMixer")
    print("=" * 60)
    
    test_tensor_lifecycle()
    test_partial_closure_leak()
    
    print("\n" + "=" * 60)
    print("Test complete. Check for warnings above.")
