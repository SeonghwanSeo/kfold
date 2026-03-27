import torch
import triton
import triton.testing

from kfold.utils.kernels.cdist import cdist


def test_correctness():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define tensor dimensions: Batch, N, M
    B, N, M = 1, 5000, 5000
    eps = 1e-8

    print("=== Correctness Test ===")

    # Initialize tensors
    x = torch.randn((B, N, 3), device="cuda", dtype=torch.float32) * 100
    y = torch.randn((B, M, 3), device="cuda", dtype=torch.float32) * 100
    x_pt = x.clone().requires_grad_(True)
    y_pt = y.clone().requires_grad_(True)
    x_tr = x.clone().requires_grad_(True)
    y_tr = y.clone().requires_grad_(True)

    # Forward pass: PyTorch baseline
    # Creates an intermediate tensor of shape [B, N, M, 3]
    diff = x_pt.unsqueeze(2) - y_pt.unsqueeze(1)
    dist_pt = torch.sqrt(torch.sum(diff**2, dim=-1) + eps)

    # Forward pass: Triton kernel
    dist_tr = cdist(x_tr, y_tr, eps)

    # Check forward pass correctness (allow small floating point tolerance)
    fwd_match = torch.allclose(dist_pt, dist_tr, atol=1e-5, rtol=1e-5)
    print(f"Forward Pass Match: {fwd_match}")
    if not fwd_match:
        print(f"Max diff (Forward): {torch.max(torch.abs(dist_pt - dist_tr)).item()}")

    # Create a random gradient output for the backward pass
    grad_out = torch.randn_like(dist_pt)

    # Backward pass: PyTorch baseline
    dist_pt.backward(grad_out)

    # Backward pass: Triton kernel
    dist_tr.backward(grad_out)

    # Check backward pass correctness for both x and y gradients
    bwd_x_match = torch.allclose(x_pt.grad, x_tr.grad, atol=1e-4, rtol=1e-4)
    bwd_y_match = torch.allclose(y_pt.grad, y_tr.grad, atol=1e-4, rtol=1e-4)

    print(f"Backward Pass Match (x grad): {bwd_x_match}")
    print(f"Backward Pass Match (y grad): {bwd_y_match}")


def benchmark_performance():
    # Use larger dimensions to clearly see the memory bandwidth bottleneck in PyTorch
    Latom = 384 * 24
    B, N, M = 8, Latom, Latom
    # chunk_size = 8  # gradient checkpointing
    eps = 1e-8

    print("\n=== Performance Benchmark of cdist(x, y) ===")

    x = torch.randn((B, N, 3), device="cuda", dtype=torch.float32, requires_grad=True)
    y = torch.randn((B, M, 3), device="cuda", dtype=torch.float32, requires_grad=True)
    print(f"Input tensors: x shape {x.shape}, y shape {y.shape}")

    grad_out = torch.randn((B, N, M), device="cuda", dtype=torch.float32)

    # Define baseline PyTorch function
    def run_pytorch():
        diff = x.unsqueeze(2) - y.unsqueeze(1)
        dist = torch.sqrt(torch.sum(diff**2, dim=-1) + eps)
        dist.backward(grad_out)

    # Define Triton function
    def run_triton():
        dist = cdist(x, y, eps)
        dist.backward(grad_out)

    print("\nRunning PyTorch benchmark on cdist(x, y)...")
    for i in range(5):
        print(f"Run {i + 1}/5...")
        pt_ms = triton.testing.do_bench(run_pytorch, quantiles=[0.5, 0.2, 0.8])
        print(f"PyTorch Baseline: {pt_ms[0]:.3f} ms")

        # Benchmark Triton
        tr_ms = triton.testing.do_bench(run_triton, quantiles=[0.5, 0.2, 0.8])
        print(f"Triton Fused Kernel: {tr_ms[0]:.3f} ms")

        print(f"Speedup: {pt_ms[0] / tr_ms[0]:.2f}x faster")


if __name__ == "__main__":
    test_correctness()
    benchmark_performance()
