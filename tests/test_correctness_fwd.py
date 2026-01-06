import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from flash_attn_bhnd import run_bhnd_flash_fwd

def test_fwd_correctness(B, H, N, D):
    """
        Verifies the numerical correctness of the custom BHND Flash Attention forward kernel
        by comparing it against PyTorch's native implementation.

        This test generates random inputs in FP16, runs both the custom kernel and
        `torch.nn.functional.scaled_dot_product_attention` (using the Flash Attention backend),
        and checks if the outputs match within a specific tolerance.

        Args:
            B (int): Batch size.
            H (int): Number of attention heads.
            N (int): Sequence length (context size).
            D (int): Head dimension.

        Returns:
            None: The function prints "✅ Triton and Torch match" or
                "❌ Triton and Torch differ" directly to stdout.
    """

    dtype = torch.float16
    q = torch.randn((B, H, N, D), dtype=dtype, device="cuda", requires_grad=False)
    k = torch.randn((B, H, N, D), dtype=dtype, device="cuda", requires_grad=False)
    v = torch.randn((B, H, N, D), dtype=dtype, device="cuda", requires_grad=False)

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        torch_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False, dropout_p=0.0
        )
    trit_out, _ = run_bhnd_flash_fwd(q, k, v)

    if torch.allclose(trit_out, torch_out, atol=1e-2, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")


if __name__ == "__main__":
    test_fwd_correctness(4, 8, 2048, 64)
