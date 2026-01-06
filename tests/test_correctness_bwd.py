import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from flash_attn_bhnd import run_bhnd_flash_bwd, run_bhnd_flash_fwd


def test_bwd_correctness(B, H, N, D):
    """
        Verifies the numerical correctness of the custom BHND Flash Attention backward kernel
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

    q = torch.randn((B, H, N, D), dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn((B, H, N, D), dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn((B, H, N, D), dtype=dtype, device="cuda", requires_grad=True)
    
    dy = torch.randn((B, H, N, D), dtype=dtype, device="cuda")


    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        ref_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=False, dropout_p=0.0
        )

    ref_out.backward(dy)
    
    ref_dq = q.grad.clone()
    ref_dk = k.grad.clone()
    ref_dv = v.grad.clone()

    q.grad = None
    k.grad = None
    v.grad = None

    q_det = q.detach()
    k_det = k.detach()
    v_det = v.detach()
    dy_det = dy.detach()


    _, l_save = run_bhnd_flash_fwd(q_det, k_det, v_det)
    trit_dq, trit_dk, trit_dv = run_bhnd_flash_bwd(q_det, k_det, v_det, l_save, dy_det)

    for name, tri, ref in [("dQ", trit_dq, ref_dq), 
                           ("dK", trit_dk, ref_dk), 
                           ("dV", trit_dv, ref_dv)]:
        if not torch.allclose(tri, ref, atol=1e-2, rtol=0.0):
            diff = torch.abs(tri - ref).max().item()
            print(f"❌ {name} mismatch! Max diff: {diff:.6f}")
            passed = False
        else:
            print(f"✅ {name} match")


if __name__ == "__main__":
    test_bwd_correctness(4, 8, 2048, 64)
