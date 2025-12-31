import triton
import torch
from .fwd_kernel import fwd_kernel


# 64 32 4 3
def run_bhnd_flash_fwd(q, k, v):

    BATCH, HEADS, N_CTX, D_HEAD = q.shape

    assert q.shape[0] == k.shape[0] == v.shape[0] == BATCH
    assert q.shape[1] == k.shape[1] == v.shape[1] == HEADS
    assert q.shape[2] == k.shape[2] == v.shape[2] == N_CTX
    assert q.shape[3] == k.shape[3] == v.shape[3] == D_HEAD

    sm_scale = 1.0 / (D_HEAD**0.5)
    BLOCK_D = triton.next_power_of_2(D_HEAD)
    o = torch.empty_like(q)
    l_save = torch.empty((BATCH, HEADS, N_CTX), device=q.device, dtype=torch.float32)

    Block_M = 64
    Block_N = 32
    Num_Warps = 4
    Num_Stages = 3

    grid = (triton.cdiv(N_CTX, Block_M), BATCH * HEADS)

    fwd_kernel[grid](
        q, k, v,
        sm_scale,
        o, l_save,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        l_save.stride(0), l_save.stride(1), l_save.stride(2),
        BATCH, HEADS, N_CTX, D_HEAD,
        BLOCK_M=Block_M,
        BLOCK_N=Block_N,
        BLOCK_D=BLOCK_D,
        num_warps=Num_Warps, num_stages=Num_Stages,
    )

    return o, l_save
