import triton
import torch
from .fwd_kernel import fwd_kernel
from .bwd_kernel import bwd_kernel

def run_bhnd_flash_fwd(q, k, v):
    """
    Entry point for the custom Triton Flash Attention Forward Kernel (BHND layout).

    This function sets up the grid, calculates the scaling factor, allocates
    necessary output buffers, and launches the compiled Triton kernel.
    It is specifically optimized for input tensors in the (Batch, Heads, N_Ctx, D_Head)
    memory layout, avoiding the need for transposing typically required by
    standard PyTorch attention implementations (which prefer BSHD).

    Args:
        q (torch.Tensor): Query tensor of shape (Batch, Heads, N_Ctx, D_Head).
        k (torch.Tensor): Key tensor of shape (Batch, Heads, N_Ctx, D_Head).
        v (torch.Tensor): Value tensor of shape (Batch, Heads, N_Ctx, D_Head).

    Returns:
        tuple: A tuple containing:
            - o (torch.Tensor): Output attention values, same shape and dtype as `q`.
            - l_save (torch.Tensor): LogSumExp values of shape (Batch, Heads, N_Ctx).
              Stored in fp32, required for the backward pass gradient computation.

    Note:
        - This implementation uses a fixed configuration (Block_M=64, Block_N=32,
          Warps=4, Stages=3) tuned for general performance on this specific kernel logic.
          The configuration has showed to have grreat potential for smaller context lengths.
        - The scaling factor is automatically set to 1 / sqrt(D_Head).
    """
    
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


def run_bhnd_flash_bwd(q, k, v, l_save, dy):
    """
    Backward pass for the custom Triton Flash Attention Forward Kernel (BHND layout).

    This function sets up the grid, calculates the scaling factor, allocates
    necessary output buffers, and launches the compiled Triton kernel.
    It is specifically optimized for input tensors in the (Batch, Heads, N_Ctx, D_Head)
    memory layout, avoiding the need for transposing typically required by
    standard PyTorch attention implementations (which prefer BSHD).

    Args:
        q (torch.Tensor): Query tensor of shape (Batch, Heads, N_Ctx, D_Head).
        k (torch.Tensor): Key tensor of shape (Batch, Heads, N_Ctx, D_Head).
        v (torch.Tensor): Value tensor of shape (Batch, Heads, N_Ctx, D_Head).
        dy (torch.Tensor): Output's gradients tensor of shape (Batch, Heads, N_Ctx, D_Head).
        l_save (torch.Tensor): Saved logsum tensor of shape (Batch, Heads, N_Ctx).

    Returns:
        tuple: A tuple containing:
            - o (torch.Tensor): Output attention values, same shape and dtype as `q`.
            - l_save (torch.Tensor): LogSumExp values of shape (Batch, Heads, N_Ctx).
              Stored in fp32, required for the backward pass gradient computation.

    Note:
        - This implementation uses a fixed configuration (Block_M=64, Block_N=32,
          Warps=4, Stages=3) tuned for general performance on this specific kernel logic.
          The configuration has showed to have grreat potential for smaller context lengths.
        - The scaling factor is automatically set to 1 / sqrt(D_Head).
    """
    
    BATCH, HEADS, N_CTX, D_HEAD = q.shape

    assert q.shape[0] == k.shape[0] == v.shape[0] == l_save.shape[0] == dy.shape[0] == BATCH
    assert q.shape[1] == k.shape[1] == v.shape[1] == l_save.shape[1] == dy.shape[1] == HEADS
    assert q.shape[2] == k.shape[2] == v.shape[2] == l_save.shape[2] == dy.shape[2] == N_CTX
    assert q.shape[3] == k.shape[3] == v.shape[3] == dy.shape[3] == D_HEAD

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    dy = dy.contiguous()
    l_save = l_save.contiguous()

    sm_scale = 1.0 / (D_HEAD**0.5)
    BLOCK_D = triton.next_power_of_2(D_HEAD)
    dQ_out = torch.zeros_like(q)
    dK_out = torch.zeros_like(k)
    dV_out = torch.zeros_like(v)

    Block_M = 32
    Block_N = 64
    Num_Warps = 4
    Num_Stages = 1

    V_locks = torch.zeros([BATCH, HEADS, triton.cdiv(N_CTX, Block_N)], dtype=torch.int32, device="cuda")
    K_locks = torch.zeros([BATCH, HEADS, triton.cdiv(N_CTX, Block_N)], dtype=torch.int32, device="cuda")

    grid = (triton.cdiv(N_CTX, Block_M), BATCH * HEADS)

    bwd_kernel[grid](
        q, k, v,
        dy,
        l_save,
        V_locks, V_locks.stride(0), V_locks.stride(1), V_locks.stride(2),
        K_locks, K_locks.stride(0), K_locks.stride(1), K_locks.stride(2),
        sm_scale,
        dQ_out, dK_out, dV_out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        dy.stride(0), dy.stride(1), dy.stride(2), dy.stride(3),
        l_save.stride(0), l_save.stride(1), l_save.stride(2),
        dQ_out.stride(0), dQ_out.stride(1), dQ_out.stride(2), dQ_out.stride(3),
        dK_out.stride(0), dK_out.stride(1), dK_out.stride(2), dK_out.stride(3),
        dV_out.stride(0), dV_out.stride(1), dV_out.stride(2), dV_out.stride(3),
        BATCH, HEADS, N_CTX, D_HEAD,
        BLOCK_M=Block_M,
        BLOCK_N=Block_N,
        BLOCK_D=BLOCK_D,
        num_warps=Num_Warps, num_stages=Num_Stages,
    )

    return dQ_out, dK_out, dV_out