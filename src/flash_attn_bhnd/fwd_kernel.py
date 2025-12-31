import triton
import triton.language as tl

@triton.jit
def fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    Sm_scale,
    Out_ptr, L_ptr,
    stride_qz, stride_qh, stride_qm, stride_qd,  # Strides for Query
    stride_kz, stride_kh, stride_km, stride_kd,  # Strides for Key
    stride_vz, stride_vh, stride_vm, stride_vd,  # Strides for Value
    stride_oz, stride_oh, stride_om, stride_od,  # Strides for Output
    stride_lz, stride_lh, stride_lm,             # Strides for saving L values
    Z, H, N_CTX, D,                              # Batch, Heads, Context Length
    BLOCK_M: tl.constexpr,  # Block size for Queries (Rows)
    BLOCK_N: tl.constexpr,  # Block size for Keys and Values (Cols in transposed)
    BLOCK_D: tl.constexpr,  # Block size for columns
):
    """
    Triton kernel for the forward pass of Flash Attention, optimized for BHND layout.

    This kernel implements the IO-aware exact attention algorithm (FlashAttention).
    It computes Attention(Q, K, V) = Softmax(Q @ K.T / sqrt(D)) @ V using
    tiling to keep intermediate matrices (Q blocks) in SRAM, reducing global memory access.

    Algorithm:
        The kernel uses the 'Online Softmax' trick to compute the softmax normalization
        factors (L_i) and maximums (M_i) iteratively without materializing the full
        NxN attention matrix.

    Grid Structure:
        - Axis 0 (pid_m): Handles the sequence length (N) dimension, split into tiles of size BLOCK_M.
        - Axis 1 (pid_z * pid_h): Handles the batch (Z) and heads (H) dimensions.

    Args:
        Q_ptr, K_ptr, V_ptr: Pointers to input tensors.
        Sm_scale (float): Scaling factor, usually 1 / sqrt(D_HEAD).
        Out_ptr: Pointer to the output tensor buffer.
        L_ptr: Pointer to the buffer for storing LogSumExp values (needed for backward pass).
        stride_*: Strides for all input/output tensors. Crucial for handling BHND layout
                  correctly without reshaping.
        Z (int): Batch size.
        H (int): Number of heads.
        N_CTX (int): Sequence length (Context size).
        D (int): Head dimension.
        BLOCK_M (constexpr): Tile size for the Query sequence dimension (rows of attention matrix).
                             Larger values improve arithmetic intensity but require more SRAM.
        BLOCK_N (constexpr): Tile size for the Key/Value sequence dimension (cols of attention matrix).
        BLOCK_D (constexpr): Tile size for the head dimension. Must be a power of 2.
    """

    pid_z = tl.program_id(axis=1) % Z
    pid_h = tl.program_id(axis=1) // Z
    pid_m = tl.program_id(axis=0)

    Q_ptr += pid_z * stride_qz + pid_h * stride_qh
    K_ptr += pid_z * stride_kz + pid_h * stride_kh
    V_ptr += pid_z * stride_vz + pid_h * stride_vh
    Out_ptr += pid_z * stride_oz + pid_h * stride_oh
    L_ptr += pid_z * stride_lz + pid_h * stride_lh

    cols_offset = tl.arange(0, BLOCK_D)      # Used for all matrices
    cols_mask = cols_offset < D              # Used for all matrices

    rows_offset_Q = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    rows_mask_Q = rows_offset_Q < N_CTX

    Q_mask = rows_mask_Q[:, None] & cols_mask[None, :]
    Q_ptrs = Q_ptr + (
        rows_offset_Q[:, None] * stride_qm + cols_offset[None, :] * stride_qd
    )
    Q = tl.load(Q_ptrs, mask=Q_mask, other=0.0)
    Sm_scale = Sm_scale.to(tl.float16)
    Q = Q * Sm_scale

    M_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float("inf")
    L_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for i in range(0, N_CTX, BLOCK_N):
        rows_offset_KV = tl.arange(0, BLOCK_N) + i
        row_mask_KV = rows_offset_KV < N_CTX

        KV_mask = row_mask_KV[:, None] & cols_mask[None, :]
        K_ptrs = K_ptr + (
            rows_offset_KV[:, None] * stride_km + cols_offset[None, :] * stride_kd
        )
        V_ptrs = V_ptr + (
            rows_offset_KV[:, None] * stride_vm + cols_offset[None, :] * stride_vd
        )

        K = tl.load(K_ptrs, mask=KV_mask, other=0.0)
        V = tl.load(V_ptrs, mask=KV_mask, other=0.0)

        S = tl.dot(Q, K.T)
        cur_max = tl.max(S, axis=1)
        new_max = tl.maximum(M_i, cur_max)

        S = S - new_max[:, None]
        S = tl.exp(S).to(tl.float16)
        cur_L = tl.sum(S, axis=1)

        acc = (tl.exp(M_i - new_max)[:, None] * acc) + tl.dot(S, V)
        L_i = (L_i * tl.exp(M_i - new_max)) + cur_L
        M_i = new_max

    acc = acc / L_i[:, None]

    O_ptrs = Out_ptr + (
        rows_offset_Q[:, None] * stride_om + cols_offset[None, :] * stride_od
    )
    O_mask = rows_mask_Q[:, None] & cols_mask[None, :]

    tl.store(O_ptrs, acc.to(tl.float16), mask=O_mask)

    L_ptrs = L_ptr + (rows_offset_Q[:] * stride_lm)
    tl.store(L_ptrs, tl.log(L_i) + M_i, mask=rows_mask_Q)
