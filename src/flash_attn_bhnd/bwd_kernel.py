import triton
import triton.language as tl

@triton.jit
def bwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    dY_ptr,
    L_save_ptr,
    V_locks_ptr, V_locks_stride_z, V_locks_stride_h, V_locks_stride_m,
    K_locks_ptr, K_locks_stride_z, K_locks_stride_h, K_locks_stride_m,
    Sm_scale,
    dQ_out_ptr, dK_out_ptr, dV_out_ptr,
    stride_qz, stride_qh, stride_qm, stride_qd,          # Strides for Query
    stride_kz, stride_kh, stride_km, stride_kd,          # Strides for Key
    stride_vz, stride_vh, stride_vm, stride_vd,          # Strides for Value
    stride_dyz, stride_dyh, stride_dym, stride_dyd,      # Strides for Gradients of Y = softmax(Q.KT * Sm_scale) . V
    stride_lz, stride_lh, stride_lm,                     # Strides for Query
    stride_dqoz, stride_dqoh, stride_dqom, stride_dqod,  # Strides for d-Query output
    stride_dkoz, stride_dkoh, stride_dkom, stride_dkod,  # Strides for d-Key output
    stride_dvoz, stride_dvoh, stride_dvom, stride_dvod,  # Strides for Query 0 d
    Z, H, N_CTX, D,                                      # Batch, Heads, Context Length
    BLOCK_M: tl.constexpr,  # Block size for Rows
    BLOCK_N: tl.constexpr,  # Block size for Columns
    BLOCK_D: tl.constexpr,  # 2^n Bigger than head dimention
):
    pid_z = tl.program_id(axis=1) %  Z
    pid_h = tl.program_id(axis=1) // Z
    pid_m = tl.program_id(axis=0) 

    Q_ptr += pid_z * stride_qz + pid_h * stride_qh
    K_ptr += pid_z * stride_kz + pid_h * stride_kh
    V_ptr += pid_z * stride_vz + pid_h * stride_vh
    L_save_ptr += pid_z * stride_lz + pid_h * stride_lh
    dY_ptr += pid_z * stride_dyz + pid_h * stride_dyh

    dQ_out_ptr += pid_z * stride_dqoz + pid_h * stride_dqoh
    dK_out_ptr += pid_z * stride_dkoz + pid_h * stride_dkoh
    dV_out_ptr += pid_z * stride_dvoz + pid_h * stride_dvoh

    V_locks_ptr += pid_z * V_locks_stride_z + pid_h * V_locks_stride_h
    K_locks_ptr += pid_z * K_locks_stride_z + pid_h * K_locks_stride_h

    full_cols_offset = tl.arange(0, BLOCK_D)
    full_cols_mask = full_cols_offset < D
    rows_offset = tl.arange(0, BLOCK_M) + pid_m * BLOCK_M
    rows_mask = rows_offset < N_CTX

    Q_ptrs = Q_ptr + (rows_offset[:, None] * stride_qm + full_cols_offset[None, :] * stride_qd) 
    QdYdQ_mask = rows_mask[:, None] & full_cols_mask[None, :]
    Q_values = tl.load(Q_ptrs, mask=QdYdQ_mask, other=0.0)

    dY_ptrs = dY_ptr + (rows_offset[:, None] * stride_dym + full_cols_offset[None, :] * stride_dyd)
    dY_values = tl.load(dY_ptrs, mask=QdYdQ_mask, other=0.0) 
    
    L_ptrs = L_save_ptr + rows_offset * stride_lm
    L_i = tl.load(L_ptrs, mask=rows_mask, other=0.0)

    ## Calculation of rowsum for P * dP && dV
    row_sum_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for i in range(0, N_CTX, BLOCK_N):
        cols_offset = tl.arange(0, BLOCK_N) + i
        cols_mask = cols_offset < N_CTX
        
        K_ptrs = K_ptr + (cols_offset[:, None] * stride_km + full_cols_offset[None, :] * stride_kd)
        KVdKdV_mask = cols_mask[:, None] & full_cols_mask[None, :]
        K_values = tl.load(K_ptrs, mask=KVdKdV_mask, other=0.0)

        P_block = tl.exp(tl.dot(Q_values, K_values.T) * Sm_scale - L_i[:, None]).to(tl.float16)
        prod = tl.dot(P_block.T, dY_values)
        
        V_ptrs = V_ptr + (cols_offset[:, None] * stride_vm + full_cols_offset[None, :] * stride_vd)
        V_values = tl.load(V_ptrs, mask=KVdKdV_mask, other=0.0)

        dp_block = tl.dot(dY_values, V_values.T)
        partial_sum = tl.sum(P_block * dp_block, axis=1)
        row_sum_acc += partial_sum

        ## Part where atomic add prod to dV output rows shown by cols_offset
        dV_ptrs = dV_out_ptr + (cols_offset[:, None] * stride_dvom + full_cols_offset[None, :] * stride_dvod)
        V_lock_ptr = V_locks_ptr + (i // BLOCK_N) * V_locks_stride_m
        while tl.atomic_cas(V_lock_ptr, 0, 1) == 1:
            pass
        prod += tl.load(dV_ptrs, mask=KVdKdV_mask, other=0.0)
        tl.store(dV_ptrs, prod, mask=KVdKdV_mask)
        tl.debug_barrier()
        tl.atomic_xchg(V_lock_ptr, 0)
        #tl.atomic_add(dV_ptrs, prod, mask=KVdKdV_mask)
        ## Atomic done


    ## Calculations for dK, dQ, 
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for i in range(0, N_CTX, BLOCK_N):
        cols_offset = tl.arange(0, BLOCK_N) + i
        cols_mask = cols_offset < N_CTX
        
        K_ptrs = K_ptr + (cols_offset[:, None] * stride_km + full_cols_offset[None, :] * stride_kd)
        KVdKdV_mask = cols_mask[:, None] & full_cols_mask[None, :]
        K_values = tl.load(K_ptrs, mask=KVdKdV_mask, other=0.0)

        P_block = tl.exp(tl.dot(Q_values, K_values.T) * Sm_scale - L_i[:, None])

        V_ptrs = V_ptr + (cols_offset[:, None] * stride_vm + full_cols_offset[None, :] * stride_vd)
        V_values = tl.load(V_ptrs, mask=KVdKdV_mask, other=0.0)

        dp_block = tl.dot(dY_values, V_values.T)
        ds_block = (P_block * (dp_block - row_sum_acc[:, None])).to(tl.float16)
        
        ## Part where atomic add prod to dk output rows shown by cols_offset
        prod = tl.dot(ds_block.T, Q_values) * Sm_scale
        dK_ptrs = dK_out_ptr + (cols_offset[:, None] * stride_dkom + full_cols_offset[None, :] * stride_dkod)
        K_lock_ptr = K_locks_ptr + (i // BLOCK_N) * K_locks_stride_m
        while tl.atomic_cas(K_lock_ptr, 0, 1) == 1:
            pass
        prod += tl.load(dK_ptrs, mask=KVdKdV_mask, other=0.0)
        tl.store(dK_ptrs, prod, mask=KVdKdV_mask)
        tl.debug_barrier()
        tl.atomic_xchg(K_lock_ptr, 0)
        #tl.atomic_add(dK_ptrs, prod, mask=KVdKdV_mask)
        ## Atomic done

        acc = tl.dot(ds_block, K_values, acc=acc)

    dQ_ptrs = dQ_out_ptr + (rows_offset[:, None] * stride_dqom + full_cols_offset[None, :] * stride_dqod) 
    tl.store(dQ_ptrs, acc * Sm_scale, mask=QdYdQ_mask)


        
        
