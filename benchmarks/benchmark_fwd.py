import triton 
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from flash_attn_bhnd import fwd_kernel

def _run_bhnd_flash_fwd(q, k, v):
    
    BATCH, HEADS, N_CTX, D_HEAD = q.shape    
    assert q.shape[0] == k.shape[0] == v.shape[0] == BATCH
    assert q.shape[1] == k.shape[1] == v.shape[1] == HEADS
    assert q.shape[2] == k.shape[2] == v.shape[2] == N_CTX
    assert q.shape[3] == k.shape[3] == v.shape[3] == D_HEAD

    sm_scale = 1.0 / (D_HEAD ** 0.5)
    BLOCK_D = triton.next_power_of_2(D_HEAD)
    o = torch.empty_like(q)
    l_save =  torch.empty((BATCH, HEADS, N_CTX), device=q.device, dtype=torch.float32)

    Block_M_list = [8, 16, 32, 64, 128]
    Block_N_list = [8, 16, 32, 64, 128, 256]
    warp_list = [2, 4, 8]
    num_stages_list = [1, 2, 3, 4, 5, 6]
    best_ms = float('inf')
    best_config = None    

    for BM in Block_M_list:
        for BN in Block_N_list:
            for NW in warp_list:
                for NS in num_stages_list:
                    grid = (triton.cdiv(N_CTX, BM), BATCH * HEADS)

                    def run_current_config():
                        fwd_kernel[grid](
                            q, k, v, sm_scale,
                            o, l_save,
                            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                            l_save.stride(0), l_save.stride(1), l_save.stride(2),
                            BATCH, HEADS, N_CTX, D_HEAD,
                            BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BLOCK_D,
                            num_warps=NW, num_stages=NS
                        )

                    try:
                        ms = triton.testing.do_bench(run_current_config, warmup=25, rep=100)
                                    
                        if ms < best_ms:
                            best_ms = ms
                            best_config = f"BLOCK_M = {BM}, BLOCK_N = {BN}, NUM_WARPS = {NW}, NUM_STAGES = {NS}"
                            
                    except Exception as e:
                       pass
                    
    print(f"Best Config: {best_config} @ {best_ms:.4f} ms")
                
    return best_ms, best_config, o


def benchmark():
    
    # Format: (BATCH, HEADS, N_CTX, D_HEAD)
    configs = [
        (4, 8, 1024, 64),   # Short sequence
        (4, 8, 4096, 64),   # Longer sequsdpa_kernelence
        (1, 8, 8192, 64),   # Very long sequence
        (8, 16, 512, 64),   # High batch/heads, short sequence (Memory bound test)
        (2, 8,  512, 128),  # Larger Head Dimension, smaller Sequence length (Check register pressure)
        (2, 8, 1024, 128),  # Larger Head Dimension (Check register pressure)
        (2, 8, 2048, 128),  # Larger Head Dimension and Sequence length (Check extreme register pressure)
    ]

    results = []
    print(f"{'Config (B, H, N, D)':<25} | {'Provider':<20} | {'Time (ms)':<10} | {'TFLOPS':<10} | {'Speedup vs Torch (Def)':<20}")
    print("-" * 100)

    for B, H, N, D in configs:
        
            dtype = torch.float16
            
            q = torch.randn((B, H, N, D), dtype=dtype, device="cuda", requires_grad=False)
            k = torch.randn((B, H, N, D), dtype=dtype, device="cuda", requires_grad=False)
            v = torch.randn((B, H, N, D), dtype=dtype, device="cuda", requires_grad=False)

            q_opt = q.transpose(1, 2).contiguous()
            k_opt = k.transpose(1, 2).contiguous()
            v_opt = v.transpose(1, 2).contiguous()

            # FLOPs formula: 4 * B * H * N^2 * D
            flops = 4.0 * B * H * N * N * D

            # --- RUN 1: PyTorch Default ---
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                ms_torch_def = triton.testing.do_bench(
                    lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False, dropout_p=0.0)
                )
                tflops_torch_def = (flops / (ms_torch_def * 1e-3)) * 1e-12

            # --- RUN 2: PyTorch Optimized ---
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                ms_torch_opt = triton.testing.do_bench(
                    lambda: torch.nn.functional.scaled_dot_product_attention(q_opt, k_opt, v_opt, is_causal=False, dropout_p=0.0)
                )
                tflops_torch_opt = (flops / (ms_torch_opt * 1e-3)) * 1e-12

            # --- RUN 3: bhnd Kernel ---
            try:
                ms_triton, best_cfg, _ = _run_bhnd_flash_fwd(q, k, v)
                tflops_triton = (flops / (ms_triton * 1e-3)) * 1e-12
            except Exception as e:
                print(f"Triton Failed: {e}")
                ms_triton = 1e9
                tflops_triton = 0

            cfg_str = f"({B}, {H}, {N}, {D})"
            
            results.append([cfg_str, "PyTorch (Default)", ms_torch_def, tflops_torch_def, 1.0])
            results.append([cfg_str, "PyTorch (Opt)", ms_torch_opt, tflops_torch_opt, ms_torch_def/ms_torch_opt])
            results.append([cfg_str, "Your Triton", ms_triton, tflops_triton, ms_torch_def/ms_triton])

            print(f"{cfg_str:<25} | {'PyTorch (Default)':<20} | {ms_torch_def:<20.3f} | {tflops_torch_def:<10.2f} | 1.00x")
            print(f"{'':<25} | {'PyTorch (Opt)':<20} | {ms_torch_opt:<20.3f} | {tflops_torch_opt:<10.2f} | {ms_torch_def/ms_torch_opt:.2f}x")
            print(f"{'':<25} | {'Your Triton':<20} | {ms_triton:<20.3f} | {tflops_triton:<10.2f} | {ms_torch_def/ms_triton:.2f}x")
            print("-" * 100)
        

    return results


if __name__ == "__main__":
    benchmark()