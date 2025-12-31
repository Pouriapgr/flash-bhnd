# Flash Attention for BHND Layout (Triton)

##  Introduction

This repository contains a custom **Flash Attention** forward pass implementation written in **OpenAI Triton**. 

Unlike standard PyTorch implementations that often optimize for `(Batch, Sequence, Heads, Dim)` or require reshaping, this kernel is natively optimized for the **BHND** memory layout:
$$ (\text{Batch}, \text{Heads}, \text{Sequence}, \text{Dim}) $$

This layout is the natural output of many Multi-Head Attention implementations in PyTorch (e.g., Vision Transformers, MoE) before the `transpose` operation. By calculating attention directly on BHND, we avoid the **"Transpose Tax"**—the memory overhead and latency cost of permuting dimensions to make tensors contiguous.

---

##  Performance Benchmarks

**Hardware:** NVIDIA GeForce RTX 4070  
**Precision:** FP16  
**Comparison:** PyTorch 2.x `sdpa` (Flash Attention Backend) vs. This Custom Kernel

### The "Transpose Tax" Advantage
For sequences where the computation is not completely compute-bound, the overhead of memory permutation in PyTorch becomes visible.

| Configuration $(B, H, N, D)$ | Seq Length | Speedup vs PyTorch Default |                         Notes                           |
| :----------------------------| :--------- | :------------------------- | :------------------------------------------------------ |
|      `(4, 8, 1024, 64)`      |    1024    |     **~1.15x Faster**      | Removes `transpose(1, 2)` overhead                      |
|      `(8, 16, 512, 64)`      |    512     |     **~1.17x Faster**      | High batch/head count benefits significantly            |
|      `(1, 8, 8192, 64)`      |    8192    |      **0.3x Slower**       | Long sequences require L2 cache swizzling               |

**Conclusion:**  
This kernel is highly effective for **small to medium sequence lengths** (common in ViT, Image processing, and standard Transformer layers) where the cost of reshaping memory dominates the actual matrix multiplication time. Also the projects where LLMs are needed on consumer level hardware like RAG systems can benefit from this kernel.

---

##  Setup & Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+ (with CUDA support)
- Triton 
- GPU with Compute Capability 8.0+ (Ampere or newer recommended)


### Running Tests
To run the verification script (validates numerical accuracy against PyTorch):
bash
PYTHONPATH=. python test_correctness_fwd.py

To run the performance benchmarks:
bash
PYTHONPATH=. python benchmark_fwd.py

---

##  Technical Deep Dive

Many Triton tutorials show code but don't explain the *strategy*. Here is exactly how this kernel works and why it is fast.

### 1. The Memory Layout (BHND)
Standard Flash Attention implementations usually expect `(Batch, Seq, Heads, Dim)`. To use them, PyTorch performs:
python
# PyTorch internal logic
q = q.transpose(1, 2).contiguous() # Costly memory copy!
flash_attn_func(q, ...)
Our kernel iterates directly over the `(Batch, Heads)` dimensions as the primary grid axis, meaning we read data exactly as it sits in memory for BHND tensors. No copies, no reshapes.

### 2. Tiling Strategy & Loop Order
The kernel solves the $O(N^2)$ memory bottleneck of Attention using **Tiling** and **Online Softmax**.

1.  **Grid Launch:** We launch a 2D grid of programs.
*   Axis 0 handles the `M` dimension (Sequence Length of Queries), split into blocks of size `BLOCK_M`.
*   Axis 1 handles the `Batch * Heads` dimension.
2.  **Outer Loop (Queries):** Each program loads a block of Queries ($Q_{block}$) of size `[BLOCK_M, D]`. This block stays in SRAM (L1 Cache).
3.  **Inner Loop (Keys/Values):** We iterate through the Key ($K$) and Value ($V$) matrices in blocks of size `[BLOCK_N, D]`.
*   We compute $S = Q \cdot K^T$
*   We update the running maximum ($m_i$) and running sum ($l_i$) for Softmax.
*   We compute the partial output $O = O + \text{softmax}(S) \cdot V$ using the **online rescaling trick** to ensure numerical stability.

### 3. Optimization Details
*   **Pointers & Strides:** We calculate memory offsets manually using strides (`stride_qm`, `stride_qh`, etc.). This allows the kernel to be agnostic to the physical memory layout, provided the strides are correct.
*   **Fused Operations:** The Softmax is not computed as a separate layer. It is fused directly into the matrix multiplication loop, reducing global memory reads/writes.
*   **Register Pressure:** We carefully chose `BLOCK_M=64` and `BLOCK_N=32` (tunable) to balance register usage per thread and occupancy on the Streaming Multiprocessors (SMs). These values are derived with respect to the results of benchmarking different configs. The benchmark_fwd.py file was used to perform the brute-force search over the parameters.

---

##  Testing & Verification

### Correctness (`test_correctness_fwd.py`)
This script compares the output of our Triton kernel against `torch.nn.functional.scaled_dot_product_attention`.
*   **Success Criteria:** `torch.allclose` with `atol=1e-2` (FP16 precision).
*   **Scope:** Verifies that the numerical output is identical to the standard implementation.

### Benchmarking (`benchmark_fwd.py`)
This script measures wall-clock time in milliseconds using `triton.testing.do_bench`.
*   **Metrics:** Time (ms) and TFLOPS.
*   **Scenarios:** Tested across varying Batch sizes, Head counts, and Sequence lengths.
*   **Warmup:** Includes warmup runs to ensure the GPU clock is stable.

---

##  Roadmap & Future Work

This project is under active development. The current focus is on maximizing the forward pass efficiency before expanding features.

- [ ] **L2 Cache Swizzling:** Currently, for very large sequences ($N > 4096$), performance degrades compared to CuDNN/Cutlass. Implementing L2 swizzling (processing tiles in a Hilbert curve or specific order) will increase cache hits.
- [ ] **Backward Pass:** Implement the gradient calculation (`bwd_kernel`) to allow full training support.
- [ ] **Causal Masking:** Add support for `is_causal=True` (triangular masking) for autoregressive models like GPT.
- [ ] **Variable Sequence Lengths:** Support for ragged batches / nested tensors.

---

##  Contributing

Contributions are welcome! If you find a better `BLOCK` config for H100s or A100s, please open a PR with your benchmark results.

