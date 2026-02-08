
Dense Matrix–Vector Multiply on CPU (OpenMP) and GPU

> **Data policy:** Use `data/generate_dense.py` to create all inputs (no CSVs are included).

Implement \(y = A x\) for **dense** real matrices.

## Tasks
1. **Correctness baseline:** Implement a single‑threaded reference. Validate within relative error \(<10^{-12}\).
2. **OpenMP version:** Parallelize across rows; discuss scheduling, `simd` vectorization, and NUMA effects. Provide scaling vs `OMP_NUM_THREADS`.
3. **GPU version:** Implement a CUDA kernel and time it using CUDA events. Discuss memory access (coalescing, shared tiles) and launch configuration rationale.
4. **Performance analysis:** Report time and estimated bandwidth (GB/s). Explain memory‑bound vs compute‑bound behavior (roofline angle).
5. **Extensions (bonus):** Cache blocking / shared‑memory tiling; pinned H2D/D2H; stream overlap.

## Datasets to generate
- Warm‑up: `--n 512`
- Performance: `--n 2048` (or larger if your GPU memory allows)

## Deliverables
- Your source + a brief performance note (hardware, compiler flags, timing method, results table).
