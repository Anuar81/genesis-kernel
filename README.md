# Genesis Kernel

Runtime x86-64 machine code generation for LLM inference on CPU. No compilers, no CUDA required.

Two kernel families:
- **NF4**: Fused dequantization + matrix multiplication (AVX-512 ZMM). For MoE CPU offload.
- **Q4_K**: ggml-compatible `vec_dot_q4_K_q8_K` replacement (AVX-512 VNNI in YMM). Drop-in for llama.cpp via `LD_PRELOAD`.

Built with AI assistance + reverse engineering of ggml internals. The NF4 kernels were discovered through genetic evolution of instruction orderings. The Q4_K kernel was hand-engineered through 7 iterations; the final optimization (a prefetch hint) was found by brute-force injection of 5,000 random instructions.

## Reproduce in 5 Minutes

Requirements: Linux x86-64, Python 3.10+, CPU with AVX-512 (AMD Zen 4+, Intel Skylake-X+).

```bash
# Check AVX-512 support
grep -c avx512 /proc/cpuinfo  # should be > 0

# Install
git clone https://github.com/Anuar81/genesis-kernel.git
cd genesis-kernel
pip install -e .

# Run tests
pytest tests/ -v

# NF4 quick check
python -c "
from genesis_kernel import compile_best_nf4_matmul, quantize_nf4, reorder_activations
import numpy as np, time
M, K = 2048, 1536
w = np.random.randn(M*K).astype(np.float32)
nf4, scales = quantize_nf4(w)
act = reorder_activations(np.random.randn(K).astype(np.float32))
kernel = compile_best_nf4_matmul(M, K)
# warmup
kernel(nf4, scales, act, M, K)
t0 = time.perf_counter()
for _ in range(100): kernel(nf4, scales, act, M, K)
t1 = time.perf_counter()
print(f'NF4 {M}x{K}: {(t1-t0)/100*1000:.3f} ms/call ({kernel._origin})')
"
```

## Two Kernels

### NF4: Fused Dequant + Matmul

Standard CPU inference dequantizes the full weight matrix, then multiplies:
```
Traditional:  NF4 weights → dequantize to float32 (M×K) → matmul → output
Genesis:      NF4 weights + activations → fused dequant+dot → output (one pass, all in registers)
```

For hybrid GPU/CPU MoE inference, this means copying only the result vector over PCIe instead of the full weight matrix.

Four dimension-specific kernels were discovered through genetic evolution of x86 instruction orderings (16,460 mutations across 25 runs). The evolved schedules exploit Zen 4 microarchitectural properties: cache-line-aligned NOPs, early scale broadcasts, reverse-distance activation loading, and interleaved pipeline execution.

| Dimensions | Use case | Improvement vs baseline |
|---|---|---|
| 1024×2048 | 80B gate_up | +2.93% |
| 2048×512 | 80B down_proj | +19.25% |
| 3072×2048 | 30B gate_up | +3.90% |
| 2048×1536 | 30B down_proj | +2.89% |

For other dimensions, the base 2-pipeline kernel is used automatically.

### Q4_K: ggml-Compatible Dot Product

Drop-in replacement for `ggml_vec_dot_q4_K_q8_K_generic` in llama.cpp. Uses AVX-512 VNNI (`VPDPWSSD`) in YMM registers (256-bit) to avoid Zen 4's double-pumping penalty on 512-bit operations.

Seven iterations, each targeting a specific bottleneck:

| Version | ISA | ns/blk (N=1024) | vs ggml | What changed |
|---|---|---|---|---|
| turbo3 | AVX2 | 13.6 | 2.39× slower | First working kernel |
| turbo4 | AVX-512 F | 12.27 | 2.13× slower | YMM registers, 4 accumulators |
| turbo5 | AVX-512 F | 5.90 | 1.026× slower | Prologue rewrite (signbit, fp16 inline, no SIB) |
| turbo6 | AVX-512 F+VNNI | 5.71 | 1.007× slower | VPDPWSSD replaces VPMADDUBSW+VPMADDWD |
| **turbo7** | **AVX-512 F+VNNI** | **5.51** | **0.985× (wins)** | **+1 PREFETCHNTA hint (brute-force found)** |
| ggml | VNNI+VBMI | 5.59 | 1.00× | Compiled with `-march=native` (uses VPDPWSSD, VPERMT2B) |

Turbo7 = turbo6 + `PREFETCHNTA [R9+358]` at position 19 of the inner loop. R9+358 points to the 2nd cache line of the next Q8_K block. The hardware prefetcher brings sequential lines from the block start; this manual prefetch targets a specific line the HW hasn't reached yet. Non-temporal hint avoids polluting L1/L2 with single-use data.

Found by brute-force: 5,000 random instruction injections → 3-stage verification (measure → re-measure ×5 → A/B ×20 rounds) → 29 confirmed improvements → best one baked into turbo7.

The evolutionary mutator (reordering the 33 existing ops) found only +0.56% — the inner loop was already near-optimal for this ISA. Injection of new instructions had a much larger search space.

## Benchmarks

All measurements on AMD Ryzen 9 7900 (Zen 4, AVX-512), 32GB DDR5, EndeavourOS.

### Microbenchmarks (what Genesis improves directly)

**NF4** — per-expert latency, Genesis vs bitsandbytes CPU fallback:

| Metric | Genesis AVX-512 | bitsandbytes CPU | Speedup |
|---|---|---|---|
| Per-expert latency (2048×1536) | 0.15 ms | 24.8 ms | 165× |

**Q4_K** — isolated A/B benchmark, turbo7 vs ggml (llama.cpp b8184), 20 rounds × 20,000 reps, `taskset -c 0`, governor=performance:

| N (blocks) | turbo7 ns/blk | ggml ns/blk | Delta | Wins |
|---|---|---|---|---|
| 32 (LLaMA 70B token gen) | 32.05 | 36.59 | +12.56% | 20/20 |
| 128 | 11.69 | 12.85 | +9.05% | 20/20 |
| 512 | 6.40 | 6.56 | +2.52% | 20/20 |
| 1024 | 5.51 | 5.59 | +1.46% | 20/20 |

Correctness: bit-exact vs ggml (`diff = 0.000000e+00` for N=1 to N=1024).

### End-to-End (context — depends on model and offload ratio)

**NF4** — MoE inference with hybrid GPU/CPU offload (RTX 4090 + CPU):

| Model | Experts | VRAM | Speed | CPU layers |
|---|---|---|---|---|
| Qwen3-Coder-30B-A3B | 128 (8 active) | 13.4 GB | 5.7 tok/s | 8 |
| Qwen3-Next-80B-A3B | 512 (10 active) | 20.7 GB | 2.7–3.3 tok/s | 24 |

The 80B model does not fit in a single RTX 4090 without CPU offload.

**Q4_K** — llama.cpp with `LD_PRELOAD=./genesis_turbo7.so` (Qwen3-Coder-30B, b8184):

| Config | ggml stock | genesis turbo7 | Delta |
|---|---|---|---|
| Prompt tok/s (-c 2048) | 97.5 | 100.4 | +3.0% |
| Prompt tok/s (-c 4096, -t 12) | 89.7 | 97.2 | +8.4% |
| Gen tok/s (-c 4096, -t 12) | 40.4 | 41.8 | +3.5% |

Genesis improves one operation (CPU dot product). End-to-end impact depends on how much of the workload runs on CPU. The 30B model with most layers on GPU shows modest gains; a fully CPU-bound model would show more.

## Installation

From source (recommended):
```bash
git clone https://github.com/Anuar81/genesis-kernel.git
cd genesis-kernel
pip install -e .
```

Dependencies: NumPy. That's it. The kernels use only Python stdlib (`ctypes`, `mmap`, `struct`) to emit and execute machine code.

For the Q4_K llama.cpp integration, you also need `gcc` to compile the shared library.

## Usage

### NF4 Matmul

```python
import numpy as np
from genesis_kernel import compile_best_nf4_matmul, quantize_nf4, reorder_activations

# Quantize weights to NF4
weights = np.random.randn(2048 * 1536).astype(np.float32)
nf4_weights, scales = quantize_nf4(weights)

# Reorder activations (required — kernel expects interleaved even/odd layout)
activations = np.random.randn(1536).astype(np.float32)
act_reord = reorder_activations(activations)

# Compile and run
kernel = compile_best_nf4_matmul(M=2048, K=1536)
output = kernel(nf4_weights, scales, act_reord, M=2048, K=1536)
```

For a complete example with hybrid GPU/CPU MoE inference, see [`examples/hybrid_moe_offload.py`](examples/hybrid_moe_offload.py).

### Q4_K in llama.cpp

```bash
# Generate the shared library
python -c "from genesis_kernel import generate_turbo7_so; generate_turbo7_so('.')"

# Run llama.cpp with Genesis kernel
LD_PRELOAD=./genesis_turbo7.so llama-cli -m model.gguf -p "Hello" -n 128

# Optional: verbose mode to confirm Genesis loaded
GENESIS_VERBOSE=1 LD_PRELOAD=./genesis_turbo7.so llama-cli -m model.gguf -p "Hello" -n 128
# Should print: [genesis] turbo7 loaded: 1078 bytes at 0x...
```

The `.so` intercepts `ggml_vec_dot_q4_K_q8_K_generic` — the non-repacked path that handles standard `block_q4_K` data. Tested with llama.cpp b8184.

**Important: disabling repack may be required.** llama.cpp has a repack system (`ggml_backend_cpu_repack_buffer_type` in `ggml/src/ggml-cpu/repack.cpp`) that converts `block_q4_K` into an interleaved `block_q4_Kx8` format for optimized GEMM. When repack is active, some code paths may bypass `_generic` entirely, meaning our kernel never gets called. During our testing we had to disable repack to get consistent results:

```bash
# In your llama.cpp source directory, disable repack:
sed -i '/^ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void) {/,/^}/ c\
ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void) {\
    return NULL;\
}' ggml/src/ggml-cpu/repack.cpp

# Rebuild
cmake --build build -j$(nproc)

# Now LD_PRELOAD will intercept all Q4_K dot products:
LD_PRELOAD=./genesis_turbo7.so llama-cli -m model.gguf -p "Hello" -n 128

# To restore original repack, re-checkout the file:
git checkout ggml/src/ggml-cpu/repack.cpp
cmake --build build -j$(nproc)
```

Without this patch, `LD_PRELOAD` may still work for some operations, but the repacked GEMM path will use ggml's own kernel instead of ours. Our benchmarks were run with repack disabled.

## How It Works

Genesis generates x86-64 machine code at runtime using a Python-based emitter (`x86_emitter.py`). No LLVM, no GCC, no compiler toolchain — just `struct.pack` to encode instruction bytes, `mmap` with `PROT_EXEC` to make them executable, and `ctypes` to call them.

The emitter supports AVX-512 Foundation, VNNI, and VEX-encoded instructions. Each kernel is defined as a sequence of operations that the emitter translates to raw bytes. The Q4_K kernel (turbo7) is 1,078 bytes / 34 inner loop operations.

### Why not just use a compiler?

Because the goal is to search the instruction space programmatically. A compiler makes fixed decisions about instruction ordering, register allocation, and scheduling. Genesis represents the kernel as a mutable sequence of operations, enabling:
- Genetic evolution of instruction orderings (NF4 kernels)
- Brute-force injection of hint instructions (Q4_K turbo7)
- Direct control over every byte emitted

The evolutionary system and brute-force injector are not included in this repository. The kernels here are their output, baked into deterministic functions. See "How the Evolutionary System Works" below for details on the process.

## How the Evolutionary System Works

The evolutionary factory is not open-source, but here's exactly how it works — no black boxes.

### NF4 kernel evolution

**Genome representation**: Each kernel's inner loop is a list of Python function calls to the emitter (e.g., `_emit_vpmovzxbd`, `_emit_vpermps`, `_emit_vfmadd231ps`). You can see the evolved orderings directly in `nf4_kernel.py` — the `_baked_kernel_*` functions show every instruction in order.

**Fitness function**: Wall-clock execution time. `time.perf_counter_ns()`, median of 1,000 repetitions, pinned to core 0 with `taskset`. Lower is better. No proxy metrics.

**Mutation operators** (4 types):
- `swap`: Pick two random instructions, swap their positions
- `insert`: Add a NOP or prefetch at a random position
- `delete`: Remove one instruction (only NOPs/prefetches, never compute ops)
- `replace`: Swap one instruction for a different one of the same type

**Selection**: (1+1) evolutionary strategy. Mutate the current best → benchmark → if faster, replace. No population, no crossover. Simple hill climbing.

**Statistics**: 25 runs, 16,460 total mutations evaluated, 4 winners baked (one per dimension). Best improvement: +19.25% (2048×512).

### Q4_K brute-force injection

Not evolution — pure random search. For each attempt:
1. Pick a random instruction type (prefetcht0/t1/t2/nta, multi_nop, lfence)
2. Pick a random position in the 33-op inner loop
3. Pick random parameters (offset 0–4096 for prefetch, 1–15 bytes for NOP)
4. Insert it, compile, run
5. If it crashes or gives wrong results → discard
6. If it's faster → 3-stage verification: measure → re-measure ×5 → A/B ×20 rounds
7. If it survives all 3 stages → save to `hallazgos_inyeccion.jsonl`

5,000 attempts → 29 confirmed improvements → best one (PREFETCHNTA [R9+358] at position 19) baked into turbo7.

## FAQ

**Q: Can I reproduce the benchmark numbers?**
A: Yes. You need an AVX-512 CPU (ideally Zen 4 for comparable results). The Q4_K benchmark requires llama.cpp b8184 built with `cmake -B build -DGGML_NATIVE=ON && cmake --build build`. We pin to core 0 with `taskset`, set governor to `performance`, and run 20 rounds × 20,000 reps. Median of medians.

**Q: What commit of llama.cpp?**
A: b8184. GCC 14.2.1 with `-march=native`. The ggml function `ggml_vec_dot_q4_K_q8_K_generic` compiles to AVX-512 VNNI + VBMI instructions despite its "generic" name — verified by `objdump`.

**Q: Why `perf_counter_ns` and not `perf stat` / `rdtsc`?**
A: We pin to core 0 (`taskset -c 0`), set governor to `performance`, and run enough repetitions (20,000 per round, 20 rounds) that timer resolution is irrelevant. Raw data available for independent analysis.

**Q: Does this work on Intel?**
A: Should work on any CPU with AVX-512F + VNNI (Intel Ice Lake+, Sapphire Rapids, etc.). Only tested on AMD Zen 4. Intel benchmarks welcome as PRs.

**Q: Is this "genetic programming" / "evolutionary AI"?**
A: The NF4 kernels: yes, instruction orderings were evolved over 16,460 mutations (swap, insert, delete, replace). The Q4_K kernel: no, it was hand-engineered through 7 iterations with AI assistance. The final prefetch hint was found by brute force (5,000 random injections, not evolution).

**Q: Why is the evolutionary system not published?**
A: The kernel code is open (Apache 2.0). The mutation engine and injection framework are in the private development repo, but the algorithm is fully documented above in "How the Evolutionary System Works" — genome representation, fitness function, mutation operators, selection strategy, and exact statistics. Nothing is hidden about the method, only the implementation code.

**Q: "Beats llama.cpp" — really?**
A: In our isolated microbenchmarks on AMD Zen 4, turbo7 outperformed ggml's `vec_dot_q4_K_q8_K_generic` (compiled with `-march=native`) by 1.5–12.6% depending on N. End-to-end gains in llama.cpp are smaller (3–8% prompt, 0–3.5% generation) because the dot product is one part of the pipeline. We do not claim Genesis is faster than llama.cpp as a whole.

## Requirements

- CPU: AVX-512 (AMD Zen 4+, Intel Skylake-X+ for NF4; Zen 4+ or Ice Lake+ for Q4_K VNNI)
- OS: Linux x86-64 (uses `mmap` with `PROT_EXEC`)
- Python: 3.10+
- NumPy

For Q4_K `.so` generation: `gcc`
For hybrid GPU/CPU examples: PyTorch, bitsandbytes, transformers, safetensors, NVIDIA GPU

## Troubleshooting

**"Illegal instruction"**: Your CPU lacks AVX-512. Check: `grep avx512 /proc/cpuinfo`

**Kernel returns zeros (NF4)**: Activations must be reordered with `reorder_activations()`. The kernel expects interleaved even/odd layout.

**Q4_K `.so` not loading**: Verify with `GENESIS_VERBOSE=1`. If no `[genesis]` message appears, the symbol name may not match your llama.cpp version. Check with `nm -D libggml-cpu.so | grep q4_K_q8_K`.

**Slow first call**: The first `compile_*` call JIT-compiles the kernel (milliseconds). Subsequent calls with the same parameters are cached.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Acknowledgments

- [ggml](https://github.com/ggerganov/ggml) / [llama.cpp](https://github.com/ggerganov/llama.cpp) — the Q4_K format and reference implementation we benchmark against
- [QLoRA](https://arxiv.org/abs/2305.14314) (Dettmers et al.) — the NF4 quantization scheme
- [uops.info](https://uops.info/) — instruction latency/throughput data for Zen 4

## Supporting This Project

If Genesis Kernel is useful to you:
- ⭐ Star this repo
- 🗣️ Mention it in your project or paper
- ☕ [Buy me a coffee](https://buymeacoffee.com/alarrama2)
- 💛 [Sponsor on GitHub](https://github.com/sponsors/Anuar81)

---

# Genesis Kernel (Español)

Generación de código máquina x86-64 en tiempo de ejecución para inferencia de LLMs en CPU. Sin compiladores, sin CUDA.

Dos familias de kernels:
- **NF4**: Dequantización fusionada + multiplicación de matrices (AVX-512 ZMM). Para offload MoE a CPU.
- **Q4_K**: Reemplazo compatible con ggml de `vec_dot_q4_K_q8_K` (AVX-512 VNNI en YMM). Drop-in para llama.cpp via `LD_PRELOAD`.

Construido con asistencia de IA + ingeniería inversa de ggml. Los kernels NF4 fueron descubiertos por evolución genética del orden de instrucciones. El kernel Q4_K fue diseñado a mano en 7 iteraciones; la optimización final (un prefetch hint) fue encontrada por inyección bruta de 5,000 instrucciones aleatorias.

## Reproducir en 5 Minutos

Requisitos: Linux x86-64, Python 3.10+, CPU con AVX-512 (AMD Zen 4+, Intel Skylake-X+).

```bash
# Verificar soporte AVX-512
grep -c avx512 /proc/cpuinfo  # debe ser > 0

# Instalar
git clone https://github.com/Anuar81/genesis-kernel.git
cd genesis-kernel
pip install -e .

# Correr tests
pytest tests/ -v

# Check rápido NF4
python -c "
from genesis_kernel import compile_best_nf4_matmul, quantize_nf4, reorder_activations
import numpy as np, time
M, K = 2048, 1536
w = np.random.randn(M*K).astype(np.float32)
nf4, scales = quantize_nf4(w)
act = reorder_activations(np.random.randn(K).astype(np.float32))
kernel = compile_best_nf4_matmul(M, K)
kernel(nf4, scales, act, M, K)
t0 = time.perf_counter()
for _ in range(100): kernel(nf4, scales, act, M, K)
t1 = time.perf_counter()
print(f'NF4 {M}x{K}: {(t1-t0)/100*1000:.3f} ms/call ({kernel._origin})')
"
```

## Dos Kernels

### NF4: Dequant Fusionada + Matmul

La inferencia CPU estándar dequantiza la matriz completa de pesos y después multiplica:
```
Tradicional:  pesos NF4 → dequantizar a float32 (M×K) → matmul → output
Genesis:      pesos NF4 + activaciones → dequant+dot fusionado → output (una pasada, todo en registros)
```

Para inferencia MoE híbrida GPU/CPU, esto significa copiar solo el vector resultado por PCIe en vez de la matriz completa de pesos.

Cuatro kernels por dimensión fueron descubiertos por evolución genética del orden de instrucciones x86 (16,460 mutaciones en 25 corridas). Los ordenamientos evolucionados explotan propiedades microarquitecturales de Zen 4: NOPs alineados a cache line, broadcast de escala anticipado, carga de activaciones en orden inverso de distancia, y ejecución intercalada de pipelines.

| Dimensiones | Caso de uso | Mejora vs baseline |
|---|---|---|
| 1024×2048 | 80B gate_up | +2.93% |
| 2048×512 | 80B down_proj | +19.25% |
| 3072×2048 | 30B gate_up | +3.90% |
| 2048×1536 | 30B down_proj | +2.89% |

Para otras dimensiones, se usa automáticamente el kernel base de 2 pipelines.

### Q4_K: Dot Product Compatible con ggml

Reemplazo drop-in de `ggml_vec_dot_q4_K_q8_K_generic` en llama.cpp. Usa AVX-512 VNNI (`VPDPWSSD`) en registros YMM (256-bit) para evitar la penalización de double-pumping de Zen 4 en operaciones de 512-bit.

Siete iteraciones, cada una atacando un cuello de botella específico:

| Versión | ISA | ns/blk (N=1024) | vs ggml | Qué cambió |
|---|---|---|---|---|
| turbo3 | AVX2 | 13.6 | 2.39× más lento | Primer kernel funcional |
| turbo4 | AVX-512 F | 12.27 | 2.13× más lento | Registros YMM, 4 acumuladores |
| turbo5 | AVX-512 F | 5.90 | 1.026× más lento | Reescritura del prólogo |
| turbo6 | AVX-512 F+VNNI | 5.71 | 1.007× más lento | VPDPWSSD reemplaza VPMADDUBSW+VPMADDWD |
| **turbo7** | **AVX-512 F+VNNI** | **5.51** | **0.985× (gana)** | **+1 PREFETCHNTA (fuerza bruta)** |
| ggml | VNNI+VBMI | 5.59 | 1.00× | Compilado con `-march=native` |

Turbo7 = turbo6 + `PREFETCHNTA [R9+358]` en posición 19 del inner loop. R9+358 apunta a la 2da cache line del siguiente bloque Q8_K. El prefetcher de hardware trae líneas secuenciales desde el inicio del bloque; este prefetch manual apunta a una línea específica que el HW todavía no alcanzó.

Encontrado por fuerza bruta: 5,000 inyecciones aleatorias → verificación en 3 etapas → 29 mejoras confirmadas → la mejor se horneó en turbo7.

El mutador evolutivo (reordenar las 33 ops existentes) encontró solo +0.56% — el inner loop ya estaba cerca del óptimo para este ISA.

## Benchmarks

Todas las mediciones en AMD Ryzen 9 7900 (Zen 4, AVX-512), 32GB DDR5, EndeavourOS.

### Microbenchmarks (lo que Genesis mejora directamente)

**NF4** — latencia por experto, Genesis vs bitsandbytes CPU:

| Métrica | Genesis AVX-512 | bitsandbytes CPU | Speedup |
|---|---|---|---|
| Latencia por experto (2048×1536) | 0.15 ms | 24.8 ms | 165× |

**Q4_K** — benchmark aislado A/B, turbo7 vs ggml (llama.cpp b8184), 20 rondas × 20,000 reps, `taskset -c 0`, governor=performance:

| N (bloques) | turbo7 ns/blk | ggml ns/blk | Delta | Wins |
|---|---|---|---|---|
| 32 (LLaMA 70B token gen) | 32.05 | 36.59 | +12.56% | 20/20 |
| 128 | 11.69 | 12.85 | +9.05% | 20/20 |
| 512 | 6.40 | 6.56 | +2.52% | 20/20 |
| 1024 | 5.51 | 5.59 | +1.46% | 20/20 |

Correctitud: bit-exact vs ggml (`diff = 0.000000e+00` para N=1 a N=1024).

### End-to-End (contexto — depende del modelo y ratio de offload)

**NF4** — inferencia MoE con offload híbrido GPU/CPU (RTX 4090 + CPU):

| Modelo | Expertos | VRAM | Velocidad | Capas CPU |
|---|---|---|---|---|
| Qwen3-Coder-30B-A3B | 128 (8 activos) | 13.4 GB | 5.7 tok/s | 8 |
| Qwen3-Next-80B-A3B | 512 (10 activos) | 20.7 GB | 2.7–3.3 tok/s | 24 |

El modelo de 80B no entra en una sola RTX 4090 sin offload a CPU.

**Q4_K** — llama.cpp con `LD_PRELOAD=./genesis_turbo7.so` (Qwen3-Coder-30B, b8184):

| Config | ggml stock | genesis turbo7 | Delta |
|---|---|---|---|
| Prompt tok/s (-c 2048) | 97.5 | 100.4 | +3.0% |
| Prompt tok/s (-c 4096, -t 12) | 89.7 | 97.2 | +8.4% |
| Gen tok/s (-c 4096, -t 12) | 40.4 | 41.8 | +3.5% |

Genesis mejora una operación (dot product CPU). El impacto end-to-end depende de cuánto del workload corre en CPU. El 30B con la mayoría de capas en GPU muestra ganancias modestas; un modelo 100% CPU mostraría más.

## Instalación

Desde source (recomendado):
```bash
git clone https://github.com/Anuar81/genesis-kernel.git
cd genesis-kernel
pip install -e .
```

Dependencias: NumPy. Los kernels usan solo stdlib de Python (`ctypes`, `mmap`, `struct`) para emitir y ejecutar código máquina.

Para la integración Q4_K con llama.cpp, también se necesita `gcc`.

## Uso

### NF4 Matmul

```python
import numpy as np
from genesis_kernel import compile_best_nf4_matmul, quantize_nf4, reorder_activations

# Cuantizar pesos a NF4
weights = np.random.randn(2048 * 1536).astype(np.float32)
nf4_weights, scales = quantize_nf4(weights)

# Reordenar activaciones (requerido — el kernel espera layout intercalado par/impar)
activations = np.random.randn(1536).astype(np.float32)
act_reord = reorder_activations(activations)

# Compilar y ejecutar
kernel = compile_best_nf4_matmul(M=2048, K=1536)
output = kernel(nf4_weights, scales, act_reord, M=2048, K=1536)
```

Para un ejemplo completo con inferencia MoE híbrida GPU/CPU, ver [`examples/hybrid_moe_offload.py`](examples/hybrid_moe_offload.py).

### Q4_K en llama.cpp

```bash
# Generar la biblioteca compartida
python -c "from genesis_kernel import generate_turbo7_so; generate_turbo7_so('.')"

# Correr llama.cpp con kernel Genesis
LD_PRELOAD=./genesis_turbo7.so llama-cli -m model.gguf -p "Hola" -n 128

# Opcional: modo verbose para confirmar que Genesis cargó
GENESIS_VERBOSE=1 LD_PRELOAD=./genesis_turbo7.so llama-cli -m model.gguf -p "Hola" -n 128
# Debería imprimir: [genesis] turbo7 loaded: 1078 bytes at 0x...
```

**Importante: puede ser necesario desactivar repack.** llama.cpp tiene un sistema de repack que convierte `block_q4_K` a formato intercalado `block_q4_Kx8`. Cuando está activo, algunos paths no pasan por `_generic` y nuestro kernel no se ejecuta. En nuestras pruebas tuvimos que desactivarlo:

```bash
# En el directorio de llama.cpp, desactivar repack:
sed -i '/^ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void) {/,/^}/ c\
ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void) {\
    return NULL;\
}' ggml/src/ggml-cpu/repack.cpp

# Recompilar
cmake --build build -j$(nproc)

# Ahora LD_PRELOAD intercepta todos los dot products Q4_K:
LD_PRELOAD=./genesis_turbo7.so llama-cli -m model.gguf -p "Hola" -n 128

# Para restaurar el repack original:
git checkout ggml/src/ggml-cpu/repack.cpp
cmake --build build -j$(nproc)
```

## Cómo Funciona

Genesis genera código máquina x86-64 en tiempo de ejecución usando un emitter en Python (`x86_emitter.py`). Sin LLVM, sin GCC, sin toolchain de compilación — solo `struct.pack` para codificar bytes de instrucciones, `mmap` con `PROT_EXEC` para hacerlos ejecutables, y `ctypes` para llamarlos.

El emitter soporta AVX-512 Foundation, VNNI, e instrucciones codificadas en VEX. Cada kernel se define como una secuencia de operaciones que el emitter traduce a bytes crudos. El kernel Q4_K (turbo7) son 1,078 bytes / 34 operaciones de inner loop.

### ¿Por qué no usar un compilador?

Porque el objetivo es buscar en el espacio de instrucciones programáticamente. Un compilador toma decisiones fijas sobre orden de instrucciones, asignación de registros y scheduling. Genesis representa el kernel como una secuencia mutable de operaciones, permitiendo:
- Evolución genética del orden de instrucciones (kernels NF4)
- Inyección bruta de instrucciones hint (Q4_K turbo7)
- Control directo sobre cada byte emitido

## Cómo Funciona el Sistema Evolutivo

La fábrica evolutiva no es open-source, pero acá está exactamente cómo funciona — sin cajas negras.

### Evolución de kernels NF4

**Representación del genoma**: El inner loop de cada kernel es una lista de llamadas a funciones del emitter (ej: `_emit_vpmovzxbd`, `_emit_vpermps`, `_emit_vfmadd231ps`). Los ordenamientos evolucionados se pueden ver directamente en `nf4_kernel.py` — las funciones `_baked_kernel_*` muestran cada instrucción en orden.

**Función de fitness**: Tiempo de ejecución real. `time.perf_counter_ns()`, mediana de 1,000 repeticiones, afinidad a core 0 con `taskset`. Menor es mejor. Sin métricas proxy.

**Operadores de mutación** (4 tipos):
- `swap`: Elegir dos instrucciones al azar, intercambiar posiciones
- `insert`: Agregar un NOP o prefetch en posición aleatoria
- `delete`: Eliminar una instrucción (solo NOPs/prefetches, nunca ops de cómputo)
- `replace`: Cambiar una instrucción por otra del mismo tipo

**Selección**: Estrategia evolutiva (1+1). Mutar el mejor actual → benchmarkear → si es más rápido, reemplazar. Sin población, sin crossover. Hill climbing simple.

**Estadísticas**: 25 corridas, 16,460 mutaciones evaluadas, 4 ganadores horneados (uno por dimensión). Mejor mejora: +19.25% (2048×512).

### Inyección bruta Q4_K

No es evolución — búsqueda aleatoria pura. Para cada intento:
1. Elegir tipo de instrucción al azar (prefetcht0/t1/t2/nta, multi_nop, lfence)
2. Elegir posición aleatoria en el inner loop de 33 ops
3. Elegir parámetros aleatorios (offset 0–4096 para prefetch, 1–15 bytes para NOP)
4. Insertar, compilar, ejecutar
5. Si crashea o da resultado incorrecto → descartar
6. Si es más rápido → verificación en 3 etapas: medir → re-medir ×5 → A/B ×20 rondas
7. Si sobrevive las 3 etapas → guardar en `hallazgos_inyeccion.jsonl`

5,000 intentos → 29 mejoras confirmadas → la mejor (PREFETCHNTA [R9+358] en posición 19) horneada en turbo7.

## Preguntas Frecuentes

**P: ¿Puedo reproducir los números del benchmark?**
R: Sí. Necesitás una CPU con AVX-512 (idealmente Zen 4 para resultados comparables). El benchmark Q4_K requiere llama.cpp b8184 compilado con `cmake -B build -DGGML_NATIVE=ON && cmake --build build`. Afinidad a core 0 con `taskset`, governor en `performance`, 20 rondas × 20,000 reps. Mediana de medianas.

**P: ¿Qué commit de llama.cpp?**
R: b8184. GCC 14.2.1 con `-march=native`. La función ggml `ggml_vec_dot_q4_K_q8_K_generic` compila a instrucciones AVX-512 VNNI + VBMI a pesar de su nombre "generic" — verificado con `objdump`.

**P: ¿Funciona en Intel?**
R: Debería funcionar en cualquier CPU con AVX-512F + VNNI (Intel Ice Lake+, Sapphire Rapids, etc.). Solo testeado en AMD Zen 4. PRs con benchmarks de Intel son bienvenidos.

**P: ¿Es "programación genética" / "IA evolutiva"?**
R: Los kernels NF4: sí, los ordenamientos de instrucciones fueron evolucionados en 16,460 mutaciones. El kernel Q4_K: no, fue diseñado a mano en 7 iteraciones con asistencia de IA. El prefetch final fue encontrado por fuerza bruta (5,000 inyecciones aleatorias, no evolución).

**P: ¿Por qué el sistema evolutivo no se publica?**
R: El código de los kernels es abierto (Apache 2.0). El motor de mutaciones y el framework de inyección están en el repo privado de desarrollo, pero el algoritmo está completamente documentado arriba en "Cómo Funciona el Sistema Evolutivo" — representación del genoma, función de fitness, operadores de mutación, estrategia de selección, y estadísticas exactas. Nada está oculto sobre el método, solo el código de implementación.

**P: ¿"Le gana a llama.cpp" — en serio?**
R: En nuestros microbenchmarks aislados en AMD Zen 4, turbo7 superó a `vec_dot_q4_K_q8_K_generic` de ggml (compilado con `-march=native`) por 1.5–12.6% dependiendo de N. Las ganancias end-to-end en llama.cpp son menores (3–8% prompt, 0–3.5% generación) porque el dot product es una parte del pipeline. No afirmamos que Genesis sea más rápido que llama.cpp como un todo.

## Requisitos

- CPU: AVX-512 (AMD Zen 4+, Intel Skylake-X+ para NF4; Zen 4+ o Ice Lake+ para Q4_K VNNI)
- OS: Linux x86-64 (usa `mmap` con `PROT_EXEC`)
- Python: 3.10+
- NumPy

Para generación del `.so` Q4_K: `gcc`
Para ejemplos de GPU/CPU híbrido: PyTorch, bitsandbytes, transformers, safetensors, GPU NVIDIA

## Solución de Problemas

**"Illegal instruction"**: Tu CPU no tiene AVX-512. Verificar: `grep avx512 /proc/cpuinfo`

**El kernel devuelve ceros (NF4)**: Las activaciones deben reordenarse con `reorder_activations()`. El kernel espera layout intercalado par/impar.

**El `.so` Q4_K no carga**: Verificar con `GENESIS_VERBOSE=1`. Si no aparece mensaje `[genesis]`, el nombre del símbolo puede no coincidir con tu versión de llama.cpp. Verificar con `nm -D libggml-cpu.so | grep q4_K_q8_K`.

**Primera llamada lenta**: La primera llamada a `compile_*` compila el kernel JIT (milisegundos). Las llamadas siguientes con los mismos parámetros usan cache.

## Licencia

Apache 2.0 — ver [LICENSE](LICENSE).

## Reconocimientos

- [ggml](https://github.com/ggerganov/ggml) / [llama.cpp](https://github.com/ggerganov/llama.cpp) — el formato Q4_K y la implementación de referencia contra la que benchmarkeamos
- [QLoRA](https://arxiv.org/abs/2305.14314) (Dettmers et al.) — el esquema de cuantización NF4
- [uops.info](https://uops.info/) — datos de latencia/throughput de instrucciones para Zen 4

## Apoyar Este Proyecto

Si Genesis Kernel te resulta útil:
- ⭐ Darle estrella al repo
- 🗣️ Mencionarlo en tu proyecto o paper
- ☕ [Invitame un café](https://buymeacoffee.com/alarrama2)
- 💛 [Sponsorear en GitHub](https://github.com/sponsors/Anuar81)
