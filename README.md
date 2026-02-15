# Genesis Kernel

AVX-512 fused NF4 dequantization + matrix multiplication for local LLM inference.

Genesis generates x86-64 machine code at runtime — no compilers, no CUDA required for CPU layers. It fuses weight dequantization with the dot product in a single pass, avoiding materialization of the full weight matrix. Works with any NF4-quantized model — dense or MoE. Particularly effective for MoE CPU offload on a single GPU.

## Results

Measured on AMD Ryzen 9 7900 (Zen 4, AVX-512) + NVIDIA RTX 4090 (24GB):

| Metric | Genesis AVX-512 | bitsandbytes CPU | Speedup |
|---|---|---|---|
| Per-expert latency | 0.15 ms | 24.8 ms | **165x** |
| 30B MoE (8 layers on RAM) | 5.7 tok/s, 13.4GB VRAM | — | — |
| 80B MoE (24 layers on RAM) | 2.7–3.3 tok/s, 20.7GB VRAM | impossible | — |

The 80B model (Qwen3-Next-80B-A3B) does not fit in a single RTX 4090 without CPU offload. With Genesis, it runs at conversational speed.

## Installation

```bash
pip install genesis-kernel
```

Or from source:

```bash
git clone https://github.com/Anuar81/genesis-kernel.git
cd genesis-kernel
pip install -e .
```

## Quick Start

```python
import numpy as np
from genesis_kernel import compile_best_nf4_matmul, quantize_nf4, reorder_activations

# Quantize weights to NF4
weights = np.random.randn(2048 * 1536).astype(np.float32)
nf4_weights, scales = quantize_nf4(weights)

# Prepare activations
activations = np.random.randn(1536).astype(np.float32)
act_reord = reorder_activations(activations)

# Compile and run kernel
kernel = compile_best_nf4_matmul(M=2048, K=1536)
output = kernel(nf4_weights, scales, act_reord, M=2048, K=1536)
```

For a complete example loading a real MoE model with hybrid GPU/CPU inference, see [`examples/hybrid_moe_offload.py`](examples/hybrid_moe_offload.py).

## How It Works

Standard CPU inference dequantizes the full weight matrix, then multiplies:

```
Traditional:  NF4 weights → dequantize to float32 (M×K matrix) → matmul → output
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
              This intermediate matrix is huge and slow to produce
```

Genesis fuses both steps into one AVX-512 kernel:

```
Genesis:  NF4 weights + activations → fused dequant+dot → output
          No intermediate matrix. One pass. All in registers.
```

For hybrid GPU/CPU MoE inference, this means copying only the result vector over PCIe instead of the full weight matrix — roughly 1000x less data transfer per expert.

## The Story: Genetically Evolved x86 Kernels

The kernels in this package were not written by hand. They were discovered through genetic evolution of x86 instruction orderings.

Starting from a baseline 2-pipeline AVX-512 kernel, an evolutionary system:

1. Represented the kernel's inner loop as a sequence of x86 operations
2. Applied random mutations: swaps, insertions, deletions, replacements
3. Benchmarked each mutant on real hardware (AMD Zen 4)
4. Selected the fastest variants and repeated

Over 25 evolutionary runs, the system evaluated 16,460 mutations across 8 mutation types. The best evolved kernels outperform the hand-tuned baseline by up to 19.25%.

### What evolution discovered

The evolved instruction orderings exploit Zen 4 microarchitectural properties that would be difficult to find by hand:

- **NOP alignment**: Inserting NOPs at specific positions to align subsequent instructions to cache line boundaries, improving instruction fetch throughput
- **Early scale broadcast**: Moving the scale broadcast 9 positions earlier than the baseline, giving the memory subsystem time to fulfill the load before the value is needed
- **Reverse activation loading**: Loading activations in reverse distance order, which the hardware prefetcher handles more efficiently
- **Interleaved computation**: Replacing a multiply with a NOP and reordering surrounding instructions to reduce port contention between the two pipelines

### Baked kernel dimensions

Four dimension-specific kernels are included, matching common MoE projection sizes:

| Dimensions | Use case | Improvement vs baseline |
|---|---|---|
| 1024×2048 | 80B gate_up projection | +2.93% |
| 2048×512 | 80B down projection | +19.25% |
| 3072×2048 | 30B gate_up projection | +3.90% |
| 2048×1536 | 30B down projection | +2.89% |

For other dimensions, the base 2-pipeline kernel is used automatically.

### Why the evolutionary system is not published

The kernel code is open (Apache 2.0). The evolutionary factory — the mutation engine, the fitness evaluator, the learned mutation selector — is not included in this repository. The kernels here are the output of that process, baked into functions that emit optimized x86 directly.

## Tested Models

| Model | Experts | VRAM | Speed | RAM layers |
|---|---|---|---|---|
| Qwen3-Coder-30B-A3B-Instruct | 128 (8 active) | 13.4 GB | 5.7 tok/s | 8 |
| Qwen3-Next-80B-A3B-Instruct | 512 (10 active) | 20.7 GB | 2.7–3.3 tok/s | 24 |

## Requirements

- CPU with AVX-512 support (AMD Zen 4+, Intel Skylake-X+)
- Python 3.10+
- NumPy
- Linux (x86-64) — uses `mmap` with `PROT_EXEC` for JIT

For hybrid GPU/CPU inference (examples):
- NVIDIA GPU with CUDA
- PyTorch, bitsandbytes, transformers, safetensors

## Troubleshooting

**"Illegal instruction" crash**: Your CPU does not support AVX-512. Check with:
```bash
grep avx512 /proc/cpuinfo
```

**Kernel returns zeros**: Ensure activations are reordered with `reorder_activations()` before passing to the kernel. The kernel expects interleaved even/odd layout.

**Slow performance**: The first call to `compile_best_nf4_matmul()` JIT-compiles the kernel (a few milliseconds). Subsequent calls with the same dimensions are cached.

**OOM with large models**: For 80B models, 64GB system RAM is recommended. With 32GB, the OS may OOM-kill the process during weight loading.

## Supporting This Project

If Genesis Kernel is useful to you:
- ⭐ Star this repo — it helps with visibility
- 🗣️ Mention it in your project or paper
- ☕ [Buy me a coffee](https://buymeacoffee.com/alarrama2)
- 💛 [Sponsor on GitHub](https://github.com/sponsors/Anuar81)

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

# Genesis Kernel (Español)

Kernels AVX-512 de dequantización NF4 + multiplicación de matrices fusionada para inferencia local de LLMs.

Genesis genera código máquina x86-64 en tiempo de ejecución — sin compiladores, sin CUDA para las capas en CPU. Fusiona la dequantización de pesos con el producto punto en una sola pasada, evitando materializar la matriz completa de pesos. Funciona con cualquier modelo cuantizado a NF4 — denso o MoE. Particularmente efectivo para offload MoE a CPU en una sola GPU.

## Resultados

Medido en AMD Ryzen 9 7900 (Zen 4, AVX-512) + NVIDIA RTX 4090 (24GB):

| Métrica | Genesis AVX-512 | bitsandbytes CPU | Speedup |
|---|---|---|---|
| Latencia por experto | 0.15 ms | 24.8 ms | **165x** |
| 30B MoE (8 capas en RAM) | 5.7 tok/s, 13.4GB VRAM | — | — |
| 80B MoE (24 capas en RAM) | 2.7–3.3 tok/s, 20.7GB VRAM | imposible | — |

El modelo de 80B (Qwen3-Next-80B-A3B) no entra en una sola RTX 4090 sin offload a CPU. Con Genesis, corre a velocidad conversacional.

## Instalación

```bash
pip install genesis-kernel
```

## Uso Rápido

```python
import numpy as np
from genesis_kernel import compile_best_nf4_matmul, quantize_nf4, reorder_activations

# Cuantizar pesos a NF4
weights = np.random.randn(2048 * 1536).astype(np.float32)
nf4_weights, scales = quantize_nf4(weights)

# Preparar activaciones
activations = np.random.randn(1536).astype(np.float32)
act_reord = reorder_activations(activations)

# Compilar y ejecutar kernel
kernel = compile_best_nf4_matmul(M=2048, K=1536)
output = kernel(nf4_weights, scales, act_reord, M=2048, K=1536)
```

Para un ejemplo completo cargando un modelo MoE real con inferencia híbrida GPU/CPU, ver [`examples/hybrid_moe_offload.py`](examples/hybrid_moe_offload.py).

## La Historia: Kernels x86 Evolucionados Genéticamente

Los kernels de este paquete no fueron escritos a mano. Fueron descubiertos por evolución genética del orden de instrucciones x86.

Partiendo de un kernel base AVX-512 de 2 pipelines, un sistema evolutivo:

1. Representó el inner loop del kernel como una secuencia de operaciones x86
2. Aplicó mutaciones aleatorias: intercambios, inserciones, eliminaciones, reemplazos
3. Benchmarkeó cada mutante en hardware real (AMD Zen 4)
4. Seleccionó los más rápidos y repitió

En 25 corridas evolutivas, el sistema evaluó 16,460 mutaciones de 8 tipos distintos. Los mejores kernels evolucionados superan al baseline optimizado a mano por hasta 19.25%.

### Lo que la evolución descubrió

Los ordenamientos de instrucciones evolucionados explotan propiedades microarquitecturales de Zen 4 que serían difíciles de encontrar a mano:

- **Alineación con NOPs**: Insertar NOPs en posiciones específicas para alinear instrucciones a líneas de cache
- **Broadcast de escala anticipado**: Mover el broadcast de escala 9 posiciones antes, dándole tiempo al subsistema de memoria
- **Carga inversa de activaciones**: Cargar activaciones en orden inverso de distancia, que el prefetcher maneja mejor
- **Cómputo intercalado**: Reemplazar un multiply por NOP y reordenar instrucciones para reducir contención de puertos

### Kernels horneados por dimensión

| Dimensiones | Caso de uso | Mejora vs baseline |
|---|---|---|
| 1024×2048 | 80B gate_up | +2.93% |
| 2048×512 | 80B down_proj | +19.25% |
| 3072×2048 | 30B gate_up | +3.90% |
| 2048×1536 | 30B down_proj | +2.89% |

Para otras dimensiones, se usa automáticamente el kernel base de 2 pipelines.

### Por qué el sistema evolutivo no se publica

El código de los kernels es abierto (Apache 2.0). La fábrica evolutiva — el motor de mutaciones, el evaluador de fitness, el selector de mutaciones aprendido — no está incluida en este repositorio. Los kernels aquí son la salida de ese proceso, horneados en funciones que emiten x86 optimizado directamente.

## Requisitos

- CPU con soporte AVX-512 (AMD Zen 4+, Intel Skylake-X+)
- Python 3.10+
- NumPy
- Linux (x86-64) — usa `mmap` con `PROT_EXEC` para JIT

## Solución de Problemas

**Crash "Illegal instruction"**: Tu CPU no soporta AVX-512. Verificar con:
```bash
grep avx512 /proc/cpuinfo
```

**El kernel devuelve ceros**: Asegurate de reordenar las activaciones con `reorder_activations()` antes de pasarlas al kernel. El kernel espera layout intercalado par/impar.

**Rendimiento lento**: La primera llamada a `compile_best_nf4_matmul()` compila el kernel JIT (unos milisegundos). Las llamadas siguientes con las mismas dimensiones usan cache.

**OOM con modelos grandes**: Para modelos de 80B, se recomiendan 64GB de RAM. Con 32GB, el OS puede matar el proceso por OOM durante la carga de pesos.

## Apoyar Este Proyecto

Si Genesis Kernel te resulta útil:
- ⭐ Darle estrella al repo — ayuda con la visibilidad
- 🗣️ Mencionarlo en tu proyecto o paper
- ☕ [Invitame un café](https://buymeacoffee.com/alarrama2)
- 💛 [Sponsorear en GitHub](https://github.com/sponsors/Anuar81)

## Licencia

Apache 2.0 — ver [LICENSE](LICENSE).
