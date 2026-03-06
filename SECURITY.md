# Security Policy

## What this project does

Genesis Kernel generates x86-64 machine code at runtime and executes it via `mmap` with `PROT_EXEC`. This is inherently security-sensitive — it's JIT compilation without a compiler.

## Scope

The generated code runs with the same privileges as the calling Python process. There is no sandboxing. If you run Genesis, you are trusting it to emit correct x86 instructions.

## What we do

- All emitted code is deterministic: the same inputs always produce the same bytes
- No user-supplied data is ever interpreted as instructions
- No network access, no file system writes (except the optional `.so` generation)
- The `.so` generation (`generate_turbo7_so`) writes a C file and compiles it with `gcc` — review the generated `.c` file if you want to audit it

## What we don't do

- We don't sandbox the JIT code
- We don't sign or verify the emitted bytes
- We don't protect against a modified `x86_emitter.py` emitting malicious code

## Reporting a vulnerability

If you find a security issue, please email alarrama@gmail.com instead of opening a public issue.

We'll respond within 7 days and work with you on a fix before any public disclosure.

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | ✅        |
| 0.1.x   | ❌        |
