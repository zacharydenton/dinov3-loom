# Working notes

## `where [range(...)]` on a launch argument is part of HRX's dispatch ABI

A kernel written like the ggml-hrx corpus does:

```
} launch(%token_count: index, %input: buffer, %output: buffer) where [range(%token_count, 1, 4096)] {
```

compiles fine, and `loom-compile --backend=amdgpu-hal` emits a clean HSACO with the
expected kernarg layout. Launched through `hipModuleLaunchKernel` it dies with:

```
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The agent attempted to access memory
beyond the largest legal address.
```

Bisected to exactly that clause. The same kernel with the clause removed runs
correctly; a kernel that declares the `index` argument but never *uses* it also runs
(the argument is dead-code-eliminated before the constraint matters). The
disassembly of the failing kernel shows the guard lowered to a branch that clobbers
`s[0:1]` — the kernarg segment pointer — via a wave64 `s_and_saveexec_b64` in a
wave32 kernel.

Read: those constraints are a contract HRX's own dispatch layer upholds when it
builds the dispatch packet, not a free-standing hint. Kernels meant to be launched
by an ordinary HIP/CUDA host must not carry them. Everything here is written without
`where` clauses on launch arguments; `index.assume` inside the body is fine.

Worth reporting upstream — silently miscompiling rather than refusing to compile is
the bad failure mode.

## Config values must be bound at compile time

`config.get` results carry a `no_ordinary_uses` constraint. Compiling without
`--config=<key>=<value>` fails with:

```
target 'amdgpu-rdna3-5' ... rejected 'config.get' value 'result' ... constraint
'no_ordinary_uses' is not satisfied
```

So `hidden_size` and `epsilon` are specialization constants, not runtime arguments.
That is the point of the design — one HSACO per shape, JIT-compiled in ~2 ms — but it
means the host has to compile per configuration rather than pass a struct.

## `iree-test-loom` segfaults at teardown

`iree-test-loom <kernel> --device=amdgpu` runs the `check.case` samples, then
segfaults inside `hsa_executable_destroy` in the ROCm 7.14 loader, sometimes before
the JSON result reaches stdout. That is why validation here goes through
`host/loomrun` and NumPy instead of the `check` dialect. The `check.case` blocks are
kept in the kernels because they document intent and `loom-format --check` still
validates them.

## Arch's ROCm aborts under HRX

`hsa-rocr` on Arch is built with `-D_GLIBCXX_ASSERTIONS` (from `/etc/makepkg.conf`),
and HSA's completion thread trips a `std::vector` bounds assert as soon as HRX opens
a queue. `scripts/env.sh` puts `~/.local/rocm-hrx` (ROCm 7.14, extracted from the
kyuz0 Strix Halo toolbox) ahead of it. `host/loomrun` links real ROCm HIP and is
unaffected.
