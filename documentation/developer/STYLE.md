# Style Guide

Conventions for OCTproEngine. Keep changes consistent with the code around them.

## Formatting
- Indent with **tabs**, not spaces.
- Match the surrounding file (there is no clang-format; formatting is by convention).

## Naming
- Types: `PascalCase` — `ProcessorConfiguration`, `CpuBackend`.
- Methods and variables: `camelCase` — `enableResampling`, `signalLength`.
- Enum values and compile-time constants: `UPPER_CASE` — `DataType::UINT16`.
- Access members through `this->`. Never use `m_` or other prefixes.
- Python bindings use `snake_case`.

## Comments
- **Never remove or change existing comments.** Add new ones instead.
- Comment the *why*, not the obvious *what*.

## Structure
- Public headers live in `include/`; everything else is internal.
- Hide implementation behind the pImpl idiom: a private `struct Impl` reached via `this->impl`.
- Keep the four backends (CPU, CUDA, OpenCL, Vulkan) behaviourally consistent — a change to one usually needs the same change to the others.

## Errors
- Throw `std::invalid_argument` for bad arguments, `std::runtime_error` for runtime failures.
- Validate before mutating state, so a rejected call leaves no partial changes.

## Always
- Must build and run on **Windows, Linux, and Jetson Nano**.
- **High performance, low memory.** Keep the per-buffer hot path allocation-free; do setup in `initialize()`.
- `process(inputBuffer)` runs **asynchronously**. Fill input via `getNextAvailableBuffer()`, which blocks until a buffer is free.

## Commits
`<tag>: <present-tense description>` — see [COMMITS.md](COMMITS.md).
