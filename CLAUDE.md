# CLAUDE.md — Instructions for Claude Code

This file is read automatically at the start of every session in this repository.
Use it to set preferences, conventions, and constraints.

This repository implements an equivariant operator parameterization, primarily
intended for Hamiltonian learning.

Implementation plans and notes are stored in the `agents/` folder.

---

## Language & Style

- This is a Julia project. All source code is Julia unless otherwise noted.
- Follow existing code style in each file rather than imposing a uniform style
  across the whole codebase.
- Add docstrings to most functions, but keep them brief. If I need more
  clarification I will ask.
- Do not refactor surrounding code when fixing a bug or adding a feature — keep
  changes focused.
- Do not add or remove any whitespaces except in the lines you are editing already.
- Try to keep lines under 80 characters, with 92 characters the absolute maximum.

---

## Workflow

- Do not commit changes unless explicitly asked to.
- Do not push to remote unless explicitly asked to.
- Before editing a file, read it first.
- Prefer editing existing files over creating new ones.
- Try to avoid code duplication. Create utility functions whenever appropriate
  to share functionality.

---

## Julia-specific

- Prefer in-place (`!`) variants of functions when performance matters.
- Type annotations are only for dispatch, not performance. Keep code as
  type-agnostic as possible.
- It is ok to add new dependencies if they are needed, but flag this so it can
  be reviewed.
- GPU kernels use KernelAbstractions — do not introduce CUDA.jl-specific code
  in shared paths.
- It is ok to use features from the latest Julia versions. If you do, provide
  brief summaries and comments.

---

## Commit style

- Use short imperative commit messages (e.g. "Remove dead simpletrans.jl").
- Do not add "Co-Authored-By" lines unless asked.

---

## What NOT to do

- Do not silently change behaviour — if a refactor changes observable output,
  flag it before proceeding.
- Do not delete files without confirmation, even if they appear unused.
- Do not open pull requests without being asked.
