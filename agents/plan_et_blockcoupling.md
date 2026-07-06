# Plan — Upstream `BlockCoupling` into EquivariantTensors, drop ACEpotentials

Status: **draft for review; no ET changes made yet.**

Goal: move the Wigner–Eckart block coupling (`src/linear2c/coupling.jl`,
PR #4 / commit `4aca4cc`) into `EquivariantTensors.O3`, where it naturally
belongs, and refactor ACEoperators to consume it from ET. Combined with the
in-progress radials move (ET branch `radials-move-impl`, design in ET
`agents/radials.md`), this removes the ACEpotentials dependency entirely.

All claims below were verified against the local checkouts
(`~/claude_projects/EquivariantTensors.jl`, this repo) — references inline.

---

## 0. Why single-track on ET `main` (no 0.4.x backport)

* ET `main` is at version `0.5.0` with breaking changes since the registered
  `0.4.3` (63 commits; e.g. `NTtransformST → DPTransform`,
  `QuadO3 → QuadSO3`).
* ACEpotentials 0.10.1 pins `EquivariantTensors = "0.4.3"` (i.e. `< 0.5`), so
  ACEoperators cannot use ET-main while ACEpotentials remains a dependency.
* The radials move eliminates that dependency, so the only consumer constraint
  disappears. A `0.4.4` backport branch would be a bridge with no destination.

The only inputs `BlockCoupling` needs from ET — `O3.cg(l,m,l',m',λ,μ,⋅)` and
`O3.Ctran(l)` — exist unchanged on `main` (`src/O3/O3_utils.jl:44,165`), so
the code transfers verbatim.

---

## Phase A — ET PR: `O3.BlockCoupling` (branch `co/block-coupling` off `main`)

Independent of, and parallel to, the radials PR. Purely additive.

### A.1 `src/O3/O3_block_coupling.jl` (new)

Contents = ACEoperators `src/linear2c/coupling.jl` @ `4aca4cc`, adapted:

* Lives inside `module O3`: add
  `include("O3_block_coupling.jl")` to `src/O3/O3.jl` after
  `include("O3_utils.jl")` (needs `Ctran`, `cg`). `using LinearAlgebra` is
  already in scope in `O3` (provides `I`, `mul!`); the local
  `using EquivariantTensors: O3` / `using LinearAlgebra: I, mul!` lines are
  dropped.
* API moved as-is (names verified collision-free across ET `src/`):
  - `cg_block([T=Float64], l, l', λ) -> Array{T,3}` — dense real CG block,
    complex CG mapped to the real-SH basis via `Ctran`, even channels real /
    odd channels imaginary part; orthonormal slices.
  - `struct BlockCoupling{T}` with `l, lp, λs::UnitRange{Int},
    C::Vector{Array{T,3}}`; constructors `BlockCoupling([T,] l, l')`.
  - `couple(bc, X)`, `decouple(bc, Xλ)` — complete orthonormal change of
    basis between an `(2l+1)×(2l'+1)` block and its λ-components; pure
    mat-vec products (Zygote-differentiable, no mutation).
  - `transform_λ(bc, λ, v)` and accumulating in-place `transform_λ!(X, bc, λ, v)`.
  - `channel_parity(l, l', λ) -> :even/:odd` classifier.
* **No new exports.** `O3` currently exports only `coupling_coeffs`
  (`src/O3/O3.jl:9`); consumers use qualified `O3.BlockCoupling` etc. The
  export surface can be revisited when the API is declared public.
* File header: keep the construction/convention notes (the complex→real
  mapping, the even/odd `l+l'+λ` channel structure, why this is NOT the
  Gaunt coupling of `coupling_coeffs`/`cgmatrix` and carries both parities);
  reworded self-contained — drop ACEoperators `twocenter.md` §-references
  and the "candidate for upstreaming" note.
* Docstrings: kept; they appear automatically in ET docs
  (`docs/src/docstrings.md` uses `@autodocs Modules = [EquivariantTensors,
  EquivariantTensors.O3]`). Optionally add the main entry points to the
  curated `docs/src/api.md` — suggest deferring until declared public API.

### A.2 `test/O3/test_block_coupling.jl` (new)

Contents = ACEoperators `test/linear2c/test_coupling.jl` (the full Stage 0
suite: all admissible channels present with both parities; parametric element
type / Float32; complete orthonormal change of basis `WWᵀ = WᵀW = I`;
couple/decouple exact mutual inverses; SO(3) intertwining
`D^l X (D^{l'})ᵀ ↔ D^λ v`; improper-rotation sign structure (even channels
true λ-irreps, odd channels pseudo-λ); orbital-swap `(-1)^{l+l'+λ}` sign;
`transform_λ!` accumulation; inadmissible-λ errors). Adaptations:

* imports → `using EquivariantTensors: O3` and
  `using EquivariantTensors.O3: BlockCoupling, couple, decouple, transform_λ,
  transform_λ!, cg_block, channel_parity`;
* "Stage 0" / twocenter.md §-references removed from comments (math comments
  kept).

Hook into `test/runtests.jl` inside the existing `"O3-Coupling"` testset:

```julia
@testset "Block Coupling" begin include("O3/test_block_coupling.jl"); end
```

### A.3 Process

* Work in `~/claude_projects/EquivariantTensors.jl` on a new branch
  `co/block-coupling` off `main` (the checkout currently sits on
  `radials-move-impl`, clean — leave that branch untouched).
* Run the full ET test suite.
* Per ET `CLAUDE.md`: no commit/push/PR until the user signs off on the
  implementation.

---

## Phase B — ET radials PR (user-driven, in progress)

`radials-move-impl`: `Radials` submodule (`src/radials/`), exporting
`LearnableRnlBasis, SplineRnlBasis, splinify, learnable_Rnl_basis,
PolyEnvelope1sR, PolyEnvelope2sX, agnesi_transform`;
`learnable_Rnl_basis` is the renamed `ace_learnable_Rnlrzz`
(`src/radials/constructors.jl:3`); `_convert_zlist` & friends live in ET
utils. Not planned here — Phase C assumes A and B are available on one ET
rev (merged to main, or merged locally for development).

---

## Phase C — ACEoperators migration (branch `et-migration` off `stage2-onsite`)

One atomic switch — not via the stage0→stage1→stage2 merge-forward flow,
because the radials swap touches Stage 1–2 files that don't exist on
`stage0-coupling`, and ET-0.5 + ACEpotentials cannot coexist in one
environment.

1. **`Project.toml`**: remove ACEpotentials from `[deps]`/`[compat]`;
   `EquivariantTensors = "0.5"`; temporary
   `[sources] EquivariantTensors = {url = "...", rev = "..."}` (or local
   `path` during development) until an ET 0.5 release containing A+B is
   registered, then delete the entry.
2. **`src/linear2c/coupling.jl`** shrinks to the import line
   `using EquivariantTensors.O3: BlockCoupling, couple, decouple,
   transform_λ, transform_λ!, cg_block, channel_parity`
   plus a brief pointer comment. The names land in the ACEoperators
   namespace, so `onsite.jl`, `overlap.jl`, `blocks.jl` and the tests keep
   working unchanged. The file stays: Stage 3's three-way contraction layer
   is planned to live there (plan_lin2c.md Stage 3).
3. **Radials swap** (all ACEpotentials call sites, verified by grep):
   * `src/linear2c/basis.jl:17,56–59,82` and
     `src/linear2c/overlap.jl:16,61–65,138`:
     `import ACEpotentials.Models as _M` → `import EquivariantTensors.Radials`;
     `_M.ace_learnable_Rnlrzz(...)` → `Radials.learnable_Rnl_basis(...)`
     (verify kwarg parity — `elements`, `spec`, `level = _M.TotalDegree()`,
     `max_level`, `maxl`, `maxn`, `rin0cuts` — against
     `src/radials/constructors.jl` at implementation time);
     `_M.initialparameters` / `_M.initialstates` → the LuxCore methods
     Radials extends; `_M.evaluate` / `_M.evaluate_batched` → the
     `Radials`/ACEbase equivalents.
   * `src/linear2c/blocks.jl:27`:
     `ACEpotentials.Models._convert_zlist` → `EquivariantTensors._convert_zlist`.
   * `src/ACEoperators.jl:8`: drop `import ACEpotentials`.
4. **`test/linear2c/test_coupling.jl`** slims to a smoke test (one
   couple/decouple round-trip, one SO(3) intertwining check, one
   `transform_λ!` accumulation) noting the full suite now lives in ET.
5. Full test suite must stay green. **Flag for review:** if ET `Radials`
   changed any defaults (transforms, envelopes, initialisation) relative to
   ACEpotentials, radial *values* change. The symmetry tests are invariant
   to this; any value-pinning tests are not. Check and report explicitly
   during implementation.

---

## Sequencing & risks

* Phase A is mergeable into ET main independently of Phase B; only Phase C
  needs both on a single rev.
* Until ET 0.5 (with A+B) is registered, ACEoperators CI needs the
  `[sources]` entry (Julia ≥ 1.11 Pkg) pointing at a pushed ET rev — i.e.
  the ET branches must be pushed before ACEoperators CI can pass.
* PR #4 stays as-is; the thin-wrapper switch arrives in the migration PR,
  and PR #4's description gets a note once that lands.
* After the move, ET owns the coupling's correctness tests; ACEoperators'
  smoke test only guards the integration. Behavioural changes to the
  coupling must then be caught by ET's suite — another reason to move the
  full suite (decided).
