# Implementation Plan — Linear 2-Center Hamiltonian Model (`linear2c`)

Status: **approved; in progress**. This plan implements the model specified in
`docs/src/twocenter.md`. Section references (§N) point into that document.
This revision folds in the review comments (single graph, ACEpotentials radial
dependency, N-way coupling via `ET.O3`, multi-species params, stop-and-test per
stage).

**Revision (2026-07-06):** `H` and `S` are **separate, independent models**
(no single model producing the pair) — see §3.6 / §6.7. Earlier references to
a unified `(H, S)` model are superseded.

The plan is grounded in the actual APIs of the installed dependencies
(EquivariantTensors `2d207`, ACEpotentials `nR4od`, Polynomials4ML, SpheriCart,
WignerD, NeighbourLists, Lux). Verified call signatures are quoted inline so we
do not design against an imagined API.

---

## 0. Workflow: staged, stop-and-test after each stage

**Build in stages; stop after each stage, run its tests, and hand back for
review** — the unit tests in particular are to be inspected before moving on.

| Stage | Deliverable | Main risk retired |
|-------|-------------|-------------------|
| 0 ✅ | scaffolding, `coupling.jl` (`transform_λ`), block index bookkeeping | CG/Wigner-Eckart conventions |
| 1 ✅ | **Overlap `S`** model (on-site const + off-site bond-only) end-to-end, with rotation/inversion/Hermiticity tests | graph→bond-embedding→block-assembly→symmetrize loop |
| 2 | **On-site `H_ii`** model (one-center ACE + `transform_λ`) | ACE-feature reuse, parity selection |
| 3 | **Three-way coupling** `B_i⊗B_j⊗φ_b→Λ` in isolation + equivariance test | the off-site coupling machinery |
| 4 | **Off-site `H_ij`** model assembled from Stage 3 | full bond model + bond-flip Hermiticity |
| 5 | `LinearH2C` (H-only) model struct + standalone overlap model, hypers/heuristics, sanity checks, full §12 suite, docs | integration, UX |

---

## 1. What already exists (reuse, do not rebuild)

Verified against source. These are the load-bearing APIs.

### 1.1 One-particle → A → B (one-center equivariant ACE) — EquivariantTensors

```julia
import EquivariantTensors as ET
# many-body spec: list of [(n=,l=), ...] correlation blocks, total-degree truncated
mb_spec = ET.sparse_nnll_set(; L, ORD, minn=0, maxn, maxl,
                               level = bb -> sum(b.n + b.l for b in bb; init=0),
                               maxlevel)
# single target irrep L:
𝔹 = ET.sparse_equivariant_tensor(; L, mb_spec, Rnl_spec, Ylm_spec, basis = real)
# multiple target irreps at once (what we need — all Λ up to L_max):
𝔹 = ET.sparse_equivariant_tensors(; LL = (0,1,2,...), mb_spec, Rnl_spec, Ylm_spec, basis = real)
```

`sparse_equivariant_tensor(s)` returns a `SparseACEbasis` (a Lux layer) holding
`abasis::PooledSparseProduct` (pools neighbours → `A_{i,nlm}`), `aabasis::
SparseSymmProd` (→ symmetric `AA`), and `A2Bmaps` (one sparse coupling matrix
per `L`, built via `symmetrisation_matrix(L, mb_spec; PI=true, basis)`).
`evaluate(tensor, Rnl, Ylm, ps, st)` returns a tuple `BB` with one equivariant
block per requested `L`; `Rnl, Ylm` are `(n_neigh × n_basis)` matrices, or
`(maxneigs × nnodes × n_basis)` over a graph (pooling is over the neighbour
dim). Pullbacks/rrules are provided → Zygote works.

This **is** the `𝓑_i = {B_{i v}^{(ν)LM}}` of §3, including parity bookkeeping
(`sparse_nnll_set`'s `evenfilter` enforces `(-1)^{Σ l} = (-1)^L`).

### 1.2 O3 coupling + Wigner-D — `EquivariantTensors.O3`

```julia
U, MM = O3.coupling_coeffs(L, ll, nn=nothing; PI, basis=real)  # generalized CG, any length(ll)
D     = O3.D_from_angles(l, θ::SVector{3}, real)               # real-SH Wigner-D
Q, D  = O3.QD_from_angles(l, θ, real)                          # rotation + its D
T     = O3.Ctran(l)                                            # real↔complex SH
```

`coupling_coeffs` works for **arbitrary `length(ll)`** (verified: `_coupling_coeffs`
takes `ll::SVector{N,Int}`, sums over all admissible intermediate couplings via
`SetLl`). So:
* `coupling_coeffs(λ, (l, l'); PI=false, basis=real)` is the two-index
  Wigner–Eckart CG array `⟨l m; l' m' | λ μ⟩` of §4 — the heart of `transform_λ`.
* `coupling_coeffs(Λ, (L1, L2, lb); PI=false, basis=real)` is the **three-way**
  coupling `C^{(Λ)}_{(L1)(L2)(lb)}` of §7.2 — the κ-sum is internal. This is the
  ET.O3 functionality to use for the off-site coupling coefficients (§1.5).

`D_from_angles`/`QD_from_angles` give the §12 test machinery directly (pattern in
`test/O3/test_O3_transforms.jl`, `test/.../test_coupling.jl`).

### 1.3 Structure → graph → embedding — EquivariantTensors + extensions

```julia
using NeighbourLists                       # triggers the extension
G = ET.Atoms.interaction_graph(sys, rcut)  # AtomsBase system → ETGraph
# ETGraph: ii (centers, sorted), jj (neighbours), first, node_data, edge_data,
#          graph_data, maxneigs.  edge_data[e] carries 𝐫, z0, z1, 𝐒 (shift).
ytrans = ET.NTtransform(x -> x.𝐫)
yembed = ET.EdgeEmbed( ET.EmbedDP(ytrans, P4ML.real_sphericalharmonics(maxl)) )
rembed = ET.EdgeEmbed( ET.EmbedDP(rtrans, rbasis) )
# EdgeEmbed maps edge features → (maxneigs × nnodes × nfeat) via reshape_embedding
forces_from_edge_grads(sys, G, ∇E.edge_data)   # adjoint back to atoms
```

This is the §11 step-1/step-2 plumbing.

**Single graph, filtered (decided).** We use **one** `ETGraph` built with
`rcut = r_cut^on` (the larger cutoff). Bond quantities that need the shorter
`r_cut^b` are obtained by **filtering edges** by `|r_ji| ≤ r_cut^b` (the bond
envelope already → 0 there, so filtering only avoids wasted work; off-site blocks
for `r_cut^b < |r_ji| ≤ r_cut^on` are zero by §9). **Default: `r_cut^b =
r_cut^on`** — the two cutoffs are kept as separate hyperparameters but coincide
unless explicitly set otherwise. Environment pooling uses all edges; bond/overlap
assembly uses the filtered subset.

### 1.4 Radial basis with transforms + envelopes — ACEpotentials (decided)

**Depend on `ACEpotentials.Models` now**; spin out a dedicated `ACEradials.jl`
later. Use:

```julia
import ACEpotentials.Models as M
rbasis = M.ace_learnable_Rnlrzz(; elements, level, max_level, maxl, maxn, ...)
#   -> LearnableRnlrzzBasis: species-pair SMatrix of GeneralizedAgnesiTransform
#      + PolyEnvelope2sX, weights Wnlq[n,q,zi,zj]; optional M.splinify(...)
```

This gives species-resolved radial channels (§2: index `n` runs over
(species, radial-order) pairs) and proper distance transforms + envelopes for
free, for **both** the environment basis `R_nl` and the bond basis `R^b_nl`
(two independent `LearnableRnlrzzBasis` instances with their own `maxn/maxl` and
`rin0cuts`). New dependency: **ACEpotentials** (flagged per CLAUDE.md).

### 1.5 What does NOT exist and must be built

* **`transform_λ`** (§4) — thin wrapper of `O3.coupling_coeffs` into a
  couple/inverse-couple pair specialised for output `(l,m;l',m')` blocks.
  **Implement here (`coupling.jl`) first; consider upstreaming to
  EquivariantTensors later.**
* **Three-way coupling layer** `T^{Λμ}_{ij} = C^{(Λ)} : (B_i^{L1}⊗B_j^{L2}⊗φ^b_{lb})`
  (§7.2). The *coupling coefficients* come from
  `O3.coupling_coeffs(Λ, (L1,L2,lb); PI=false, basis=real)` (§1.2) — use this.
  What we add is the **contraction layer** that applies them over the selected
  output channels, Lux/Zygote-compatible. A bespoke optimized triple-product
  (not routing through generic ET contraction) is a worthwhile experiment to try
  *after* a correct reference version exists — but the coefficient construction
  must come from `ET.O3`.
* **Block assembly / bookkeeping**: orbital list → target block types
  `(nl, n'l')` → admissible `λ` ranges → global matrix index map; on-site vs
  off-site assembly loops; post-hoc `X ← ½(X + Xᵀ)` (§7.3).
* **Model struct + hyperparameter resolution** (§10) and sanity checks (§14).

---

## 2. Module / file layout

Following §14 (with `model.jl` allowed to spawn an `assemble.jl` if it grows):

```
src/
  ACEoperators.jl          # add: include linear2c, re-export public API
  linear2c/
    coupling.jl            # transform_λ (§4) + 3-way coupling contraction (§7.2)
    basis.jl               # env ACE basis 𝓑_i, bond basis φ^b, radial bases
    blocks.jl              # orbital list, block-type enumeration, λ ranges,
                           #   global index map, couple↔block helpers
    onsite.jl              # H_ii layer (§5)            [could live in model.jl]
    offsite.jl             # H_ij layer (§7)            [could live in model.jl]
    overlap.jl             # standalone overlap model S (§8)
    hypers.jl              # defaults + heuristics (§10)
    params.jl              # parameter packing/unpacking, initialisation
    model.jl               # LinearH2C (H-only) struct, Lux glue, assembly, symmetrize
test/
  runtests.jl              # include linear2c suite
  linear2c/
    test_coupling.jl       # transform_λ round-trip + 3-way coupling equivariance
    test_overlap.jl        # Stage 1
    test_onsite.jl         # Stage 2
    test_offsite.jl        # Stage 4
    test_symmetry.jl       # full §12 suite on the assembled H and S models
```

---

## 3. Core design decisions

1. **Real spherical harmonics throughout** (`basis = real`), matching §1 and the
   SpheriCart/EquivariantTensors convention. All CG and Wigner-D use the real
   convention (`O3.coupling_coeffs(...; basis=real)`, `O3.D_from_angles(l,θ,real)`).

2. **Couple all needed `Λ` at once** via `sparse_equivariant_tensors(; LL=...)`,
   where `LL = 0:L_max` and `L_max = max over the orbital list of (l + l')`
   (resolved automatically from the orbital list, §10). One `𝓑_i` per site,
   shared by on-site and off-site (§10 implementation note). The effective
   `mb_spec` is the union of the on-site and off-site specs (element-wise max of
   ORD / degree), resolved at construction.

3. **Single graph as model input** (decided). Each model's
   `forward(G, ps, st)` takes one `ETGraph`; a thin helper
   `graph(model, sys)` builds it from an AtomsBase system. For the H model the
   graph uses `r_cut^on` and bond assembly filters edges to `r_cut^b`; the
   standalone overlap model needs only its own (cheaper) bond graph. Keeping
   the graph build outside the Lux layer matches `mlip.jl` and lets us reuse
   `forces_from_edge_grads`.

4. **Hermiticity via post-hoc symmetrization** `X ← ½(X+Xᵀ)` (§7.3, §8) for all
   stages (decided — no feature-level symmetrization for now). The
   selection-rule tests it implies (§12.6) are still written and checked.

5. **Weights are scalar (invariant) per `(nl, n'l'; λ)` channel** (§4, §5.1):
   equivariance lives entirely in `B^{λμ}`/`T^{Λμ}` and the inverse-CG transform,
   so regression weights are plain vectors — a linear model.
   **Parameter handling is multi-species from the start** (decided): weights are
   indexed by species (on-site) and ordered species-pair / shell-pair type
   (off-site), even though the first working model targets single-species Si.
   `params.jl` owns this indexing so adding elements needs no structural change.

6. **Separate `S` and `H` models (decided 2026-07-06).** The overlap and
   Hamiltonian models are independent structs with independent hypers,
   parameters, and forward passes — there is **no** single model producing the
   pair `(H, S)`. Rationale: the initial focus is *experimentation*, where the
   functional forms of `S` and `H` will be varied independently and may be
   entirely unrelated; and `S` is normally much cheaper to build and fit than
   `H`, so coupling their evaluation would force the expensive machinery onto
   the cheap target. The models still share code-level building blocks
   (`coupling.jl`, radial/bond embeddings, block bookkeeping) but no model
   object. Stage 1's `OverlapModel` already follows this design.

---

## 4. Stages in detail

### Stage 0 — scaffolding + `coupling.jl`
* Create `src/linear2c/` and the includes in `ACEoperators.jl`.
* Implement `transform_λ` (§4):
  * `cgmat(l, l', λ; basis=real) = O3.coupling_coeffs(λ, (l,l'); PI=false, basis)`
    returns `⟨l m; l' m'|λ μ⟩`; cache per `(l,l',λ)`.
  * `couple(X_block, l, l') -> Dict λ => X^{λμ}` and inverse
    `decouple(Xλ..., l, l') -> X_block` (the `Σ_λ ⟨..|λμ⟩ X^{λμ}` of §4).
  * `transform_λ(weights, Bλ, l, l') -> (m,m')` block per §4 eq.
* **Test (`test_coupling.jl`, part 1):** round-trip `decouple∘couple ≈ id` on a
  random block; CG ranges `|l−l'|≤λ≤l+l'`; parity sign `(-1)^{l+l'+λ}`.
* **Stop, run tests, hand back.**

### Stage 1 — Overlap `S` (§8)  ← simplest full pipeline
* Single graph (§3.3); bond basis `φ^b_{nlm}(r_ji) = R^b_{nl}(r) Y_{lm}(r̂)` via
  §1.3 EdgeEmbed on edges filtered to `r_cut^b`, radial via
  `ACEpotentials.Models` (§1.4). (At Stage 1 begin, also look up the N-way
  `O3.coupling_coeffs` usage from §1.2/§1.5 so Stage 3 is de-risked early.)
* Off-site: for each ordered shell pair `(nl,n'l')` and admissible `Λ`, contract
  bond harmonics `φ^b_{·Λ·}` (so `l_q=Λ`, no internal CG — §8) with scalar
  weights `u^{(nl,n'l';Λ)}` (species-pair indexed), then `transform_Λ` →
  `(m,m')` block.
* On-site `S_ii`: identity or fixed Gram (no learning).
* Assemble full `S`, then `S ← ½(S+Sᵀ)`.
* **Tests (`test_overlap.jl`):** rotation, inversion (+parity signs),
  Hermiticity, permutation, cutoff smoothness — restricted to `S`.
  Equivariance pattern: rotate the system, rebuild `S`, compare to
  `𝓓(Q) S 𝓓(Q)ᵀ`, `𝓓 = blockdiag(D^{l}(Q))` from `O3.D_from_angles`.
* **Stop, run tests, hand back.**

### Stage 2 — On-site `H_ii` (§5)
* Build env ACE basis `𝓑_i` via `sparse_equivariant_tensors(; LL=0:L_max, ...)`
  on the (single) graph.
* For each species, each unordered shell pair `(nl)≤(n'l')`, each admissible `λ`
  with matching parity (§5.2): scalar weights `w^{(nl,n'l';λ)}` · `B_i^{λμ}` →
  `transform_λ` → block; diagonal-pair `√2` normalisation (§5.2).
* **Tests (`test_onsite.jl`):** §12.1–3,5 on `H_ii`, plus §12.6 vanishing odd-λ
  diagonal-shell components, plus parity selection (§5.2: structure vs inversion
  image differ by the expected sign pattern).
* **Stop, run tests, hand back.**

### Stage 3 — Three-way coupling layer (§7.2)
* Coefficients from `O3.coupling_coeffs(Λ, (L1,L2,lb); PI=false, basis=real)`
  (§1.2) over the selected `(Λ, v1, v2, n_b l_b)` output channels.
* `apply_couple3(C, Bi, Bj, φb) -> T^{Λμ}` — a Lux/Zygote-compatible contraction.
  A bespoke optimized triple-product is an optional follow-up; the reference
  version routes coefficients through `ET.O3`.
* **Test (`test_coupling.jl`, part 2):** random `Bi,Bj,φb` and random `Q`; check
  `T^{Λ}(Q·) ≈ D^{Λ}(Q) T^{Λ}` and parity. Isolates the novel code from assembly.
* **Stop, run tests, hand back.**

### Stage 4 — Off-site `H_ij` (§7)
* For each ordered shell-pair type and admissible `Λ`: enumerate retained
  `q=(v1,v2,n_b l_b)` under `ν_i+ν_j ≤ ν_off−1` (§10); scalar weights
  `w^{(nl,n'l';Λ)}` (species-pair indexed) · `T^{Λμ}_{ij,q}` → `transform_Λ` →
  block. Reuse `𝓑_i` (per-site, once) and bond `φ^b_{ij}`.
* Assemble off-site blocks for all bonds with `r_ji = r_j − r_i`; add to `H`;
  `H ← ½(H+Hᵀ)` (§7.3). No canonical bond ordering required.
* **Tests (`test_offsite.jl`):** §12.1–3,5 on `H_ij`, especially Hermiticity
  under bond flip.
* **Stop, run tests, hand back.**

### Stage 5 — Integration, hypers, full model, docs
* `LinearH2C` struct (§14): owns env basis, bond basis, on/off-site weight sets,
  orbital list, cutoffs; Lux layer exposing `forward(G, ps, st) -> H`;
  assembly loop + symmetrization (→ `assemble.jl` if `model.jl` grows past
  ~300 lines). The overlap model remains the separate `OverlapModel` (Stage 1)
  with its own `forward -> S` — no combined `(H, S)` model (§3.6).
* `hypers.jl`/`params.jl`: defaults + heuristics (§10); auto-resolve `L_max`,
  union `mb_spec`, `l_max^b ≥ l_max^orb` check, `r_cut^b ≤ r_cut^on` default
  coincidence, construction-time sanity checks (§14). Multi-species parameter
  layout throughout.
* `test_symmetry.jl`: the §12 suite (minus translation, §5) on the full `H`
  model and the standalone `S` model separately, including cutoff-smoothness
  `C^p` across `r_cut^b`/`r_cut^on`.
* Minimal docs/example mirroring `examples/atoms/mlip.jl` (small Si system,
  assemble `H` and `S` with their respective models, run the symmetry checks).
* **Stop, run tests, hand back.**

---

## 5. §12 symmetry tests — concrete approach

Each property (except translation, see below) gets a test; the reusable kernel is
the block-Wigner conjugation:

```julia
θ = 2π .* rand(3);  Q, _ = O3.QD_from_angles(0, θ, real)
𝓓 = blockdiag([ O3.D_from_angles(l, θ, real) for (n,l) in orbital_list_expanded ]...)
H_rot = assemble(rotate(sys, Q));   @test H_rot ≈ 𝓓 * H * 𝓓'
```

* **Rotation (§12.1):** as above.
* **Inversion (§12.2):** `Q` with `det = −1`; expect `(-1)^l` per-block parity on
  top of conjugation; check the predicted sign pattern.
* **Hermiticity (§12.3):** `H ≈ Hᵀ`, `S ≈ Sᵀ` after symmetrization.
* **Translation (§12.4):** **skipped** — trivial (positions enter only via
  relative bond vectors).
* **Permutation (§12.5):** relabel atoms ⇒ `H` rows/cols permuted, values equal.
* **Vanishing blocks (§12.6):** odd-λ on diagonal shell pairs `≡ 0`.
* **Cutoff smoothness (§12.7):** sweep a bond length across each cutoff, check the
  Frobenius norm is `C^p` (envelope order `p`) via finite-difference continuity.

---

## 6. Review decisions (resolved)

1. **Radial basis:** depend on `ACEpotentials.Models` now; spin out `ACEradials.jl`
   later. (§1.4)
2. **Graph input:** single graph, filtered to `r_cut^b` for bonds; `r_cut^b =
   r_cut^on` by default. (§1.3, §3.3)
3. **`transform_λ` home:** local in `coupling.jl` for now; upstream later. (§1.5)
4. **Three-way coupling:** coefficients from `O3.coupling_coeffs` (N-way); a
   bespoke optimized contraction is an optional later experiment. (§1.2, §1.5, Stage 3)
5. **Hermiticity:** post-hoc symmetrization only. (§3.4)
6. **Multi-species:** first model is single-species Si, but parameter handling is
   multi-species from the start. (§3.5)
7. **Separate `S` and `H` models** (2026-07-06): independent structs, hypers,
   parameters, and forward passes; shared building blocks only. Motivated by
   experimentation with unrelated forms and the much lower cost of `S`. (§3.6)

---

## 7. Reused-API quick reference (verified)

| Need | Call | Source (verified) |
|------|------|--------|
| env ACE basis 𝓑_i (multi-L) | `ET.sparse_equivariant_tensors(; LL, mb_spec, Rnl_spec, Ylm_spec, basis=real)` | `src/ace/sparse_ace_utils.jl:3` |
| mb spec | `ET.sparse_nnll_set(; L, ORD, minn, maxn, maxl, level, maxlevel)` | `src/utils/sparseprod.jl:117` |
| evaluate basis | `ET.evaluate(tensor, Rnl, Ylm, ps, st)` → tuple of `BB` per L | `src/ace/sparse_ace_basis.jl:94` |
| 2-way CG / Wigner-Eckart | `O3.coupling_coeffs(λ, (l,l'); PI=false, basis=real)` | `src/O3/O3.jl:294` |
| N-way coupling (off-site) | `O3.coupling_coeffs(Λ, (L1,L2,lb); PI=false, basis=real)` | `src/O3/O3.jl:294` (N-way `_coupling_coeffs`) |
| real Wigner-D | `O3.D_from_angles(l, θ, real)`, `O3.QD_from_angles` | `src/O3/O3_utils.jl` |
| structure→graph (single) | `ET.Atoms.interaction_graph(sys, rcut)` | `src/extensions/atoms.jl:15`, ext |
| edge embedding | `ET.EdgeEmbed(ET.EmbedDP(ET.NTtransform(f), basis))` | `src/embed/embeddings.jl`, `examples/atoms/mlip.jl` |
| radial w/ transform+env | `ACEpotentials.Models.ace_learnable_Rnlrzz` / `LearnableRnlrzzBasis` | `ACEpotentials/.../models/ace_heuristics.jl:12` |
| forces adjoint | `ET.Atoms.forces_from_edge_grads(sys, G, ∇.edge_data)` | `src/extensions/atoms.jl:19` |

---

*Revised per review. Proceeding with Stage 0.*
