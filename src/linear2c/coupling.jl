#
# coupling.jl — Wigner–Eckart recoupling for matrix blocks (§4)
#
# Implements the `transform_λ` step that converts equivariant (coupled) features
# X^{λμ} into an (l,m;l',m') matrix block and back, using the real-spherical-
# harmonic Clebsch–Gordan coupling consistent with `EquivariantTensors.O3`.
#
# IMPORTANT — this is the *Clebsch–Gordan coupling of two independent orbital
# indices* (a tensor product), NOT the Gaunt/product coupling of two harmonics
# at the same point. The latter (`O3.coupling_coeffs` / `O3.cgmatrix`) carries
# the selection rule `l+l'+λ` even and would drop the odd-`l+l'+λ` channels. As
# a pure change of basis on the block, ALL admissible `λ ∈ |l-l'|:l+l'` are
# present. NOTE on terminology: the LCAO Hamiltonian of a scalar operator is a
# PROPER O(3) tensor (transforming as D^l ⊗ D^{l'}); the odd-`l+l'+λ` λ-blocks
# carry inversion sign (-1)^{l+l'}, which is simply the correct behaviour of
# that component of l⊗l' — they are not "pseudotensors of H". Whether an
# odd-`l+l'+λ` component is nonzero is a question of selection rules and the
# feature set that sources it (see §5.2 / the model files): for on-site and
# single-bond blocks the odd channels vanish identically (Gaunt rule / one bond
# vector), so they are unused there; they can be nonzero only for environment-
# dependent off-site blocks with genuine axial/chiral geometry, which the
# real-CG ACE features cannot form.
#
# Construction (verified numerically, see test_coupling.jl):
#   * Start from the COMPLEX CG block  Cc[μ,m,m'] = ⟨l m; l' m' | λ μ⟩  (all λ,
#     all parities nonzero), which intertwines  Dc^l X (Dc^{l'})ᵀ ↔ Dc^λ v.
#   * Map to the real basis via the vector law  v_real = Ctran(l) · v_complex
#     (verified: `D_from_angles(l,·,real) = Ctran(l) D_complex Ctran(l)'`), so a
#     covariant block converts as  X_r = T_l X_c T_{l'}ᵀ  with  T = Ctran.
#   * The resulting coupling is purely real on even-`l+l'+λ` channels and purely
#     imaginary on odd ones; since `D_from_angles(·,real)` is real, real and
#     imaginary parts intertwine separately. We take the real part on even
#     channels and the imaginary part on odd channels — a single real, complete,
#     orthonormal change of basis over all λ.
#
# Conventions: ascending indices, m = -l:l ↦ 1:2l+1, μ = -λ:λ ↦ 1:2λ+1; a block
# transforms as  X ↦ D^l(Q) X D^{l'}(Q)ᵀ  and a coupled component as v ↦ D^λ v,
# all with `O3.D_from_angles(·, real)`.
#
# This file currently lives in ACEoperators; it is a candidate for upstreaming
# into EquivariantTensors once stable (see agents/plan_lin2c.md §1.5).
#

import EquivariantTensors as ET
using EquivariantTensors: O3
using LinearAlgebra: I

"""
    channel_parity(l, l', λ) -> Symbol

Classifies a coupled channel by `l+l'+λ`: `:even` channels are sourced by the
ordinary (geometric, proper-tensor) ACE/bond features; `:odd` channels would
require axial/chiral features (parity `(-1)^{λ+1}`) that the real-CG ACE
construction cannot form, and vanish identically for on-site and single-bond
blocks. The on-/off-site models populate the `:even` channels (§5.2).

(The names refer to the parity of `l+l'+λ`, not to the Hamiltonian itself, which
is a proper O(3) tensor — see the file header.)
"""
channel_parity(l::Integer, lp::Integer, λ::Integer) =
      iseven(l + lp + λ) ? :even : :odd

"""
    cg_block(l, l', λ) -> Array{Float64,3}

Dense real Clebsch–Gordan block coupling `C[μ+λ+1, m+l+1, m'+l'+1]` coupling the
orbital pair `(l, l')` to the output irrep `λ`, in the ascending-index real
convention consistent with `O3.D_from_angles(·, real)` (see file header).

Returns a `(2λ+1, 2l+1, 2l'+1)` array; nonzero for every triangle-admissible
`λ ∈ |l-l'|:l+l'` (both parities). The slices are orthonormal:
`Σ_{m,m'} C[μ,m,m'] C[ν,m,m'] = δ_{μν}`.
"""
function cg_block(l::Integer, lp::Integer, λ::Integer)
   C = zeros(Float64, 2λ + 1, 2l + 1, 2lp + 1)
   (abs(l - lp) <= λ <= l + lp) || return C
   Tl = O3.Ctran(l); Tlp = O3.Ctran(lp); Tλ = O3.Ctran(λ)
   even = iseven(l + lp + λ)
   for ν = -λ:λ, a = -l:l, b = -lp:lp
      s = 0.0im
      for m = -l:l, mp = -lp:lp
         μ = m + mp
         abs(μ) <= λ || continue
         cc = O3.cg(l, m, lp, mp, λ, μ, complex)        # complex CG ⟨lm;l'm'|λμ⟩
         iszero(cc) && continue
         s += conj(Tλ[ν + λ + 1, μ + λ + 1]) *
              Tl[a + l + 1, m + l + 1] * Tlp[b + lp + 1, mp + lp + 1] * cc
      end
      # even-parity channel is purely real, odd-parity purely imaginary
      C[ν + λ + 1, a + l + 1, b + lp + 1] = even ? real(s) : imag(s)
   end
   return C
end

"""
    BlockCoupling(l, l')

Precomputed Wigner–Eckart coupling for the orbital pair `(l, l')`. Holds the CG
block `cg_block(l, l', λ)` for every triangle-admissible channel
`λ ∈ |l-l'|:l+l'`, and provides `couple` / `decouple` between an
`(2l+1)×(2l'+1)` matrix block and its complete set of coupled `λ`-components.
"""
struct BlockCoupling
   l::Int
   lp::Int
   λs::Vector{Int}                  # all triangle-admissible channels
   parities::Vector{Symbol}         # :even / :odd (parity of l+l'+λ) per channel
   C::Vector{Array{Float64, 3}}     # C[k] = cg_block(l, l', λs[k])
end

function BlockCoupling(l::Integer, lp::Integer)
   λs = collect(abs(l - lp):(l + lp))
   parities = [ channel_parity(l, lp, λ) for λ in λs ]
   C = [ cg_block(l, lp, λ) for λ in λs ]
   return BlockCoupling(Int(l), Int(lp), λs, parities, C)
end

Base.length(bc::BlockCoupling) = length(bc.λs)

"""
    couple(bc::BlockCoupling, X) -> Vector{Vector{Float64}}

Decompose a `(2l+1)×(2l'+1)` matrix block `X` into its coupled components,
`Xλ[k][μ+λ+1] = Σ_{m,m'} C[μ,m,m'] X[m,m']` for `λ = bc.λs[k]`. Exact inverse of
[`decouple`](@ref) (the coupling is a complete orthonormal change of basis).
"""
function couple(bc::BlockCoupling, X::AbstractMatrix)
   @assert size(X) == (2 * bc.l + 1, 2 * bc.lp + 1)
   Xλ = Vector{Vector{Float64}}(undef, length(bc.λs))
   for (k, λ) in enumerate(bc.λs)
      C = bc.C[k]
      v = zeros(Float64, 2λ + 1)
      for μ = 1:(2λ + 1), mi = 1:size(X, 1), mpi = 1:size(X, 2)
         v[μ] += C[μ, mi, mpi] * X[mi, mpi]
      end
      Xλ[k] = v
   end
   return Xλ
end

"""
    decouple(bc::BlockCoupling, Xλ) -> Matrix{Float64}

Reassemble the `(2l+1)×(2l'+1)` matrix block from its coupled components,
`X[m,m'] = Σ_λ Σ_μ C[μ,m,m'] Xλ[λ][μ]`. Exact inverse of [`couple`](@ref).
Equal to `Σ_λ transform_λ(bc, λ, Xλ[λ])`.
"""
function decouple(bc::BlockCoupling, Xλ::AbstractVector)
   @assert length(Xλ) == length(bc.λs)
   X = zeros(Float64, 2 * bc.l + 1, 2 * bc.lp + 1)
   for (k, λ) in enumerate(bc.λs)
      _accumulate_block!(X, bc.C[k], Xλ[k])
   end
   return X
end

"""
    transform_λ(bc::BlockCoupling, λ, vλ) -> Matrix{Float64}

The §4 `transform_λ[...]_{mm'}` operation for a single channel `λ`: given the
λ-equivariant vector `vλ` (length `2λ+1`, e.g. `Σ_q w_q B^{λμ}_q`), return its
contribution to the `(l,m;l',m')` block,
`transform_λ(vλ)[m,m'] = Σ_μ C[μ,m,m'] vλ[μ]`. The full block is the sum over
all channels, i.e. `decouple(bc, [v_{λ}...])`.
"""
function transform_λ(bc::BlockCoupling, λ::Integer, vλ::AbstractVector)
   k = findfirst(==(Int(λ)), bc.λs)
   k === nothing && error(
         "λ = $λ is not admissible for (l,l') = ($(bc.l),$(bc.lp))")
   @assert length(vλ) == 2λ + 1
   X = zeros(Float64, 2 * bc.l + 1, 2 * bc.lp + 1)
   _accumulate_block!(X, bc.C[k], vλ)
   return X
end

# X[m,m'] += Σ_μ C[μ,m,m'] v[μ]
function _accumulate_block!(X, C, v)
   for μ = 1:length(v), mi = 1:size(X, 1), mpi = 1:size(X, 2)
      X[mi, mpi] += C[μ, mi, mpi] * v[μ]
   end
   return X
end
