#
# wignereckart.jl — Wigner–Eckart recoupling for matrix blocks (§4)
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
# This file currently lives in ACEoperators.Transforms; it is a candidate for
# upstreaming into EquivariantTensors once stable (see agents/plan_lin2c.md §1.5).
#

using EquivariantTensors: O3
using LinearAlgebra: I, mul!

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
    cg_block([T=Float64], l, l', λ) -> Array{T,3}

Dense real Clebsch–Gordan block coupling `C[μ+λ+1, m+l+1, m'+l'+1]` coupling the
orbital pair `(l, l')` to the output irrep `λ`, in the ascending-index real
convention consistent with `O3.D_from_angles(·, real)` (see file header). The
element type `T` selects the working precision of the stored coupling.

Returns a `(2λ+1, 2l+1, 2l'+1)` array; nonzero for every triangle-admissible
`λ ∈ |l-l'|:l+l'` (both parities). The slices are orthonormal:
`Σ_{m,m'} C[μ,m,m'] C[ν,m,m'] = δ_{μν}`.

The construction is carried out internally in `ComplexF64`; `T` only selects
the storage precision (a `T` wider than `Float64` gains no accuracy).
"""
function cg_block(::Type{T}, l::Integer, lp::Integer, λ::Integer) where {T}
   C = zeros(T, 2λ + 1, 2l + 1, 2lp + 1)
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
      # even-(l+l'+λ) channel is purely real, odd-(l+l'+λ) purely imaginary
      C[ν + λ + 1, a + l + 1, b + lp + 1] = even ? T(real(s)) : T(imag(s))
   end
   return C
end

cg_block(l::Integer, lp::Integer, λ::Integer) = cg_block(Float64, l, lp, λ)

"""
    BlockCoupling(l, l')

Precomputed Wigner–Eckart coupling for the orbital pair `(l, l')`. Holds the CG
block `cg_block(l, l', λ)` for every triangle-admissible channel
`λ ∈ |l-l'|:l+l'`, and provides `couple` / `decouple` between an
`(2l+1)×(2l'+1)` matrix block and its complete set of coupled `λ`-components.
"""
struct BlockCoupling{T}
   l::Int
   lp::Int
   λs::UnitRange{Int}               # all triangle-admissible channels
   C::Vector{Array{T, 3}}           # C[k] = cg_block(T, l, l', λs[k])
end

function BlockCoupling(::Type{T}, l::Integer, lp::Integer) where {T}
   λs = abs(l - lp):(l + lp)
   C = [ cg_block(T, l, lp, λ) for λ in λs ]
   return BlockCoupling{T}(Int(l), Int(lp), λs, C)
end

BlockCoupling(l::Integer, lp::Integer) = BlockCoupling(Float64, l, lp)

Base.length(bc::BlockCoupling) = length(bc.λs)

# channel index of λ in bc.λs (contiguous range ⇒ O(1) arithmetic)
function _λindex(bc::BlockCoupling, λ::Integer)
   k = Int(λ) - first(bc.λs) + 1
   (1 <= k <= length(bc.λs)) || error(
         "λ = $λ is not admissible for (l,l') = ($(bc.l),$(bc.lp))")
   return k
end

# the CG block of channel λs[k], reshaped to a (2λ+1) × (2l+1)(2l'+1)
# matrix; `reshape` of an `Array` shares the underlying data (no copy)
_cgmat(bc::BlockCoupling, k::Integer) =
      reshape(bc.C[k], size(bc.C[k], 1), :)

"""
    couple(bc::BlockCoupling, X) -> Vector{<:AbstractVector}

Decompose a `(2l+1)×(2l'+1)` matrix block `X` into its coupled components,
`Xλ[k][μ+λ+1] = Σ_{m,m'} C[μ,m,m'] X[m,m']` for `λ = bc.λs[k]`. Exact inverse of
[`decouple`](@ref) (the coupling is a complete orthonormal change of basis).
The element type follows `promote_type(eltype(bc.C), eltype(X))`.
"""
function couple(bc::BlockCoupling, X::AbstractMatrix)
   @assert size(X) == (2 * bc.l + 1, 2 * bc.lp + 1)
   x = vec(X)
   return [ _cgmat(bc, k) * x for k in eachindex(bc.λs) ]
end

"""
    decouple(bc::BlockCoupling, Xλ) -> Matrix

Reassemble the `(2l+1)×(2l'+1)` matrix block from its coupled components,
`X[m,m'] = Σ_λ Σ_μ C[μ,m,m'] Xλ[λ][μ]`. Exact inverse of [`couple`](@ref).
Equal to `Σ_λ transform_λ(bc, λ, Xλ[λ])`.
"""
function decouple(bc::BlockCoupling, Xλ::AbstractVector)
   @assert length(Xλ) == length(bc.λs)
   return sum( transform_λ(bc, λ, Xλ[k]) for (k, λ) in enumerate(bc.λs) )
end

"""
    transform_λ(bc::BlockCoupling, λ, vλ) -> Matrix

The §4 `transform_λ[...]_{mm'}` operation for a single channel `λ`: given the
λ-equivariant vector `vλ` (length `2λ+1`, e.g. `Σ_q w_q B^{λμ}_q`), return its
contribution to the `(l,m;l',m')` block,
`transform_λ(vλ)[m,m'] = Σ_μ C[μ,m,m'] vλ[μ]`. The full block is the sum over
all channels, i.e. `decouple(bc, [v_{λ}...])`. See [`transform_λ!`](@ref) for
an in-place accumulating variant.
"""
function transform_λ(bc::BlockCoupling, λ::Integer, vλ::AbstractVector)
   k = _λindex(bc, λ)
   @assert length(vλ) == 2λ + 1
   x = transpose(_cgmat(bc, k)) * vλ
   return reshape(x, 2 * bc.l + 1, 2 * bc.lp + 1)
end

"""
    transform_λ!(X, bc::BlockCoupling, λ, vλ) -> X

In-place variant of [`transform_λ`](@ref): accumulates the channel-`λ`
contribution into the `(2l+1)×(2l'+1)` block, `X[m,m'] += Σ_μ C[μ,m,m'] vλ[μ]`,
without allocating. Intended for the matrix assembly loops.
"""
function transform_λ!(X::AbstractMatrix, bc::BlockCoupling, λ::Integer,
                      vλ::AbstractVector)
   k = _λindex(bc, λ)
   @assert size(X) == (2 * bc.l + 1, 2 * bc.lp + 1)
   @assert length(vλ) == 2λ + 1
   mul!(vec(X), transpose(_cgmat(bc, k)), vλ, true, true)
   return X
end
