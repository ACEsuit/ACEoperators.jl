#
# overlap.jl — the two-center overlap model S (§8, Stage 1)
#
# S is a purely geometric two-center quantity: S_ii is a fixed on-site block
# (identity / orthonormal basis here) and S_ij (i≠j) depends only on the bond
# vector r_ji. For each ordered shell pair (n l, n' l') and each admissible
# coupled angular momentum Λ, the Λ-block of S_ij is a linear combination of
# bond harmonics φ^b_{n_q Λ}(r_ji) = R^b_{n_q Λ}(r) Y_{Λ·}(r̂), recoupled into the
# (m,m') block via transform_Λ (§4). Only even-`l+l'+Λ` channels appear, and this
# is COMPLETE for a two-center quantity: a single bond vector supplies only those
# channels, and the count matches the Slater–Koster integrals (e.g. p–p gives
# Λ=0,2 = ppσ,ppπ; the odd Λ=1 axial term is correctly absent). Hermiticity is
# enforced post-hoc by S ← ½(S + Sᵀ) (§7.3).
#

import ACEpotentials.Models as _M
import Polynomials4ML as _P4ML
using LinearAlgebra: norm, I
using StaticArrays: SVector

# one off-site contribution: for species pair (iz,jz), shell pair (a,b) with
# angular momenta (la,lb), coupled channel Λ, the radial columns feeding it and
# the slice of the parameter vector that weights them.
struct OffsiteEntry
   iz::Int; jz::Int
   a::Int;  b::Int
   la::Int; lb::Int
   Λ::Int
   radialcols::Vector{Int}        # columns of Rnl with l == Λ
   wrange::UnitRange{Int}         # slice of the flat parameter vector
end

"""
    OverlapModel(orbitals; maxn_bond=…, kwargs...)

Linear two-center overlap model (§8) for the orbital list `orbitals`. Builds a
bond radial basis (`ACEpotentials.Models.ace_learnable_Rnlrzz`), a real spherical
harmonic basis up to `maxL(orbitals)`, the Wigner–Eckart couplings, and the
off-site parameter layout. The bond radial weights are fixed at construction
(geometry-only); the learnable parameters are the off-site coefficients `W`.
"""
struct OverlapModel{TR, TY, TRP, TRS}
   orbitals::OrbitalBasis
   bondradial::TR                  # LearnableRnlrzzBasis
   bondradial_ps::TRP              # fixed radial weights
   bondradial_st::TRS
   ybasis::TY                      # real_sphericalharmonics(maxL)
   couplings::Dict{Tuple{Int,Int}, BlockCoupling}
   entries::Vector{OffsiteEntry}
   group::Dict{Tuple{Int,Int}, Vector{Int}}   # (iz,jz) -> entry indices
   nparam::Int
   rcut::Float64
end

function OverlapModel(orbitals::OrbitalBasis;
                      maxn_bond::Integer = 6,
                      rng = Random.default_rng())
   elements = Tuple(orbitals._i2z)
   Lmax = maxL(orbitals)
   # bond radial basis: needs angular channels up to Lmax (every coupled Λ)
   bondradial = _M.ace_learnable_Rnlrzz(; elements = collect(elements),
                  level = _M.TotalDegree(), max_level = maxn_bond + Lmax,
                  maxl = Lmax, maxn = maxn_bond)
   bps = _M.initialparameters(rng, bondradial)
   bst = _M.initialstates(rng, bondradial)
   ybasis = _P4ML.real_sphericalharmonics(Lmax)
   rspec = bondradial.spec                      # Vector of (n=,l=)
   rcut = maximum(c.rcut for c in bondradial.rin0cuts)

   couplings = Dict{Tuple{Int,Int}, BlockCoupling}()
   entries = OffsiteEntry[]
   w = 0
   for iz = 1:length(orbitals._i2z), jz = 1:length(orbitals._i2z)
      shi = orbitals.shells[iz]; shj = orbitals.shells[jz]
      for a = 1:length(shi), b = 1:length(shj)
         la = shi[a].l; lb = shj[b].l
         haskey(couplings, (la, lb)) || (couplings[(la, lb)] = BlockCoupling(la, lb))
         bc = couplings[(la, lb)]
         for Λ in bc.λs
            channel_parity(la, lb, Λ) == :even || continue   # single bond ⇒ even l+l'+Λ
            cols = [ q for q = 1:length(rspec) if rspec[q].l == Λ ]
            isempty(cols) && continue
            push!(entries, OffsiteEntry(iz, jz, a, b, la, lb, Λ, cols,
                                        (w + 1):(w + length(cols))))
            w += length(cols)
         end
      end
   end
   group = Dict{Tuple{Int,Int}, Vector{Int}}()
   for (e, ent) in enumerate(entries)
      push!(get!(group, (ent.iz, ent.jz), Int[]), e)
   end
   return OverlapModel(orbitals, bondradial, bps, bst, ybasis,
                       couplings, entries, group, w, rcut)
end

nparams(model::OverlapModel) = model.nparam

"""random off-site parameter vector for `model`."""
init_params(model::OverlapModel; rng = Random.default_rng()) =
      randn(rng, model.nparam)

# Y-block (2Λ+1 ascending-m values) for degree Λ from the full SH vector.
_yblock(Y, Λ) = @view Y[(Λ^2 + 1):((Λ + 1)^2)]

"""
    assemble_S(model, Zs, bonds, W) -> Matrix

Assemble the overlap matrix for atoms with species `Zs` (atomic numbers) and
`bonds`, a list of `(i, j, 𝐫)` ordered bond triples (`𝐫 = r_j - r_i`, only pairs
within `model.rcut` need be included). `W` is the flat off-site parameter vector.
On-site blocks are the identity; the result is symmetrized `S ← ½(S+Sᵀ)`.
"""
function assemble_S(model::OverlapModel, Zs::AbstractVector{<:Integer},
                    bonds, W::AbstractVector)
   ob = model.orbitals
   layout = OrbitalLayout(ob, Zs)
   S = zeros(Float64, layout.ntot, layout.ntot)

   # on-site: orthonormal basis ⇒ identity on each atom's orbital block
   for i = 1:length(Zs)
      shs = species_shells(ob, Zs[i])
      for a = 1:length(shs)
         r = shell_range(layout, i, a)
         @views S[r, r] .= I(length(r))
      end
   end

   # off-site: bond contributions
   for (i, j, 𝐫) in bonds
      zi = Zs[i]; zj = Zs[j]
      iz = z2i(ob, zi); jz = z2i(ob, zj)
      ents = get(model.group, (iz, jz), nothing)
      ents === nothing && continue
      r = norm(𝐫)
      r > model.rcut && continue
      r̂ = 𝐫 / r
      Rnl = _M.evaluate(model.bondradial, r, zi, zj,
                        model.bondradial_ps, model.bondradial_st)
      Y = _P4ML.evaluate(model.ybasis, SVector{3}(r̂))
      for e in ents
         ent = model.entries[e]
         yΛ = _yblock(Y, ent.Λ)
         w = @view W[ent.wrange]
         rad = sum(w[q] * Rnl[ent.radialcols[q]] for q = 1:length(w))
         vΛ = rad .* yΛ
         blk = transform_λ(model.couplings[(ent.la, ent.lb)], ent.Λ, vΛ)
         ri = shell_range(layout, i, ent.a)
         rj = shell_range(layout, j, ent.b)
         @views S[ri, rj] .+= blk
      end
   end

   return (S + S') ./ 2
end

# ---------------------------------------------------------------------------
# helpers for assembling a configuration

"""
    bondlist(Rs, rcut) -> Vector{(i,j,𝐫)}

Brute-force list of ordered bonds (both directions) with `|r_j - r_i| ≤ rcut`,
for a free (non-periodic) cluster of positions `Rs`. PBC handling via a proper
neighbour list / graph is added in a later stage.
"""
function bondlist(Rs::AbstractVector, rcut::Real)
   bonds = Tuple{Int, Int, SVector{3, Float64}}[]
   for i = 1:length(Rs), j = 1:length(Rs)
      i == j && continue
      𝐫 = SVector{3, Float64}(Rs[j] - Rs[i])
      norm(𝐫) <= rcut && push!(bonds, (i, j, 𝐫))
   end
   return bonds
end
