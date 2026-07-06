#
# twocenter.jl — generic linear two-center (bond-only) model (Stage 1)
#
# A linear, equivariant model for any purely geometric two-center quantity
# X_ij that depends only on the bond vector r_ji = r_j - r_i and the species
# and orbital content of the two sites. For each ordered shell pair
# (n l, n' l') and each admissible coupled angular momentum Λ, the Λ-block of
# X_ij is a linear combination of bond harmonics
# φ^b_{n_q Λ}(r_ji) = R^b_{n_q Λ}(r) Y_{Λ·}(r̂), recoupled into the (m,m')
# block via transform_Λ (§4). Only even-`l+l'+Λ` channels appear, and this is
# COMPLETE for any single-bond quantity: one bond vector supplies only those
# channels, and the count matches the Slater–Koster integrals (e.g. p–p gives
# Λ=0,2 = ppσ,ppπ; the odd Λ=1 axial term is correctly absent). Hermiticity
# is enforced post-hoc by X ← ½(X + Xᵀ) (§7.3).
#
# The assembled matrix is purely OFF-SITE (zero diagonal blocks); callers add
# whatever on-site term their target requires. First use: the overlap matrix
# (§8), S = I + assemble(model, ...) for an orthonormal orbital basis.
#

import ACEpotentials.Models as _M
import Polynomials4ML as _P4ML
import EquivariantTensors as ET
using LinearAlgebra: norm
using StaticArrays: SVector
using Unitful: @u_str

# one off-site contribution: for species pair (iz,jz), shell pair (a,b) with
# angular momenta (la,lb), coupled channel Λ, the radial columns feeding it and
# the slice of the flat parameter vector that weights them.
struct OffsiteEntry
   iz::Int; jz::Int
   a::Int;  b::Int
   la::Int; lb::Int
   Λ::Int
   radialcols::Vector{Int}        # columns of Rnl with l == Λ
   wrange::UnitRange{Int}         # slice of the flat parameter vector
end

"""
    TwoCenterModel(orbitals; maxn_bond=…, kwargs...)

Generic linear two-center (bond-only) model for the orbital list `orbitals`.
Builds a bond radial basis (`ACEpotentials.Models.ace_learnable_Rnlrzz`), a
real spherical harmonic basis up to `maxL(orbitals)`, the Wigner–Eckart
couplings, and the off-site parameter layout. The bond radial weights are
fixed at construction (geometry-only); the learnable parameters are the
off-site coefficients `W`. The assembled matrix is purely off-site — e.g.
the overlap matrix of an orthonormal basis is `S = I + assemble(model, …)`.
"""
struct TwoCenterModel{TR, TY, TRP, TRS}
   orbitals::OrbitalBasis
   bondradial::TR                  # LearnableRnlrzzBasis
   bondradial_ps::TRP              # fixed radial weights
   bondradial_st::TRS
   ybasis::TY                      # real_sphericalharmonics(maxL)
   couplings::Dict{Tuple{Int,Int}, BlockCoupling{Float64}}
   entries::Vector{OffsiteEntry}
   group::Dict{Tuple{Int,Int}, Vector{Int}}   # (iz,jz) -> entry indices
   nparam::Int
   rcut::Float64
end

function TwoCenterModel(orbitals::OrbitalBasis;
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

   couplings = Dict{Tuple{Int,Int}, BlockCoupling{Float64}}()
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
   return TwoCenterModel(orbitals, bondradial, bps, bst, ybasis,
                         couplings, entries, group, w, rcut)
end

nparams(model::TwoCenterModel) = model.nparam

"""random off-site parameter vector for `model`."""
init_params(model::TwoCenterModel; rng = Random.default_rng()) =
      randn(rng, model.nparam)

# Y-block (2Λ+1 ascending-m values) for degree Λ from the full SH vector.
_yblock(Y, Λ) = @view Y[(Λ^2 + 1):((Λ + 1)^2)]

# species → Int atomic number (ETGraph node data may carry AtomsBase
# ChemicalSpecies rather than plain integers)
_atomic_number(z::Integer) = Int(z)
_atomic_number(z) = Int(NeighbourLists.AtomsBase.atomic_number(z))

# accumulate all bond contributions into the global matrix X (kernel shared
# by the bond-list and graph entry points); bonds iterates (i, j, 𝐫) triples
function _assemble_bonds!(X, model::TwoCenterModel, layout, Zs, bonds, W)
   ob = model.orbitals
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
         @views X[ri, rj] .+= blk
      end
   end
   return X
end

"""
    assemble(model::TwoCenterModel, Zs, bonds, W) -> Matrix

Assemble the off-site two-center matrix for atoms with species `Zs` (atomic
numbers) and `bonds`, a list of `(i, j, 𝐫)` ordered bond triples
(`𝐫 = r_j - r_i`, only pairs within `model.rcut` need be included). `W` is
the flat off-site parameter vector. No on-site term is added (diagonal
blocks are zero for a free cluster; under PBC, self-image bonds `i == j`
contribute). The result is symmetrized `X ← ½(X+Xᵀ)`. E.g. the overlap
matrix of an orthonormal orbital basis is
`S = I + assemble(model, Zs, bonds, W)`.
"""
function assemble(model::TwoCenterModel, Zs::AbstractVector{<:Integer},
                  bonds, W::AbstractVector)
   layout = OrbitalLayout(model.orbitals, Zs)
   X = zeros(Float64, layout.ntot, layout.ntot)
   _assemble_bonds!(X, model, layout, Zs, bonds, W)
   return (X + X') ./ 2
end

"""
    graph(model::TwoCenterModel, sys) -> ETGraph

Interaction graph of the AtomsBase system `sys` at the model cutoff, built
via `EquivariantTensors.Atoms.interaction_graph` (NeighbourLists-backed;
periodic boundary conditions are handled by the neighbour list, §3.3).
"""
graph(model::TwoCenterModel, sys) =
      ET.Atoms.interaction_graph(sys, model.rcut * u"Å")

"""
    assemble(model::TwoCenterModel, G::ETGraph, W) -> Matrix

Assemble the off-site two-center matrix from an interaction graph (see
[`graph`](@ref)). Periodic-image bonds contribute additively to the same
`(i, j)` block (Γ-point convention).
"""
function assemble(model::TwoCenterModel, G::ET.ETGraph, W::AbstractVector)
   Zs = [ _atomic_number(x.z) for x in G.node_data ]
   bonds = ( (i, j, e.𝐫) for (i, j, e) in zip(G.ii, G.jj, G.edge_data) )
   return assemble(model, Zs, bonds, W)
end

# ---------------------------------------------------------------------------
# helpers for assembling a configuration

"""
    bondlist(Rs, rcut) -> Vector{(i,j,𝐫)}

Brute-force list of ordered bonds (both directions) with `|r_j - r_i| ≤ rcut`,
for a free (non-periodic) cluster of positions `Rs`. Reference / cross-check
path for the graph-based assembly (`assemble(model, G, W)`), which is the
primary entry point and handles PBC.
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
