#
# onsite.jl — on-site Hamiltonian blocks H_ii (§5, Stage 2)
#
# H_ii depends on the local atomic environment of site i. For each shell pair
# (n l, n' l') of the center species and each matching-parity coupled angular
# momentum λ (l+l'+λ even, §5.2), the λ-block of H_ii is a linear combination of
# the equivariant environment features B_i^{λ} (from EnvBasis), recoupled into
# the (m,m') block via transform_λ (§4). The on-site contribution is
# block-diagonal in the atom index; Hermiticity is enforced post-hoc by
# H ← ½(H + Hᵀ) (§7.3).
#
# Like the overlap model, only proper-parity channels are populated: EnvBasis
# yields proper-tensor features, so the pseudotensor (odd l+l'+λ) channels are
# not representable here and are left at zero — which also gives the §12.6
# vanishing of odd-λ diagonal-shell components.
#

using LinearAlgebra: norm, I
using StaticArrays: SVector

# one on-site contribution: center species iz, shell pair (a,b) with angular
# momenta (la,lb), coupled channel λ, the number of degree-λ env features and
# the slice of the parameter vector weighting them.
struct OnsiteEntry
   iz::Int
   a::Int;  b::Int
   la::Int; lb::Int
   λ::Int
   nfeat::Int
   wrange::UnitRange{Int}
end

"""
    OnsiteModel(orbitals; ORD=2, totaldegree=6, rng=…)

On-site Hamiltonian model (§5) for the orbital list `orbitals`. Builds the
environment ACE basis `EnvBasis` (degrees `0:maxL(orbitals)`), the Wigner–Eckart
couplings, and the on-site parameter layout. Learnable parameters are the
on-site coefficients `W`; the environment radial weights are fixed at
construction.
"""
struct OnsiteModel{TE}
   orbitals::OrbitalBasis
   env::TE                          # EnvBasis
   couplings::Dict{Tuple{Int,Int}, BlockCoupling}
   entries::Vector{OnsiteEntry}
   group::Dict{Int, Vector{Int}}    # center-species index -> entry indices
   nparam::Int
   rcut::Float64
end

function OnsiteModel(orbitals::OrbitalBasis;
                     ORD::Integer = 2, totaldegree::Integer = 6,
                     rng = Random.default_rng())
   Lmax = maxL(orbitals)
   env = EnvBasis(orbitals._i2z; Lmax = Lmax, ORD = ORD,
                  totaldegree = totaldegree, rng = rng)
   couplings = Dict{Tuple{Int,Int}, BlockCoupling}()
   entries = OnsiteEntry[]
   w = 0
   for iz = 1:length(orbitals._i2z)
      shi = orbitals.shells[iz]
      for a = 1:length(shi), b = 1:length(shi)        # all ordered shell pairs
         la = shi[a].l; lb = shi[b].l
         haskey(couplings, (la, lb)) || (couplings[(la, lb)] = BlockCoupling(la, lb))
         bc = couplings[(la, lb)]
         for (k, λ) in enumerate(bc.λs)
            bc.parities[k] == :proper || continue     # proper-tensor features only
            nf = nfeatures(env, λ)
            nf == 0 && continue
            push!(entries, OnsiteEntry(iz, a, b, la, lb, λ, nf, (w + 1):(w + nf)))
            w += nf
         end
      end
   end
   group = Dict{Int, Vector{Int}}()
   for (e, ent) in enumerate(entries)
      push!(get!(group, ent.iz, Int[]), e)
   end
   return OnsiteModel(orbitals, env, couplings, entries, group, w, env.rcut)
end

nparams(model::OnsiteModel) = model.nparam

init_params(model::OnsiteModel; rng = Random.default_rng()) =
      randn(rng, model.nparam)

"""
    assemble_Honsite(model, Zs, Rs, W) -> Matrix

Assemble the on-site (block-diagonal) part of `H` for atoms with species `Zs`
(atomic numbers) at positions `Rs` (`Vector{SVector{3}}`). For each atom the
environment features `B_i^{λ}` are contracted with the on-site weights `W` and
recoupled into the orbital blocks. Off-diagonal atom blocks are zero (added by
the off-site model in Stage 4). The result is symmetrized `H ← ½(H+Hᵀ)`.
"""
function assemble_Honsite(model::OnsiteModel, Zs::AbstractVector{<:Integer},
                          Rs::AbstractVector, W::AbstractVector)
   ob = model.orbitals
   layout = OrbitalLayout(ob, Zs)
   H = zeros(Float64, layout.ntot, layout.ntot)

   for i = 1:length(Zs)
      zi = Zs[i]
      iz = z2i(ob, zi)
      ents = get(model.group, iz, nothing)
      ents === nothing && continue
      # local environment of site i (neighbours within the env cutoff)
      Renv = SVector{3, Float64}[]
      Zenv = Int[]
      for j = 1:length(Zs)
         i == j && continue
         𝐫 = SVector{3, Float64}(Rs[j] - Rs[i])
         if norm(𝐫) <= model.rcut
            push!(Renv, 𝐫); push!(Zenv, Zs[j])
         end
      end
      isempty(Renv) && continue
      BB = site_features(model.env, zi, Renv, Zenv)
      for e in ents
         ent = model.entries[e]
         Bλ = BB[ent.λ + 1]                       # degree-λ features (LL = 0:Lmax)
         w = @view W[ent.wrange]
         vλ = sum(w[q] * Bλ[q] for q = 1:ent.nfeat)
         vλv = vλ isa Number ? SVector(vλ) : vλ    # degree-0 features are scalars
         blk = transform_λ(model.couplings[(ent.la, ent.lb)], ent.λ, vλv)
         ri = shell_range(layout, i, ent.a)
         rj = shell_range(layout, i, ent.b)
         @views H[ri, rj] .+= blk
      end
   end

   return (H + H') ./ 2
end
