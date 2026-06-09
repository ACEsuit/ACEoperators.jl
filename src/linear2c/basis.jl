#
# basis.jl — environment ACE features 𝓑_i (§2, §3) and bond features
#
# `EnvBasis` builds, per atomic site, the equivariant ACE features
# B_i^{Lμ} = Σ_v C^{Lμ}_v ∏ A_{i,nlm} from the local environment, using
# EquivariantTensors' `sparse_equivariant_tensors` (pooling + symmetrisation) on
# top of an ACEpotentials radial basis and real spherical harmonics. The same
# 𝓑_i is shared by the on-site model (Stage 2) and the off-site model (Stage 4).
#
# Only PROPER-tensor features are produced (the real-CG construction in
# EquivariantTensors couples to proper tensors, parity (-1)^L). Pseudotensor
# (odd-parity) features are not available from this construction; consequently
# the on-/off-site models populate only the matching-parity (l+l'+λ even)
# channels — see coupling.jl and §5.2.
#

import ACEpotentials.Models as _M
import Polynomials4ML as _P4ML
import EquivariantTensors as _ET
using LinearAlgebra: norm
using StaticArrays: SVector

"""
    EnvBasis(elements; Lmax, ORD=2, maxl=Lmax, totaldegree=6, rng=…)

Equivariant environment ACE basis producing features of degrees `L = 0:Lmax` at
each site, for the given `elements` (atomic numbers or symbols). `ORD` is the
correlation order, `totaldegree` the total-degree truncation, and `maxl` the
angular truncation of the one-particle basis (raised to at least `Lmax`).
"""
struct EnvBasis{TR, TY, TT, TRP, TRS}
   _i2z::Vector{Int}
   radial::TR
   radial_ps::TRP
   radial_st::TRS
   ybasis::TY
   tensor::TT
   Lmax::Int
   rcut::Float64
end

function EnvBasis(elements;
                  Lmax::Integer,
                  ORD::Integer = 2,
                  maxl::Integer = Lmax,
                  totaldegree::Integer = 6,
                  rng = Random.default_rng())
   zs = Int[ _asz(e) for e in elements ]
   maxl_env = max(Int(maxl), Int(Lmax))           # Y must reach every output L
   level = bb -> sum((b.n + b.l) for b in bb; init = 0)
   mb_spec = _ET.sparse_nnll_set(; ORD = ORD, minn = 1, maxn = totaldegree,
                                   maxl = maxl_env, level = level,
                                   maxlevel = totaldegree)
   rspec = sort(unique([ (n = b.n, l = b.l) for bb in mb_spec for b in bb ]))
   maxn = maximum(s.n for s in rspec)
   radial = _M.ace_learnable_Rnlrzz(; elements = zs, spec = collect(rspec),
                                      maxn = maxn, maxl = maxl_env)
   rps = _M.initialparameters(rng, radial)
   rst = _M.initialstates(rng, radial)
   ybasis = _P4ML.real_sphericalharmonics(maxl_env)
   Ylm_spec = [ (l = l, m = m) for l = 0:maxl_env for m = -l:l ]
   tensor = _ET.sparse_equivariant_tensors(; LL = Tuple(0:Int(Lmax)),
                  mb_spec = mb_spec, Rnl_spec = collect(rspec),
                  Ylm_spec = Ylm_spec, basis = real)
   rcut = maximum(c.rcut for c in radial.rin0cuts)
   return EnvBasis(zs, radial, rps, rst, ybasis, tensor, Int(Lmax), rcut)
end

"""number of equivariant features of degree `L` produced per site."""
nfeatures(eb::EnvBasis, L::Integer) = length(eb.tensor, L)

"""
    site_features(eb, z0, Rs_env, Zs_env) -> NTuple

Equivariant ACE features at a site of species `z0` (atomic number) with neighbour
displacements `Rs_env` (`Vector{SVector{3}}`, relative to the site) and neighbour
species `Zs_env`. Returns the tuple `BB` with `BB[L+1]` the degree-`L` features
(each an `SVector{2L+1}`; degree 0 are scalars).
"""
function site_features(eb::EnvBasis, z0::Integer,
                       Rs_env::AbstractVector, Zs_env::AbstractVector)
   Rnl = _M.evaluate_batched(eb.radial, norm.(Rs_env), z0, Zs_env,
                             eb.radial_ps, eb.radial_st)
   Ylm = _P4ML.evaluate(eb.ybasis, Rs_env)
   return _ET.evaluate(eb.tensor, Rnl, Ylm, NamedTuple(), NamedTuple())
end
