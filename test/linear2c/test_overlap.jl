#
# Stage 1 tests — two-center overlap model S (§8) and the §12 symmetry suite
# restricted to S: rotation, inversion (+parity), Hermiticity, permutation,
# cutoff smoothness. (Translation §12.4 is skipped per the plan: S depends only
# on relative bond vectors.)
#

using Test, LinearAlgebra, StaticArrays, Random
using EquivariantTensors: O3
import ACEoperators as A

include("symmetry_utils.jl")

##

@info("Stage 1: two-center overlap model S")

rng = MersenneTwister(2024)

assemble(model, Zs, Rs, W) = A.assemble_S(model, Zs, A.bondlist(Rs, model.rcut), W)

# a small mixed cluster
ob = A.OrbitalBasis(:Si => [(n=3, l=0), (n=3, l=1), (n=3, l=2)],
                    :O  => [(n=2, l=0), (n=2, l=1)])
model = A.OverlapModel(ob; maxn_bond = 5, rng = rng)
W = A.init_params(model; rng = rng)

Rs0 = [SVector(0.0, 0.0, 0.0), SVector(2.2, 0.3, 0.0),
       SVector(-0.4, 2.0, 0.5), SVector(1.1, -1.3, 1.8)]
Zs  = [14, 8, 14, 8]

@testset "assembly basics" begin
   S = assemble(model, Zs, Rs0, W)
   n = sum(A.species_norb(ob, z) for z in Zs)
   @test size(S) == (n, n)
   @test S ≈ S'                                   # Hermiticity (§12.3)
   # on-site blocks are the identity
   layout = A.OrbitalLayout(ob, Zs)
   for i = 1:length(Zs)
      r = layout.offset[i] .+ (1:A.species_norb(ob, Zs[i]))
      @test S[r, r] ≈ I(length(r))
   end
   # off-site blocks are nonzero (the model is doing something)
   r1 = layout.offset[1] .+ (1:A.species_norb(ob, Zs[1]))
   r2 = layout.offset[2] .+ (1:A.species_norb(ob, Zs[2]))
   @test norm(S[r1, r2]) > 1e-6
end

@testset "rotational equivariance (§12.1)" begin
   S = assemble(model, Zs, Rs0, W)
   for _ in 1:3
      θ = 2π .* rand(rng, 3)
      Q = O3.Q_from_angles(θ)
      Rs_rot = [SVector{3}(Q * r) for r in Rs0]
      S_rot = assemble(model, Zs, Rs_rot, W)
      𝓓 = blockwigner(model.orbitals, Zs, θ)
      @test S_rot ≈ 𝓓 * S * 𝓓'
   end
end

@testset "inversion equivariance + parity (§12.2)" begin
   S = assemble(model, Zs, Rs0, W)
   Rs_inv = [-r for r in Rs0]
   S_inv = assemble(model, Zs, Rs_inv, W)
   P = parity_matrix(model.orbitals, Zs)
   @test S_inv ≈ P * S * P'
   # the parity action is nontrivial (some l are odd) — guard against a trivial pass
   @test !(P ≈ I(size(P, 1)))
   @test S_inv ≉ S
end

@testset "permutation equivariance (§12.5)" begin
   # single species so the orbital layout is uniform and the atom permutation
   # lifts to a clean block permutation of S
   ob1 = A.OrbitalBasis(:Si => [(n=3, l=0), (n=3, l=1)])
   m1 = A.OverlapModel(ob1; maxn_bond = 4, rng = rng)
   W1 = A.init_params(m1; rng = rng)
   Rs = [SVector(0.0,0.0,0.0), SVector(2.0,0.1,0.0),
         SVector(0.2,1.9,0.3), SVector(-1.0,1.0,1.5)]
   Z1 = fill(14, 4)
   S = assemble(m1, Z1, Rs, W1)
   π = [3, 1, 4, 2]
   Perm = atom_perm_matrix(π, A.species_norb(ob1, 14))
   S_perm = assemble(m1, Z1[π], Rs[π], W1)
   @test S_perm ≈ Perm * S * Perm'
end

@testset "cutoff smoothness (§12.7)" begin
   ob1 = A.OrbitalBasis(:Si => [(n=3, l=0), (n=3, l=1)])
   m1 = A.OverlapModel(ob1; maxn_bond = 4, rng = rng)
   W1 = A.init_params(m1; rng = rng)
   norb = A.species_norb(ob1, 14)
   # off-site block norm as a bond stretches across the cutoff
   f(d) = begin
      Rs = [SVector(0.0,0.0,0.0), SVector(d, 0.0, 0.0)]
      S = assemble(m1, fill(14, 2), Rs, W1)
      norm(S[1:norb, (norb+1):(2norb)])
   end
   ds = range(m1.rcut - 0.6, m1.rcut + 0.3; length = 200)
   vals = f.(ds)
   # vanishes beyond the cutoff
   @test all(abs.(vals[ds .>= m1.rcut]) .< 1e-9)
   # continuous (C^0): no jumps along the sweep
   @test maximum(abs.(diff(vals))) < 1e-2
   # first difference also goes smoothly to zero at the cutoff (C^1 envelope)
   d1 = diff(vals) ./ step(ds)
   @test abs(d1[end]) < 1e-3
end
