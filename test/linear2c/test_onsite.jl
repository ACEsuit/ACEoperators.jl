#
# Stage 2 tests — on-site Hamiltonian model H_ii (§5) and the §12 symmetry suite
# restricted to the on-site (block-diagonal) part: rotation, inversion (+parity),
# Hermiticity, permutation, plus the §12.6 vanishing of odd-λ components on
# diagonal shell pairs.
#

using Test, LinearAlgebra, StaticArrays, Random
using EquivariantTensors: O3
import ACEoperators as A

include("symmetry_utils.jl")

##

@info("Stage 2: on-site Hamiltonian model H_ii")

rng = MersenneTwister(7)

ob = A.OrbitalBasis(:Si => [(n=3, l=0), (n=3, l=1)],
                    :O  => [(n=2, l=0), (n=2, l=1)])
model = A.OnsiteModel(ob; ORD = 2, totaldegree = 6, rng = rng)
W = A.init_params(model; rng = rng)

# a cluster with neighbours inside the env cutoff for every atom
Rs0 = [SVector(0.0, 0.0, 0.0), SVector(2.2, 0.3, 0.0),
       SVector(-0.4, 2.0, 0.5), SVector(1.1, -1.3, 1.8)]
Zs  = [14, 8, 14, 8]

@testset "assembly basics" begin
   H = A.assemble_Honsite(model, Zs, Rs0, W)
   n = sum(A.species_norb(ob, z) for z in Zs)
   @test size(H) == (n, n)
   @test H ≈ H'                                       # Hermiticity (§12.3)
   layout = A.OrbitalLayout(ob, Zs)
   # on-site is block-diagonal in the atom index, and nonzero
   r1 = layout.offset[1] .+ (1:A.species_norb(ob, Zs[1]))
   r2 = layout.offset[2] .+ (1:A.species_norb(ob, Zs[2]))
   @test norm(H[r1, r2]) == 0
   @test norm(H[r1, r1]) > 1e-6
end

@testset "rotational equivariance (§12.1)" begin
   H = A.assemble_Honsite(model, Zs, Rs0, W)
   for _ in 1:3
      θ = 2π .* rand(rng, 3)
      Q = O3.Q_from_angles(θ)
      Rs_rot = [SVector{3}(Q * r) for r in Rs0]
      H_rot = A.assemble_Honsite(model, Zs, Rs_rot, W)
      𝓓 = blockwigner(model.orbitals, Zs, θ)
      @test H_rot ≈ 𝓓 * H * 𝓓'
   end
end

@testset "inversion equivariance + parity (§12.2)" begin
   H = A.assemble_Honsite(model, Zs, Rs0, W)
   H_inv = A.assemble_Honsite(model, Zs, [-r for r in Rs0], W)
   P = parity_matrix(model.orbitals, Zs)
   @test H_inv ≈ P * H * P'
   @test !(P ≈ I(size(P, 1)))
   @test H_inv ≉ H
end

@testset "permutation equivariance (§12.5)" begin
   ob1 = A.OrbitalBasis(:Si => [(n=3, l=0), (n=3, l=1)])
   m1 = A.OnsiteModel(ob1; ORD = 2, totaldegree = 6, rng = rng)
   W1 = A.init_params(m1; rng = rng)
   Rs = [SVector(0.0,0.0,0.0), SVector(2.0,0.1,0.0),
         SVector(0.2,1.9,0.3), SVector(-1.0,1.0,1.5)]
   Z1 = fill(14, 4)
   H = A.assemble_Honsite(m1, Z1, Rs, W1)
   π = [3, 1, 4, 2]
   Perm = atom_perm_matrix(π, A.species_norb(ob1, 14))
   H_perm = A.assemble_Honsite(m1, Z1[π], Rs[π], W1)
   @test H_perm ≈ Perm * H * Perm'
end

@testset "odd-λ vanishing on diagonal shell pairs (§12.6)" begin
   # On a diagonal shell pair (l, l) the even-`l+l'+λ` channels are λ even
   # (2l+λ even). The odd-λ coupled components of an on-site block vanish
   # identically (Gaunt rule: both orbitals share the center), so the assembled
   # block must have zero projection onto them.
   H = A.assemble_Honsite(model, Zs, Rs0, W)
   layout = A.OrbitalLayout(ob, Zs)
   for i = 1:length(Zs)
      shs = A.species_shells(ob, Zs[i])
      for a = 1:length(shs)
         l = shs[a].l
         r = A.shell_range(layout, i, a)
         blk = H[r, r]
         bc = A.BlockCoupling(l, l)
         Xλ = A.couple(bc, blk)
         for (k, λ) in enumerate(bc.λs)
            if isodd(2l + λ)               # odd-l+l'+λ channel (must vanish)
               @test norm(Xλ[k]) < 1e-10
            end
         end
      end
   end
end
