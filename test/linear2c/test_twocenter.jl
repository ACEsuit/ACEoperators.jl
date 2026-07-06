#
# Stage 1 tests — generic two-center (bond-only) model and the §12 symmetry
# suite restricted to it: rotation, inversion (+parity), Hermiticity,
# permutation, cutoff smoothness, graph input + PBC. The overlap matrix S is
# the first instantiation: S = I + assemble(model, …). (Translation §12.4 is
# skipped for free clusters per the plan — the model depends only on relative
# bond vectors — but IS tested through the PBC wrap below.)
#

using Test, LinearAlgebra, StaticArrays, Random
using EquivariantTensors: O3
using AtomsBase, AtomsBuilder, Unitful
import ACEoperators as A

##

@info("Stage 1: generic two-center model (overlap S = I + 2C)")

rng = MersenneTwister(2024)

# --- helpers ---------------------------------------------------------------

# block-Wigner matrix 𝓓(Q) = blockdiag over (atom, shell) of D^l(Q)
function blockwigner(model, Zs, θ)
   ob = model.orbitals
   layout = A.OrbitalLayout(ob, Zs)
   D = zeros(layout.ntot, layout.ntot)
   for i = 1:length(Zs)
      shs = A.species_shells(ob, Zs[i])
      for a = 1:length(shs)
         r = A.shell_range(layout, i, a)
         D[r, r] = O3.D_from_angles(shs[a].l, θ, real)
      end
   end
   return D
end

# parity matrix for inversion: blockdiag over (atom, shell) of (-1)^l I
function parity_matrix(model, Zs)
   ob = model.orbitals
   layout = A.OrbitalLayout(ob, Zs)
   P = zeros(layout.ntot, layout.ntot)
   for i = 1:length(Zs)
      shs = A.species_shells(ob, Zs[i])
      for a = 1:length(shs)
         r = A.shell_range(layout, i, a)
         P[r, r] = (-1)^(shs[a].l) * I(length(r))
      end
   end
   return P
end

assemble(model, Zs, Rs, W) = A.assemble(model, Zs, A.bondlist(Rs, model.rcut), W)

# free cluster as an AtomsBase system: periodic box with a large vacuum
function vacuum_system(Zs, Rs; L = 100.0)
   atoms = [ Atom(Int(Zs[i]), SVector{3}(Rs[i])u"Å") for i = 1:length(Zs) ]
   box = (SVector(L, 0.0, 0.0)u"Å", SVector(0.0, L, 0.0)u"Å",
          SVector(0.0, 0.0, L)u"Å")
   return periodic_system(atoms, box)
end

# a small mixed cluster
ob = A.OrbitalBasis(:Si => [(n=3, l=0), (n=3, l=1), (n=3, l=2)],
                    :O  => [(n=2, l=0), (n=2, l=1)])
model = A.TwoCenterModel(ob; maxn_bond = 5, rng = rng)
W = A.init_params(model; rng = rng)

Rs0 = [SVector(0.0, 0.0, 0.0), SVector(2.2, 0.3, 0.0),
       SVector(-0.4, 2.0, 0.5), SVector(1.1, -1.3, 1.8)]
Zs  = [14, 8, 14, 8]

@testset "assembly basics" begin
   X = assemble(model, Zs, Rs0, W)
   n = sum(A.species_norb(ob, z) for z in Zs)
   @test size(X) == (n, n)
   @test X ≈ X'                                   # Hermiticity (§12.3)
   # off-site only: on-site (diagonal) blocks vanish for a free cluster
   layout = A.OrbitalLayout(ob, Zs)
   for i = 1:length(Zs)
      r = layout.offset[i] .+ (1:A.species_norb(ob, Zs[i]))
      @test norm(X[r, r]) == 0
   end
   # off-site blocks are nonzero (the model is doing something)
   r1 = layout.offset[1] .+ (1:A.species_norb(ob, Zs[1]))
   r2 = layout.offset[2] .+ (1:A.species_norb(ob, Zs[2]))
   @test norm(X[r1, r2]) > 1e-6
   # the overlap instantiation: S = I + X is Hermitian with identity on-site
   S = I + X
   @test S ≈ S'
   for i = 1:length(Zs)
      r = layout.offset[i] .+ (1:A.species_norb(ob, Zs[i]))
      @test S[r, r] ≈ I(length(r))
   end
end

@testset "rotational equivariance (§12.1)" begin
   X = assemble(model, Zs, Rs0, W)
   for _ in 1:3
      θ = 2π .* rand(rng, 3)
      Q = O3.Q_from_angles(θ)
      Rs_rot = [SVector{3}(Q * r) for r in Rs0]
      X_rot = assemble(model, Zs, Rs_rot, W)
      𝓓 = blockwigner(model, Zs, θ)
      @test X_rot ≈ 𝓓 * X * 𝓓'
   end
end

@testset "inversion equivariance + parity (§12.2)" begin
   X = assemble(model, Zs, Rs0, W)
   Rs_inv = [-r for r in Rs0]
   X_inv = assemble(model, Zs, Rs_inv, W)
   P = parity_matrix(model, Zs)
   @test X_inv ≈ P * X * P'
   # the parity action is nontrivial (some l are odd) — guard against a trivial pass
   @test !(P ≈ I(size(P, 1)))
   @test X_inv ≉ X
end

@testset "permutation equivariance (§12.5)" begin
   # single species so the orbital layout is uniform and the atom permutation
   # lifts to a clean block permutation of X
   ob1 = A.OrbitalBasis(:Si => [(n=3, l=0), (n=3, l=1)])
   m1 = A.TwoCenterModel(ob1; maxn_bond = 4, rng = rng)
   W1 = A.init_params(m1; rng = rng)
   Rs = [SVector(0.0,0.0,0.0), SVector(2.0,0.1,0.0),
         SVector(0.2,1.9,0.3), SVector(-1.0,1.0,1.5)]
   Z1 = fill(14, 4)
   X = assemble(m1, Z1, Rs, W1)
   π = [3, 1, 4, 2]
   norb = A.species_norb(ob1, 14)
   # orbital permutation matrix induced by π
   Perm = zeros(Int, length(π) * norb, length(π) * norb)
   for k = 1:length(π)
      dst = (k - 1) * norb
      src = (π[k] - 1) * norb
      Perm[dst .+ (1:norb), src .+ (1:norb)] = I(norb)
   end
   X_perm = assemble(m1, Z1[π], Rs[π], W1)
   @test X_perm ≈ Perm * X * Perm'
end

@testset "cutoff smoothness (§12.7)" begin
   ob1 = A.OrbitalBasis(:Si => [(n=3, l=0), (n=3, l=1)])
   m1 = A.TwoCenterModel(ob1; maxn_bond = 4, rng = rng)
   W1 = A.init_params(m1; rng = rng)
   norb = A.species_norb(ob1, 14)
   # off-site block norm as a bond stretches across the cutoff
   f(d) = begin
      Rs = [SVector(0.0,0.0,0.0), SVector(d, 0.0, 0.0)]
      X = assemble(m1, fill(14, 2), Rs, W1)
      norm(X[1:norb, (norb+1):(2norb)])
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

@testset "graph input + PBC (§3.3)" begin
   # (a) free cluster: the ETGraph path reproduces the brute-force bondlist path
   sys = vacuum_system(Zs, Rs0)
   G = A.graph(model, sys)
   X_graph = A.assemble(model, G, W)
   X_ref = assemble(model, Zs, Rs0, W)
   @test X_graph ≈ X_ref

   # a rattled periodic Si cell; the cutoff spans the cell boundary, so
   # periodic-image bonds (including self-images) genuinely contribute
   ob1 = A.OrbitalBasis(:Si => [(n=3, l=0), (n=3, l=1)])
   m1 = A.TwoCenterModel(ob1; maxn_bond = 4, rng = rng)
   W1 = A.init_params(m1; rng = rng)
   sys0 = rattle!(bulk(:Si, cubic = true), 0.1)
   nat = length(sys0)
   Z1 = fill(14, nat)
   cell = cell_vectors(sys0)
   Lcell = ustrip(u"Å", cell[1][1])
   @assert m1.rcut > Lcell / 2      # boundary-crossing bonds present
   G0 = A.graph(m1, sys0)
   X0 = A.assemble(m1, G0, W1)

   # (b) PBC Hermiticity: wrapped and self-image bonds must symmetrize
   @test X0 ≈ X0'
   @test norm(X0) > 1e-6

   # (c) PBC translation invariance (§12.4 with wrap): rigidly translating
   # all atoms (wrapped back into the cell) leaves the matrix unchanged
   t = SVector(0.7, -1.3, 2.1)
   pos = [ ustrip.(u"Å", position(sys0, i)) for i = 1:nat ]
   pos_t = [ mod.(p .+ t, Lcell) for p in pos ]
   sys_t = periodic_system(
         [ Atom(14, SVector{3}(p)u"Å") for p in pos_t ], cell)
   X_t = A.assemble(m1, A.graph(m1, sys_t), W1)
   @test X_t ≈ X0

   # (d) rotational equivariance with PBC: rotate positions AND cell
   θ = 2π .* rand(rng, 3)
   Q = O3.Q_from_angles(θ)
   cell_rot = ntuple(k -> SVector{3}(Q * ustrip.(u"Å", cell[k]))u"Å", 3)
   sys_rot = periodic_system(
         [ Atom(14, SVector{3}(Q * p)u"Å") for p in pos ], cell_rot)
   X_rot = A.assemble(m1, A.graph(m1, sys_rot), W1)
   𝓓 = blockwigner(m1, Z1, θ)
   @test X_rot ≈ 𝓓 * X0 * 𝓓'
end
