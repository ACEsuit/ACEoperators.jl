#
# Stage 0 tests — Wigner–Eckart recoupling `transform_λ` (§4)
#
# The block coupling X_{lm,l'm'} ↔ X^{λμ} is the Clebsch–Gordan coupling of two
# independent orbital indices, so ALL triangle-admissible λ appear (both
# parities of l+l'+λ) — the coupling is a pure, complete change of basis. (The
# even/odd l+l'+λ distinction matters only when the model SOURCES a channel from
# geometric features; the coupling itself carries all λ.) The coupling must be
# (a) a complete orthonormal change of basis (couple/decouple exact inverses
# both ways), and (b) intertwine the block transform D^l X (D^{l'})ᵀ with the
# single-irrep transform D^λ v (the property the §12 equivariance of H/S rests
# on).
#
# Part 2 (three-way coupling equivariance) is added in Stage 3.
#

using Test, LinearAlgebra
using EquivariantTensors: O3
using ACEoperators: BlockCoupling, couple, decouple, transform_λ, cg_block,
                    channel_parity

##

@info("Stage 0: transform_λ / Wigner–Eckart recoupling")

ls = 0:2     # s, p, d orbitals
θ = [0.7, 1.1, 0.3]

@testset "all admissible channels present (both parities)" begin
   for l in ls, lp in ls
      bc = BlockCoupling(l, lp)
      @test bc.λs == collect(abs(l - lp):(l + lp))      # full triangle, no parity drop
      for (k, λ) in enumerate(bc.λs)
         @test size(bc.C[k]) == (2λ + 1, 2l + 1, 2lp + 1)
         @test channel_parity(l, lp, λ) == (iseven(l + lp + λ) ? :even : :odd)
         # every admissible channel carries a genuinely nonzero coupling,
         # including the odd-(l+l'+λ) ones
         @test norm(bc.C[k]) > 1e-8
      end
      # out-of-range λ couples to zero
      @test all(iszero, cg_block(l, lp, l + lp + 1))
   end
end

@testset "parametric element type" begin
   bc64 = BlockCoupling(1, 2)                 # default Float64
   @test bc64 isa BlockCoupling{Float64}
   bc32 = BlockCoupling(Float32, 1, 2)
   @test bc32 isa BlockCoupling{Float32}
   @test all(C -> eltype(C) === Float32, bc32.C)
   @test cg_block(Float32, 1, 1, 2) ≈ Float32.(cg_block(1, 1, 2))
   # transform_λ / decouple follow the coupling's precision
   v = randn(Float32, 5)
   @test eltype(transform_λ(bc32, 2, v)) === Float32
   @test eltype(decouple(bc32, [randn(Float32, 2λ + 1) for λ in bc32.λs])) === Float32
end

@testset "complete orthonormal change of basis" begin
   for l in ls, lp in ls
      bc = BlockCoupling(l, lp)
      # stacked coupling over ALL λ must be a full orthogonal D×D map
      Dtot = (2l + 1) * (2lp + 1)
      rows = sum(2λ + 1 for λ in bc.λs)
      @test rows == Dtot
      M = zeros(rows, Dtot)
      r = 0
      for (k, λ) in enumerate(bc.λs), μ = 1:(2λ + 1)
         r += 1
         M[r, :] = vec(bc.C[k][μ, :, :])
      end
      @test M * M' ≈ I(rows)        # orthonormal coupled basis
      @test M' * M ≈ I(Dtot)        # complete
   end
end

@testset "couple/decouple are exact mutual inverses" begin
   for l in ls, lp in ls
      bc = BlockCoupling(l, lp)

      # decouple ∘ couple ≈ id on the FULL block space (no projection loss)
      for _ in 1:5
         X = randn(2l + 1, 2lp + 1)
         @test decouple(bc, couple(bc, X)) ≈ X
      end

      # couple ∘ decouple ≈ id on coupled components, and the block equals the
      # sum of per-λ transforms
      for _ in 1:5
         Xλ = [ randn(2λ + 1) for λ in bc.λs ]
         X = decouple(bc, Xλ)
         @test sum(transform_λ(bc, λ, Xλ[k]) for (k, λ) in enumerate(bc.λs)) ≈ X
         Xλ2 = couple(bc, X)
         @test all(Xλ2[k] ≈ Xλ[k] for k in eachindex(Xλ))
      end
   end
end

@testset "equivariance: decouple intertwines D^l X (D^{l'})ᵀ with D^λ" begin
   # The property the whole §12 equivariance of H/S rests on — holds for both
   # proper and pseudotensor channels.
   for l in ls, lp in ls
      bc  = BlockCoupling(l, lp)
      Dl  = O3.D_from_angles(l,  θ, real)
      Dlp = O3.D_from_angles(lp, θ, real)
      for (k, λ) in enumerate(bc.λs)
         Dλ = O3.D_from_angles(λ, θ, real)
         v  = randn(2λ + 1)
         X     = transform_λ(bc, λ, v)
         Xrot  = transform_λ(bc, λ, Dλ * v)
         @test Xrot ≈ Dl * X * transpose(Dlp)
      end
   end
end

@testset "orbital-swap symmetry carries the (-1)^{l+l'+λ} sign" begin
   # ⟨l m; l' m' | λ μ⟩ = (-1)^{l+l'-λ} ⟨l' m'; l m | λ μ⟩ : transposing the
   # orbital block maps a coupled component to ± itself with σ = (-1)^{l+l'+λ}.
   # Even-`l+l'+λ` channels are symmetric (σ=+1), odd ones antisymmetric (σ=-1)
   # — the seed of the §12.6 / Hermiticity selection rules.
   for l in ls, lp in ls
      bc  = BlockCoupling(l, lp)
      bcT = BlockCoupling(lp, l)
      for (k, λ) in enumerate(bc.λs)
         σ = (-1)^(l + lp + λ)
         @test (σ == 1) == (channel_parity(l, lp, λ) == :even)
         @test bc.C[k] ≈ σ .* permutedims(bcT.C[k], (1, 3, 2))
      end
   end
end
