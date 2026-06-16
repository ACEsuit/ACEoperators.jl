using ACEoperators
using Test

@testset "ACEoperators.jl" begin
   @testset "linear2c/coupling" begin
      include("linear2c/test_coupling.jl")
   end
   @testset "linear2c/overlap" begin
      include("linear2c/test_overlap.jl")
   end
   @testset "linear2c/onsite" begin
      include("linear2c/test_onsite.jl")
   end
end
