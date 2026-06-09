using ACEoperators
using Test

@testset "ACEoperators.jl" begin
   @testset "linear2c/coupling" begin
      include("linear2c/test_coupling.jl")
   end
end
