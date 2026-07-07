using ACEoperators
using Test

@testset "ACEoperators.jl" begin
   @testset "transforms/wignereckart" begin
      include("transforms/test_wignereckart.jl")
   end
end
