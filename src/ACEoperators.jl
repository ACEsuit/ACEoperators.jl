module ACEoperators

# ---------------------------------------------------------------------------
# Linear 2-center Hamiltonian model (see agents/plan_lin2c.md, docs/twocenter.md)
# ---------------------------------------------------------------------------

import Random
import ACEpotentials
import NeighbourLists    # activates EquivariantTensors' graph extension

include("linear2c/coupling.jl")   # Wigner-Eckart recoupling, transform_λ (§4)
include("linear2c/blocks.jl")     # orbital list + global block bookkeeping (§10)
include("linear2c/twocenter.jl")  # generic bond-only 2C model (used for S, §8)

end
