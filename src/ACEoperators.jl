module ACEoperators

# ---------------------------------------------------------------------------
# Linear 2-center Hamiltonian model (see agents/plan_lin2c.md, docs/twocenter.md)
# ---------------------------------------------------------------------------

import Random
import ACEpotentials

include("linear2c/coupling.jl")   # Wigner-Eckart recoupling, transform_λ (§4)
include("linear2c/blocks.jl")     # orbital list + global block bookkeeping (§10)
include("linear2c/overlap.jl")    # two-center overlap model S (§8, Stage 1)

end
