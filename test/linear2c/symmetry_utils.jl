#
# Shared helpers for the §12 symmetry tests (used by test_overlap, test_onsite, …)
#
using EquivariantTensors: O3
import ACEoperators as A

# block-Wigner matrix 𝓓(Q) = blockdiag over (atom, shell) of D^l(Q), for a proper
# rotation given by ZYZ Euler angles θ.
function blockwigner(orbitals, Zs, θ)
   layout = A.OrbitalLayout(orbitals, Zs)
   D = zeros(layout.ntot, layout.ntot)
   for i = 1:length(Zs)
      shs = A.species_shells(orbitals, Zs[i])
      for a = 1:length(shs)
         r = A.shell_range(layout, i, a)
         D[r, r] = O3.D_from_angles(shs[a].l, θ, real)
      end
   end
   return D
end

# parity matrix for inversion: blockdiag over (atom, shell) of (-1)^l I.
function parity_matrix(orbitals, Zs)
   layout = A.OrbitalLayout(orbitals, Zs)
   P = zeros(layout.ntot, layout.ntot)
   for i = 1:length(Zs)
      shs = A.species_shells(orbitals, Zs[i])
      for a = 1:length(shs)
         r = A.shell_range(layout, i, a)
         P[r, r] = (-1)^(shs[a].l) * I(length(r))
      end
   end
   return P
end

# orbital permutation matrix induced by an atom permutation π (single species:
# uniform per-atom orbital count `norb`).
function atom_perm_matrix(π, norb)
   n = length(π) * norb
   P = zeros(Int, n, n)
   for k = 1:length(π)
      P[(k-1)*norb .+ (1:norb), (π[k]-1)*norb .+ (1:norb)] = I(norb)
   end
   return P
end
