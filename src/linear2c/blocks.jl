#
# blocks.jl — orbital list and global matrix block bookkeeping (§10, §14)
#
# The H/S matrices are expressed in a fixed, atom-centered basis of real atomic
# orbitals. Each species carries an ordered list of shells (n, l); a shell of
# angular momentum l contributes 2l+1 orbitals (m = -l:l). This file owns the
# mapping from (atom, shell) to global row/column ranges, used by the on-site
# and off-site assembly loops.
#

const Shell = @NamedTuple{n::Int, l::Int}

"""
    OrbitalBasis(elements => shells, ...)

The per-species orbital list. `elements` are atomic numbers (Int) or symbols;
`shells` is an ordered list of `(n=…, l=…)` shells for that species. Example:

    OrbitalBasis(:Si => [(n=3,l=0), (n=3,l=1)],
                 :O  => [(n=2,l=0), (n=2,l=1)])
"""
struct OrbitalBasis
   _i2z::Vector{Int}                 # species atomic numbers, in index order
   shells::Vector{Vector{Shell}}     # shells[iz] = ordered shells of species iz
end

_sym2z(s::Symbol) = ACEpotentials.Models._convert_zlist((s,))[1]
_asz(z::Integer) = Int(z)
_asz(s::Symbol) = _sym2z(s)

function OrbitalBasis(pairs::Vararg{Pair})
   zs = Int[ _asz(first(p)) for p in pairs ]
   shells = Vector{Shell}[ Shell[ (n = Int(s.n), l = Int(s.l)) for s in last(p) ]
                           for p in pairs ]
   return OrbitalBasis(zs, shells)
end

"""species index (1-based) of atomic number `z` in the orbital basis."""
function z2i(ob::OrbitalBasis, z::Integer)
   i = findfirst(==(Int(z)), ob._i2z)
   i === nothing && error("species z=$z not in orbital basis")
   return i
end

species_shells(ob::OrbitalBasis, z::Integer) = ob.shells[z2i(ob, z)]

shell_size(s::Shell) = 2 * s.l + 1

"""number of orbitals (m-components) carried by species `z`."""
species_norb(ob::OrbitalBasis, z::Integer) =
      sum(shell_size, species_shells(ob, z); init = 0)

"""maximum orbital angular momentum across all species."""
lmax_orb(ob::OrbitalBasis) =
      maximum(s.l for shs in ob.shells for s in shs; init = 0)

"""maximum coupled angular momentum Λ that can arise (= 2 lmax_orb)."""
maxL(ob::OrbitalBasis) = 2 * lmax_orb(ob)

# ---------------------------------------------------------------------------
# Layout of a concrete configuration

"""
    OrbitalLayout(ob, Zs)

Global row/column layout of the H/S matrix for atoms with species `Zs` (atomic
numbers). `offset[i]` is the 0-based global index where atom `i`'s orbitals
start; `ntot` is the matrix dimension.
"""
struct OrbitalLayout
   ob::OrbitalBasis
   Zs::Vector{Int}
   offset::Vector{Int}     # 0-based global start of each atom's orbital block
   ntot::Int
end

function OrbitalLayout(ob::OrbitalBasis, Zs::AbstractVector{<:Integer})
   nat = length(Zs)
   offset = Vector{Int}(undef, nat)
   o = 0
   for i = 1:nat
      offset[i] = o
      o += species_norb(ob, Zs[i])
   end
   return OrbitalLayout(ob, Vector{Int}(Zs), offset, o)
end

"""
    shell_range(layout, i, a)

Global index range (`UnitRange`) of the `a`-th shell of atom `i`.
"""
function shell_range(layout::OrbitalLayout, i::Integer, a::Integer)
   shs = species_shells(layout.ob, layout.Zs[i])
   start = layout.offset[i]
   for k = 1:(a - 1)
      start += shell_size(shs[k])
   end
   return (start + 1):(start + shell_size(shs[a]))
end
