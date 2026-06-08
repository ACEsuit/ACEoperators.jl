# A Linear ACE Model for Effective Hamiltonians

This note sketches a linear, equivariant ACE-based model for an effective
single-particle Hamiltonian $H$ and overlap matrix $S$ written in a basis of
real atomic orbitals. It is meant as a discussion document: the goal is to pin
down notation and the overall architecture before writing any code. The
construction follows the general philosophy of Drautz's Atomic Cluster
Expansion (ACE) for the equivariant ("density-correlation") features, and the
symmetrization / Wigner--Eckart strategy for mapping equivariant features onto
Hamiltonian matrix blocks follows Nigam, Willatt & Ceriotti, *Equivariant
representations for molecular Hamiltonians and $N$-center atomic-scale
properties*, J. Chem. Phys. **156**, 014115 (2022) [arXiv:2109.12083], which
generalizes atom-centered density-correlation ($\lambda$-SOAP / ACE) features
to two- and $N$-center quantities. Related background: Drautz, Phys. Rev. B
**99**, 014104 (2019) (ACE); Zhang, Onat, Dusson, McSloy, Anand, Maurer, Ortner & Kermode, *Equivariant analytical mapping of first principles Hamiltonians to accurate and transferable materials models*, npj Comput. Mater. **8**, 158 (2022) [arXiv:2111.13736] (ACE-based equivariant linear model for $H$ and $S$); 

## 1. Setup and general strategy

We assume the electronic-structure reference produces, for each atomic
configuration, a Hamiltonian matrix $H$ and overlap matrix $S$ expressed in a
minimal (or at least fixed, atom-centered) basis of *real* atomic orbitals,

```math
   \psi_{i, nlm}, \qquad i = 1, \dots, N_{\rm at},\;\; l = 0,1,\dots, l_{\max}^{(i)},\;\; m = -l,\dots,l,
```

where $i$ labels the atomic site, $n$ is a radial/angular-momentum-channel
("shell") index (e.g. $1s, 2s, 2p, \dots$) and $(l,m)$ the angular momentum and
its real-harmonic projection. Each orbital is centered at the position
$\mathbf r_i$ of its host atom and transforms under a rotation/improper
rotation $Q \in O(3)$ as a real spherical harmonic of degree $l$,

```math
   \hat Q\, \psi_{i, nlm} = \sum_{m'} D^{l}_{m'm}(Q)\, \psi_{i, nlm'} ,
```

with $D^l(Q)$ the (real) Wigner matrix of degree $l$ (including the parity
factor $(-1)^l$ under inversion). The matrix elements

```math
  H_{i n l m, j n' l' m'} = \langle \psi_{i, nlm} | \hat H | \psi_{j, n'l'm'} \rangle, \qquad
  S_{i n l m, j n' l' m'} = \langle \psi_{i, nlm} | \psi_{j, n'l'm'} \rangle
```

therefore transform as the tensor product $|lm\rangle \otimes |l'm'\rangle$ of
two irreducible representations of $O(3)$ under a *simultaneous* rotation of
the whole configuration (atoms **and** orbitals). This is the central symmetry
constraint that the model must respect, and it is most conveniently handled by
recoupling the $(l,m;l',m')$ indices into irreducible blocks $(\lambda, \mu)$
via Clebsch--Gordan (CG) coefficients (§4).

We split the construction into two essentially independent pieces:

* **Overlap $S$.** Because $S$ is a strictly two-center, geometry-only
  quantity (it depends only on the relative position $\mathbf r_j - \mathbf
  r_i$ and the orbital content of the two sites, not on the chemical
  environment), we represent it with a *purely two-center* model: an
  equivariant expansion in the bond vector $\mathbf r_{ij} = \mathbf r_j -
  \mathbf r_i$ alone (§6). In particular $S_{ii} $ is just the fixed
  on-site overlap (orthonormal basis, or a fixed atomic-orbital Gram matrix),
  and $S_{ij}$, $i\neq j$, is a sum of two-center ("Slater--Koster"-like)
  terms.

* **Hamiltonian $H$.** $H$ does depend on the full chemical environment (it is
  an *effective* operator that encodes everything that has been integrated
  out), so we need the richer many-body machinery of ACE:

    * **on-site blocks** $H_{ii}$ are represented as an equivariant function of
      the atomic environment of site $i$ alone — a one-center ACE expansion
      (§5);
    * **off-site blocks** $H_{ij}$, $i \neq j$, are represented as an
      equivariant function of the environments of *both* sites $i,j$ **and**
      of the bond $\mathbf r_{ij}$ — effectively a two-center ACE expansion in the sense of Nigam *et al.* (§7).

Both pieces reuse the same fundamental building blocks: a one-particle basis
$\phi_{nlm} = R_{nl} Y_{lm}$, atomic-density features $A_{i,nlm}$ obtained by
pooling $\phi_{nlm}$ over neighbours, equivariant ACE bases $B_{i}^{(\nu)}$
built from tensor products of the $A_{i, nlm}$, and a final
Clebsch--Gordan "(re)coupling" step that turns invariant linear-regression
problems into the correctly-transforming $(l,m;l',m')$ blocks. We describe
each of these in turn.

## 2. One-particle basis and pooled $A$-features

Fix a radial basis $R_{nl}(r)$ (e.g. splines, polynomials, or Gaussian-type
orbitals over $[0, r_{\rm cut}]$) and real spherical harmonics $Y_{lm}(\hat
{\mathbf r})$, and define the one-particle ("atomic orbital-like") basis
functions

```math
   \phi_{nlm}(\mathbf r) = R_{nl}(|\mathbf r|)\, Y_{lm}(\hat{\mathbf r}),
   \qquad |\mathbf r| \le r_{\rm cut}.
```

For each atomic site $i$ we pool the contributions of the neighbours $j \in
\mathcal N_i = \{ j : |\mathbf r_{ji}| \le r_{\rm cut}\}$ (where $\mathbf
r_{ji} = \mathbf r_j - \mathbf r_i$) into the (one-particle, $\nu=1$)
*atomic density features*

```math
   A_{i, nlm} = \sum_{j \in \mathcal N_i} z_{j}\, \phi_{nlm}(\mathbf r_{ji}),
```

where $z_j$ encodes the chemical species of $j$ (and possibly the species of
$i$, in the usual ACE "bond basis" sense). Under rotation/inversion of the
whole structure, $A_{i,nlm}$ transforms exactly like $\phi_{nlm}$, i.e. as
$Y_{lm}$:

```math
   \hat Q\, A_{i,nlm} = \sum_{m'} D^l_{m'm}(Q)\, A_{i,nlm'} .
```

## 3. Equivariant ACE basis $B_i^{(\nu)}$

Higher-order ($\nu$-correlation) equivariant features are obtained, as usual
in ACE, by forming tensor products of $\nu$ copies of $A_{i,\cdot}$ and
projecting onto irreducible representations of $O(3)$ using generalized
Clebsch--Gordan coefficients — equivalently, by the iterative coupling scheme
of Nigam *et al.* (their eq. (13)/(14), the "NICE" recursion) or the standard
ACE generalized-CG construction (Drautz 2019; Dusson *et al.* 2022; Darby
*et al.*). Schematically, an order-$\nu$ equivariant basis function carrying
total angular momentum $L$ and projection $M$ can be written

```math
   B^{(\nu) L M}_{i \mathbf{v}}
   = \sum_{m_1,\dots,m_\nu}
     C^{L M}_{l_1 m_1 \cdots l_\nu m_\nu}\;
     \prod_{t=1}^{\nu} A_{i, n_t l_t m_t},
```

where $\mathbf v = (n_1 l_1, \dots, n_\nu l_\nu)$ collects the
"$\nu$-correlation order" indices (subject to the usual symmetrisation over
permutations of identical $(n_t l_t)$ that ACE performs to remove
redundancies), and $C^{LM}_{\cdots}$ is the generalized CG array that projects
the $\nu$-fold tensor product $\bigotimes_t Y_{l_t}$ onto the irreducible
component $L$. By construction

```math
   \hat Q\, B^{(\nu) L M}_{i \mathbf v} = \sum_{M'} D^L_{M'M}(Q)\, B^{(\nu) L M'}_{i \mathbf v},
```

i.e. $B_i^{(\nu) L \bullet}$ transforms as an $O(3)$ irrep of degree $L$ (with
the appropriate parity $(-1)^{l_1+\cdots+l_\nu}$, which must match $(-1)^L$ for
a "proper" tensor and is opposite for a pseudotensor — see §5.3). We write

```math
   \mathcal B_i = \big\{ B_{i \mathbf v}^{(\nu) L \bullet} \big\}_{\nu \le \nu_{\max},\, L \le L_{\max}}
```

for the full equivariant ACE basis at site $i$, truncated at correlation order
$\nu_{\max}$ and angular momentum $L_{\max}$. These are exactly the one-center
($N=1$) members of the symmetrized $N$-center feature hierarchy of Nigam *et
al.* (their $|\overline{\rho_i^{\otimes \nu}; \sigma; L\mu}\rangle$, eq. (6)).

## 4. Wigner--Eckart recoupling: from invariants to matrix blocks

The crucial trick that converts an ordinary (rotation-*invariant*) linear ACE
model into an equivariant model of a tensorial target — here, an
$(l,m;l',m')$ block of $H$ or $S$ — is a change of basis on the *output* side
via Clebsch--Gordan coefficients (this is the Wigner--Eckart construction used
throughout Nigam *et al.*, eqs. (21)-(23), and in symmetry-adapted GPR
/ equivariant tensorial-property models more generally). Concretely, define
the *coupled* representation of a generic two-index block $X_{nlm,n'l'm'}$
(standing for $H$ or $S$, on a pair of shells $(n,l)$ and $(n',l')$) by

```math
   X_{nl;n'l'}^{\lambda \mu}
      = \sum_{m m'} \langle l m;\, l' m' \,|\, \lambda \mu \rangle \; X_{nlm, n'l'm'},
   \qquad |l-l'| \le \lambda \le l+l',
```

with inverse

```math
   X_{nlm, n'l'm'} = \sum_{\lambda \mu} \langle lm;\, l'm' \,|\, \lambda \mu\rangle\; X^{\lambda \mu}_{nl;n'l'} .
```

Each fixed-$\lambda$ slice $X^{\lambda \bullet}_{nl;n'l'}$ transforms as a
*single* $O(3)$ irrep of degree $\lambda$ (a "proper" tensor if
$(-1)^{l+l'+\lambda} = +1$ and a pseudotensor — picking up an extra sign under
improper rotations — otherwise). This is exactly what the equivariant ACE
basis $B^{(\nu) \lambda \bullet}$ produces, so a *linear, equivariant* model
for $X^{\lambda \bullet}_{nl;n'l'}$ can be written directly as a linear
combination of $B^{\lambda \bullet}$'s with **scalar** (rotation-invariant)
regression weights, and then mapped back to the $(m,m')$ block via the inverse
CG transform above. We will refer to the pair of operations
"couple $\to$ contract with weights $\to$ invert the CG transform" compactly
as

```math
  \texttt{transform}_{\lambda}\Big[ \textstyle\sum_q w_q\, B^{(\cdot)\lambda \mu}\Big]_{m m'}
  \;:=\;
  \sum_{\mu} \langle lm; l'm' | \lambda \mu\rangle \sum_q w_q\, B^{(\cdot) \lambda \mu} ,
```

i.e. exactly the user-level "`transform(...)`" referred to informally in the
introduction. Note that $\texttt{transform}_\lambda$ only makes sense (gives a
nonzero, symmetry-consistent contribution) when $\lambda$ lies in the range
$|l - l'| \le \lambda \le l+l'$ allowed by the CG selection rule for the target
shell pair $(l,l')$; in particular the *full* $(l,m;l',m')$ block is obtained
by summing the transform over **all** admissible $\lambda$,

```math
   X_{nlm,n'l'm'} = \sum_{\lambda = |l-l'|}^{l+l'} \texttt{transform}_\lambda[\cdots]_{mm'} .
```

## 5. On-site blocks $H_{ii}$

### 5.1 General form

The on-site (block-diagonal) part $H_{ii}$ depends, by translational and
permutational symmetry, only on the *local atomic environment* of site $i$
(species and positions of neighbours within $r_{\rm cut}^{\rm on}$), exactly
as in a standard ACE potential. We therefore propose

```math
   H_{i;\, nlm,\, n'l'm'}
   \;=\; \sum_{\lambda=|l-l'|}^{l+l'}
   \texttt{transform}_\lambda\Big[ \sum_{q} w^{(a_i; nl, n'l'; \lambda)}_{q}\;
        B^{(\cdot)\, \lambda \mu}_{i, q} \Big]_{m m'} ,
```

i.e. for each shell pair $(nl, n'l')$ and each admissible coupled angular
momentum $\lambda$, the corresponding $\lambda$-equivariant block of $H_{ii}$
is a linear combination — with scalar, $\lambda$- and species-($a_i$,
the chemical element of $i$)-dependent weights $w_q$ — of equivariant ACE
basis functions $B_{i,q}^{(\cdot)\lambda \mu}$ evaluated on the environment of
$i$, recoupled back into the $(m,m')$ block via $\texttt{transform}_\lambda$.

### 5.2 Practical remarks

* **Selection rules.** Only basis functions $B^{(\cdot)\lambda \bullet}_i$ with
  the matching parity $\sigma = (-1)^{l + l' + \lambda}$ contribute (improper
  vs. proper tensor); this halves the number of independent weight sets.
* **Hermiticity.** $H_{ii}$ is Hermitian, and (working with real orbitals) in
  fact symmetric: $H_{i;nlm,n'l'm'} = H_{i;n'l'm',nlm}$. It is therefore
  sufficient to build models for lexicographically-ordered shell pairs $(nl)
  \le (n'l')$ and to symmetrize, exactly as discussed by Nigam *et al.* around
  their eq. (19)-(20). In the coupled representation this means we only need
  one weight vector per *unordered* shell pair and $\lambda$ (with a
  $\sqrt{2}$-type normalisation on the diagonal $nl = n'l'$ terms).
* **Truncation.** $\lambda$ ranges over $|l-l'|,\dots,l+l'$, but the size of
  the regression problem is controlled mainly by $\nu_{\max}$ and $L_{\max}$
  in $\mathcal B_i$ (the maximal correlation order and angular momentum kept
  in the ACE expansion), which can be tuned independently of $(l_{\max},
  l'_{\max})$ of the orbital basis.

### 5.3 Relation to "weights $\times\ B^{L''}$"

In the notation used in the original sketch, "$L''$" is exactly the coupled
angular momentum $\lambda$ above: the on-site block is a transform of a sum,
over the *coupling* channel $\lambda = L''$, of weighted equivariant basis
functions $B_i^{\lambda}$. Note that $L'' = \lambda$ is **not** a free choice —
it is fixed (within $|l-l'|,\dots,l+l'$) by the orbital pair $(l,l')$ that
defines the target block; what *is* free is which members of $\mathcal B_i$
(which $(\nu, \mathbf v)$, i.e. which radial/angular "shapes" of the
environment) enter the linear combination for each $\lambda$.

## 6. Two-center (bond) descriptors $\phi_{nlm}(\mathbf r_{ij})$

For both $S_{ij}$ and the off-site part of $H_{ij}$ we need an equivariant
description of the *bond* itself. We reuse the same functional form as the
one-particle basis, evaluated at the bond vector,

```math
   \phi^{\rm b}_{nlm}(\mathbf r_{ij}) = R^{\rm b}_{nl}(|\mathbf r_{ij}|)\, Y_{lm}(\hat{\mathbf r}_{ij}),
   \qquad |\mathbf r_{ij}| \le r_{\rm cut}^{\rm b},
```

i.e. the $\nu=0$-neighbour, two-center feature
$\langle n00 | \mathbf r_{ji}; g\rangle$ generalised to $l>0$ — the building
block of the "two-centers, one-neighbour" features of Nigam *et al.* (their
eqs. (10)-(11)). We deliberately use a *separate* radial basis $R^{\rm b}_{nl}$
and cutoff $r^{\rm b}_{\rm cut}$, possibly different from the ones used to
build $A_{i,nlm}$ — see the discussion in §8.

## 7. Off-site blocks $H_{ij}$, $i \neq j$

### 7.1 General form

For $i \neq j$, $H_{ij}$ depends on (i) the environment of $i$, (ii) the
environment of $j$, and (iii) the bond $\mathbf r_{ij}$ connecting them — a
genuine *three-piece*, "two-center-plus-environments" object, in the spirit of
Nigam *et al.*'s $N$-center construction (their two-center, one-neighbour
features, eq. (11), generalised to higher correlation order on each side and
combined with a richer bond descriptor). The natural equivariant ansatz is to
first form a triple tensor product

```math
  T^{(\Lambda M)}_{i j; \mathbf v_1 \mathbf v_2 b}
  = \sum_{\mu_1 \mu_2 \mu_b}
    \langle L_1 \mu_1; L_2 \mu_2 | \kappa \nu \rangle\,
    \langle \kappa \nu; l_b \mu_b | \Lambda M \rangle\;
    B^{(\cdot) L_1 \mu_1}_{i \mathbf v_1}\;
    B^{(\cdot) L_2 \mu_2}_{j \mathbf v_2}\;
    \phi^{\rm b}_{n_b l_b \mu_b}(\mathbf r_{ij}),
```

i.e. couple an equivariant feature $B_i^{L_1}$ of the environment of $i$, an
equivariant feature $B_j^{L_2}$ of the environment of $j$, and a bond
descriptor $\phi^{\rm b}_{n_b l_b}$, through an intermediate channel $\kappa$
into a final irreducible block of degree $\Lambda$. (As usual, the
intermediate coupling $\kappa$ may be summed over, or — to limit the basis
size — restricted to a small set, e.g. $\kappa \in \{|L_1-L_2|, \dots,
L_1+L_2\}$ with a cutoff on $\kappa$ itself; this is a hyperparameter exactly
analogous to the internal-coupling truncations used in higher-order ACE/MACE
bases). The off-site block is then

```math
   H_{ij;\, nlm,\, n'l'm'}
   = \sum_{\Lambda = |l-l'|}^{l+l'}
     \texttt{transform}_\Lambda\Big[
        \sum_{q} w^{(a_i a_j; nl, n'l'; \Lambda)}_{q}\;
        T^{(\Lambda \bullet)}_{ij,\, q}
     \Big]_{mm'},
```

with $q$ ranging over the retained combinations $(\mathbf v_1, \mathbf v_2,
n_b l_b, \kappa)$, and weights depending on the *ordered* species pair $(a_i,
a_j)$ and on the shell pair $(nl, n'l')$. This is the direct off-site
generalisation of eq. (23)/(28) of Nigam *et al.*, with the pair feature
$A_{ii'}$ there replaced here by the explicit triple coupling
$B_i \otimes B_j \otimes \phi^{\rm b}_{ij}$.

### 7.2 Coupling strategy: direct three-way contraction

The coupling in §7.1 is written with an explicit intermediate channel $\kappa$,
which is an artifact of expressing a three-way product as a sequence of
two-way CG steps. In practice we use the coupling-matrix machinery of
EquivariantTensors, which constructs a single sparse coupling tensor

```math
  \mathbf C^{(\Lambda)}_{(L_1 \mu_1)(L_2 \mu_2)(l_b \mu_b)}
```

that contracts $B_i^{L_1} \otimes B_j^{L_2} \otimes \phi^{\rm b}_{l_b}$
directly onto the target irrep $\Lambda$ in one step, summing over all
intermediate channels $\kappa$ internally. Crucially, the coupling matrix is
constructed *after* specifying exactly which output combinations
$(\Lambda, \mathbf v_1, \mathbf v_2, n_b l_b)$ are needed (e.g. those with
$\Lambda \in [|l-l'|, l+l']$ for each target shell pair), so feature selection
is applied at the coupling stage — before any arithmetic on feature values.
The intermediate channel $\kappa$ never needs to be materialized.

This approach is strictly preferable to coupling in a fixed sequential order
(e.g. $B_i \otimes \phi^{\rm b}$ first, then with $B_j$, or $B_i \otimes B_j$
first, then with $\phi^{\rm b}$): sequential orderings impose an arbitrary
structural bias and generate intermediate objects whose size is set by the
intermediate coupling rather than by the actually-needed output channels. The
direct three-way coupling is the most general linear model within the chosen
feature space, and the coupling-matrix sparsity (imposed by the CG triangle
rules $|L_1 - L_2| \le \kappa \le L_1 + L_2$, $|\kappa - l_b| \le \Lambda
\le \kappa + l_b$) keeps the contraction cost modest.

One implementation-level optimization is worth noting: since $B_i^{L_1}$
appears in all bonds $(i,j)$ incident on atom $i$, it can be precomputed once
per atom rather than per bond. Whether to exploit this by first contracting
$B_i \otimes \phi^{\rm b}_{ij}$ (bond-dressed site features, computed once per
bond per site) and then contracting with $B_j$, or to evaluate the full
coupling matrix per bond, is an evaluation-order choice that does not affect
the model itself — the coupling matrix and the selected feature set are
identical either way.

### 7.3 Index-permutation (Hermiticity) symmetry

$H$ is Hermitian (and, with real orbitals, $H_{ji} = H_{ij}^{T}$), so the model
must satisfy

```math
   H_{ij;\, nlm,\, n'l'm'} = H_{ji;\, n'l'm',\, nlm} .
```

Following Nigam *et al.* (eqs. (19)-(20) and surrounding discussion), we
distinguish:

* **cross-species pairs** ($a_i \neq a_j$): we may fix a canonical order of
  the species (e.g. atomic number) and only ever build/evaluate $H_{ij}$ for
  $i$ the heavier atom; Hermiticity then determines $H_{ji}$ without any
  additional modelling.
* **same-species, off-diagonal pairs** ($a_i = a_j$, $i \neq j$): there is no
  canonical order, so the feature triple $(B_i, B_j, \phi^{\rm b}_{ij})$ must
  be combined into manifestly symmetric/antisymmetric combinations under
  $i \leftrightarrow j$ (which simultaneously sends $\mathbf r_{ij} \to
  -\mathbf r_{ij} = \mathbf r_{ji}$, picking up a parity factor
  $(-1)^{l_b}$ on the bond harmonics), e.g.
  ```math
     T^{(\Lambda \bullet)}_{\{ij\},\, q}{}^{\pm}
        = T^{(\Lambda \bullet)}_{ij,\, q} \pm T^{(\Lambda \bullet)}_{ji,\, q} ,
  ```
  and only the combination with the correct symmetry contributes to a given
  (lexicographically-ordered) shell-pair block, exactly mirroring the
  symmetric/antisymmetric pair-feature construction of Nigam *et al.*, eq.
  (16)/(20). As they note, Hermiticity then forces some of these
  symmetry-adapted blocks to vanish identically — a useful, free consistency
  check on the implementation.

## 8. The overlap matrix $S$

Because $S_{ij}$ is a purely geometric, two-center quantity (no dependence on
the wider chemical environment beyond the identity/orbital content of $i$ and
$j$ themselves), we propose to model it with a *much smaller* architecture
than $H$: drop the $B_i, B_j$ environment dependence entirely and retain only
the bond descriptor,

```math
   S_{ij;\, nlm,\, n'l'm'}
     = \sum_{\Lambda=|l-l'|}^{l+l'}
       \texttt{transform}_\Lambda\Big[
         \sum_{q} u^{(a_i a_j; nl, n'l'; \Lambda)}_q\;
         \phi^{\rm b}_{n_q l_q \bullet}(\mathbf r_{ij})
       \Big]_{mm'}, \qquad i \neq j,
```

i.e. a genuinely "two-center"/Slater--Koster-like expansion: for each
admissible $\Lambda$, a linear combination of bond-harmonics
$\phi^{\rm b}_{n_q l_q \Lambda \bullet}$ (so $l_q = \Lambda$ — no internal CG
coupling is needed because there is only one tensorial object), recoupled into
the $(m,m')$ block. (Exactly as for $H$, Hermiticity / index-exchange symmetry
must be enforced for same-species pairs, §7.3, and a canonical species
ordering used for cross-species pairs.) The on-site part $S_{ii}$ is either
the identity (orthonormal basis) or a fixed, geometry-independent Gram matrix
of the atomic-orbital basis (which can typically be computed analytically and
needs no learning at all).

This two-center-only ansatz for $S$ is the same simplification used (for
different quantities) by the "two-centers, zero/one-neighbour" features in
Nigam *et al.*, eqs. (10)-(11) — in our case we only need the $\nu = 0$
("zero-neighbour"), pure-bond level, since $S$ has no environment-dependence
to capture.

## 9. Two independent basis sets and cutoffs

The model has two entirely separate one-particle bases, each with its own
radial functions and cutoff radius:

* $\phi_{nlm}(\mathbf r) = R_{nl}(|\mathbf r|)\,Y_{lm}(\hat{\mathbf r})$,
  $|\mathbf r| \le r_{\rm cut}^{\rm on}$ — the **environment basis**, used to
  build the pooled $A_{i,nlm}$ features and thence the equivariant ACE basis
  $\mathcal B_i$ at each site. The cutoff $r_{\rm cut}^{\rm on}$ controls how
  far the chemical environment that shapes on-site electronic structure
  extends, and is typically set by nearsightedness / locality arguments.

* $\phi^{\rm b}_{nlm}(\mathbf r) = R^{\rm b}_{nl}(|\mathbf r|)\,Y_{lm}(\hat{\mathbf r})$,
  $|\mathbf r| \le r_{\rm cut}^{\rm b}$ — the **bond basis**, used to embed
  the bond vector $\mathbf r_{ij}$ in the off-site features $T^{(\Lambda)}_{ij}$
  and in the two-center overlap model $S_{ij}$. The cutoff $r_{\rm cut}^{\rm b}$
  sets the range beyond which all off-site blocks are taken to vanish, and is
  typically controlled by the decay of orbital overlap / hopping integrals.

The two cutoffs play physically distinct roles and can therefore differ:
$r_{\rm cut}^{\rm b} < r_{\rm cut}^{\rm on}$ is natural for short-ranged
(tight-binding-like) bases where hopping decays faster than environmental
influence; $r_{\rm cut}^{\rm b} > r_{\rm cut}^{\rm on}$ can arise in
delocalised or metallic systems. The radial functions $R_{nl}$ and
$R^{\rm b}_{nl}$ are likewise independent — they may be of the same
parametric family (e.g. both polynomial envelopes) but are fitted/truncated
independently, with their own $(n_{\max}, l_{\max})$ truncations.

One consequence worth noting: an atom $j$ with $r_{\rm cut}^{\rm b} <
|\mathbf r_{ij}| \le r_{\rm cut}^{\rm on}$ contributes to the environment
$\mathcal B_i$ (and hence to the on-site model for $H_{ii}$) but has a
vanishing off-site block $H_{ij} = 0$. This is physically reasonable —
distant neighbours shape the local electronic structure without having
non-negligible direct hopping — and does not introduce any discontinuity
provided both basis envelopes go smoothly to zero at their respective cutoffs.

## 10. Model parameters

At model-construction time the user specifies:

* **Orbital list**: the sequence of shells $(nl)$ present at each species, e.g.
  $(1s, 2s, 2p, 3d, \dots)$. This determines the target block types
  $(a_i nl; a_j n'l')$ and, in particular, the maximum orbital angular momentum
  $l_{\max}^{\rm orb}$ — which in turn fixes the range of CG coupling channels
  $\Lambda \in [|l-l'|, l+l']$ needed in the transform steps.

* **Environment basis** (used to build $A_{i,nlm}$ and $\mathcal B_i$):
  - $n_{\max}$, $l_{\max}$ — radial and angular truncations of $\phi_{nlm}$
  - `totaldegree` — a total-degree truncation combining $n$ and $l$ to control
    basis size
  - $r_{\rm cut}^{\rm on}$ — environment cutoff radius
  - $\nu_{\max}$ — maximum correlation order (body order minus one) of $\mathcal B_i$

* **Bond basis** (used to embed $\mathbf r_{ij}$ in off-site features and $S_{ij}$):
  - $n_{\max}^{\rm b}$, $l_{\max}^{\rm b}$ — radial and angular truncations of
    $\phi^{\rm b}_{nlm}$
  - $r_{\rm cut}^{\rm b}$ — bond cutoff radius

Note that $l_{\max}^{\rm b}$ must be at least $l_{\max}^{\rm orb}$ (since the
bond basis must be able to contribute to every admissible $\Lambda$ channel), but
may be larger if higher-angular-momentum bond features are found useful.
The equivariant ACE basis $\mathcal B_i$ is coupled up to
$L_{\max} = l_{\max}^{\rm orb} + l_{\max}^{\rm b}$ (the maximum $\Lambda$ that
can arise from coupling a bond descriptor with an environment feature in the
off-site model), though in practice a tighter truncation is applied via the
total-degree criterion.

## 11. Network structure

The model is implemented as a [Lux.jl](https://lux.csail.mit.edu)-compatible
neural-network layer graph. The forward pass for a single configuration is:

1. **Neighbour lists**: compute the environment neighbour list (cutoff
   $r_{\rm cut}^{\rm on}$) and the bond neighbour list (cutoff
   $r_{\rm cut}^{\rm b}$).

2. **Embedding** (fully parallel):
   - (i) $R_{nl}(r_{ji})$ for all environment pairs $(i,j)$
   - (ii) $Y_{lm}(\hat{\mathbf r}_{ji})$ for all environment pairs
   - (iii) $R^{\rm b}_{nl}(r_{ij})$ for all bond pairs $(i,j)$
   - (iv) $Y_{lm}(\hat{\mathbf r}_{ij})$ for all bond pairs

3. **Feature construction** (parallel branches):
   - (i) **ACE layer**: pool and symmetrize to produce $A_{i,nlm}$ and
     then $\mathcal B_i = \{B_{i\mathbf v}^{(\nu)LM}\}$ at every site
   - (ii) **Bond embedding**: form $\phi^{\rm b}_{nlm}(\mathbf r_{ij})$ at
     every bond pair from outputs (iii)+(iv)

4. **Matrix assembly** (parallel branches):
   - (i) **On-site**: for each site $i$, contract $\mathcal B_i$ with
     species- and shell-pair-specific weight vectors via
     $\texttt{transform}_\Lambda$ to produce $H_{ii}$
   - (ii) **Off-site**: for each bond pair $(i,j)$, apply the three-way
     EquivariantTensors coupling matrix to
     $(B_i^{L_1}, B_j^{L_2}, \phi^{\rm b}_{ij})$, contract with weights,
     and apply $\texttt{transform}_\Lambda$ to produce $H_{ij}$ and $S_{ij}$
   - (iii) On-site overlap : analogous, but just constant
   - (iv) Off-site overlap : analogous, built only from \phi_{nlm}^{\rm b}

Steps 3(i) and 3(ii) can run in parallel since both depend only on step 2
outputs; steps 4(i) and 4(ii) can likewise run in parallel. The off-site
branch (step 4(ii)) is where the equivariance machinery of §§4, 7 is
concentrated.

During development decide whether 4. should be split into 4. Matrix block assembly; 5. global matrix assembly. 

## 12. Out of scope

The following questions concern training and fitting strategy rather than
model architecture, and are deferred to a separate discussion:

* **Orthogonalization**: whether to fit $H$ directly (paired with a learned or
  analytic $S$, solving the generalized eigenproblem $HU = SU\,\mathrm{diag}\,\epsilon$)
  or the Löwdin-orthogonalized $\bar H = S^{-1/2}HS^{-1/2}$. Both targets
  have identical equivariance; the choice affects loss landscape and
  learnability, not the feature architecture described here.

* **Regression method**: the feature layer described above is compatible with
  linear regression, symmetry-adapted GPR, and equivariant neural-network
  readouts. The choice among these will be made once the feature layer is
  implemented and benchmarked.