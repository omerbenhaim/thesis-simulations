# Vertices of the n×n×n tristochastic polytope

Context handoff for continuing this work (e.g. from Claude on the web).

## Math

A tristochastic tensor `x = (x_ijk) ∈ ℝ^{n³}` has `x_ijk ≥ 0` and every **line**
sums to 1:

```
Σ_k x_ijk = 1   Σ_j x_ijk = 1   Σ_i x_ijk = 1
```

There are `3n²` line constraints; the constraint matrix `A` has rank
`3n²−3n+1`. A feasible point is a **vertex** iff the columns of `A` on its
support `S = supp(x)` are linearly independent (nullity 0) and `A_S x_S = 1` has
a unique **strictly positive** solution. Hence a vertex support has size
`n² ≤ |S| ≤ 3n²−3n+1`. The 0/1 vertices are Latin squares; the interesting ones
are fractional.

Dependencies: `numpy`, `scipy`, `networkx`. Note: several filenames contain
spaces — quote them on the command line.

## Code files (`polytope/`)

### `polytope simulation.py`
Samples vertices by maximizing a random linear objective over the polytope (an
LP), then studies the support. Uses `highs-ds` (dual simplex) so the optimum is
a genuine vertex. Subcommands:
- `demo` — sanity check for n=3,4,5 (support size ≤ bound, nullity 0, residuals).
- `sample --n N --trials T --seed S --out FILE.jsonl` — bulk sample to JSONL.
- `summarize FILE...` — aggregate stats per n.
- `enrich --n N --target K --out FILE.jsonl` — collect only **fractional
  (non-0/1)** vertices, dedup by support, rank by support size; flags a "simple"
  (full-support) vertex. Supports `--exclude-existing FILE...` to never repeat
  supports from earlier runs.
- `show FILE --index I | --seed S | --all` — render a vertex's "quasi-Latin-
  square" support map `Q[i][j] = {supported k}` as a bordered grid.

### `probabilistic simulation.py`
Nati's **random line-choice** experiment. For each of the `3n²` lines pick one
coordinate uniformly; `S = union`. By construction `E|S| = 3n²−3n+1` (the vertex
bound). For each `S` it solves ONE LP

```
maximize t   subject to   A_S y = b,  y_i ≥ t
```

and classifies by the optimum `t*`:
- LP infeasible → **not solvable** (`b ∉ Im(A_S)`).
- `t* > 0` → **exact-positive** (S is the support of a real polytope point).
- `t* ≈ 0` → solvable but a coordinate is forced to 0 (boundary).
- `t* < 0` → **forced negative** (every solution has a negative entry).
- nullity(A_S) = 0 → unique solution → **vertex** when also exact-positive.

Prints `prob_positive_support_in_polytope`, `prob_vertex_support`, and a full
solvability breakdown (positive / not-exact / forced-negative, each with a
"uniquely solvable" sub-count). Flags: `--n --trials --seed --out
--exclude-existing --eps --save-failed --failed-out`.

### `computer search.py`
**Exact** enumeration of *all* n=3 vertices (rational arithmetic). Since
`dim(P) = 27 − 19 = 8`, every vertex has ≥8 zero coordinates with independent
normals, so enumerating all `C(27,8) = 2,220,075` zero-sets, solving exactly,
and keeping unique strictly-positive supports finds every vertex. Also
`--show-support-map ID` (recomputes deterministically, prints one vertex's map)
and `--mode random_lp` (n≥4 sampling fallback). Correct but slow (~11 min).

### `_crosscheck_n3.py`
Independent completeness check: a second (determinant-gated) enumeration plus a
random-LP subset test.

## Results files (`polytope/`)

- **`frac_n5.jsonl`, `frac_n6.jsonl`** — 200 fractional vertices each (from
  `enrich`). Per record: `seed`, `distinct_values` (fraction strings, e.g.
  `["1/2","1"]`), `support_size`, `is_half_integral`, three support histograms
  (`shaft_size_histogram`, `row_symbol_support_histogram`,
  `column_symbol_support_histogram`), `component_sizes`,
  `num_bipartite_components`, `num_nonbipartite_components`, `support_map`.
- **`line_support_n3.jsonl`** — 40 exact-positive supports found by the random
  line-choice method at n=3 (all `positive_nonvertex`).
  `line_support_n4.jsonl`, `line_support_n5.jsonl` are empty (none found).
- **`failed_n5.jsonl`** — a couple of *failed* supports (throwaway test artifact).
- **`results_n3/vertices_n3.json`** — the exact enumeration output:
  `total_vertices`, size/value distributions, and one compact line per vertex
  `{vertex_id, support_size, unique_values}`. Supports are NOT stored (recompute
  with `computer search.py --show-support-map ID`).

## Key findings

- Random-objective LP mostly lands on **integral** (Latin-square) vertices;
  fractional ones are rare — hence the `enrich` filter. n=5 fractional values are
  mostly half-integral `{1/2, 1}` plus some `1/3`, `1/4`; n=6 reaches much richer
  denominators (up to `1/13`, `1/17`).
- Random line-choice supports hit the expected **size** exactly
  (`E|S| = 3n²−3n+1`) but are almost never a valid support:
  `prob_positive ≈ 0.5%` at n=3, `≈ 0%` at n=4/5; `prob_vertex ≈ 0`.
- **n=3 has exactly 66 vertices**: 12 permutation (support 9, value `{1}`) + 54
  half-integral (support 17, values `{1/2, 1}`). No full-support/simple vertices.
  Verified complete two independent ways.

## Open items

- `computer search.py` is correct but slow (~11 min) — most `C(27,8)` iterations
  raise a singular-matrix exception; a determinant gate (as in
  `_crosscheck_n3.py`) gives roughly a 10× speedup.
- The repo has committed `.pyc` files and a UTF-16-encoded `.gitignore` (git
  misreads it, so nothing is actually ignored) — minor cleanup available.
