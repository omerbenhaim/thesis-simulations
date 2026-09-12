# Vertices of the n×n×n tristochastic polytope

Context handoff for continuing this work (e.g. from Claude on the web). It
summarizes every experiment built, the files they produce, and the findings.

## Math

A tristochastic tensor `x = (x_ijk) ∈ ℝ^{n³}` has `x_ijk ≥ 0` and every **line**
sums to 1:

```
Σ_k x_ijk = 1     Σ_j x_ijk = 1     Σ_i x_ijk = 1
```

There are `3n²` line constraints; the incidence/constraint matrix `A` has rank
`r = 3n²−3n+1`. A feasible point is a **vertex** iff the columns of `A` on its
support `S = supp(x)` are linearly independent (nullity 0) and `A_S x_S = 1` has
a unique **strictly positive** solution. So a vertex support has
`n² ≤ |S| ≤ r`, and it is **simple** (non-degenerate) exactly when `|S| = r`
(maximal support). The 0/1 vertices are Latin squares; the interesting ones are
fractional, and the prized ones are the large-support / simple fractional
vertices.

Dependencies: `numpy`, `scipy`, `networkx`. Note: some filenames contain spaces
— quote them. Work is on git branch `21.08.2026` (remote `origin`).

## The experiments (in `polytope/`)

### 1. `polytope simulation.py` — random-objective LP sampler
Maximize a random linear objective over the polytope (LP, dual simplex) → a
vertex; study its support. Subcommands: `demo` (sanity n=3,4,5), `sample`
(bulk → JSONL), `summarize` (aggregate stats), `enrich` (keep only fractional
non-0/1 vertices, dedup, rank by support size, flag simple), `show` (render a
vertex's "quasi-Latin-square" support map as a grid). Shared serialization
(`build_vertex_record`) and helpers live here and are reused by the walks.
- **Finding:** a random objective mostly lands on **integral** (Latin-square)
  vertices; fractional ones are rarer (hence `enrich`). At n=5/6 the fractional
  vertices are mostly half-integral `{1/2,1}` with some thirds/quarters; n=6
  reaches richer denominators (up to `1/13`, `1/17`). At n=10 objective-LP
  reaches support up to ~259/271 but **never simple**.

### 2. `tristochastic_lp_walk.py` — support-shrinking walk to a vertex
Start at the uniform tensor `U = 1/n` (deep interior). Repeatedly pick a
direction `D ∈ ker(A_S)` (support- and line-sum-preserving) and step to the
boundary, zeroing ≥1 coordinate; the support strictly shrinks until
`ker(A_S) = {0}` — exactly the vertex condition. Direction via a small LP
(`RandomLPNullspaceDirectionStrategy`); vertex certified exactly (mod-p full
column rank). CLI: `--n --walks/--target --out ...`; `times_reached` counter,
fresh unused walk seeds per run, crash-safe incremental writes,
`--exclude-existing`/merge.
- **Finding:** this **reliably reaches the large-support / simple fractional
  vertices** the objective sampler misses. n=10 walks land at support 269–271
  of 271 (several simple); n=13/15 similar. Radius from the uniform center grows
  ~linearly with n (n=10→15 ≈ 6.0 → 9.3). Vertices are richly fractional with
  large denominators (e.g. n=10 values on a `/1373` grid).

### 3. `tristochastic_basis_walk.py` — random basis-exchange walk
A *basis* `B` is `r` independent columns of `A`; then `A_B x = 1` has a unique
solution. Start from a simple vertex (loaded from `lp_walk_n{n}.jsonl`), and
random-walk through bases: pick an entering column `e ∉ B`, compute the
fundamental circuit `A_e = A_B α`, pick a leaving column with `α_i ≠ 0`, move to
`B'`. Classify each basis solution **positive / zero-containing / negative**
(residual- and rank-checked). Two output files: distinct new positive vertices
(with `times_reached`) and one stats row per walk. Exits cleanly if no simple
start vertex exists for that n.
- **Finding:** across **150,000 random bases (n=10–15, 25,000 each): 0
  positive, 0 zero-containing, 100% negative.** Strictly-positive bases (feasible
  vertices) are vanishingly rare in the basis graph — a random exchange from a
  vertex jumps deep into infeasibility (`min x ≈ −0.8`) and never returns.

### 4. `probabilistic simulation.py` — random line-choice supports (Nati's idea)
For each of the `3n²` lines pick one coordinate uniformly; `S = union`. By
construction `E|S| = r`. One max-`t` LP classifies whether `S` is the support of
a feasible polytope point / a vertex; reports `prob_positive_support`,
`prob_vertex_support`, and a solvability breakdown. Flags include
`--save-failed`/`--failed-out` to inspect failed supports.
- **Finding:** the construction gets the expected **size** exactly, but the
  support is almost never valid: `prob_positive ≈ 0.5%` at n=3, `≈ 0%` at n=4/5;
  `prob_vertex ≈ 0`. The right size does not give a real support by chance.

### 5. `computer search.py` — exact n=3 vertex enumeration
Enumerates **all** vertices for n=3 with exact rational arithmetic by iterating
all `C(27,8)` zero-sets (dim `P` = 8). `--show-support-map ID` reprints a vertex;
`--mode random_lp` is an n≥4 sampling fallback. `_crosscheck_n3.py` independently
verifies completeness (second enumeration + random-LP subset test).
- **Finding:** n=3 has **exactly 66 vertices** = 12 permutation (support 9,
  value `{1}`) + 54 half-integral (support 17, values `{1/2,1}`); **no simple
  vertices**. Verified complete two independent ways. (Correct but slow, ~11 min.)

### 6. `tristochastic_latin_cube_search.py` — 3-Latin-square + provenance cube switch
Sample 3 Latin squares, average their permutation tensors (`X = (P1+P2+P3)/3`,
support = union, values 1/3, 2/3, 1). Work in exact integer units of 1/3.
Admit a triple only if its support size is exactly `r`; then repeatedly apply a
`2×2×2` cube switch that keeps the support at exactly `r`: one checkerboard
class (all value 1/3, with **provenance** covering L1,L2,L3) goes 1/3→0 while the
opposite class (all 0) goes 0→1/3. Provenance = the single Latin square a 1/3
cell came from; any cell touched by a switch becomes None. Exhaustive cube
search (`C(n,2)³`, both orientations), cycle-prevention on visited supports,
random choice among valid moves. After each switch, test `rank(A_S)=r` (vertex).
Replaces an earlier random-cube version (removed). CLI: `--n --walks --seed …`.
- **Finding:** the most productive **simple-vertex** generator — at every n from
  10 to 15, ~33–44 of every 100 admitted walks reach a full/simple vertex
  (support = r), all distinct, no duplicates (~220 simple vertices total).
  Admit rate (fraction of triples with support exactly r) ≈ 6–8%; mean ~9–12
  switches to a vertex. (At small n=4/5 it mostly stalls — valid cubes are rare
  there.)

## Result files

| files | from | contents |
|---|---|---|
| `frac_n5.jsonl`, `frac_n6.jsonl` | `enrich` | 200 fractional vertices each |
| `enrich_n10.jsonl` | `enrich` | n=10 objective-LP fractional vertices (support ≤ 259, none simple) — comparison baseline |
| `lp_walk_n3…n15.jsonl` | LP walk | distinct vertices reached descending from the center; mostly large-support / simple fractional |
| `basis_walk_stats_n10…n15.jsonl` | basis walk | one row per walk (positive/zero/negative counts + %); all 100% negative |
| `basis_walk_vertices_n10…n15.jsonl` | basis walk | new positive vertices found — all **empty** (none found) |
| `line_support_n3.jsonl` (+ empty n4/n5) | line-choice | exact-positive supports found by random line-choice |
| `latin_cube_vertices_n10…n15.jsonl` | latin-cube | distinct simple vertices found by cube switching (values 1/3, 2/3) |
| `latin_cube_stats_n10…n15.jsonl` | latin-cube | per-walk rows + a final summary row (admit %, histogram, counts) |
| `results_n3/vertices_n3.json` | exact enum | 66 vertices (compact: id, support_size, unique_values) |

## Shared record conventions

Vertex records (from `build_vertex_record`) lead with, in order,
`seed → [times_reached] → support_size → radius_from_uniform → distinct_values`,
then support histograms, `is_integral`/`is_half_integral`/`is_simple_vertex`,
support graph info, and the `support_map`. Notably:
- `distinct_values` are **exact ratios** (e.g. `["1/2","1"]`), recovered by
  re-solving `A_S y = 1` cleanly — correct even for large denominators.
- `radius_from_uniform = ‖x − U‖_F`.
- `times_reached` counts revisits; identity is the **canonical support**
  (`frozenset` of triples). Files dedup on this, write crash-safe, support
  `--exclude-existing` merges, and reruns draw fresh unused seeds.

## Overall picture

- **Finding simple / large-support fractional vertices:** two effective methods —
  the **support-shrinking LP walk from the center**, and (most productively) the
  **3-Latin-square cube switch** (~33–44% of admitted walks give a simple vertex,
  n=10–15). Objective-LP sampling and random line-choice do not reach them.
- **Basis graph:** feasible (positive) bases are essentially isolated —
  0 in 150,000 random bases.
- **Small n exactly known:** n=3 = 66 vertices (no simple ones).

## Open items

- `computer search.py` exact enumeration is slow (~11 min); a determinant gate
  (as in `_crosscheck_n3.py`) would give ~10×.
- Larger n for the LP walk gets expensive (n=15 ≈ 7 min/walk, n=20 > 12 min/walk)
  — a nullspace-projection direction strategy could speed it up; not yet built.
- Long runs must keep the machine on AC power (no-sleep settings only apply on
  AC; on battery it sleeps and suspends the job).
- Minor repo cleanup: committed `.pyc` files and a UTF-16 `.gitignore` (git
  misreads it, so nothing is actually ignored).
