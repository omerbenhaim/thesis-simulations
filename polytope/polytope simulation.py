"""
Vertices of the n x n x n tristochastic polytope (3D Birkhoff polytope).

We study tensors x = (x_ijk) in R^{n^3} with x_ijk >= 0 and every line summing
to 1:

    sum_k x[i,j,k] = 1   for every (i,j)     (n^2 vertical-shaft constraints)
    sum_j x[i,j,k] = 1   for every (i,k)      (n^2 row-symbol constraints)
    sum_i x[i,j,k] = 1   for every (j,k)      (n^2 col-symbol constraints)

There are n^3 variables and 3 n^2 line-sum constraints.  The constraint matrix
A has rank 3 n^2 - 3 n + 1 (the rows are linearly dependent), so a vertex has
support of size at most 3 n^2 - 3 n + 1.

A feasible point x is a VERTEX  <=>  the columns of A indexed by the support
S = supp(x) are linearly independent  <=>  ker(A_S) = {0}  <=>  nullity 0.

We sample vertices by maximizing a random linear objective <v, x> over the
polytope (an LP), extract the support, and study its combinatorial structure:
the "quasi Latin square" support map and the line-incidence support graph.

CLI:
    python "polytope simulation.py" demo
    python "polytope simulation.py" sample --n 5 --trials 100 --seed 0 --out results_n5.jsonl
    python "polytope simulation.py" summarize results_n5.jsonl
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.linalg import lstsq
from scipy.optimize import linprog
import networkx as nx


# ------------------------------------------------------------
# 1. Constraint matrix
# ------------------------------------------------------------

def build_tristochastic_constraints(n: int):
    """
    Build the line-sum constraint system A x = b for the n x n x n
    tristochastic polytope.

    Variables are indexed by triples with the flat convention
        idx = i*n*n + j*n + k
    which matches numpy's C-order reshape((n, n, n)).

    Returns
    -------
    A : scipy.sparse.csr_matrix, shape (3*n*n, n**3)
    b : np.ndarray, shape (3*n*n,), all ones
    triple_to_var : dict[(i,j,k)] -> col
    var_to_triple : list[(i,j,k)]  (indexed by col)
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    N = n ** 3
    triple_to_var: Dict[Tuple[int, int, int], int] = {}
    var_to_triple: List[Tuple[int, int, int]] = [(-1, -1, -1)] * N
    for i in range(n):
        for j in range(n):
            for k in range(n):
                idx = i * n * n + j * n + k
                triple_to_var[(i, j, k)] = idx
                var_to_triple[idx] = (i, j, k)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    row = 0

    # Block 1: sum_k x[i,j,k] = 1   for every (i,j)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                rows.append(row)
                cols.append(i * n * n + j * n + k)
                data.append(1.0)
            row += 1

    # Block 2: sum_j x[i,j,k] = 1   for every (i,k)
    for i in range(n):
        for k in range(n):
            for j in range(n):
                rows.append(row)
                cols.append(i * n * n + j * n + k)
                data.append(1.0)
            row += 1

    # Block 3: sum_i x[i,j,k] = 1   for every (j,k)
    for j in range(n):
        for k in range(n):
            for i in range(n):
                rows.append(row)
                cols.append(i * n * n + j * n + k)
                data.append(1.0)
            row += 1

    A = sp.coo_matrix((data, (rows, cols)), shape=(3 * n * n, N)).tocsr()
    b = np.ones(3 * n * n)
    return A, b, triple_to_var, var_to_triple


def constraint_rank_expected(n: int) -> int:
    """Known rank of A (and max support size of a vertex)."""
    return 3 * n * n - 3 * n + 1


# ------------------------------------------------------------
# 2. Sample a random vertex via LP
# ------------------------------------------------------------

def solve_random_vertex(n: int, seed: int, A=None, b=None,
                        method: str = "highs-ds", eps: float = 1e-9) -> Dict[str, Any]:
    """
    Maximize <v, x> over the polytope for a random gaussian direction v.
    For generic v the optimum is a vertex.

    Returns a dict with x (tensor), support S, objective value, LP status, seed.
    """
    if A is None or b is None:
        A, b, _, _ = build_tristochastic_constraints(n)

    rng = np.random.default_rng(seed)
    v = rng.normal(size=n ** 3)

    res = linprog(c=-v, A_eq=A, b_eq=b, bounds=(0, None), method=method)

    out: Dict[str, Any] = {
        "seed": seed,
        "lp_status": int(res.status),
        "lp_message": str(res.message),
        "success": bool(res.success),
        "x": None,
        "support": [],
        "objective_value": None,
    }
    if res.success and res.x is not None:
        x = res.x.reshape((n, n, n))
        S = [(i, j, k)
             for i in range(n) for j in range(n) for k in range(n)
             if x[i, j, k] > eps]
        out["x"] = x
        out["support"] = S
        out["objective_value"] = float(-res.fun)  # we maximized <v,x>
    return out


# ------------------------------------------------------------
# 3. 2D support map (quasi Latin square) + histograms
# ------------------------------------------------------------

def build_support_map(x: np.ndarray, eps: float = 1e-9) -> List[List[List[int]]]:
    """
    Q[i][j] = sorted list of supported k for the vertical shaft (i, j, *).
    Instead of one symbol per cell, each cell carries a set of symbols.
    """
    n = x.shape[0]
    return [[[k for k in range(n) if x[i, j, k] > eps] for j in range(n)]
            for i in range(n)]


def support_histograms(x: np.ndarray, eps: float = 1e-9):
    """
    Three line-direction support-size histograms:
      shaft : over k for fixed (i,j)   -- the vertical shafts
      rowk  : over j for fixed (i,k)
      colk  : over i for fixed (j,k)
    """
    n = x.shape[0]
    shaft: Counter = Counter()
    rowk: Counter = Counter()
    colk: Counter = Counter()
    for i in range(n):
        for j in range(n):
            shaft[int(np.sum(x[i, j, :] > eps))] += 1
    for i in range(n):
        for k in range(n):
            rowk[int(np.sum(x[i, :, k] > eps))] += 1
    for j in range(n):
        for k in range(n):
            colk[int(np.sum(x[:, j, k] > eps))] += 1
    return shaft, rowk, colk


# ------------------------------------------------------------
# 4. Vertex certificate: rank / nullity of A_S
# ------------------------------------------------------------

def analyze_support_rank(A, S, triple_to_var, b, x=None,
                         eps_rank: float = 1e-8) -> Dict[str, Any]:
    """
    Restrict A to the support columns and check the vertex certificate.

    nullity == 0 (full column rank) certifies that the support determines a
    unique point, i.e. x is a vertex.  We also solve A_S y = b in least squares
    and confirm y matches the LP's positive values.
    """
    support_size = len(S)
    result: Dict[str, Any] = {
        "support_size": support_size,
        "rank_AS": 0,
        "nullity": 0,
        "residual_norm": None,
        "min_y": None,
        "max_y": None,
        "smallest_singular_value": None,
        "max_abs_diff_from_original": None,
    }
    if support_size == 0:
        return result

    cols = [triple_to_var[t] for t in S]
    A_S_dense = A[:, cols].toarray()

    # Numerical rank via SVD with a relative tolerance (floored by eps_rank).
    s = np.linalg.svd(A_S_dense, compute_uv=False)
    tol = max(eps_rank, s.max() * max(A_S_dense.shape) * np.finfo(float).eps)
    rank_AS = int(np.sum(s > tol))
    nullity = support_size - rank_AS

    # Solve the restricted (rank-deficient rows, consistent) system.
    y, _, _, _ = lstsq(A_S_dense, b)
    residual_norm = float(np.linalg.norm(A_S_dense @ y - b))

    result.update({
        "rank_AS": rank_AS,
        "nullity": int(nullity),
        "residual_norm": residual_norm,
        "min_y": float(np.min(y)),
        "max_y": float(np.max(y)),
        "smallest_singular_value": float(s[rank_AS - 1]) if rank_AS > 0 else 0.0,
    })
    if x is not None:
        x_S = np.array([x[i, j, k] for (i, j, k) in S], dtype=float)
        result["max_abs_diff_from_original"] = float(np.max(np.abs(y - x_S)))
    return result


# ------------------------------------------------------------
# 5. Value classification + numerical-stability probes
# ------------------------------------------------------------

def _value_to_fraction_str(v: float, tol: float = 1e-6, max_den: int = 100) -> str:
    """
    Render v as 'p/q' if it snaps to a small-denominator rational within tol,
    else as a decimal string (visibly not clean).

    Vertices of this polytope are always rational (integer A, integer b), so a
    truly irrational value cannot occur.  The guard instead protects against a
    value with a large denominator or solver noise being shown as a bogus clean
    fraction: with a modest max_den, only genuinely small-denominator values
    snap; anything else keeps its decimal.  (max_den is deliberately small --
    a large cap would let some p/q approximate almost any value within tol.)
    """
    f = Fraction(v).limit_denominator(max_den)
    if abs(float(f) - v) <= tol:
        return str(f)
    return f"{v:.6g}"


def value_stats(pos_vals, eps_value: float = 1e-6) -> Dict[str, Any]:
    """Classify the positive support values (integral / half-integral / other)."""
    arr = np.asarray(list(pos_vals), dtype=float)
    if arr.size == 0:
        return {
            "min_positive_value": None, "max_value": None,
            "num_values_near_1": 0, "num_values_near_half": 0,
            "distinct_values": [], "num_distinct_values_rounded": 0,
            "is_integral": False, "is_half_integral": False,
            "distinct_denominators": [], "denominator_guess": 1,
        }

    near_1 = int(np.sum(np.abs(arr - 1.0) < eps_value))
    near_half = int(np.sum(np.abs(arr - 0.5) < eps_value))
    distinct_rounded = sorted(set(np.round(arr, 6).tolist()))
    distinct_values = [_value_to_fraction_str(v, tol=eps_value)
                       for v in distinct_rounded]
    num_distinct = len(distinct_rounded)

    is_integral = bool(np.all(np.abs(arr - 1.0) < eps_value))
    is_half_integral = bool(np.all(
        (np.abs(arr - 0.5) < eps_value) | (np.abs(arr - 1.0) < eps_value)))

    denoms = sorted({Fraction(float(v)).limit_denominator(10000).denominator
                     for v in arr})
    common_den = 1
    for d in denoms:
        common_den = common_den * d // math.gcd(common_den, d)

    return {
        "min_positive_value": float(arr.min()),
        "max_value": float(arr.max()),
        "num_values_near_1": near_1,
        "num_values_near_half": near_half,
        "distinct_values": distinct_values,
        "num_distinct_values_rounded": num_distinct,
        "is_integral": is_integral,
        "is_half_integral": is_half_integral,
        "distinct_denominators": denoms,
        "denominator_guess": int(common_den),
    }


def support_size_by_eps(x: np.ndarray,
                        eps_list=(1e-7, 1e-8, 1e-9, 1e-10)) -> Dict[str, int]:
    """How many entries exceed each threshold -- probes LP numerical noise."""
    return {("%g" % e): int(np.sum(x > e)) for e in eps_list}


# ------------------------------------------------------------
# 6. Support graph (line incidences)
# ------------------------------------------------------------

def build_support_graph(S) -> nx.Graph:
    """
    Graph on the support triples.  Two triples are adjacent iff they share a
    line (agree in two coordinates): same (i,j), or same (i,k), or same (j,k).
    Each line becomes a clique on its supported triples.
    """
    G = nx.Graph()
    G.add_nodes_from(S)

    by_ij: Dict[Tuple[int, int], List] = defaultdict(list)
    by_ik: Dict[Tuple[int, int], List] = defaultdict(list)
    by_jk: Dict[Tuple[int, int], List] = defaultdict(list)
    for t in S:
        i, j, k = t
        by_ij[(i, j)].append(t)
        by_ik[(i, k)].append(t)
        by_jk[(j, k)].append(t)

    for groups in (by_ij, by_ik, by_jk):
        for members in groups.values():
            if len(members) >= 2:
                for a, c in combinations(members, 2):
                    G.add_edge(a, c)
    return G


def analyze_support_graph(G: nx.Graph) -> Dict[str, Any]:
    """Component / bipartite / cycle / degree summary of the support graph."""
    comps = list(nx.connected_components(G))
    comp_sizes = sorted((len(c) for c in comps), reverse=True)
    num_bip = sum(1 for c in comps if nx.is_bipartite(G.subgraph(c)))

    V = G.number_of_nodes()
    E = G.number_of_edges()
    C = len(comps)
    deg_hist = Counter(d for _, d in G.degree())

    return {
        "num_nodes": V,
        "num_edges": E,
        "num_components": C,
        "component_sizes": comp_sizes,
        "num_bipartite_components": num_bip,
        "num_nonbipartite_components": C - num_bip,
        # cyclomatic number = independent cycles (E - V + C)
        "num_independent_cycles": E - V + C,
        "degree_histogram": {str(k): int(v) for k, v in sorted(deg_hist.items())},
    }


# ------------------------------------------------------------
# 7. One full vertex record
# ------------------------------------------------------------

def solution_is_integral(sol: Dict[str, Any], eps_value: float = 1e-6) -> bool:
    """
    True iff the vertex is 0/1 (a Latin square): every positive entry is ~1.
    Cheap check used to reject integral vertices before heavy analysis.
    """
    if not sol.get("success"):
        return False
    x = sol["x"]
    S = sol["support"]
    # A Latin-square vertex has exactly one 1 per (i,j)-shaft, so |S| == n*n.
    return len(S) == x.shape[0] ** 2 and all(
        abs(x[i, j, k] - 1.0) < eps_value for (i, j, k) in S)


def build_vertex_record(n: int, seed: int, sol: Dict[str, Any],
                        A, b, triple_to_var, var_to_triple,
                        eps_support: float = 1e-9, eps_rank: float = 1e-8,
                        eps_value: float = 1e-6,
                        include_support_map: bool = True) -> Dict[str, Any]:
    """Assemble the full JSON-serializable record from an already-solved LP."""
    t0 = time.time()
    rec: Dict[str, Any] = {
        "n": n,
        "seed": seed,
        "lp_status": sol["lp_status"],
        "lp_message": sol["lp_message"],
        "success": sol["success"],
        "constraint_rank_expected": constraint_rank_expected(n),
    }
    if not sol["success"]:
        rec["error"] = "LP did not solve to optimality"
        rec["solve_time_sec"] = time.time() - t0
        return rec

    x = sol["x"]
    S = sol["support"]
    rec["objective_value"] = sol["objective_value"]
    rec["support_size"] = len(S)
    rec["average_vertical_shaft_size"] = len(S) / (n * n)
    # A vertex is SIMPLE (non-degenerate) iff its support is maximal.
    rec["is_simple_vertex"] = (len(S) == constraint_rank_expected(n))

    # histograms
    shaft, rowk, colk = support_histograms(x, eps_support)
    rec["shaft_size_histogram"] = {str(k): int(v) for k, v in sorted(shaft.items())}
    rec["row_symbol_support_histogram"] = {str(k): int(v) for k, v in sorted(rowk.items())}
    rec["column_symbol_support_histogram"] = {str(k): int(v) for k, v in sorted(colk.items())}

    # vertex certificate
    rank_info = analyze_support_rank(A, S, triple_to_var, b, x=x, eps_rank=eps_rank)
    for key in ("rank_AS", "nullity", "residual_norm", "min_y", "max_y",
                "smallest_singular_value", "max_abs_diff_from_original"):
        rec[key] = rank_info[key]
    rec["min_positive_value"] = rank_info["min_y"]  # == min positive x value

    # value classification
    pos_vals = [x[i, j, k] for (i, j, k) in S]
    rec.update(value_stats(pos_vals, eps_value))

    # numerical stability
    ssbe = support_size_by_eps(x)
    rec["support_size_by_eps"] = ssbe
    spread = max(ssbe.values()) - min(ssbe.values())
    rec["support_size_spread_over_eps"] = spread
    rec["numerically_unstable"] = spread > 1

    # support graph
    G = build_support_graph(S)
    rec["support_graph"] = analyze_support_graph(G)

    if include_support_map:
        rec["support_map"] = build_support_map(x, eps_support)
        rec["support"] = [[i, j, k] for (i, j, k) in S]

    rec["solve_time_sec"] = time.time() - t0
    return rec


def sample_vertex_record(n: int, seed: int, A, b, triple_to_var, var_to_triple,
                         eps_support: float = 1e-9, eps_rank: float = 1e-8,
                         eps_value: float = 1e-6,
                         include_support_map: bool = True) -> Dict[str, Any]:
    """Solve one random-objective LP and assemble its full record."""
    sol = solve_random_vertex(n, seed, A=A, b=b, eps=eps_support)
    return build_vertex_record(n, seed, sol, A, b, triple_to_var, var_to_triple,
                               eps_support, eps_rank, eps_value, include_support_map)


# ------------------------------------------------------------
# JSON helpers
# ------------------------------------------------------------

def _jsonable(o: Any) -> Any:
    """Recursively convert numpy scalars/arrays to native python for json.dump."""
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def _slim_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Compact enrich record: only the fields worth studying per vertex."""
    g = rec.get("support_graph", {})
    slim = {
        "seed": rec["seed"],
        "distinct_values": rec.get("distinct_values"),
        "support_size": rec["support_size"],
        "is_half_integral": rec["is_half_integral"],
        "shaft_size_histogram": rec["shaft_size_histogram"],
        "row_symbol_support_histogram": rec["row_symbol_support_histogram"],
        "column_symbol_support_histogram": rec["column_symbol_support_histogram"],
        "component_sizes": g.get("component_sizes"),
        "num_bipartite_components": g.get("num_bipartite_components"),
        "num_nonbipartite_components": g.get("num_nonbipartite_components"),
    }
    if "support_map" in rec:  # present only if --no-support-map was off
        slim["support_map"] = rec["support_map"]
    return slim


def _extract_support(rec: Dict[str, Any]):
    """Frozenset of support triples from a saved record (support_map or support)."""
    if "support_map" in rec:
        Q = rec["support_map"]
        return frozenset((i, j, k)
                         for i, row in enumerate(Q)
                         for j, ks in enumerate(row) for k in ks)
    if "support" in rec:
        return frozenset(tuple(t) for t in rec["support"])
    return None


# ------------------------------------------------------------
# CLI: sample
# ------------------------------------------------------------

def cmd_sample(args) -> None:
    A, b, ttv, vtt = build_tristochastic_constraints(args.n)
    max_supp = constraint_rank_expected(args.n)
    n_ok = 0
    t_start = time.time()

    with open(args.out, "w", encoding="utf-8") as f:
        for t in range(args.trials):
            seed = args.seed + t
            rec = sample_vertex_record(
                args.n, seed, A, b, ttv, vtt,
                eps_support=args.eps,
                include_support_map=not args.no_support_map,
            )
            f.write(json.dumps(_jsonable(rec)) + "\n")

            if rec.get("success"):
                n_ok += 1
                resid = rec.get("residual_norm")
                print(f"[n={args.n} {t + 1:>4}/{args.trials} seed={seed}] "
                      f"support={rec['support_size']:>3}/{max_supp} "
                      f"nullity={rec['nullity']} "
                      f"resid={resid:.1e} "
                      f"minpos={rec['min_positive_value']:.2e} "
                      f"comp={rec['support_graph']['num_components']} "
                      f"{'INT' if rec['is_integral'] else ('HALF' if rec['is_half_integral'] else 'frac')}")
            else:
                print(f"[n={args.n} {t + 1:>4}/{args.trials} seed={seed}] "
                      f"LP FAILED status={rec['lp_status']}")

    print(f"\nwrote {args.trials} records ({n_ok} successful) to {args.out} "
          f"in {time.time() - t_start:.1f}s")


# ------------------------------------------------------------
# CLI: enrich  (direction B -- keep ONLY fractional vertices)
# ------------------------------------------------------------

def cmd_enrich(args) -> None:
    """
    Keep sampling random-objective LP vertices, reject the integral (0/1)
    ones, and collect the fractional vertices.  Rank the collection by support
    size, highlighting the SIMPLE (full-support) fractional vertices.
    """
    A, b, ttv, vtt = build_tristochastic_constraints(args.n)
    max_supp = constraint_rank_expected(args.n)
    max_attempts = (args.max_attempts if args.max_attempts is not None
                    else args.target * 500)

    collected: List[dict] = []
    seen_supports: set = set()
    existing_records: List[dict] = []
    out_in_exclude = False
    n_integral = n_fail = n_dup = n_small = attempts = 0
    seed = args.seed
    t_start = time.time()

    if args.exclude_existing:
        out_abs = os.path.normcase(os.path.abspath(args.out))
        for path in args.exclude_existing:
            same_as_out = os.path.normcase(os.path.abspath(path)) == out_abs
            out_in_exclude = out_in_exclude or same_as_out
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    key = _extract_support(rec)
                    if key is not None:
                        seen_supports.add(key)
                    if same_as_out:
                        existing_records.append(rec)
        print(f"excluding {len(seen_supports)} supports from "
              f"{len(args.exclude_existing)} existing file(s)")

    print(f"enriching for fractional vertices  (n={args.n}, "
          f"target={args.target}, max_attempts={max_attempts})\n")

    while len(collected) < args.target and attempts < max_attempts:
        used_seed = seed
        sol = solve_random_vertex(args.n, used_seed, A=A, b=b, eps=args.eps)
        attempts += 1
        seed += 1

        if not sol["success"]:
            n_fail += 1
            continue
        if solution_is_integral(sol, args.eps_value):
            n_integral += 1
            continue
        if len(sol["support"]) < args.min_support:
            n_small += 1
            continue
        if not args.no_dedup:
            key = frozenset(sol["support"])
            if key in seen_supports:
                n_dup += 1
                continue
            seen_supports.add(key)

        rec = build_vertex_record(args.n, used_seed, sol, A, b, ttv, vtt,
                                  eps_support=args.eps, eps_value=args.eps_value,
                                  include_support_map=not args.no_support_map)
        collected.append(rec)
        print(f"  [{len(collected):>4}/{args.target}] "
              f"support={rec['support_size']:>3}/{max_supp} "
              f"{'SIMPLE' if rec['is_simple_vertex'] else '      '} "
              f"null={rec['nullity']} "
              f"minpos={rec['min_positive_value']:.2e} "
              f"denom={rec['denominator_guess']:>3} "
              f"{'half' if rec['is_half_integral'] else 'frac':>4} "
              f"nonbip={rec['support_graph']['num_nonbipartite_components']:>2} "
              f"(seed {used_seed}, attempt {attempts})")

    # Rank by support size (desc), then average shaft size (desc).
    collected.sort(key=lambda r: (-r["support_size"],
                                  -r["average_vertical_shaft_size"]))
    out_records = [_slim_record(r) for r in collected]
    if out_in_exclude and existing_records:
        # Grow the same file: keep old records, add the new ones, re-sort.
        out_records = sorted(existing_records + out_records,
                             key=lambda r: -r.get("support_size", 0))

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(_jsonable(rec)) + "\n")
    if out_in_exclude and existing_records:
        print(f"merged into {args.out}: {len(existing_records)} existing + "
              f"{len(collected)} new = {len(out_records)} total")

    _report_enrichment(args, collected, max_supp, attempts,
                       n_integral, n_fail, n_dup, n_small, time.time() - t_start)


def _report_enrichment(args, collected, max_supp, attempts,
                       n_integral, n_fail, n_dup, n_small, elapsed) -> None:
    successful = attempts - n_fail
    n_frac = successful - n_integral  # fractional vertices encountered

    print("\n" + "=" * 64)
    print(f"ENRICHMENT SUMMARY   (n={args.n}, max support {max_supp})")
    print("=" * 64)
    print(f"  LP solves (attempts)   : {attempts}")
    if n_fail:
        print(f"  failed LPs             : {n_fail}")
    print(f"  integral (rejected)    : {n_integral}")
    print(f"  fractional encountered : {n_frac}"
          + (f"   ({n_frac / successful:.1%} of successful)" if successful else ""))
    if not args.no_dedup:
        print(f"  duplicate supports     : {n_dup}")
    if args.min_support > 0:
        print(f"  below min-support {args.min_support:<4} : {n_small}")
    print(f"  collected (kept)       : {len(collected)}")
    print(f"  wall time              : {elapsed:.1f}s")

    if not collected:
        print("\n  (no fractional vertices collected -- raise --max-attempts)")
        return

    sizes = [r["support_size"] for r in collected]
    n_simple = sum(1 for r in collected if r["is_simple_vertex"])
    n_half = sum(1 for r in collected if r["is_half_integral"])
    n_other = len(collected) - n_half

    print(f"\n  support size : mean {np.mean(sizes):.1f}, "
          f"min {min(sizes)}, max {max(sizes)}  (of {max_supp})")
    print(f"  support-size distribution : {dict(sorted(Counter(sizes).items()))}")
    print(f"  simple (full-support)     : {n_simple}")
    print(f"  half-integral / other     : {n_half} / {n_other}")
    print(f"  denominator guesses       : "
          f"{dict(sorted(Counter(r['denominator_guess'] for r in collected).items()))}")

    topk = min(args.top, len(collected))
    print(f"\n  top {topk} by support size:")
    print(f"    {'#':>3} {'support':>7} {'null':>4} {'simple':>6} {'minpos':>9} "
          f"{'denom':>5} {'class':>5} {'nonbip':>6}  shaft_hist")
    for idx, r in enumerate(collected[:topk], 1):
        print(f"    {idx:>3} {r['support_size']:>7} "
              f"{r['nullity']:>4} "
              f"{'yes' if r['is_simple_vertex'] else 'no':>6} "
              f"{r['min_positive_value']:>9.2e} "
              f"{r['denominator_guess']:>5} "
              f"{'half' if r['is_half_integral'] else 'frac':>5} "
              f"{r['support_graph']['num_nonbipartite_components']:>6}  "
              f"{r['shaft_size_histogram']}")

    print(f"\n  wrote {len(collected)} fractional vertices to {args.out} "
          f"(sorted by support size, desc)")


# ------------------------------------------------------------
# CLI: demo  (first milestone)
# ------------------------------------------------------------

def cmd_demo(args) -> None:
    ns = args.ns if getattr(args, "ns", None) else [3, 4, 5]
    trials = getattr(args, "trials", None) or 10

    for n in ns:
        A, b, ttv, vtt = build_tristochastic_constraints(n)
        max_supp = constraint_rank_expected(n)
        print(f"\n===== n={n}   (max support = 3n^2-3n+1 = {max_supp}) =====")

        out_path = f"demo_n{n}.jsonl"
        supp_sizes: List[int] = []
        shaft_avgs: List[float] = []
        all_ok = True

        with open(out_path, "w", encoding="utf-8") as f:
            for t in range(trials):
                seed = 1000 * n + t
                rec = sample_vertex_record(n, seed, A, b, ttv, vtt,
                                           include_support_map=True)
                f.write(json.dumps(_jsonable(rec)) + "\n")

                if not rec.get("success"):
                    print(f"  seed={seed:>5}  LP FAILED status={rec['lp_status']}  CHECK!")
                    all_ok = False
                    continue

                ok = (rec["nullity"] == 0
                      and rec["support_size"] <= max_supp
                      and (rec["residual_norm"] or 0.0) < 1e-6)
                all_ok = all_ok and ok
                supp_sizes.append(rec["support_size"])
                shaft_avgs.append(rec["average_vertical_shaft_size"])

                print(f"  seed={seed:>5}  support={rec['support_size']:>3}  "
                      f"avg_shaft={rec['average_vertical_shaft_size']:.2f}  "
                      f"rank={rec['rank_AS']:>3}  nullity={rec['nullity']}  "
                      f"resid={rec['residual_norm']:.1e}  "
                      f"shaft_hist={rec['shaft_size_histogram']}  "
                      f"{'OK' if ok else 'CHECK!'}")

        print(f"  --> support sizes: min={min(supp_sizes)} max={max(supp_sizes)} "
              f"(<= {max_supp}); avg_shaft in "
              f"[{min(shaft_avgs):.2f}, {max(shaft_avgs):.2f}]; "
              f"all sanity checks {'PASSED' if all_ok else 'FAILED'}; "
              f"support maps saved to {out_path}")


# ------------------------------------------------------------
# CLI: summarize  (second milestone)
# ------------------------------------------------------------

def cmd_summarize(args) -> None:
    by_n: Dict[int, List[dict]] = defaultdict(list)
    for path in args.files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                by_n[rec["n"]].append(rec)

    for n in sorted(by_n):
        recs = by_n[n]
        succ = [r for r in recs if r.get("success")]
        max_supp = constraint_rank_expected(n)

        print(f"\n================ n = {n}  (max support {max_supp}) ================")
        print(f"  trials                 : {len(recs)}")
        print(f"  successful LPs         : {len(succ)}")
        if not succ:
            continue

        sizes = [r["support_size"] for r in succ]
        shafts = [r["average_vertical_shaft_size"] for r in succ]
        print(f"  distinct support sizes : {len(set(sizes))}  {sorted(set(sizes))}")
        print(f"  support size mean/min/max: {np.mean(sizes):.2f} / {min(sizes)} / {max(sizes)}")
        print(f"  mean vertical shaft size: {np.mean(shafts):.3f}")

        # aggregate shaft-size distribution
        shaft_agg: Counter = Counter()
        for r in succ:
            for k, v in r.get("shaft_size_histogram", {}).items():
                shaft_agg[int(k)] += v
        print(f"  shaft-size distribution : {dict(sorted(shaft_agg.items()))}")

        # nullity distribution
        null_agg = Counter(r["nullity"] for r in succ)
        print(f"  nullity distribution    : {dict(sorted(null_agg.items()))}")

        # value classes
        n_int = sum(1 for r in succ if r.get("is_integral"))
        n_half = sum(1 for r in succ if r.get("is_half_integral") and not r.get("is_integral"))
        n_other = len(succ) - n_int - n_half
        print(f"  fraction integral       : {n_int / len(succ):.3f}  ({n_int})")
        print(f"  fraction half-integral  : {n_half / len(succ):.3f}  ({n_half})")
        print(f"  fraction outside 0,1/2,1: {n_other / len(succ):.3f}  ({n_other})")

        dens = Counter(r.get("denominator_guess") for r in succ)
        print(f"  denominator guesses     : {dict(sorted(dens.items()))}")

        # support graph stats
        comps = [r["support_graph"]["num_components"] for r in succ if "support_graph" in r]
        bip = [r["support_graph"]["num_bipartite_components"] for r in succ if "support_graph" in r]
        nonbip = [r["support_graph"]["num_nonbipartite_components"] for r in succ if "support_graph" in r]
        if comps:
            print(f"  graph components mean   : {np.mean(comps):.2f} "
                  f"(bipartite {np.mean(bip):.2f}, non-bipartite {np.mean(nonbip):.2f})")

        unstable = sum(1 for r in succ if r.get("numerically_unstable"))
        if unstable:
            print(f"  numerically unstable    : {unstable} sample(s) flagged")


# ------------------------------------------------------------
# CLI: show  (render the quasi Latin square support map)
# ------------------------------------------------------------

def _print_quasi_latin_square(Q) -> None:
    """Print Q[i][j] = {supported k} as a bordered n x n grid (quasi Latin square)."""
    n = len(Q)
    cells = [[",".join(str(k) for k in Q[i][j]) or "." for j in range(n)]
             for i in range(n)]
    w = max(3, max(len(c) for row in cells for c in row))
    sep = "     +" + "+".join("-" * (w + 2) for _ in range(n)) + "+"
    print("     " + "".join(f"  {('j' + str(j)).center(w)} " for j in range(n)))
    print(sep)
    for i in range(n):
        body = "|".join(f" {cells[i][j].center(w)} " for j in range(n))
        print(f" i{i:<2} |{body}|")
        print(sep)


def cmd_show(args) -> None:
    with open(args.file, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    if not recs:
        print("no records in file")
        return

    if args.seed is not None:
        indices = [i for i, r in enumerate(recs) if r.get("seed") == args.seed]
        if not indices:
            print(f"no record with seed={args.seed}")
            return
    elif args.all:
        indices = range(len(recs))
    else:
        indices = [args.index]

    for idx in indices:
        if not 0 <= idx < len(recs):
            print(f"index {idx} out of range (0..{len(recs) - 1})")
            continue
        rec = recs[idx]
        Q = rec.get("support_map")
        if Q is None:
            print(f"\n=== record {idx}: seed={rec.get('seed')} "
                  f"(no support_map -- re-run enrich WITHOUT --no-support-map) ===")
            continue
        max_supp = constraint_rank_expected(len(Q))
        print(f"\n=== record {idx}   seed={rec.get('seed')}   "
              f"support={rec.get('support_size')}/{max_supp}   "
              f"half_integral={rec.get('is_half_integral')}   "
              f"nonbip_components={rec.get('num_nonbipartite_components')} ===")
        print(f"    shaft-size histogram: {rec.get('shaft_size_histogram')}")
        _print_quasi_latin_square(Q)


# ------------------------------------------------------------
# argparse
# ------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sample and study vertices of the n x n x n tristochastic polytope.")
    sub = p.add_subparsers(dest="cmd")

    ps = sub.add_parser("sample", help="sample many vertices to a JSONL file")
    ps.add_argument("--n", type=int, required=True)
    ps.add_argument("--trials", type=int, default=100)
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--out", type=str, required=True)
    ps.add_argument("--eps", type=float, default=1e-9, help="support threshold")
    ps.add_argument("--no-support-map", action="store_true",
                    help="omit bulky support_map/support (for large n)")
    ps.set_defaults(func=cmd_sample)

    pe = sub.add_parser("enrich",
                        help="direction B: collect ONLY fractional (non-0/1) vertices")
    pe.add_argument("--n", type=int, required=True)
    pe.add_argument("--target", type=int, default=50,
                    help="number of fractional vertices to collect")
    pe.add_argument("--max-attempts", type=int, default=None,
                    help="cap on LP solves (default target*500)")
    pe.add_argument("--seed", type=int, default=0)
    pe.add_argument("--out", type=str, required=True)
    pe.add_argument("--eps", type=float, default=1e-9, help="support threshold")
    pe.add_argument("--eps-value", type=float, default=1e-6,
                    help="integrality / value-classification tolerance")
    pe.add_argument("--min-support", type=int, default=0,
                    help="keep only fractional vertices with support >= this")
    pe.add_argument("--no-dedup", action="store_true",
                    help="keep duplicate supports (default: dedup distinct vertices)")
    pe.add_argument("--exclude-existing", nargs="+", default=None, metavar="FILE",
                    help="preload supports from existing JSONL file(s) and never re-save "
                         "them; if --out is one of these files, new vertices are merged in")
    pe.add_argument("--no-support-map", action="store_true")
    pe.add_argument("--top", type=int, default=15,
                    help="how many vertices to show in the ranking table")
    pe.set_defaults(func=cmd_enrich)

    pd = sub.add_parser("demo", help="milestone 1: n=3,4,5 x 10 trials with sanity checks")
    pd.add_argument("--ns", type=int, nargs="+", default=None, help="which n values")
    pd.add_argument("--trials", type=int, default=10)
    pd.set_defaults(func=cmd_demo)

    pm = sub.add_parser("summarize", help="milestone 2: aggregate JSONL files")
    pm.add_argument("files", nargs="+")
    pm.set_defaults(func=cmd_summarize)

    psh = sub.add_parser("show",
                         help="print the quasi Latin square (support map) of saved vertices")
    psh.add_argument("file")
    psh.add_argument("--index", type=int, default=0, help="which record (0-based)")
    psh.add_argument("--seed", type=int, default=None, help="select the record by seed")
    psh.add_argument("--all", action="store_true", help="print every record")
    psh.set_defaults(func=cmd_show)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        args = parser.parse_args(["demo"])  # default action
    args.func(args)


if __name__ == "__main__":
    main()
