"""
Probabilistic "random line-choice" support simulation for the n x n x n
tristochastic polytope  (Nati's idea).

Construction
------------
There are 3 n^2 lines in the tensor (the shafts (i,j,*), (i,*,k), (*,j,k)).
For each line pick ONE coordinate uniformly at random; let S be the union of
the chosen coordinates.  A coordinate lies on 3 lines, each choosing it with
probability 1/n, so

    Pr[(i,j,k) in S] = 1 - (1 - 1/n)^3   ==>   E|S| = 3 n^2 - 3 n + 1,

which is exactly the maximum possible vertex support.  That is why this is
interesting: a random support already has the right expected size.

Then we ask, for each random S:
  * is S the support of a feasible polytope point?   ("exact-positive")
  * is S a vertex support?                            (also nullity(A_S) = 0)

Both are decided with ONE linear program (NOT least squares -- least squares
could return a negative solution even when a nonnegative one exists):

    maximize t   subject to   A_S y = b ,   y_i >= t  for all i.

Let t* be the optimum.  Because every line contributes a coordinate to S, each
line-sum constraint pins sum_{L cap S} y = 1, so t* <= 1 (bounded).  Then:

    LP infeasible          ->  b not in Im(A_S)         : NOT solvable
    t* <= 1e-9             ->  no strictly positive soln: solvable, not exact
    t* >  1e-9             ->  y > 0 exists             : EXACT positive support
    exact positive & nullity(A_S)==0                    : VERTEX support

CLI
---
    python "probabilistic simulation.py" --n 5 --trials 1000 --seed 0 --out line_support_n5.jsonl
    # never repeat supports from earlier runs (or the LP experiments):
    python "probabilistic simulation.py" --n 5 --trials 1000 --seed 1000 \
        --out line_support_n5.jsonl --exclude-existing line_support_n5.jsonl frac_n5.jsonl

Saved records carry `support_map`, so the existing viewer works on them too:
    python "polytope simulation.py" show line_support_n5.jsonl --index 0
"""

import argparse
import json
import os
import time
from collections import Counter
from fractions import Fraction
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


# ------------------------------------------------------------
# Constraint matrix + lines  (copied from "polytope simulation.py")
# ------------------------------------------------------------

def build_tristochastic_constraints(n: int):
    """A x = b for the n x n x n tristochastic polytope; idx = i*n*n + j*n + k."""
    N = n ** 3
    triple_to_var: Dict[Tuple[int, int, int], int] = {}
    for i in range(n):
        for j in range(n):
            for k in range(n):
                triple_to_var[(i, j, k)] = i * n * n + j * n + k

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    row = 0
    for i in range(n):                       # sum_k x[i,j,k] = 1
        for j in range(n):
            for k in range(n):
                rows.append(row); cols.append(i * n * n + j * n + k); data.append(1.0)
            row += 1
    for i in range(n):                       # sum_j x[i,j,k] = 1
        for k in range(n):
            for j in range(n):
                rows.append(row); cols.append(i * n * n + j * n + k); data.append(1.0)
            row += 1
    for j in range(n):                       # sum_i x[i,j,k] = 1
        for k in range(n):
            for i in range(n):
                rows.append(row); cols.append(i * n * n + j * n + k); data.append(1.0)
            row += 1

    A = sp.coo_matrix((data, (rows, cols)), shape=(3 * n * n, N)).tocsr()
    b = np.ones(3 * n * n)
    return A, b, triple_to_var


def constraint_rank_expected(n: int) -> int:
    """rank(A) = maximum vertex support size = E|S| of the random construction."""
    return 3 * n * n - 3 * n + 1


def build_lines(n: int) -> List[List[Tuple[int, int, int]]]:
    """The 3 n^2 lines, each as its list of n coordinates."""
    lines: List[List[Tuple[int, int, int]]] = []
    for i in range(n):
        for j in range(n):
            lines.append([(i, j, k) for k in range(n)])   # (i,j,*)
    for i in range(n):
        for k in range(n):
            lines.append([(i, j, k) for j in range(n)])   # (i,*,k)
    for j in range(n):
        for k in range(n):
            lines.append([(i, j, k) for i in range(n)])   # (*,j,k)
    return lines


def random_line_choice_support(lines, rng) -> frozenset:
    """Pick one coordinate per line uniformly; return the union as a frozenset."""
    S = set()
    for line in lines:
        S.add(line[rng.integers(len(line))])
    return frozenset(S)


# ------------------------------------------------------------
# The two miracles + positivity, via one max-t LP
# ------------------------------------------------------------

def _maxt_lp(A_S, b: np.ndarray, m: int):
    """
    maximize t  s.t.  A_S y = b,  y_i >= t.
    Variables z = [y_0..y_{m-1}, t] (all free).  Returns the linprog result.
    """
    c = np.zeros(m + 1)
    c[-1] = -1.0                                             # minimize -t
    A_eq = sp.hstack([A_S, sp.csr_matrix((A_S.shape[0], 1))], format="csr")
    A_ub = sp.hstack([-sp.identity(m, format="csr"),        # -y_i + t <= 0
                      sp.csr_matrix(np.ones((m, 1)))], format="csr")
    b_ub = np.zeros(m)
    bounds = [(None, None)] * (m + 1)
    return linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b,
                   bounds=bounds, method="highs")


def analyze_support(S, A, triple_to_var, b,
                    eps_pos: float = 1e-9, eps_rank: float = 1e-8) -> Dict[str, Any]:
    """
    Classify a candidate support S.  nullity is computed only when the support
    is exact-positive (that is the only case where the vertex test matters).
    """
    cols = [triple_to_var[t] for t in S]
    m = len(S)
    A_S = A[:, cols]

    res = _maxt_lp(A_S, b, m)
    out: Dict[str, Any] = {
        "support_size": m, "lp_success": bool(res.success),
        "lp_status": int(res.status), "t_star": None,
        "min_y": None, "max_y": None, "nullity": None, "y": None,
    }
    if not res.success:                       # b not in Im(A_S): not solvable
        return out

    y = np.asarray(res.x[:m], dtype=float)
    out["t_star"] = float(res.x[m])
    out["min_y"] = float(np.min(y))
    out["max_y"] = float(np.max(y))

    # rank/nullity for EVERY solvable support (tells unique vs many solutions)
    A_S_dense = A_S.toarray()
    s = np.linalg.svd(A_S_dense, compute_uv=False)
    tol = max(eps_rank, s.max() * max(A_S_dense.shape) * np.finfo(float).eps)
    out["nullity"] = m - int(np.sum(s > tol))
    if out["t_star"] > eps_pos:               # exact-positive -> keep the solution
        out["y"] = y
    return out


# ------------------------------------------------------------
# Records + dedup helpers
# ------------------------------------------------------------

def _value_to_fraction_str(v: float, tol: float = 1e-6, max_den: int = 100) -> str:
    """Render v as 'p/q' if it snaps to a small-denominator rational, else decimal."""
    f = Fraction(v).limit_denominator(max_den)
    return str(f) if abs(float(f) - v) <= tol else f"{v:.6g}"


def _support_map_from_S(S, n: int) -> List[List[List[int]]]:
    """Q[i][j] = sorted supported k, rebuilt from the support set."""
    Q = [[[] for _ in range(n)] for _ in range(n)]
    for (i, j, k) in S:
        Q[i][j].append(k)
    for i in range(n):
        for j in range(n):
            Q[i][j].sort()
    return Q


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


def build_record(n: int, S, info: Dict[str, Any], max_supp: int,
                 is_vertex: bool) -> Dict[str, Any]:
    """Compact record for an exact-positive support (vertex or not)."""
    rec: Dict[str, Any] = {
        "support_size": info["support_size"],
        "category": "vertex" if is_vertex else "positive_nonvertex",
        "is_simple_vertex": bool(is_vertex and info["support_size"] == max_supp),
        "nullity": int(info["nullity"]),
        "t_star": info["t_star"],
        "min_y": info["min_y"],
    }
    if is_vertex and info.get("y") is not None:      # unique solution -> its values
        distinct = sorted(set(np.round(info["y"], 6).tolist()))
        rec["distinct_values"] = [_value_to_fraction_str(v) for v in distinct]
    rec["support_map"] = _support_map_from_S(S, n)
    return rec


def failed_record(n: int, category: str, S, info: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal record for a failed support (just enough to view its map later)."""
    return {
        "n": n,
        "category": category,
        "support_size": len(S),
        "t_star": info.get("t_star"),
        "lp_status": info.get("lp_status"),
        "support_map": _support_map_from_S(S, n),
    }


# ------------------------------------------------------------
# Main run
# ------------------------------------------------------------

def run(args) -> None:
    A, b, ttv = build_tristochastic_constraints(args.n)
    lines = build_lines(args.n)
    max_supp = constraint_rank_expected(args.n)
    rng = np.random.default_rng(args.seed)

    seen = set()
    existing_records: List[dict] = []
    out_in_exclude = False
    if args.exclude_existing:
        out_abs = os.path.normcase(os.path.abspath(args.out)) if args.out else None
        for path in args.exclude_existing:
            same = out_abs is not None and os.path.normcase(os.path.abspath(path)) == out_abs
            out_in_exclude = out_in_exclude or same
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    key = _extract_support(rec)
                    if key is not None:
                        seen.add(key)
                    if same:
                        existing_records.append(rec)
        print(f"excluding {len(seen)} supports from {len(args.exclude_existing)} file(s)\n")

    collected: List[dict] = []
    failed: List[dict] = []                          # up to args.save_failed failed supports
    trials_attempted = n_dup = new_unique = 0
    n_not_solvable = n_solvable_no_pos = n_positive = n_vertex = n_simple = 0
    n_unique_mixed = 0                                # uniquely solvable but not all-positive
    n_solvable_forced_negative = 0                   # solvable, but every solution has a negative
    n_unique_negative = 0                            # unique solution with an actual negative entry
    support_sum = 0
    vertex_sizes: List[int] = []
    t_start = time.time()

    for _ in range(args.trials):
        trials_attempted += 1
        S = random_line_choice_support(lines, rng)
        if S in seen:
            n_dup += 1
            continue
        seen.add(S)
        new_unique += 1
        support_sum += len(S)

        info = analyze_support(S, A, ttv, b, eps_pos=args.eps)
        if not info["lp_success"]:
            n_not_solvable += 1
            if len(failed) < args.save_failed:
                failed.append(failed_record(args.n, "not_solvable", S, info))
            continue
        if info["t_star"] is None or info["t_star"] <= args.eps:
            n_solvable_no_pos += 1
            if info["nullity"] == 0:                  # unique solution, not exact-positive
                n_unique_mixed += 1
            if info["t_star"] is not None and info["t_star"] < -args.eps:
                n_solvable_forced_negative += 1       # every solution has a negative entry
                if info["nullity"] == 0 and info["min_y"] < -args.eps:
                    n_unique_negative += 1
            if len(failed) < args.save_failed:
                failed.append(failed_record(args.n, "solvable_no_positive", S, info))
            continue

        n_positive += 1
        is_vertex = info["nullity"] == 0
        collected.append(build_record(args.n, S, info, max_supp, is_vertex))
        if is_vertex:
            n_vertex += 1
            vertex_sizes.append(len(S))
            if len(S) == max_supp:
                n_simple += 1

    elapsed = time.time() - t_start

    n_saved = 0
    if args.out:
        out_records = (existing_records + collected
                       if out_in_exclude and existing_records else collected)
        with open(args.out, "w", encoding="utf-8") as f:
            for rec in out_records:
                f.write(json.dumps(rec) + "\n")
        n_saved = len(collected)

    if args.failed_out and failed:
        with open(args.failed_out, "w", encoding="utf-8") as f:
            for rec in failed:
                f.write(json.dumps(rec) + "\n")
        print(f"saved {len(failed)} failed supports to {args.failed_out}")

    _print_summary(args, max_supp, dict(
        trials_attempted=trials_attempted, new_unique=new_unique, n_dup=n_dup,
        mean_support=(support_sum / new_unique if new_unique else 0.0),
        n_not_solvable=n_not_solvable, n_solvable_no_pos=n_solvable_no_pos,
        n_positive=n_positive, n_vertex=n_vertex, n_simple=n_simple,
        n_unique_mixed=n_unique_mixed,
        n_solvable_forced_negative=n_solvable_forced_negative,
        n_unique_negative=n_unique_negative,
        vertex_sizes=vertex_sizes, elapsed=elapsed, n_saved=n_saved,
        merged=bool(out_in_exclude and existing_records), n_existing=len(existing_records)))


def _print_summary(args, max_supp, s) -> None:
    denom = max(1, s["new_unique"])
    print("=" * 62)
    print(f"RANDOM LINE-CHOICE SUPPORT   (n = {args.n})")
    print("=" * 62)
    print(f"  trials_attempted             : {s['trials_attempted']}")
    print(f"  new_unique_supports_tested   : {s['new_unique']}   "
          f"(duplicates skipped: {s['n_dup']})")
    print(f"  max_vertex_support (3n^2-3n+1): {max_supp}")
    print(f"  mean support size (tested)   : {s['mean_support']:.2f}   "
          f"(theory E|S| = {max_supp})")
    print()
    print(f"  >> prob_positive_support_in_polytope = {s['n_positive'] / denom:.4f}"
          f"   ({s['n_positive']}/{s['new_unique']})")
    print(f"  >> prob_vertex_support               = {s['n_vertex'] / denom:.4f}"
          f"   ({s['n_vertex']}/{s['new_unique']})")
    print()
    print(f"  count_exact_positive         : {s['n_positive']}")
    print(f"  count_vertex                 : {s['n_vertex']}")
    print(f"  simple / full-support verts  : {s['n_simple']}")
    if s["vertex_sizes"]:
        vs = s["vertex_sizes"]
        print(f"  mean support size (vertices) : {np.mean(vs):.2f}  "
              f"(min {min(vs)}, max {max(vs)})")
        print(f"  vertex support-size dist     : {dict(sorted(Counter(vs).items()))}")
    print()
    print(f"  breakdown of {s['new_unique']} tested supports:")
    print(f"    not_solvable               : {s['n_not_solvable']}")
    print(f"    solvable_no_positive       : {s['n_solvable_no_pos']}")
    print(f"    exact_positive             : {s['n_positive']}  "
          f"(of which vertex: {s['n_vertex']})")
    print()
    print(f"  solvability metrics (of {s['new_unique']} tested):")
    print(f"    solvable, exact-positive support exists : {s['n_positive']}")
    print(f"    solvable, not exact-positive            : {s['n_solvable_no_pos']}")
    print(f"        of which forced negative (t*<0)     : {s['n_solvable_forced_negative']}")
    print(f"    uniquely solvable & exact-positive      : {s['n_vertex']}")
    print(f"    uniquely solvable & not exact-positive  : {s['n_unique_mixed']}")
    print(f"        of which forced negative (min_y<0)  : {s['n_unique_negative']}")
    print(f"  wall time                    : {s['elapsed']:.1f}s")
    if args.out:
        extra = f" (+{s['n_existing']} merged from existing)" if s["merged"] else ""
        print(f"  saved {s['n_saved']} positive-support records to {args.out}{extra}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Random line-choice support simulation for the tristochastic polytope.")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None,
                   help="JSONL file for exact-positive support records (optional)")
    p.add_argument("--exclude-existing", nargs="+", default=None, metavar="FILE",
                   help="skip supports already present in these JSONL file(s); "
                        "if --out is one of them, new records are merged in")
    p.add_argument("--eps", type=float, default=1e-9,
                   help="positivity threshold on t* (default 1e-9)")
    p.add_argument("--save-failed", type=int, default=0, metavar="INT",
                   help="save up to INT failed supports for later inspection")
    p.add_argument("--failed-out", type=str, default=None, metavar="FILE",
                   help="JSONL file for the saved failed supports")
    return p


def main(argv=None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
