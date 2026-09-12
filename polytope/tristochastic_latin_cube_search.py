"""
3-Latin-square + provenance cube-switch search for vertices of the n x n x n
tristochastic polytope.  SEPARATE experiment; own result files.

This REPLACES the earlier "sample 3 Latin squares + random cube" script.

Idea
----
Sample 3 Latin squares, average their permutation tensors:

    X = (P1 + P2 + P3) / 3,     A X = 1,  X >= 0.

Work in EXACT integer units of 1/3, so the tensor holds values in {0,1,2,3}
(0, 1/3, 2/3, 1); no fragile float comparisons.  Admit a triple only if the
initial support size is exactly

    r = 3n^2 - 3n + 1

(the maximal vertex support).  Then repeatedly perform a special 2x2x2 cube
switch that keeps the support size at exactly r, checking after every switch
whether the point is a vertex (rank(A_S) = r).

Cube-switch conditions (exhaustive over all C(n,2)^3 cubes, both orientations)
-----------------------------------------------------------------------------
A cube splits into two checkerboard parity classes of 4 corners.  A switch is
valid when one class P has all four values == 1/3 and the opposite class Z has
all four values == 0, AND the PROVENANCE of the four P corners collectively
covers {L1, L2, L3}:
  * a value-1/3 cell that came from exactly one original Latin square is owned
    by that square (L1/L2/L3);
  * any cell touched by a previous switch has provenance None.
The switch sets P: 1/3 -> 0 and Z: 0 -> 1/3 (i.e. step 1/3 along ker(A)); all 8
cube cells then become provenance None.  Support stays exactly r.

Cycle prevention: track visited supports within a walk; never switch into an
already-visited support.  If no valid unvisited cube exists, the walk stalls and
a fresh triple is sampled.

CLI
---
    python tristochastic_latin_cube_search.py --n 10 --walks 100 --seed 1
"""

import argparse
import json
import os
import time
from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np

# ------------------------------------------------------------
# Reuse the project (via tristochastic_lp_walk, which loads polysim)
# ------------------------------------------------------------
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "lpwalk", os.path.join(_HERE, "tristochastic_lp_walk.py"))
lw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lw)
polysim = lw.polysim

_EVEN_POS = [0, 3, 5, 6]      # corners (a,b,c) with even parity  (t = 4a+2b+c)
_ODD_POS = [1, 2, 4, 7]       # corners with odd parity


# ------------------------------------------------------------
# Latin squares, permutation tensors, cube geometry
# ------------------------------------------------------------

def random_latin_square(n: int, rng) -> np.ndarray:
    """Cyclic base + random row/col/symbol permutations -> a valid Latin square."""
    base = (np.arange(n)[:, None] + np.arange(n)[None, :]) % n
    rp, cp, sp = rng.permutation(n), rng.permutation(n), rng.permutation(n)
    return sp[base[np.ix_(rp, cp)]]


def is_valid_latin(L: np.ndarray, n: int) -> bool:
    full = set(range(n))
    return all(set(L[i, :]) == full for i in range(n)) and \
           all(set(L[:, j]) == full for j in range(n))


def perm_tensor_int(L: np.ndarray, n: int) -> np.ndarray:
    """0/1 permutation tensor of a Latin square, as a flat int vector."""
    v = np.zeros(n ** 3, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            v[i * n * n + j * n + int(L[i, j])] = 1
    return v


def build_cubes(n: int) -> np.ndarray:
    """All C(n,2)^3 cubes as an (ncubes, 8) array of corner flat-indices,
    corner order t = 4a+2b+c."""
    pairs = list(combinations(range(n), 2))
    rows = []
    for i0, i1 in pairs:
        ii = (i0, i1)
        for j0, j1 in pairs:
            jj = (j0, j1)
            for k0, k1 in pairs:
                kk = (k0, k1)
                rows.append([ii[a] * n * n + jj[b] * n + kk[c]
                             for a in (0, 1) for b in (0, 1) for c in (0, 1)])
    return np.array(rows, dtype=np.int64)


# ------------------------------------------------------------
# Vertex test
# ------------------------------------------------------------

def float_rank(A_S: np.ndarray, tol: float = 1e-8) -> int:
    if A_S.shape[1] == 0:
        return 0
    R = np.linalg.qr(A_S, mode="r")
    return int(np.sum(np.abs(np.diag(R)) > tol))


def is_vertex(A, A_dense, support_sorted, r: int) -> bool:
    """support has size r; vertex iff rank(A_S) = r (float pre-filter + exact)."""
    if float_rank(A_dense[:, support_sorted]) != r:
        return False
    return lw.full_column_rank_modp(A, list(support_sorted))


# ------------------------------------------------------------
# Valid cube search (exhaustive, vectorized)
# ------------------------------------------------------------

def find_valid_cubes(X_int, prov, cubes) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Return list of (P_cells, Z_cells) for every cube+orientation where the P
    class is all 1/3 (value 1), the Z class is all 0, and P's provenance covers
    {L1,L2,L3}.  P_cells are switched 1/3->0, Z_cells 0->1/3.
    """
    vals = X_int[cubes]                       # (ncubes, 8)
    ev = vals[:, _EVEN_POS]
    od = vals[:, _ODD_POS]
    out: List[Tuple[np.ndarray, np.ndarray]] = []

    for P_pos, Z_pos, Pmask in ((_EVEN_POS, _ODD_POS, (ev == 1).all(1) & (od == 0).all(1)),
                                (_ODD_POS, _EVEN_POS, (od == 1).all(1) & (ev == 0).all(1))):
        idx = np.nonzero(Pmask)[0]
        if idx.size == 0:
            continue
        Pc = cubes[np.ix_(idx, P_pos)]        # (k, 4) positive corners
        pv = prov[Pc]
        good = (pv == 0).any(1) & (pv == 1).any(1) & (pv == 2).any(1)
        gi = np.nonzero(good)[0]
        if gi.size == 0:
            continue
        Zc = cubes[np.ix_(idx, Z_pos)]
        for g in gi:
            out.append((Pc[g], Zc[g]))
    return out


# ------------------------------------------------------------
# One admitted walk (support already == r)
# ------------------------------------------------------------

def cube_walk(X_int, prov, support_set, A, A_dense, b, r, cubes, wrng,
              max_switches, verbose=False):
    support_sorted = np.array(sorted(support_set))
    visited = {frozenset(support_set)}
    switches = 0
    valid_counts: List[int] = []
    had_valid = False

    if is_vertex(A, A_dense, support_sorted, r):
        return {"status": "vertex_at_start", "X_int": X_int, "support": support_sorted,
                "switches": 0, "valid_counts": valid_counts, "had_valid": False}

    for _step in range(max_switches):
        valid = find_valid_cubes(X_int, prov, cubes)
        valid_counts.append(len(valid))
        if valid:
            had_valid = True
        # cycle prevention: keep only switches into unvisited supports
        cands = []
        for P, Z in valid:
            newkey = frozenset((support_set - set(int(c) for c in P)) | set(int(c) for c in Z))
            if newkey not in visited:
                cands.append((P, Z, newkey))
        if not cands:
            return {"status": "stalled", "switches": switches,
                    "valid_counts": valid_counts, "had_valid": had_valid}

        P, Z, newkey = cands[int(wrng.integers(len(cands)))]
        X_int[P] -= 1                          # 1/3 -> 0
        X_int[Z] += 1                          # 0   -> 1/3
        prov[P] = -1
        prov[Z] = -1                           # all 8 cube cells -> None
        support_set = set(newkey)
        support_sorted = np.array(sorted(support_set))
        visited.add(newkey)
        switches += 1

        if verbose:
            assert len(support_set) == r, "support left size r"
            assert np.max(np.abs(A @ (X_int / 3.0) - b)) < 1e-9, "line sums broke"
            assert set(np.unique(X_int)).issubset({0, 1, 2, 3}), "values not in {0,1,2,3}"
            assert (prov[P] == -1).all() and (prov[Z] == -1).all(), "switched provenance not None"

        if is_vertex(A, A_dense, support_sorted, r):
            return {"status": "vertex", "X_int": X_int, "support": support_sorted,
                    "switches": switches, "valid_counts": valid_counts, "had_valid": had_valid}

    return {"status": "max_switches", "switches": switches,
            "valid_counts": valid_counts, "had_valid": had_valid}


# ------------------------------------------------------------
# Record building
# ------------------------------------------------------------

def make_vertex_record(n, walk_seed, walk_id, res, A, b, ttv, vtt):
    support = res["support"]
    x_flat = res["X_int"].astype(float) / 3.0
    supp_triples = [vtt[int(c)] for c in support]
    sol = {"success": True, "lp_status": 0, "lp_message": "latin_cube_switch",
           "x": x_flat.reshape((n, n, n)), "support": supp_triples,
           "objective_value": None}
    rec = polysim.build_vertex_record(n, walk_seed, sol, A, b, ttv, vtt,
                                      include_support_map=True)
    rec["radius_from_uniform"] = lw.radius_from_uniform(x_flat, n)
    rec["move_strategy"] = "latin_square_cube_switch"
    rec["first_walk"] = walk_id
    rec["num_cube_switches"] = res["switches"]
    rec["initial_support_size"] = len(support)
    rec["times_reached"] = 1
    return lw.reorder_record(rec)


# ------------------------------------------------------------
# Main run
# ------------------------------------------------------------

def run(args) -> None:
    n = args.n
    A, b, ttv, vtt = polysim.build_tristochastic_constraints(n)
    r = polysim.constraint_rank_expected(n)
    A_dense = A.toarray()
    cubes = build_cubes(n)
    print(f"latin-cube switch search  (n={n}, r={r}, {cubes.shape[0]} cubes, "
          f"target {args.walks} admitted walks)\n")

    vertices_by_key, _ = lw.load_existing(args.out_vertices)
    master = np.random.default_rng(args.seed)
    stats_f = open(args.out_stats, "a", encoding="utf-8")

    triples = admitted = already_vertex = had_valid_walks = stalled = 0
    walks_vertex = new_saved = dup_reached = 0
    init_hist: Counter = Counter()
    switches_list: List[int] = []
    max_triples = args.max_triples if args.max_triples is not None else args.walks * 500
    t0 = time.time()

    try:
        while admitted < args.walks and triples < max_triples:
            triples += 1
            walk_seed = int(master.integers(0, 2 ** 31 - 1))
            wrng = np.random.default_rng(walk_seed)

            L = [random_latin_square(n, wrng) for _ in range(3)]
            P = [perm_tensor_int(Lt, n) for Lt in L]
            X_int = P[0] + P[1] + P[2]         # values in {0,1,2,3}
            support_set = set(int(c) for c in np.nonzero(X_int)[0])
            init_hist[len(support_set)] += 1

            if len(support_set) != r:          # reject, sample again
                continue
            admitted += 1

            # provenance: a value-1 cell owned by its single Latin square, else None
            prov = np.full(n ** 3, -1, dtype=np.int64)
            for t in range(3):
                prov[(X_int == 1) & (P[t] == 1)] = t

            res = cube_walk(X_int, prov, support_set, A, A_dense, b, r, cubes, wrng,
                            args.max_switches_per_walk, verbose=args.verbose)
            if res["had_valid"]:
                had_valid_walks += 1
            found = res["status"] in ("vertex", "vertex_at_start")
            simple = vsize = radius = None

            if found:
                walks_vertex += 1
                switches_list.append(res["switches"])
                vsize = int(len(res["support"]))
                simple = (vsize == r)
                key = frozenset(vtt[int(c)] for c in res["support"])
                if key in vertices_by_key:
                    vertices_by_key[key]["times_reached"] += 1
                    dup_reached += 1
                else:
                    rec = make_vertex_record(n, walk_seed, walk_id=admitted, res=res,
                                             A=A, b=b, ttv=ttv, vtt=vtt)
                    vertices_by_key[key] = rec
                    new_saved += 1
                    radius = rec["radius_from_uniform"]
                lw.write_all(args.out_vertices, vertices_by_key)
                if radius is None:
                    radius = vertices_by_key[key]["radius_from_uniform"]
            if res["status"] == "vertex_at_start":
                already_vertex += 1
            elif res["status"] == "stalled":
                stalled += 1

            vc = res["valid_counts"]
            row = {
                "walk_id": admitted, "walk_seed": walk_seed, "n": n, "target_support": r,
                "initial_support_size": r, "already_vertex": res["status"] == "vertex_at_start",
                "num_cube_switches": res["switches"], "stop_reason": res["status"],
                "vertex_found": found, "is_simple": simple, "vertex_support_size": vsize,
                "radius_from_uniform": radius,
                "valid_cubes_mean": round(float(np.mean(vc)), 2) if vc else None,
                "valid_cubes_max": int(np.max(vc)) if vc else None,
            }
            stats_f.write(json.dumps(lw.polysim._jsonable(row)) + "\n")
            stats_f.flush()
            print(f"  walk {admitted} (triple {triples}): {res['status']}"
                  f"{f'  -> vertex size {vsize}/{r}' + (' SIMPLE' if simple else '') if found else ''}"
                  f"  switches={res['switches']}")
    finally:
        stats_f.close()

    lw.write_all(args.out_vertices, vertices_by_key)

    pct = 100 * admitted / triples if triples else 0.0
    summary = {
        "summary": True, "n": n, "target_support": r,
        "triples_sampled": triples, "support_r_count": admitted,
        "support_r_percentage": round(pct, 3),
        "initial_support_histogram": dict(sorted(init_hist.items())),
        "admitted_walks": admitted, "already_vertex": already_vertex,
        "walks_with_valid_cube": had_valid_walks, "stalled_walks": stalled,
        "walks_ending_at_vertex": walks_vertex,
        "distinct_vertices": len(vertices_by_key), "new_saved": new_saved,
        "duplicate_reaches": dup_reached,
        "mean_switches_to_vertex": round(float(np.mean(switches_list)), 2) if switches_list else None,
        "wall_time_sec": round(time.time() - t0, 1),
    }
    with open(args.out_stats, "a", encoding="utf-8") as f:
        f.write(json.dumps(lw.polysim._jsonable(summary)) + "\n")

    print("\n" + "=" * 62)
    print(f"LATIN-CUBE SWITCH SUMMARY   (n={n}, r={r})")
    print("=" * 62)
    print(f"  triples sampled            : {triples}")
    print(f"  support == r               : {admitted}  ({pct:.2f}%)")
    print(f"  init support histogram     : {dict(sorted(init_hist.items()))}")
    print(f"  admitted already vertices  : {already_vertex}")
    print(f"  walks with a valid cube    : {had_valid_walks}")
    print(f"  stalled walks              : {stalled}")
    print(f"  walks ending at a vertex   : {walks_vertex}")
    print(f"  distinct vertices          : {len(vertices_by_key)}  (new {new_saved}, dup {dup_reached})")
    if switches_list:
        print(f"  switches to vertex (mean)  : {np.mean(switches_list):.2f}")
    print(f"  vertex file / stats file   : {args.out_vertices} / {args.out_stats}")
    print(f"  wall time                  : {time.time() - t0:.1f}s")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="3-Latin-square + provenance cube-switch vertex search.")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--walks", type=int, default=100, help="target admitted walks (support == r)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-switches-per-walk", type=int, default=500)
    p.add_argument("--max-triples", type=int, default=None,
                   help="cap on total triples sampled (default walks*500)")
    p.add_argument("--tolerance", type=float, default=1e-9)
    p.add_argument("--out-vertices", type=str, default=None)
    p.add_argument("--out-stats", type=str, default=None)
    p.add_argument("--verbose", action="store_true",
                   help="assert all invariants after every cube switch")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.out_vertices is None:
        args.out_vertices = f"latin_cube_vertices_n{args.n}.jsonl"
    if args.out_stats is None:
        args.out_stats = f"latin_cube_stats_n{args.n}.jsonl"
    run(args)


if __name__ == "__main__":
    main()
