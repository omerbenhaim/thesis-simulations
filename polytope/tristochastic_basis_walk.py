"""
Random BASIS-EXCHANGE walk on the tristochastic incidence matrix A.

This is a SEPARATE experiment from the support-shrinking LP walk
(tristochastic_lp_walk.py) -- that file is not touched.  Its own result files.

Setup
-----
A has rank r = 3n^2 - 3n + 1.  A *basis* B is a set of exactly r columns of A
that are linearly independent; they span col(A), and 1 in col(A), so

    A_B x = 1

has a unique solution.  We random-walk through bases and measure how often that
unique solution is strictly positive (a full/simple vertex of the polytope).

Start
-----
Each walk starts from a random full/simple vertex taken from the EXISTING
LP-walk results `lp_walk_n{n}.jsonl` (support_size == r and is_simple_vertex).
If no such vertex exists, the script prints an instruction and exits cleanly.

One exchange step  (B -> B')
----------------------------
  1. pick an entering column e not in B (uniform);
  2. represent A_e = A_B alpha  (alpha unique);
  3. eligible leaving columns = { b_i in B : alpha_i != 0 } (the fundamental
     circuit), using a numerical tolerance -- not literal != 0;
  4. pick a leaving column f uniformly among those; B' = (B u {e}) \ {f};
  5. verify rank(A_B') = r (reject/resample if not); solve A_B' x = 1;
  6. classify x as positive / zero-containing / negative (careful tolerance,
     residual-checked); the walk CONTINUES from B' regardless.

Statistics start AFTER the first exchange -- the known starting basis is not
counted and (per project choice) not re-saved, since it already lives in the
LP-walk JSON.  Only NEW strictly-positive vertices (not already in the LP-walk
file) are written to the vertex file, with a `times_reached` counter.

CLI
---
    python tristochastic_basis_walk.py --n 10 --walks 100 --steps-per-walk 1000 \
        --seed 1 --out-vertices basis_walk_vertices_n10.jsonl \
        --out-stats basis_walk_stats_n10.jsonl
"""

import argparse
import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.linalg import solve_triangular

# ------------------------------------------------------------
# Reuse the existing project (both files loaded by path; names have no space)
# ------------------------------------------------------------
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "lpwalk", os.path.join(_HERE, "tristochastic_lp_walk.py"))
lw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lw)
polysim = lw.polysim                                     # shared project helpers


# ------------------------------------------------------------
# Basis solve (QR) + classification
# ------------------------------------------------------------

def qr_of(A_dense, cols):
    """Reduced QR of A restricted to `cols`."""
    return np.linalg.qr(A_dense[:, cols], mode="reduced")


def is_full_rank(R, rank_tol: float) -> bool:
    """Full column rank iff no tiny pivot on R's diagonal."""
    d = np.abs(np.diag(R))
    return d.size == R.shape[0] and float(d.min()) > rank_tol


def solve_qr(Q, R, rhs):
    """Solve (QR) z = rhs for the least-squares/consistent solution."""
    return solve_triangular(R, Q.T @ rhs, lower=False)


def load_simple_starts(path: str, r: int, triple_to_var):
    """
    From an LP-walk JSONL, return
      starts     : list of (basis_columns, support_key, record) for full/simple
                   vertices (support_size == r and is_simple_vertex);
      known_keys : set of support keys of ALL vertices in the file (these are
                   already saved there, so the basis walk will not re-save them).
    """
    starts: List[Tuple[List[int], frozenset, dict]] = []
    known_keys: set = set()
    if not os.path.exists(path):
        return starts, known_keys
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            Q = rec.get("support_map")
            if Q is None:
                continue
            triples = [(i, j, k)
                       for i, row in enumerate(Q)
                       for j, ks in enumerate(row) for k in ks]
            key = frozenset(triples)
            known_keys.add(key)
            if rec.get("support_size") == r and rec.get("is_simple_vertex"):
                starts.append(([triple_to_var[t] for t in triples], key, rec))
    return starts, known_keys


def load_stats_history(path: str):
    """
    Used walk seeds and the largest walk_id from an existing stats file, so a
    rerun (even with the same --seed) continues with fresh seeds / ids instead
    of repeating identical walks and appending duplicate stats.
    """
    used_seeds: set = set()
    last_id = 0
    if not os.path.exists(path):
        return used_seeds, last_id
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("walk_seed") is not None:
                used_seeds.add(int(row["walk_seed"]))
            last_id = max(last_id, int(row.get("walk_id", 0)))
    return used_seeds, last_id


def make_basis_vertex_record(n, walk_seed, walk_id, step, cols, x,
                             A, b, triple_to_var, var_to_triple):
    """Full project vertex record (build_vertex_record) + basis-walk metadata."""
    x_flat = np.zeros(n ** 3)
    x_flat[cols] = x
    supp_triples = [var_to_triple[c] for c in cols]
    sol = {"success": True, "lp_status": 0, "lp_message": "basis_walk",
           "x": x_flat.reshape((n, n, n)), "support": supp_triples,
           "objective_value": None}
    rec = polysim.build_vertex_record(n, walk_seed, sol, A, b,
                                      triple_to_var, var_to_triple,
                                      include_support_map=True)
    rec["move_strategy"] = "random_basis_exchange"
    rec["first_walk"] = walk_id
    rec["first_step"] = step
    rec["times_reached"] = 1
    return lw.reorder_record(rec)


# ------------------------------------------------------------
# Main run
# ------------------------------------------------------------

def run(args) -> None:
    A, b, ttv, vtt = polysim.build_tristochastic_constraints(args.n)
    r = polysim.constraint_rank_expected(args.n)
    A_dense = A.toarray()
    N = args.n ** 3
    tol = args.tolerance
    alpha_tol, rank_tol, resid_tol = 1e-8, 1e-8, 1e-6

    lp_file = args.lp_walk_file or f"lp_walk_n{args.n}.jsonl"
    starts, known_keys = load_simple_starts(lp_file, r, ttv)
    if not starts:
        print(f"No full/simple vertex available for n={args.n}. First run the LP "
              f"support-shrinking walk for this n until at least one vertex with "
              f"support {r} = 3n^2-3n+1 is found (e.g. "
              f"python tristochastic_lp_walk.py --n {args.n} --walks 50 "
              f"--out lp_walk_n{args.n}.jsonl).")
        return

    # basis-walk vertices persist across runs (times_reached accumulates)
    vertices_by_key, _ = lw.load_existing(args.out_vertices)
    used_walk_seeds, last_walk_id = load_stats_history(args.out_stats)
    master = np.random.default_rng(args.seed)
    agg = Counter()
    total_new = 0
    t0 = time.time()

    print(f"basis-exchange walk  (n={args.n}, r={r}, {len(starts)} simple starts "
          f"in {lp_file}, {args.walks} walks x {args.steps_per_walk} steps)\n")

    stats_f = open(args.out_stats, "a", encoding="utf-8")
    try:
        for _w in range(args.walks):
            last_walk_id += 1
            walk_id = last_walk_id
            while True:                                  # a fresh, previously-unused seed
                walk_seed = int(master.integers(0, 2 ** 31 - 1))
                if walk_seed not in used_walk_seeds:
                    used_walk_seeds.add(walk_seed)
                    break
            wrng = np.random.default_rng(walk_seed)

            # random full/simple starting basis
            B, start_key, start_rec = starts[int(wrng.integers(len(starts)))]
            B = list(B)
            Bset = set(B)
            Q, R = qr_of(A_dense, B)
            # sanity on the start: basis + strictly positive (not counted/saved)
            if not is_full_rank(R, rank_tol):
                print(f"  walk {walk_id}: start not full rank -- skipping"); continue
            x0 = solve_qr(Q, R, b)
            assert float(np.max(np.abs(A_dense[:, B] @ x0 - b))) < resid_tol
            assert x0.min() > tol, "start vertex not strictly positive"

            counts = Counter()
            minx_list: List[float] = []
            neg_coords: List[int] = []
            walk_positive_keys: set = set()             # distinct positive supports THIS walk
            pos_new = steps_done = 0

            for step in range(1, args.steps_per_walk + 1):
                moved = False
                for _attempt in range(args.max_retries):
                    e = int(wrng.integers(N))
                    while e in Bset:
                        e = int(wrng.integers(N))
                    alpha = solve_qr(Q, R, A_dense[:, e])       # A_e = A_B alpha
                    if np.max(np.abs(A_dense[:, B] @ alpha - A_dense[:, e])) > resid_tol:
                        continue                                # unreliable rep -> resample e
                    eligible = np.nonzero(np.abs(alpha) > alpha_tol)[0]
                    if eligible.size == 0:
                        continue
                    fpos = int(eligible[int(wrng.integers(eligible.size))])
                    Bp = list(B)
                    Bp[fpos] = e
                    Qp, Rp = qr_of(A_dense, Bp)
                    if not is_full_rank(Rp, rank_tol):          # not a basis -> resample
                        continue
                    x = solve_qr(Qp, Rp, b)
                    resid = float(np.max(np.abs(A_dense[:, Bp] @ x - b)))
                    if resid > resid_tol:
                        counts["numerical_failure"] += 1
                        continue

                    mn = float(x.min())
                    minx_list.append(mn)
                    if mn > tol:
                        cls = "positive"
                    elif mn >= -tol:
                        cls = "zero"
                    else:
                        cls = "negative"
                        neg_coords.append(int(np.sum(x < -tol)))
                    counts[cls] += 1

                    if cls == "positive":
                        key = frozenset(vtt[c] for c in Bp)
                        walk_positive_keys.add(key)             # per-walk distinct count
                        if key in known_keys:                   # already in LP-walk JSON
                            pass
                        elif key in vertices_by_key:            # seen earlier by basis walk
                            vertices_by_key[key]["times_reached"] += 1
                            lw.write_all(args.out_vertices, vertices_by_key)
                        else:                                   # NEW simple vertex
                            vertices_by_key[key] = make_basis_vertex_record(
                                args.n, walk_seed, walk_id, step, Bp, x, A, b, ttv, vtt)
                            pos_new += 1
                            total_new += 1
                            lw.write_all(args.out_vertices, vertices_by_key)

                    B, Bset, Q, R = Bp, set(Bp), Qp, Rp         # commit the move
                    steps_done += 1
                    moved = True
                    break
                if not moved:
                    break                                       # no valid exchange -> stop

            tested = counts["positive"] + counts["zero"] + counts["negative"]
            agg.update(counts)
            row = {
                "walk_id": walk_id, "walk_seed": walk_seed, "n": args.n,
                "steps_requested": args.steps_per_walk, "steps_completed": steps_done,
                "start_support_seed": start_rec.get("seed"),
                "start_support_size": r,
                "bases_tested": tested,
                "positive": counts["positive"], "zero": counts["zero"],
                "negative": counts["negative"],
                "numerical_failures": counts["numerical_failure"],
                "positive_percentage": round(100 * counts["positive"] / tested, 3) if tested else 0.0,
                "zero_percentage": round(100 * counts["zero"] / tested, 3) if tested else 0.0,
                "negative_percentage": round(100 * counts["negative"] / tested, 3) if tested else 0.0,
                "distinct_positive_vertices": len(walk_positive_keys),
                "positive_revisits": counts["positive"] - len(walk_positive_keys),
                "new_vertices_saved": pos_new,
                "min_x_mean": round(float(np.mean(minx_list)), 6) if minx_list else None,
                "min_x_min": round(float(np.min(minx_list)), 6) if minx_list else None,
                "neg_coords_mean": round(float(np.mean(neg_coords)), 3) if neg_coords else None,
                "neg_coords_max": max(neg_coords) if neg_coords else None,
            }
            stats_f.write(json.dumps(row) + "\n")
            stats_f.flush()
            print(f"  walk {walk_id}: tested {tested}  "
                  f"pos {counts['positive']} ({row['positive_percentage']}%)  "
                  f"zero {counts['zero']}  neg {counts['negative']}  "
                  f"distinct-pos {len(walk_positive_keys)}  new-saved {pos_new}")
    finally:
        stats_f.close()

    lw.write_all(args.out_vertices, vertices_by_key)
    _aggregate_report(args, r, agg, total_new, len(vertices_by_key), time.time() - t0)


def _aggregate_report(args, r, agg, total_new, total_vertices, elapsed) -> None:
    tested = agg["positive"] + agg["zero"] + agg["negative"]
    print("\n" + "=" * 62)
    print(f"BASIS-WALK AGGREGATE   (n={args.n}, r={r})")
    print("=" * 62)
    print(f"  bases tested (all walks)  : {tested}")
    if tested:
        print(f"  positive                  : {agg['positive']}  ({100*agg['positive']/tested:.3f}%)")
        print(f"  zero-containing           : {agg['zero']}  ({100*agg['zero']/tested:.3f}%)")
        print(f"  negative                  : {agg['negative']}  ({100*agg['negative']/tested:.3f}%)")
    print(f"  numerical failures        : {agg['numerical_failure']}")
    print(f"  NEW distinct positive verts: {total_new}  -> {args.out_vertices}")
    print(f"  (total in vertex file)    : {total_vertices}")
    print(f"  per-walk stats            : {args.out_stats}")
    print(f"  wall time                 : {elapsed:.1f}s")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Random basis-exchange walk on the tristochastic incidence matrix.")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--walks", type=int, default=100)
    p.add_argument("--steps-per-walk", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-vertices", type=str, default=None,
                   help="JSONL of NEW positive vertices (default basis_walk_vertices_n{n}.jsonl)")
    p.add_argument("--out-stats", type=str, default=None,
                   help="JSONL of per-walk stats (default basis_walk_stats_n{n}.jsonl)")
    p.add_argument("--lp-walk-file", type=str, default=None,
                   help="LP-walk results to start from (default lp_walk_n{n}.jsonl)")
    p.add_argument("--tolerance", type=float, default=1e-9,
                   help="sign-classification tolerance on the basis solution")
    p.add_argument("--max-retries", type=int, default=25,
                   help="resample attempts per step before ending the walk")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.out_vertices is None:
        args.out_vertices = f"basis_walk_vertices_n{args.n}.jsonl"
    if args.out_stats is None:
        args.out_stats = f"basis_walk_stats_n{args.n}.jsonl"
    run(args)


if __name__ == "__main__":
    main()
