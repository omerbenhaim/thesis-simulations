"""
Stochastic "descend-to-a-vertex" walk for the n x n x n tristochastic polytope.

Idea
----
Start at the uniform tensor U_ijk = 1/n (full support, deep interior).  Repeat:

    S   = supp(X)                                  current support
    D   in ker(A_S)   (A D = 0, D_a = 0 for a not in S)   support-preserving,
                                                   line-sum-preserving direction
    t*  = min_{D_a < 0}  X_a / (-D_a)              first coordinate to hit 0
    X  <- X + t* D                                 land on a lower face

Each step zeroes at least one positive coordinate, so |supp(X)| strictly
shrinks.  It stops when ker(A_S) = {0} -- which is *exactly* the vertex
condition (the support determines a unique point).  So every walk terminates
at a vertex; different random directions reach different vertices.

The direction is chosen by an LP (RandomLPNullspaceDirectionStrategy):

    maximize  c^T D        (c random gaussian on the support)
    s.t.      A_S D = 0
              -1 <= D_a <= 1                       box only fixes the free scale

Because A D = 0 forces every line of D to sum to 0, any nonzero D has both
signs, so a boundary is always hit; t* is well defined and > 0.

Vertexhood uses the project's exact rank/nullity test (nullity(A_S) == 0).
Every discovered vertex is serialized with the SAME record as
`polytope simulation.py` (via build_vertex_record), plus:
    radius_from_uniform = ||X - U||_F
and light walk metadata (walk_id, steps, trajectory, ...).

CLI
---
    python tristochastic_lp_walk.py --n 5 --target 100 --out lp_walk_n5.jsonl --seed 1
"""

import argparse
import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

# ------------------------------------------------------------
# Reuse the existing project (filename has a space -> load by path)
# ------------------------------------------------------------
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "polysim", os.path.join(_HERE, "polytope simulation.py"))
polysim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(polysim)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def uniform_tensor_flat(n: int) -> np.ndarray:
    """The center U_ijk = 1/n as a flat length-n^3 vector."""
    return np.full(n ** 3, 1.0 / n)


def radius_from_uniform(x_flat: np.ndarray, n: int) -> float:
    """Frobenius distance ||X - U||_F from the uniform tensor."""
    return float(np.sqrt(np.sum((x_flat - 1.0 / n) ** 2)))


def full_column_rank_modp(A, support, p: int = 2_147_483_647) -> bool:
    """
    EXACT vertex certificate: are the support columns of the integer 0/1 matrix
    A linearly independent (full column rank, i.e. nullity(A_S) == 0)?

    Gaussian elimination over F_p (p = 2^31 - 1).  Full column rank mod p implies
    full column rank over Q -- a rational dependence clears to a primitive
    integer one, which stays nonzero mod p -- so a True here certifies the vertex
    with no floating-point tolerance.  (A False can only be a ~1/p false negative,
    which merely restarts the walk; it never lets a non-vertex through.)
    """
    M = (A[:, support].toarray().astype(np.int64)) % p
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        if r == rows:
            return False                                  # out of rows -> dependent
        nz = np.nonzero(M[r:, c])[0]
        if nz.size == 0:
            return False                                  # column c has no pivot
        piv = r + int(nz[0])
        if piv != r:
            M[[r, piv]] = M[[piv, r]]
        M[r] = (M[r] * pow(int(M[r, c]), p - 2, p)) % p     # normalize pivot to 1
        col = M[:, c].copy()
        col[r] = 0
        if col.any():
            M = (M - np.outer(col, M[r])) % p               # clear column c elsewhere
        r += 1
    return True                                           # every column pivoted


# ------------------------------------------------------------
# Direction strategies
# ------------------------------------------------------------

class DirectionStrategy:
    """Return a nonzero D in ker(A_S) (length n^3, zero off the support), or None."""
    name = "base"

    def direction(self, x_flat, support, A, rng, tol) -> Tuple[Optional[np.ndarray], int]:
        raise NotImplementedError


class RandomLPNullspaceDirectionStrategy(DirectionStrategy):
    """
    Pick a direction by LP: max c^T D s.t. A_S D = 0, -1 <= D <= 1, random c.
    Returns a "bang-bang" extreme direction of the box intersect ker(A_S).
    Resamples c a few times in the (measure-zero) case c is orthogonal to ker.
    """
    name = "general_lp_boundary"

    def __init__(self, max_tries: int = 8):
        self.max_tries = max_tries

    def direction(self, x_flat, support, A, rng, tol):
        m = len(support)
        A_S = A[:, support]
        rhs = np.zeros(A.shape[0])
        last_status = -1
        for _ in range(self.max_tries):
            c = rng.normal(size=m)
            res = linprog(-c, A_eq=A_S, b_eq=rhs, bounds=(-1.0, 1.0), method="highs")
            last_status = int(res.status)
            if not res.success:
                continue
            d_s = np.asarray(res.x, dtype=float)
            if np.max(np.abs(d_s)) > tol:                 # nonzero direction found
                D = np.zeros(A.shape[1])
                D[support] = d_s
                return D, last_status
        return None, last_status                          # only 0 found (ker empty or unlucky)


# ------------------------------------------------------------
# Walk engine
# ------------------------------------------------------------

def run_walk(n: int, A, b, s_max: int, strategy: DirectionStrategy,
             rng, max_steps: int, tol: float, verbose: bool = False) -> Dict[str, Any]:
    """
    One walk from U down to a vertex.  Returns a dict with a `status`:
      "vertex"          -> reached a vertex (X, support, steps, trajectory)
      "lp_fail" / "zero_direction" / "no_boundary" / "no_shrink" /
      "line_sum_drift" / "max_steps"  -> restart reasons.
    """
    x = uniform_tensor_flat(n)
    init_support = int(np.sum(x > tol))
    traj = [init_support]

    for step in range(1, max_steps + 1):
        support = np.nonzero(x > tol)[0]
        m = len(support)

        # Ask the direction LP first: it returns None exactly when there is no
        # support-preserving direction, i.e. ker(A_S) = {0}.  So we detect the
        # vertex from the LP and pay the exact nullity SVD only ONCE (to
        # certify), instead of every step -- crucial for large n.
        D, lp_status = strategy.direction(x, support, A, rng, tol)
        if D is None:
            if m <= s_max and full_column_rank_modp(A, support):   # exact certify
                return {"status": "vertex", "x": x, "support": support,
                        "steps": step - 1, "trajectory": traj, "init": init_support}
            return {"status": "zero_direction", "steps": step - 1,
                    "trajectory": traj, "lp_status": lp_status}

        # (b) the direction must genuinely lie in ker(A_S): A D = 0
        if np.max(np.abs(A @ D)) > 1e-6:
            return {"status": "bad_direction", "steps": step - 1, "trajectory": traj}

        neg = D < -tol
        if not neg.any():                                 # flip orientation if needed
            D = -D
            neg = D < -tol
        if not neg.any():
            return {"status": "no_boundary", "steps": step - 1, "trajectory": traj}

        # First coordinate to reach zero.
        t_max = float(np.min(x[neg] / (-D[neg])))
        x = x + t_max * D

        # (a) a genuinely negative coordinate means the move overshot -> surface it
        # (don't let the clamp silently erase it); only then clamp near-zero dust.
        mn = float(x.min())
        if mn < -1e-7:
            return {"status": "negative_overshoot", "steps": step,
                    "trajectory": traj, "min_x": mn}
        x[x < tol] = 0.0

        drift = float(np.max(np.abs(A @ x - b)))
        if drift > 1e-6:
            return {"status": "line_sum_drift", "steps": step,
                    "trajectory": traj, "drift": drift}

        new_m = int(np.sum(x > tol))
        if new_m >= m:                                    # must strictly shrink
            return {"status": "no_shrink", "steps": step, "trajectory": traj}
        traj.append(new_m)
        if verbose:
            print(f"      step {step}: |S| {m} -> {new_m}  t*={t_max:.3g}")

    return {"status": "max_steps", "steps": max_steps, "trajectory": traj}


# ------------------------------------------------------------
# Record building + dedup
# ------------------------------------------------------------

# Fields pulled to the front of every record for readability.
_FRONT_FIELDS = ("seed", "times_reached", "support_size", "radius_from_uniform", "distinct_values")


def reorder_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Return rec with the key fields first, then everything else in place."""
    front = {k: rec[k] for k in _FRONT_FIELDS if k in rec}
    rest = {k: v for k, v in rec.items() if k not in front}
    return {**front, **rest}


def make_vertex_record(n, walk_seed, walk_id, res, A, b, ttv, vtt) -> Dict[str, Any]:
    """Full project record (build_vertex_record) + radius + walk metadata."""
    support = res["support"]
    x_flat = res["x"]
    supp_triples = [vtt[int(a)] for a in support]
    sol = {"success": True, "lp_status": 0, "lp_message": "lp_walk",
           "x": x_flat.reshape((n, n, n)), "support": supp_triples,
           "objective_value": None}

    rec = polysim.build_vertex_record(n, walk_seed, sol, A, b, ttv, vtt,
                                      include_support_map=True)  # exact ratios inside
    rec["radius_from_uniform"] = radius_from_uniform(x_flat, n)
    rec["walk_id"] = walk_id
    rec["walk_steps"] = res["steps"]
    rec["initial_support_size"] = res["init"]
    rec["final_support_size"] = len(support)
    rec["move_strategy"] = RandomLPNullspaceDirectionStrategy.name
    rec["support_size_trajectory"] = res["trajectory"]
    rec["times_reached"] = 1
    return reorder_record(rec)


def support_key(rec_or_triples) -> frozenset:
    """Canonical dedup key = frozenset of support triples (matches enrich)."""
    return frozenset(tuple(t) for t in rec_or_triples)


def write_all(path: str, vertices_by_key) -> None:
    """Rewrite the whole file (updated counters + column order). Called after
    every vertex so a long run is crash-safe."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in vertices_by_key.values():
            f.write(json.dumps(polysim._jsonable(reorder_record(rec))) + "\n")


def load_existing(path: str):
    """
    Load existing vertex records into an ordered {support_key: record} map and
    collect the walk seeds already used.  Re-runs then update a `times_reached`
    counter (instead of appending duplicates) and never reuse a walk seed.
    """
    by_key: Dict[frozenset, dict] = {}
    used_seeds: set = set()
    if not os.path.exists(path):
        return by_key, used_seeds
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            Q = rec.get("support_map")
            if Q is None:
                continue
            key = frozenset((i, j, k)
                            for i, row in enumerate(Q)
                            for j, ks in enumerate(row) for k in ks)
            rec.setdefault("times_reached", 1)
            by_key[key] = rec
            if rec.get("seed") is not None:
                used_seeds.add(int(rec["seed"]))
    return by_key, used_seeds


# ------------------------------------------------------------
# Main run
# ------------------------------------------------------------

def run(args) -> None:
    A, b, ttv, vtt = polysim.build_tristochastic_constraints(args.n)
    s_max = polysim.constraint_rank_expected(args.n)
    strategy = RandomLPNullspaceDirectionStrategy()

    # sanity: the start point is feasible and at radius 0
    U = uniform_tensor_flat(args.n)
    assert np.max(np.abs(A @ U - b)) < 1e-9, "uniform tensor is not feasible!"
    assert radius_from_uniform(U, args.n) == 0.0

    vertices_by_key, used_seeds = load_existing(args.out)
    n_existing = len(vertices_by_key)
    max_walks = args.max_walks if args.max_walks is not None else args.target * 50
    stats: Counter = Counter()
    new_saved = revisits = walk_id = 0
    radii: List[float] = []
    master = np.random.default_rng(args.seed)
    t0 = time.time()

    budget = f"{args.walks} walks" if args.walks is not None else f"target {args.target} new"
    print(f"LP-walk to vertices  (n={args.n}, {budget}, S_max={s_max}, "
          f"start=uniform, {n_existing} already known)\n")

    while True:
        if args.walks is not None:
            if walk_id >= args.walks:
                break
        elif new_saved >= args.target or walk_id >= max_walks:
            break
        walk_id += 1

        # draw a walk seed not used before (this run or earlier runs) -> distinct walks
        while True:
            walk_seed = int(master.integers(0, 2 ** 31 - 1))
            if walk_seed not in used_seeds:
                used_seeds.add(walk_seed)
                break

        res = run_walk(args.n, A, b, s_max, strategy,
                       np.random.default_rng(walk_seed),
                       args.max_steps_per_walk, args.tolerance, args.verbose)
        stats[res["status"]] += 1
        if res["status"] != "vertex":
            continue

        key = support_key(vtt[int(a)] for a in res["support"])
        if key in vertices_by_key:                       # already found -> bump counter
            existing = vertices_by_key[key]
            existing["times_reached"] = existing.get("times_reached", 1) + 1
            revisits += 1
            print(f"  walk {walk_id}: revisit support {existing['support_size']}/{s_max} "
                  f"(times_reached={existing['times_reached']})")
        else:                                            # new vertex
            rec = make_vertex_record(args.n, walk_seed, walk_id, res, A, b, ttv, vtt)
            vertices_by_key[key] = rec
            new_saved += 1
            radii.append(rec["radius_from_uniform"])
            print(f"  [{new_saved}] walk {walk_id}: NEW support {res['init']} -> "
                  f"{rec['support_size']}/{s_max} in {rec['walk_steps']} steps  "
                  f"{'SIMPLE ' if rec['is_simple_vertex'] else ''}"
                  f"{'int' if rec['is_integral'] else ('half' if rec['is_half_integral'] else 'frac')} "
                  f"radius={rec['radius_from_uniform']:.4f}")

        write_all(args.out, vertices_by_key)         # crash-safe: persist after each vertex

    write_all(args.out, vertices_by_key)

    _report(args, s_max, stats, walk_id, new_saved, revisits,
            n_existing, len(vertices_by_key), radii, time.time() - t0)


def _report(args, s_max, stats, walk_id, new_saved, revisits,
            n_existing, total, radii, elapsed) -> None:
    print("\n" + "=" * 60)
    print(f"LP-WALK SUMMARY   (n={args.n})")
    print("=" * 60)
    print(f"  walks run this session    : {walk_id}")
    print(f"  new distinct vertices     : {new_saved}")
    print(f"  revisits (already known)  : {revisits}")
    print(f"  walk outcomes             : {dict(sorted(stats.items()))}")
    print(f"  total distinct in file    : {total}  (was {n_existing})  -> {args.out}")
    if radii:
        print(f"  radius of new vertices    : mean {np.mean(radii):.4f}, "
              f"min {min(radii):.4f}, max {max(radii):.4f}")
    print(f"  wall time                 : {elapsed:.1f}s")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stochastic descend-to-a-vertex walk on the tristochastic polytope.")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--target", type=int, default=100,
                   help="distinct vertices to collect (used only if --walks not set)")
    p.add_argument("--walks", type=int, default=None,
                   help="run exactly this many walks (overrides --target)")
    p.add_argument("--out", type=str, required=True,
                   help="output JSONL (rewritten each run with updated times_reached)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps-per-walk", type=int, default=None,
                   help="step cap per walk (default n^3)")
    p.add_argument("--max-walks", type=int, default=None,
                   help="cap on total walks (default target*50)")
    p.add_argument("--tolerance", type=float, default=1e-9,
                   help="support / clamp tolerance")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_steps_per_walk is None:
        args.max_steps_per_walk = args.n ** 3
    run(args)


if __name__ == "__main__":
    main()
