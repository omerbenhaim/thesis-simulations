"""
Algebraic 3-Latin-square average + EXACTLY TWO cube switches.

Standalone experiment: own output files, no changes to the random Latin-cube
experiments (`tristochastic_latin_cube_search.py`) or their JSONL results.

Question
--------
Can the deterministic algebraic average

    L1(x,y) = x + y,  L2(x,y) = a x + b y,  L3(x,y) = c x + d y   (mod n, n prime)
    X = (P1 + P2 + P3) / 3,        |supp(X)| = r = 3n^2 - 3n + 1

be turned into a FULL-SUPPORT VERTEX (|S| = r and rank(A_S) = r) by exactly two
2x2x2 cube switches?  Values are integers in units of 1/3, so entries are in
{0,1,2,3}.  Every walk stops after the second switch, successful or not.

Switch 1 — must contain the origin
    All three squares agree at (0,0,0), so X_000 = 1 (3 units).  The first switch
    must lower it: for nonzero i, j, k the cube {0,i} x {0,j} x {0,k} has
        decrease  (0,0,0), (0,j,k), (i,0,k), (i,j,0)
        increase  (0,0,k), (0,j,0), (i,0,0), (i,j,k)
    with the three non-origin decreasing corners exactly 1/3, (i,j,k) already
    positive, the other three increasing corners 0, so |S| stays r and
    X_000 : 1 -> 2/3.  NOTE: unlike the random experiment, the four decreasing
    corners are NOT all required to be 1/3 (the origin is 1), and no provenance
    rule is applied.

Switch 2 — must kill an original L1 cell
    Any cube, either checkerboard orientation, magnitude 1/3.  The four
    decreasing corners need only be POSITIVE (not all 1/3).  Requirements: the
    result stays nonnegative and tristochastic, |S| = r, and at least one cell of
    the original L1 support drops out of the support -- otherwise all of P1 is
    still supported and the point cannot be a vertex.  Candidates are generated
    from original L1 cells currently at 1/3, since one of them must vanish.

Searching the two switches
--------------------------
First switches are cheap to enumerate ((n-1)^3 cubes), so they always are.  The
second switches of a given state cost about |L1 cells at 1/3| * (n-1)^3 to
enumerate (~1 s at n=11): while that is under --enumerate-max-ops they are
enumerated too, the walks of a construction are then DISTINCT (first, second)
pairs drawn without replacement, and "no admissible second switch" is a PROVEN
statement.  Only above that budget does the second switch fall back to sampling,
and a sampled miss is reported as `second_switch_budget_exhausted` -- never as
"none exists" (at n=11 a 500-proposal sampler missed in 4 of 60 tries although
454 valid second switches existed).

Each walk draws its `walk_seed` BEFORE making any choice and every choice the
walk makes comes from `default_rng(walk_seed)`, so a recorded walk is replayable;
the switch cells are recorded as well.

Rank reporting (honest about what is proven)
--------------------------------------------
`rank` is exact iff `rank_exact`; otherwise it is a certified LOWER bound (max
rank over two large primes, and rank mod p <= rank over Q) and
`nullity_upper_bound` is correspondingly only an upper bound -- the field names
and the `*_exact_value` fields (null when unproven) keep the two apart, and the
summary reports exact and bounded nullity distributions separately.
`nullity_lower_bound` counts kernel vectors found among 2x2x2 cube directions
fully inside the support -- those are exact integer kernel vectors, and their
independence is checked mod p, which implies independence over Q.  When the
bounds meet, the nullity is PROVEN; for |S| <= --exact-rank-max-cols an exact
Bareiss elimination closes the remaining cases.  A full-rank success is always
exact (full column rank mod p certifies it over Q, re-verified via
`full_column_rank_modp`).

CLI
---
    python tristochastic_algebraic_two_cube_search.py --regression
    python tristochastic_algebraic_two_cube_search.py --n-values 5 7 11
    python tristochastic_algebraic_two_cube_search.py --n-values 5 --exhaustive
"""

import argparse
import importlib.util
import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ------------------------------------------------------------
# Reuse the project
# ------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dlst = _load("dlst", "deterministic_latin_support_test.py")     # algebraic squares
lcs = _load("lcs", "tristochastic_latin_cube_search.py")        # cube geometry, float_rank
lw = lcs.lw                                                      # full_column_rank_modp, io
polysim = lcs.polysim                                            # constraints, vertex records

_PRIMES = (2_147_483_647, 2_147_483_629)


# ------------------------------------------------------------
# Rank / nullity with certified bounds
# ------------------------------------------------------------

def rank_modp(M: np.ndarray, p: int) -> int:
    """Rank of an integer matrix over F_p (<= rank over Q, so a lower bound)."""
    M = M.astype(np.int64) % p
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        if r == rows:
            break
        nz = np.nonzero(M[r:, c])[0]
        if nz.size == 0:
            continue
        piv = r + int(nz[0])
        if piv != r:
            M[[r, piv]] = M[[piv, r]]
        M[r] = (M[r] * pow(int(M[r, c]), p - 2, p)) % p
        col = M[:, c].copy()
        col[r] = 0
        if col.any():
            M = (M - np.outer(col, M[r])) % p
        r += 1
    return r


def cube_kernel_nullity(support_sorted: np.ndarray, n: int, cubes: np.ndarray,
                        need: int, p: int = _PRIMES[0]) -> int:
    """
    Certified nullity LOWER bound: how many independent 2x2x2 cube directions lie
    inside the support.  Each is an exact integer kernel vector of A_S (+1 on one
    checkerboard class, -1 on the other); independence is tested mod p, which
    implies independence over Q.  Stops early at `need`.
    """
    if need <= 0 or cubes is None or cubes.size == 0:
        return 0
    mem = np.zeros(n ** 3, dtype=bool)
    mem[support_sorted] = True
    inside = np.nonzero(mem[cubes].all(1))[0]
    if inside.size == 0:
        return 0
    pos = np.full(n ** 3, -1, dtype=np.int64)
    pos[support_sorted] = np.arange(support_sorted.size)

    basis: List[Tuple[int, np.ndarray]] = []            # (pivot index, reduced vector)
    for ci in inside:
        corners = cubes[ci]
        v = np.zeros(support_sorted.size, dtype=np.int64)
        v[pos[corners[lcs._EVEN_POS]]] = 1
        v[pos[corners[lcs._ODD_POS]]] = p - 1
        for piv, bv in basis:
            if v[piv]:
                v = (v - v[piv] * bv) % p
        nz = np.nonzero(v)[0]
        if nz.size:
            piv = int(nz[0])
            basis.append((piv, (v * pow(int(v[piv]), p - 2, p)) % p))
            if len(basis) >= need:
                break
    return len(basis)


def _rank_fraction(rows: List[List[int]]) -> int:
    """Exact rank over Q with Fractions (slow fallback, used only if Bareiss trips)."""
    from fractions import Fraction
    M = [[Fraction(v) for v in row] for row in rows]
    nrows, ncols = len(M), len(M[0])
    rank = 0
    for c in range(ncols):
        piv = next((i for i in range(rank, nrows) if M[i][c]), None)
        if piv is None:
            continue
        M[rank], M[piv] = M[piv], M[rank]
        pr = M[rank]
        for i in range(rank + 1, nrows):
            if M[i][c]:
                f = M[i][c] / pr[c]
                M[i] = [v - f * w for v, w in zip(M[i], pr)]
        rank += 1
    return rank


def exact_rank_int(rows: List[List[int]]) -> int:
    """
    EXACT rank over Q by fraction-free (Bareiss) elimination in Python integers.
    Every division is checked to be exact, so a wrong answer cannot slip through;
    if the invariant ever fails we fall back to Fraction elimination.
    """
    M = [row[:] for row in rows]
    nrows, ncols = len(M), len(M[0])
    prev, rank = 1, 0
    try:
        for c in range(ncols):
            piv = next((i for i in range(rank, nrows) if M[i][c]), None)
            if piv is None:
                continue
            if piv != rank:
                M[rank], M[piv] = M[piv], M[rank]
            pr = M[rank]
            pc = pr[c]
            for i in range(rank + 1, nrows):
                Mi = M[i]
                f = Mi[c]
                if f:
                    for j in range(c + 1, ncols):
                        q, rem = divmod(pc * Mi[j] - f * pr[j], prev)
                        if rem:
                            raise ArithmeticError("Bareiss division not exact")
                        Mi[j] = q
                    Mi[c] = 0
                elif prev != 1:
                    for j in range(c + 1, ncols):
                        q, rem = divmod(pc * Mi[j], prev)
                        if rem:
                            raise ArithmeticError("Bareiss division not exact")
                        Mi[j] = q
            prev = pc
            rank += 1
        return rank
    except ArithmeticError:
        return _rank_fraction(rows)


def state_rank(A_csc, support_sorted: np.ndarray, n: int, cubes: np.ndarray,
               cache: Dict[bytes, Dict[str, Any]], exact_max_cols: int = 150
               ) -> Dict[str, Any]:
    """
    Rank/nullity of A_S, cached by support.  `rank` is certified: full rank mod p
    proves full rank over Q, and otherwise rank mod p is a lower bound that the
    cube-kernel count (or, for small supports, exact Bareiss elimination) closes
    into an exact value.  `rank_exact` says which of the two it is.
    """
    key = support_sorted.tobytes()
    hit = cache.get(key)
    if hit is not None:
        return hit

    m = int(support_sorted.size)
    dense = A_csc[:, support_sorted].toarray()
    rank = rank_modp(dense, _PRIMES[0])
    if rank < m:                                        # second prime can only help
        rank = max(rank, rank_modp(dense, _PRIMES[1]))
    nullity_upper = m - rank
    nullity_lower = cube_kernel_nullity(support_sorted, n, cubes, nullity_upper)
    method = "full_rank_modp_certificate" if nullity_upper == 0 else (
        "modp_lower_bound + cube_kernel_lower_bound")
    if nullity_lower < nullity_upper and m <= exact_max_cols:
        rank = exact_rank_int([[int(v) for v in row] for row in dense])
        nullity_upper = m - rank
        nullity_lower = nullity_upper                   # exact: bounds coincide
        method = "exact_bareiss"
    exact = bool(nullity_lower == nullity_upper)
    out = {
        "support_size": m,
        # `rank` is EXACT iff rank_exact; otherwise it is a certified lower bound,
        # and `nullity_upper_bound` is correspondingly only an upper bound.
        "rank": rank,
        "rank_exact": exact,
        "rank_lower_bound": rank,
        "rank_exact_value": rank if exact else None,
        "nullity_upper_bound": nullity_upper,
        "nullity_lower_bound": nullity_lower,
        "nullity_exact_value": nullity_upper if exact else None,
        "full_rank": bool(rank == m),
        "rank_method": method,
    }
    cache[key] = out
    return out


def rank_str(st: Dict[str, Any]) -> str:
    """Human-readable rank/nullity that never hides a bound as an exact value."""
    if st["rank_exact"]:
        return f"rank={st['rank']}, nullity={st['nullity_upper_bound']}"
    return (f"rank>={st['rank_lower_bound']}, "
            f"nullity<={st['nullity_upper_bound']} (>={st['nullity_lower_bound']})")


# ------------------------------------------------------------
# Cube geometry for the two switches
# ------------------------------------------------------------

def _flat(n: int, i: int, j: int, k: int) -> int:
    return i * n * n + j * n + k


def triples(cells: Sequence[int], n: int) -> List[List[int]]:
    return [[int(c) // (n * n), (int(c) // n) % n, int(c) % n] for c in cells]


def cube_pair(n: int, x: int, i2: int, y: int, j2: int, z: int, k2: int
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Cube {x,i2}x{y,j2}x{z,k2}; P = class containing (x,y,z), Z = the other."""
    P = np.array([_flat(n, x, y, z), _flat(n, x, j2, k2),
                  _flat(n, i2, y, k2), _flat(n, i2, j2, z)], dtype=np.int64)
    Z = np.array([_flat(n, x, y, k2), _flat(n, x, j2, z),
                  _flat(n, i2, y, z), _flat(n, i2, j2, k2)], dtype=np.int64)
    return P, Z


def switch_key(P: np.ndarray, Z: np.ndarray) -> Tuple[frozenset, frozenset]:
    return frozenset(P.tolist()), frozenset(Z.tolist())


def first_switch_candidates(X: np.ndarray, n: int) -> List[Tuple[int, int, int, np.ndarray, np.ndarray]]:
    """
    All admissible origin-containing first switches: cube {0,i}x{0,j}x{0,k} with
    X_000 >= 1/3, the three other decreasing corners exactly 1/3, (i,j,k) already
    positive and the other three increasing corners 0 (so |S| is preserved).
    """
    out = []
    for i in range(1, n):
        for j in range(1, n):
            for k in range(1, n):
                P, Z = cube_pair(n, 0, i, 0, j, 0, k)
                if X[P[0]] < 1 or not (X[P[1:]] == 1).all():
                    continue
                if X[Z[3]] <= 0 or (X[Z[:3]] != 0).any():
                    continue
                out.append((i, j, k, P, Z))
    return out


def _second_ok(X: np.ndarray, P: np.ndarray, Z: np.ndarray, l1_mask: np.ndarray
               ) -> Optional[List[int]]:
    """General valid cube move that kills >=1 original L1 cell; None if invalid."""
    if (X[P] <= 0).any():                                  # decreasing corners positive
        return None
    leaving = int(np.sum(X[P] == 1))                       # 1/3 -> 0, leave support
    if leaving != int(np.sum(X[Z] == 0)):                  # must equal entering count
        return None
    removed = [int(c) for c in P.tolist() if X[c] == 1 and l1_mask[c]]
    return removed or None


def second_switch_candidates(X: np.ndarray, n: int, l1_mask: np.ndarray
                             ) -> List[Tuple[np.ndarray, np.ndarray, List[int]]]:
    """Every valid second switch, enumerated from original L1 cells at 1/3."""
    out, seen = [], set()
    for cell in np.nonzero((X == 1) & l1_mask)[0]:
        x, y, z = int(cell) // (n * n), (int(cell) // n) % n, int(cell) % n
        for i2 in range(n):
            if i2 == x:
                continue
            for j2 in range(n):
                if j2 == y:
                    continue
                for k2 in range(n):
                    if k2 == z:
                        continue
                    P, Z = cube_pair(n, x, i2, y, j2, z, k2)
                    key = switch_key(P, Z)
                    if key in seen:
                        continue
                    seen.add(key)
                    removed = _second_ok(X, P, Z, l1_mask)
                    if removed is not None:
                        out.append((P, Z, removed))
    return out


def sample_second_switch(X: np.ndarray, n: int, l1_mask: np.ndarray, rng,
                         attempts: int, used: set
                         ) -> Tuple[Optional[Tuple[np.ndarray, np.ndarray, List[int]]], int]:
    """Random valid second switch not in `used` (same conditions as the full
    enumeration); returns (candidate or None, proposals tried)."""
    cells = np.nonzero((X == 1) & l1_mask)[0]
    if cells.size == 0:
        return None, 0
    for t in range(1, attempts + 1):
        cell = int(cells[int(rng.integers(cells.size))])
        x, y, z = cell // (n * n), (cell // n) % n, cell % n
        i2, j2, k2 = (int(rng.integers(n)) for _ in range(3))
        if i2 == x or j2 == y or k2 == z:
            continue
        P, Z = cube_pair(n, x, i2, y, j2, z, k2)
        if switch_key(P, Z) in used:
            continue
        removed = _second_ok(X, P, Z, l1_mask)
        if removed is not None:
            return (P, Z, removed), t
    return None, attempts


# ------------------------------------------------------------
# Invariants
# ------------------------------------------------------------

def check_switch(X_before: np.ndarray, X: np.ndarray, A, P: np.ndarray, Z: np.ndarray,
                 r: int, n: int) -> None:
    """Assert the mandatory per-switch invariants (cheap: one sparse matvec)."""
    cells = np.concatenate([P, Z])
    assert len(set(cells.tolist())) == 8, "cube corners not distinct"
    coords = np.stack((cells // (n * n), (cells // n) % n, cells % n), axis=1)
    assert all(len(set(coords[:, ax].tolist())) == 2 for ax in range(3)), "not a 2x2x2 cube"
    D = np.zeros(n ** 3)
    D[Z], D[P] = 1.0, -1.0
    assert np.all(A @ D == 0), "cube direction not in ker(A)"
    assert (X >= 0).all(), "negative entry after switch"
    assert set(np.unique(X).tolist()) <= {0, 1, 2, 3}, "values outside {0,1,2,3}/3"
    assert np.all(A @ X.astype(float) == 3), "a line sum is not exactly 1"
    assert int(np.count_nonzero(X)) == r, "support size left r"
    assert np.array_equal(X_before[P] - 1, X[P]) and np.array_equal(X_before[Z] + 1, X[Z]), \
        "switch did not move exactly 1/3 on the eight corners"


# ------------------------------------------------------------
# Construction
# ------------------------------------------------------------

def build_construction(n: int, a: int, b: int, c: int, d: int) -> Dict[str, Any]:
    """The algebraic triple, its average X (units of 1/3) and agreement counts."""
    L = [dlst.latin_square(n, 1, 1), dlst.latin_square(n, a, b), dlst.latin_square(n, c, d)]
    latin_ok = [bool(dlst.is_latin(Lt, n)) for Lt in L]
    Pt = [dlst.perm_tensor(Lt, n) for Lt in L]
    X = Pt[0] + Pt[1] + Pt[2]
    agree = {"agree_12": int(np.sum(L[0] == L[1])), "agree_13": int(np.sum(L[0] == L[2])),
             "agree_23": int(np.sum(L[1] == L[2])),
             "agree_123": int(np.sum((L[0] == L[1]) & (L[1] == L[2])))}
    return {"L": L, "P": Pt, "X": X, "latin_ok": latin_ok, "agree": agree,
            "l1_mask": Pt[0] == 1}


def switch_record(X_before: np.ndarray, X_after: np.ndarray, P: np.ndarray, Z: np.ndarray,
                  n: int, rank: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "admissible": True,
        "cube_pairs": [sorted({int(c) // (n * n) for c in np.concatenate([P, Z])}),
                       sorted({(int(c) // n) % n for c in np.concatenate([P, Z])}),
                       sorted({int(c) % n for c in np.concatenate([P, Z])})],
        "decrease_cells": triples(P, n), "increase_cells": triples(Z, n),
        "decrease_before_thirds": [int(v) for v in X_before[P]],
        "decrease_after_thirds": [int(v) for v in X_after[P]],
        "increase_before_thirds": [int(v) for v in X_before[Z]],
        "increase_after_thirds": [int(v) for v in X_after[Z]],
        **{f"after_{k}": v for k, v in rank.items()},
    }


# ------------------------------------------------------------
# One two-switch walk
# ------------------------------------------------------------

def two_switch_walk(con: Dict[str, Any], n: int, r: int, A, A_csc, cubes,
                    first: Tuple[int, int, int, np.ndarray, np.ndarray],
                    second: Optional[Tuple[np.ndarray, np.ndarray, List[int]]],
                    init_rank: Dict[str, Any], cache, checks: bool,
                    rng=None, attempts: int = 0, used_second: Optional[set] = None,
                    exact_max_cols: int = 150) -> Dict[str, Any]:
    """Apply switch 1 then switch 2 (sampling switch 2 if not given).  Never a third."""
    X = con["X"].copy()
    l1_mask = con["l1_mask"]
    rec: Dict[str, Any] = {
        "initial_support_size": init_rank["support_size"], "initial_rank": init_rank["rank"],
        "initial_nullity_upper_bound": init_rank["nullity_upper_bound"],
        "initial_nullity_exact_value": init_rank["nullity_exact_value"],
        "initial_rank_exact": init_rank["rank_exact"],
        # trajectories hold the rank lower bound / nullity upper bound at each
        # stage; the *_exact_trajectory flags say where those are exact values
        "rank_trajectory": [init_rank["rank"]],
        "nullity_upper_bound_trajectory": [init_rank["nullity_upper_bound"]],
        "rank_exact_trajectory": [init_rank["rank_exact"]],
    }

    i, j, k, P1, Z1 = first
    before = X.copy()
    X[P1] -= 1
    X[Z1] += 1
    if checks:
        check_switch(before, X, A, P1, Z1, r, n)
    s1 = state_rank(A_csc, np.sort(np.nonzero(X)[0]), n, cubes, cache, exact_max_cols)
    rec["switch1"] = switch_record(before, X, P1, Z1, n, s1)
    rec["switch1"]["origin_value_after_thirds"] = int(X[0])
    rec["switch1"]["ijk"] = [i, j, k]
    rec["rank_trajectory"].append(s1["rank"])
    rec["nullity_upper_bound_trajectory"].append(s1["nullity_upper_bound"])
    rec["rank_exact_trajectory"].append(s1["rank_exact"])

    X1 = X.copy()
    if second is None:                                  # sampling mode (large n only)
        second, tried = sample_second_switch(X, n, l1_mask, rng, attempts, used_second or set())
        rec["second_switch_proposals"] = tried
    if second is None:
        rec["switch2"] = {"admissible": False}
        rec["success"] = False
        # a sampled miss proves nothing; "none exists" is only claimed by the
        # enumerating path in run_n
        rec["failure_reason"] = "second_switch_budget_exhausted"
        return rec, X1, X, None, None

    P2, Z2, removed = second
    before2 = X.copy()
    X[P2] -= 1
    X[Z2] += 1
    if checks:
        check_switch(before2, X, A, P2, Z2, r, n)
    support2 = np.sort(np.nonzero(X)[0])
    s2 = state_rank(A_csc, support2, n, cubes, cache, exact_max_cols)
    rec["switch2"] = switch_record(before2, X, P2, Z2, n, s2)
    rec["switch2"]["removed_l1_cells"] = triples(removed, n)
    rec["switch2"]["removed_l1_count"] = len(removed)
    rec["rank_trajectory"].append(s2["rank"])
    rec["nullity_upper_bound_trajectory"].append(s2["nullity_upper_bound"])
    rec["rank_exact_trajectory"].append(s2["rank_exact"])

    success = bool(s2["full_rank"] and support2.size == r)
    if success:                                            # exact certificate, as asked
        success = bool(lw.full_column_rank_modp(A, list(support2)))
    rec["success"] = success
    rec["failure_reason"] = None if success else (
        "not_full_rank_after_two_switches" if support2.size == r else "support_left_r")
    return rec, X1, X, support2, second


# ------------------------------------------------------------
# Main run
# ------------------------------------------------------------

def select_cases(n: int, args, A_csc, cubes, r: int, cache) -> List[Tuple]:
    """
    Coefficient quadruples to run.  `cases` = the presets from
    deterministic_latin_support_test; `promising` = scan random quadruples and
    keep those admitted (|S| = r) WITH at least one admissible first switch,
    lowest initial nullity first (two switches cannot close a large nullity).
    """
    if args.coefficients:
        a, b_, c, d = args.coefficients
        return [("explicit", a, b_, c, d)]
    if args.coefficient_mode == "cases":
        return dlst.coefficient_cases(n, args.random_coefficients, args.seed)[:args.constructions]

    rng = np.random.default_rng(args.seed + n)
    seen, scored = set(), []
    for _ in range(args.scan_quadruples):
        q = tuple(int(v) for v in rng.choice(np.arange(1, n), size=4, replace=False))
        if q in seen:
            continue
        seen.add(q)
        con = build_construction(n, *q)
        if not all(con["latin_ok"]):
            continue
        S = np.sort(np.nonzero(con["X"])[0])
        if S.size != r:
            continue
        nf = len(first_switch_candidates(con["X"], n))
        if nf == 0:
            continue
        st = state_rank(A_csc, S, n, cubes, cache, args.exact_rank_max_cols)
        scored.append((st["nullity_upper_bound"], -nf, q))
    scored.sort()
    print(f"  coefficient scan: {len(seen)} quadruples -> {len(scored)} admitted with a "
          f"first switch; initial nullities (<=) {sorted(s[0] for s in scored)[:12]}")
    return [("promising", *q) for _, _, q in scored[:args.constructions]]


def run_n(n: int, args, out_f, vertices_by_key: Dict[frozenset, dict],
          A, b, ttv, vtt, A_csc, cubes) -> List[Dict[str, Any]]:
    r = polysim.constraint_rank_expected(n)
    cache: Dict[bytes, Dict[str, Any]] = {}
    summaries = []
    cases = select_cases(n, args, A_csc, cubes, r, cache)
    print(f"\nn={n}  r={r}  ({len(cases)} constructions, "
          f"{'exhaustive pairs' if args.exhaustive else f'{args.walks} walks each'})")

    for cid, (label, a, bb, c, d) in enumerate(cases, 1):
        con = build_construction(n, a, bb, c, d)
        if not all(con["latin_ok"]):
            print(f"  [{cid}] (a,b,c,d)=({a},{bb},{c},{d}) {label}: not Latin, skipped")
            continue
        X0 = con["X"]
        support0 = np.sort(np.nonzero(X0)[0])
        if support0.size != r:
            print(f"  [{cid}] (a,b,c,d)=({a},{bb},{c},{d}) {label}: "
                  f"|S|={support0.size} != r, not admitted")
            summaries.append({"summary": True, "n": n, "construction_id": cid,
                              "coefficients": [a, bb, c, d], "case": label,
                              "admitted": False, "initial_support_size": int(support0.size),
                              "r": r})
            continue

        init_rank = state_rank(A_csc, support0, n, cubes, cache,
                               args.exact_rank_max_cols)
        firsts = first_switch_candidates(X0, n)
        base = {"n": n, "r": r, "construction_id": cid, "case": label,
                "a": a, "b": bb, "c": c, "d": d, **con["agree"]}
        print(f"  [{cid}] (a,b,c,d)=({a},{bb},{c},{d}) {label}: |S|={support0.size}=r, "
              f"{rank_str(init_rank)}, {len(firsts)} first switches")

        cnt = Counter()
        walks = 0
        used_pairs: set = set()
        master = np.random.default_rng(args.seed + 1000 * n + cid)

        if not firsts:
            row = {**base, "walk_id": 0, "walk_seed": None,
                   "initial_support_size": int(support0.size),
                   "initial_rank": init_rank["rank"],
                   "initial_nullity_upper_bound": init_rank["nullity_upper_bound"],
                   "initial_nullity_exact_value": init_rank["nullity_exact_value"],
                   "initial_rank_exact": init_rank["rank_exact"],
                   "switch1": {"admissible": False}, "switch2": {"admissible": False},
                   "success": False, "failure_reason": "no_admissible_first_switch",
                   "rank_trajectory": [init_rank["rank"]],
                   "nullity_upper_bound_trajectory": [init_rank["nullity_upper_bound"]],
                   "rank_exact_trajectory": [init_rank["rank_exact"]]}
            out_f.write(json.dumps(polysim._jsonable(row)) + "\n")
            cnt["no_admissible_first_switch"] += 1

        # Plan the (first, second) pairs.  Enumerating the second switches costs
        # about |L1 cells at 1/3| * (n-1)^3 per first switch, so enumerate while
        # that is affordable (then every "none exists" is PROVEN and the walks are
        # distinct pairs drawn without replacement) and fall back to sampling with
        # a budget only for large n.
        enum_ops = int(np.sum((X0 == 1) & con["l1_mask"])) * (n - 1) ** 3
        enumerate_mode = bool(firsts) and (args.exhaustive or enum_ops <= args.enumerate_max_ops)
        plan: List[Tuple[Any, Any]] = []
        empty_firsts = 0
        if enumerate_mode:
            t_en = time.time()
            pairs: List[Tuple[Any, Any]] = []
            for first in firsts:
                Xt = X0.copy()
                Xt[first[3]] -= 1
                Xt[first[4]] += 1
                cands = second_switch_candidates(Xt, n, con["l1_mask"])
                if not cands:
                    empty_firsts += 1
                pairs += [(first, s) for s in cands]
            print(f"       enumerated {len(pairs)} valid (first, second) pairs in "
                  f"{time.time() - t_en:.1f}s"
                  + (f"; {empty_firsts} first switches admit none" if empty_firsts else ""))
            if args.exhaustive:
                plan = pairs
            else:                                       # distinct pairs, no replacement
                take = min(args.walks, len(pairs))
                plan = [pairs[int(i)] for i in master.permutation(len(pairs))[:take]]
        elif firsts:
            plan = [(None, None)] * args.walks          # sampling mode: walk_seed decides
            print(f"       sampling mode ({enum_ops:,} ops/state > "
                  f"{args.enumerate_max_ops:,} enumeration budget)")

        for planned_first, planned_second in plan:
            walks += 1
            # the walk seed is drawn BEFORE any choice and drives every choice the
            # walk makes, so a recorded walk can be replayed from it
            walk_seed = int(master.integers(0, 2 ** 31 - 1))
            wrng = np.random.default_rng(walk_seed)
            first = planned_first if planned_first is not None else \
                firsts[int(wrng.integers(len(firsts)))]
            second = planned_second
            fkey = switch_key(first[3], first[4])
            used_second = {sk for (fk, sk) in used_pairs if fk == fkey}
            rec, X1, X2, support2, second_used = two_switch_walk(
                con, n, r, A, A_csc, cubes, first, second, init_rank, cache,
                checks=not args.no_checks, rng=wrng, attempts=args.second_attempts,
                used_second=used_second, exact_max_cols=args.exact_rank_max_cols)
            row = {**base, "walk_id": walks, "walk_seed": walk_seed,
                   "pair_source": "enumerated" if planned_second is not None else "sampled",
                   **rec}
            if second_used is not None:
                used_pairs.add((fkey, switch_key(second_used[0], second_used[1])))
                cnt["second_ok"] += 1
            cnt["first_ok"] += 1
            cnt["success"] += bool(rec["success"])
            tag = "exact" if rec["rank_exact_trajectory"][-1] else "upper_bound_only"
            cnt[f"final_nullity_{tag}_{rec['nullity_upper_bound_trajectory'][-1]}"] += 1
            if rec.get("failure_reason"):
                cnt[rec["failure_reason"]] += 1

            if args.dump_tensors == "all" or (args.dump_tensors == "success" and rec["success"]):
                row["X0_thirds"] = X0.tolist()          # all three tensors, units of 1/3
                row["X1_thirds"] = X1.tolist()
                row["X2_thirds"] = X2.tolist()
            out_f.write(json.dumps(polysim._jsonable(row)) + "\n")

            if rec["success"]:
                key = frozenset(vtt[int(x)] for x in support2)
                if key in vertices_by_key:
                    v = vertices_by_key[key]
                    v["times_reached"] = v.get("times_reached", 1) + 1
                    v.setdefault("found_by_walks", []).append([cid, walks])
                else:
                    x_flat = X2.astype(float) / 3.0
                    sol = {"success": True, "lp_status": 0,
                           "lp_message": "algebraic_two_cube_switch",
                           "x": x_flat.reshape((n, n, n)),
                           "support": [vtt[int(x)] for x in support2],
                           "objective_value": None}
                    vrec = polysim.build_vertex_record(n, walk_seed, sol, A, b, ttv, vtt,
                                                       include_support_map=True)
                    vrec["radius_from_uniform"] = lw.radius_from_uniform(x_flat, n)
                    vrec["move_strategy"] = "algebraic_two_cube_switch"
                    vrec["coefficients"] = [a, bb, c, d]
                    vrec["construction_case"] = label
                    vrec["switch1_cells"] = rec["switch1"]["decrease_cells"]
                    vrec["switch2_cells"] = rec["switch2"]["decrease_cells"]
                    vrec["rank_trajectory"] = rec["rank_trajectory"]
                    vrec["times_reached"] = 1
                    vrec["found_by_walks"] = [[cid, walks]]
                    vertices_by_key[key] = lw.reorder_record(vrec)
                print(f"       walk {walks}: SUCCESS  rank {rec['rank_trajectory']} "
                      f"nullity {rec['nullity_upper_bound_trajectory']} "
                      f"(exact {rec['rank_exact_trajectory']})")

        fin_exact = {k.rsplit("_", 1)[-1]: v for k, v in cnt.items()
                     if k.startswith("final_nullity_exact_")}
        fin_bound = {k.rsplit("_", 1)[-1]: v for k, v in cnt.items()
                     if k.startswith("final_nullity_upper_bound_only_")}
        summary = {"summary": True, "n": n, "r": r, "construction_id": cid, "case": label,
                   "coefficients": [a, bb, c, d], "admitted": True,
                   "initial_support_size": int(support0.size),
                   "initial_rank": init_rank["rank"],
                   "initial_nullity_upper_bound": init_rank["nullity_upper_bound"],
                   "initial_nullity_exact_value": init_rank["nullity_exact_value"],
                   "initial_rank_exact": init_rank["rank_exact"],
                   "first_switch_options": len(firsts),
                   "second_switch_search": ("enumerated" if enumerate_mode else "sampled"),
                   "first_switches_with_no_second": (empty_firsts if enumerate_mode else None),
                   "walks_attempted": walks,
                   "walks_with_first_switch": cnt["first_ok"],
                   "walks_with_second_switch": cnt["second_ok"],
                   "walks_reaching_vertex": cnt["success"],
                   # kept apart on purpose: a bounded nullity is not a measured one
                   "final_nullity_distribution_exact": fin_exact,
                   "final_nullity_distribution_upper_bound_only": fin_bound,
                   "failure_reasons": {k: v for k, v in cnt.items()
                                       if k.endswith("switch") or k.endswith("exhausted")
                                       or k.startswith("not_") or k.startswith("support_")},
                   **con["agree"]}
        out_f.write(json.dumps(polysim._jsonable(summary)) + "\n")
        summaries.append(summary)
        print(f"       -> {cnt['second_ok']}/{walks} reached a second switch, "
              f"{cnt['success']} full vertices, final nullity exact "
              f"{dict(sorted(fin_exact.items()))}"
              + (f", bounded {dict(sorted(fin_bound.items()))}" if fin_bound else ""))
    return summaries


def run(args) -> None:
    t0 = time.time()
    all_sum: List[Dict[str, Any]] = []
    for n in args.n_values:
        A, b, ttv, vtt = polysim.build_tristochastic_constraints(n)
        A_csc = A.tocsc()
        cubes = lcs.build_cubes(n)
        out_path = args.out or f"algebraic_two_cube_walks_n{n}.jsonl"
        vert_path = args.out_vertices or f"algebraic_two_cube_vertices_n{n}.jsonl"
        vertices_by_key, _ = lw.load_existing(vert_path)
        with open(out_path, "a", encoding="utf-8") as out_f:
            all_sum += run_n(n, args, out_f, vertices_by_key, A, b, ttv, vtt, A_csc, cubes)
        lw.write_all(vert_path, vertices_by_key)
        print(f"  -> {out_path} | {len(vertices_by_key)} distinct vertices in {vert_path}")

    adm = [s for s in all_sum if s.get("admitted")]
    print("\n" + "=" * 66)
    print("ALGEBRAIC TWO-CUBE SUMMARY")
    print("=" * 66)
    exact_init = Counter(s["initial_nullity_exact_value"] for s in adm if s["initial_rank_exact"])
    bound_init = Counter(s["initial_nullity_upper_bound"] for s in adm if not s["initial_rank_exact"])
    print(f"  constructions admitted (|S| = r) : {len(adm)}/{len(all_sum)}")
    print(f"  initial nullity (exact)          : {dict(sorted(exact_init.items()))}")
    if bound_init:
        print(f"  initial nullity (upper bound)    : {dict(sorted(bound_init.items()))}")
    print(f"  walks attempted                  : {sum(s['walks_attempted'] for s in adm)}")
    print(f"  reached an admissible switch 2   : {sum(s['walks_with_second_switch'] for s in adm)}")
    print(f"  reached a full vertex            : {sum(s['walks_reaching_vertex'] for s in adm)}")
    fin_e, fin_b = Counter(), Counter()
    for s in adm:
        for k, v in s["final_nullity_distribution_exact"].items():
            fin_e[k] += v
        for k, v in s["final_nullity_distribution_upper_bound_only"].items():
            fin_b[k] += v
    print(f"  final nullity (exact)            : {dict(sorted(fin_e.items()))}")
    if fin_b:
        print(f"  final nullity (upper bound only) : {dict(sorted(fin_b.items()))}")
    print(f"  wall time                        : {time.time() - t0:.1f}s")


# ------------------------------------------------------------
# Regression test: known n=5 success
# ------------------------------------------------------------

_REG = {
    "n": 5, "coeffs": (1, 2, 3, 4),
    "s1_dec": [(0, 0, 0), (0, 2, 3), (1, 0, 3), (1, 2, 0)],
    "s1_inc": [(0, 0, 3), (0, 2, 0), (1, 0, 0), (1, 2, 3)],
    "s2_dec": [(0, 1, 4), (0, 3, 3), (1, 1, 3), (1, 3, 4)],
    "s2_inc": [(0, 1, 3), (0, 3, 4), (1, 1, 4), (1, 3, 3)],
    "ranks": [59, 60, 61], "support": 61,
}


def regression(verbose: bool = True, exact_max_cols: int = 150) -> bool:
    n, (a, b_, c, d) = _REG["n"], _REG["coeffs"]
    A, b, ttv, vtt = polysim.build_tristochastic_constraints(n)
    A_csc, cubes = A.tocsc(), lcs.build_cubes(n)
    r = polysim.constraint_rank_expected(n)
    con = build_construction(n, a, b_, c, d)
    X = con["X"]
    cache: Dict[bytes, Dict[str, Any]] = {}
    ok = True

    def say(label, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        if verbose:
            print(f"  {'PASS' if good else 'FAIL'}  {label}: got {got}, expected {want}")

    support = np.sort(np.nonzero(X)[0])
    s0 = state_rank(A_csc, support, n, cubes, cache)
    say("initial |S|", int(support.size), _REG["support"])
    say("initial rank(A_S)", s0["rank"], _REG["ranks"][0])
    say("initial rank is exact", s0["rank_exact"], True)

    P1 = np.array([_flat(n, *t) for t in _REG["s1_dec"]], dtype=np.int64)
    Z1 = np.array([_flat(n, *t) for t in _REG["s1_inc"]], dtype=np.int64)
    firsts = first_switch_candidates(X, n)
    say("switch 1 accepted by the enumerator",
        any(switch_key(f[3], f[4]) == switch_key(P1, Z1) for f in firsts), True)
    before = X.copy()
    X = X.copy()
    X[P1] -= 1
    X[Z1] += 1
    check_switch(before, X, A, P1, Z1, r, n)
    s1 = state_rank(A_csc, np.sort(np.nonzero(X)[0]), n, cubes, cache, exact_max_cols)
    say("|S| after switch 1", int(np.count_nonzero(X)), _REG["support"])
    say("rank after switch 1", s1["rank"], _REG["ranks"][1])
    say("X_000 after switch 1 (thirds)", int(X[0]), 2)

    P2 = np.array([_flat(n, *t) for t in _REG["s2_dec"]], dtype=np.int64)
    Z2 = np.array([_flat(n, *t) for t in _REG["s2_inc"]], dtype=np.int64)
    seconds = second_switch_candidates(X, n, con["l1_mask"])
    say("switch 2 accepted by the enumerator",
        any(switch_key(p, z) == switch_key(P2, Z2) for p, z, _ in seconds), True)
    before2 = X.copy()
    X = X.copy()
    X[P2] -= 1
    X[Z2] += 1
    check_switch(before2, X, A, P2, Z2, r, n)
    support2 = np.sort(np.nonzero(X)[0])
    s2 = state_rank(A_csc, support2, n, cubes, cache, exact_max_cols)
    say("|S| after switch 2", int(support2.size), _REG["support"])
    say("rank after switch 2", s2["rank"], _REG["ranks"][2])
    say("exact full-column-rank certificate",
        bool(lw.full_column_rank_modp(A, list(support2))), True)
    removed = [t for t in _REG["s2_dec"] if con["l1_mask"][_flat(n, *t)] and before2[_flat(n, *t)] == 1]
    say("original L1 cells removed", len(removed) >= 1, True)
    if verbose:
        print(f"  rank trajectory: [{s0['rank']}, {s1['rank']}, {s2['rank']}]  "
              f"nullity: [{s0['nullity_upper_bound']}, {s1['nullity_upper_bound']}, "
              f"{s2['nullity_upper_bound']}]  exact: "
              f"[{s0['rank_exact']}, {s1['rank_exact']}, {s2['rank_exact']}]")
        print(f"\n  REGRESSION {'PASSED' if ok else 'FAILED'}")
    return ok


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Algebraic 3-Latin-square average + exactly two cube switches.")
    p.add_argument("--n-values", type=int, nargs="+", default=[5, 7, 11])
    p.add_argument("--constructions", type=int, default=4,
                   help="coefficient cases per n (from deterministic_latin_support_test)")
    p.add_argument("--random-coefficients", type=int, default=2,
                   help="extra random coefficient quadruples offered per n (cases mode)")
    p.add_argument("--coefficient-mode", choices=("cases", "promising"), default="cases",
                   help="'promising' scans quadruples for an admitted start with an "
                        "admissible first switch, lowest initial nullity first")
    p.add_argument("--scan-quadruples", type=int, default=200,
                   help="quadruples examined in 'promising' mode")
    p.add_argument("--coefficients", type=int, nargs=4, default=None,
                   metavar=("A", "B", "C", "D"), help="run exactly this quadruple")
    p.add_argument("--exact-rank-max-cols", type=int, default=150,
                   help="use exact Bareiss rank when |S| <= this (else report bounds)")
    p.add_argument("--walks", type=int, default=30, help="two-switch walks per construction")
    p.add_argument("--second-attempts", type=int, default=500,
                   help="sampled proposals per walk for the second switch (sampling mode)")
    p.add_argument("--enumerate-max-ops", type=int, default=1_000_000,
                   help="enumerate all second switches while the per-state cost "
                        "(|L1 cells at 1/3| * (n-1)^3) stays under this; above it, sample")
    p.add_argument("--exhaustive", action="store_true",
                   help="try every valid (first, second) switch pair (small n)")
    p.add_argument("--dump-tensors", choices=("none", "success", "all"), default="success")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--out-vertices", type=str, default=None)
    p.add_argument("--no-checks", action="store_true", help="skip per-switch invariant asserts")
    p.add_argument("--regression", action="store_true", help="run the known n=5 test and exit")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.regression:
        print("regression: n=5, (a,b,c,d)=(1,2,3,4), expected rank 59 -> 60 -> 61")
        raise SystemExit(0 if regression() else 1)
    run(args)


if __name__ == "__main__":
    main()
