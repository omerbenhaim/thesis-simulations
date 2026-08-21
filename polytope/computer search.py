#!/usr/bin/env python3
"""
Enumerate ALL vertices of the 3 x 3 x 3 tristochastic polytope.

Method, specialized to n=3:
  P = {x in R^27 : A x = 1, x >= 0}.
  rank(A) = 19, so dim(P) = 27 - 19 = 8.

Every vertex has at least 8 zero coordinates. Therefore, to find all vertices,
it is enough to enumerate all choices Z of 8 coordinates forced to be zero,
solve:
      A x = 1,   x_z = 0 for z in Z,
and keep the feasible unique solutions. The final vertex test is exact rational.

Saved JSON is intentionally short: total counts + one line per vertex.
The support map is not saved; --show-support-map recomputes and prints it.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from fractions import Fraction
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import sympy as sp

Cell = Tuple[int, int, int]
Vertex = Tuple[Tuple[Fraction, ...], Tuple[int, ...], Tuple[Fraction, ...]]


# ---------------------------------------------------------------------
# Basic indexing and constraints
# ---------------------------------------------------------------------

def cell_to_index(i: int, j: int, k: int, n: int = 3) -> int:
    return i * n * n + j * n + k


def index_to_cell(c: int, n: int = 3) -> Cell:
    i = c // (n * n)
    r = c % (n * n)
    j = r // n
    k = r % n
    return (i, j, k)


def build_cells(n: int = 3) -> List[Cell]:
    return [(i, j, k) for i in range(n) for j in range(n) for k in range(n)]


def build_line_constraints(n: int = 3) -> Tuple[List[Tuple[int, int, int]], int]:
    """
    coord_lines[c] = the three line-equation row indices containing cell c.

    Row blocks:
      0..n^2-1            : fixed (i,j), varying k
      n^2..2n^2-1         : fixed (i,k), varying j
      2n^2..3n^2-1        : fixed (j,k), varying i
    """
    nn = n * n
    coord_lines: List[Tuple[int, int, int]] = []
    for i, j, k in build_cells(n):
        coord_lines.append((
            i * n + j,
            nn + i * n + k,
            2 * nn + j * n + k,
        ))
    return coord_lines, 3 * nn


def build_incidence_matrix(n: int = 3) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    coord_lines, num_lines = build_line_constraints(n)
    A = np.zeros((num_lines, n ** 3), dtype=float)
    for c, lines in enumerate(coord_lines):
        for line in lines:
            A[line, c] = 1.0
    return A, coord_lines


def build_incidence_matrix_int(n: int = 3) -> List[List[int]]:
    coord_lines, num_lines = build_line_constraints(n)
    A = [[0 for _ in range(n ** 3)] for _ in range(num_lines)]
    for c, lines in enumerate(coord_lines):
        for line in lines:
            A[line][c] = 1
    return A


# ---------------------------------------------------------------------
# Exact rational support solver / vertex verifier
# ---------------------------------------------------------------------

def solve_support_exact(
    support: Sequence[int],
    coord_lines: Sequence[Tuple[int, int, int]],
    num_lines: int,
) -> Optional[List[Fraction]]:
    """
    Solve A_S x_S = 1 exactly over Q.

    Returns the unique rational solution if A_S has full column rank and the
    system is consistent. Otherwise returns None.
    """
    support = list(support)
    m = len(support)

    # Augmented matrix [A_S | 1]
    M = [[Fraction(0) for _ in range(m + 1)] for _ in range(num_lines)]
    for col, cell_idx in enumerate(support):
        for row in coord_lines[cell_idx]:
            M[row][col] = Fraction(1)
    for row in range(num_lines):
        M[row][m] = Fraction(1)

    pivot_row = 0
    pivot_cols: List[int] = []

    for col in range(m):
        sel = None
        for r in range(pivot_row, num_lines):
            if M[r][col] != 0:
                sel = r
                break
        if sel is None:
            # Not full column rank.
            return None

        M[pivot_row], M[sel] = M[sel], M[pivot_row]
        pivot = M[pivot_row][col]
        if pivot != 1:
            M[pivot_row] = [x / pivot for x in M[pivot_row]]

        for r in range(num_lines):
            if r != pivot_row and M[r][col] != 0:
                factor = M[r][col]
                M[r] = [M[r][t] - factor * M[pivot_row][t] for t in range(m + 1)]

        pivot_cols.append(col)
        pivot_row += 1

    # Check consistency of remaining rows.
    for r in range(pivot_row, num_lines):
        if all(M[r][c] == 0 for c in range(m)) and M[r][m] != 0:
            return None

    x = [Fraction(0) for _ in range(m)]
    for r, col in enumerate(pivot_cols):
        x[col] = M[r][m]
    return x


def verify_vertex_from_support(
    support: Sequence[int],
    coord_lines: Sequence[Tuple[int, int, int]],
    num_lines: int,
    n: int = 3,
) -> Optional[Tuple[Tuple[Fraction, ...], Tuple[int, ...], Tuple[Fraction, ...]]]:
    """
    Exact final verifier.

    A support gives a vertex iff A_S x_S = 1 has a unique strictly positive
    solution. Return a full rational tensor key plus support and values.
    """
    support = tuple(sorted(support))
    values = solve_support_exact(support, coord_lines, num_lines)
    if values is None:
        return None
    if any(v <= 0 for v in values):
        return None

    full = [Fraction(0) for _ in range(n ** 3)]
    for c, v in zip(support, values):
        full[c] = v
    return (tuple(full), support, tuple(values))


# ---------------------------------------------------------------------
# Complete n=3 enumeration by active zero constraints
# ---------------------------------------------------------------------

def rational_nullspace_basis_float(n: int = 3) -> np.ndarray:
    """
    Compute a nullspace basis for A using sympy, return it as a float matrix N.

    Shape: (n^3, dim), where dim=(n-1)^3. For n=3 this is (27,8).
    """
    A_int = build_incidence_matrix_int(n)
    A_sym = sp.Matrix(A_int)
    ns = A_sym.nullspace()
    if len(ns) != (n - 1) ** 3:
        raise RuntimeError(f"Expected nullity {(n - 1) ** 3}, got {len(ns)}")

    N = np.zeros((n ** 3, len(ns)), dtype=float)
    for col, vec in enumerate(ns):
        for row in range(n ** 3):
            N[row, col] = float(vec[row])
    return N


def enumerate_vertices_n3(
    tol: float = 1e-8,
    progress_every: int = 100_000,
) -> List[Vertex]:
    """
    Enumerate all vertices of the 3x3x3 tristochastic polytope.

    Guarantee idea:
      dim(P)=8. Every vertex has at least 8 active inequalities x_c=0.
      Therefore enumerating all C(27,8) choices of active zero coordinates
      must hit every vertex at least once. We deduplicate by exact rational
      full tensor.
    """
    n = 3
    Ncells = n ** 3
    dim = (n - 1) ** 3
    A, coord_lines = build_incidence_matrix(n)
    _, num_lines = build_line_constraints(n)

    # Interior equality-satisfying point.
    x0 = np.full(Ncells, 1.0 / n)

    # Equality subspace parameterization: x = x0 + N y.
    N = rational_nullspace_basis_float(n)
    if N.shape != (Ncells, dim):
        raise RuntimeError(f"Bad nullspace shape: {N.shape}")

    vertices_by_full_key: Dict[Tuple[Fraction, ...], Vertex] = {}
    tested_supports = set()

    total = 0
    feasible_zero_sets = 0
    t0 = time.time()

    for Z in combinations(range(Ncells), dim):
        total += 1
        if progress_every and total % progress_every == 0:
            print(
                f"checked {total:,} zero-sets | "
                f"candidate zero-sets {feasible_zero_sets:,} | "
                f"vertices {len(vertices_by_full_key)} | "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )

        Z_list = list(Z)
        M = N[Z_list, :]
        rhs = -x0[Z_list]

        try:
            # If M is singular, this zero-set does not define a unique point.
            y = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            continue

        x = x0 + N @ y

        # Numerical prefilter only. Exact verification comes below.
        if np.min(x) < -tol:
            continue

        support = tuple(i for i, val in enumerate(x) if val > tol)
        if support in tested_supports:
            continue
        tested_supports.add(support)

        # Vertex support size must be between n^2 and rank(A)=19.
        if not (n * n <= len(support) <= 3 * n * n - 3 * n + 1):
            continue

        feasible_zero_sets += 1
        verified = verify_vertex_from_support(support, coord_lines, num_lines, n=n)
        if verified is None:
            continue

        full_key, supp_exact, vals_exact = verified
        vertices_by_full_key[full_key] = (full_key, supp_exact, vals_exact)

    vertices = list(vertices_by_full_key.values())
    vertices.sort(key=canonical_vertex_sort_key)
    return vertices


# ---------------------------------------------------------------------
# Output and display
# ---------------------------------------------------------------------

def frac_str(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def unique_values(values: Sequence[Fraction]) -> List[str]:
    return [frac_str(v) for v in sorted(set(values))]


def canonical_vertex_sort_key(vertex: Vertex):
    _full, support, values = vertex
    coords = tuple(index_to_cell(c, 3) for c in support)
    return (len(support), unique_values(values), coords)


def vertex_records(vertices: Sequence[Vertex]) -> List[Dict[str, object]]:
    records = []
    for vid, (_full, support, values) in enumerate(vertices):
        records.append({
            "vertex_id": vid,
            "support_size": len(support),
            "unique_values": unique_values(values),
        })
    return records


def write_summary_json(vertices: Sequence[Vertex], out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    records = vertex_records(vertices)
    size_dist = Counter(r["support_size"] for r in records)
    value_dist = Counter(",".join(r["unique_values"]) for r in records)

    # Keep the JSON intentionally clean and short.
    summary = {
        "n": 3,
        "total_vertices": len(vertices),
        "support_size_distribution": {str(k): int(v) for k, v in sorted(size_dist.items())},
        "unique_value_distribution": {str(k): int(v) for k, v in sorted(value_dist.items())},
        "vertices": records,
    }

    path = os.path.join(out_dir, "vertices_n3.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path


def print_support_map(vertex_id: int, vertices: Sequence[Vertex]) -> None:
    if vertex_id < 0 or vertex_id >= len(vertices):
        raise ValueError(f"vertex_id must be in 0..{len(vertices)-1}")

    _full, support, values = vertices[vertex_id]
    print(f"vertex_id: {vertex_id}")
    print(f"support_size: {len(support)}")
    print(f"unique_values: {unique_values(values)}")

    print("\n(i,j) -> k values:")
    by_ij: Dict[Tuple[int, int], List[Tuple[int, Fraction]]] = {}
    for c, val in zip(support, values):
        i, j, k = index_to_cell(c, 3)
        by_ij.setdefault((i, j), []).append((k, val))

    for i in range(3):
        for j in range(3):
            entries = by_ij.get((i, j), [])
            text = ", ".join(f"k={k}:{frac_str(v)}" for k, v in sorted(entries))
            print(f"({i},{j}): {text}")


def print_summary(vertices: Sequence[Vertex]) -> None:
    records = vertex_records(vertices)
    size_dist = Counter(r["support_size"] for r in records)
    value_dist = Counter(",".join(r["unique_values"]) for r in records)
    n_perm = sum(1 for r in records if r["unique_values"] == ["1"])
    n_frac = len(records) - n_perm

    print("\n=== n=3 tristochastic vertices ===")
    print(f"total vertices             : {len(vertices)}")
    print(f"permutation vertices       : {n_perm}")
    print(f"fractional vertices        : {n_frac}")
    print(f"support size distribution  : {dict(sorted(size_dist.items()))}")
    print(f"unique value distribution  : {dict(sorted(value_dist.items()))}")

    if len(vertices) == 66 and dict(size_dist) == {9: 12, 17: 54}:
        print("sanity check               : OK, matches known n=3 classification")
    else:
        print("sanity check               : WARNING, does not match expected 66 / {9:12,17:54}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enumerate ALL vertices of the 3x3x3 tristochastic polytope."
    )
    parser.add_argument("--out", default="results_n3", help="output directory")
    parser.add_argument("--show-support-map", type=int, default=None, metavar="ID")
    parser.add_argument("--tol", type=float, default=1e-8, help="floating prefilter tolerance")
    parser.add_argument("--progress-every", type=int, default=100_000)
    args = parser.parse_args()

    vertices = enumerate_vertices_n3(tol=args.tol, progress_every=args.progress_every)

    if args.show_support_map is not None:
        print_support_map(args.show_support_map, vertices)
        return

    print_summary(vertices)
    path = write_summary_json(vertices, args.out)
    print(f"saved JSON                 : {path}")


if __name__ == "__main__":
    main()
