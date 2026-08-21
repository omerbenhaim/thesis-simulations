"""
Independent completeness check for GPT's n=3 enumeration.

(1) Re-enumerate with a DIFFERENT implementation (determinant-gated, no
    try/except) and confirm the same 66 / {9:12,17:54}.
(2) METHOD-INDEPENDENT: sample thousands of random-objective LP vertices and
    confirm every distinct one lies inside the enumerated set.
"""
import importlib.util
from collections import Counter
from itertools import combinations

import numpy as np
from scipy.optimize import linprog

spec = importlib.util.spec_from_file_location("cs", "computer search.py")
cs = importlib.util.module_from_spec(spec); spec.loader.exec_module(cs)

n, Ncells, dim = 3, 27, 8
A, coord_lines = cs.build_incidence_matrix(n)
_, num_lines = cs.build_line_constraints(n)
x0 = np.full(Ncells, 1.0 / n)
N = cs.rational_nullspace_basis_float(n)          # (27, 8)

# (1) independent det-gated enumeration ---------------------------------
enum = {}                                          # frozenset(support) -> support_size
tested = set()
for Z in combinations(range(Ncells), dim):
    Zl = list(Z)
    M = N[Zl, :]
    if abs(np.linalg.det(M)) < 1e-7:               # singular -> no unique point
        continue
    x = x0 + N @ np.linalg.solve(M, -x0[Zl])
    if x.min() < -1e-8:
        continue
    supp = tuple(i for i in range(Ncells) if x[i] > 1e-8)
    if supp in tested:
        continue
    tested.add(supp)
    if cs.verify_vertex_from_support(supp, coord_lines, num_lines, n) is not None:
        enum[frozenset(supp)] = len(supp)

print(f"[independent enum] vertices: {len(enum)}  "
      f"sizes: {dict(sorted(Counter(enum.values()).items()))}")

# (2) method-independent LP subset check --------------------------------
b = np.ones(num_lines)
rng = np.random.default_rng(0)
lp_seen, missing = set(), 0
for _ in range(8000):
    res = linprog(-rng.normal(size=Ncells), A_eq=A, b_eq=b,
                  bounds=(0, None), method="highs-ds")
    if not res.success:
        continue
    supp = frozenset(i for i in range(Ncells) if res.x[i] > 1e-7)
    if supp in lp_seen:
        continue
    lp_seen.add(supp)
    if supp not in enum:
        missing += 1
        print("  MISSING from enumeration:", sorted(supp))

print(f"[LP check] distinct LP vertices: {len(lp_seen)} | not in enumeration: {missing}")
print("RESULT:", "COMPLETE (every sampled vertex is enumerated)" if missing == 0
      else "INCOMPLETE")
