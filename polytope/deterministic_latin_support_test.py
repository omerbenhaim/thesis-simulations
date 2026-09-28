"""
Deterministic 3-Latin-square support test.  Standalone: no walk, no vertex test,
no rank computation, no cube search; nothing here touches the other experiments.

Construction (mainly for prime n, where every nonzero coefficient is invertible):

    L1:  z = x + y      (mod n)
    L2:  z = a*x + b*y  (mod n)
    L3:  z = c*x + d*y  (mod n)      a, b, c, d distinct and nonzero

Each square becomes its n x n x n permutation tensor; X = (L1 + L2 + L3)/3.
The only question asked: is |supp(X)| = r = 3n^2 - 3n + 1?

|supp(X)| is the size of the union of the three tensors' 1-cells, so by
inclusion-exclusion it is

    3n^2 - (agree_12 + agree_13 + agree_23) + agree_123,

where agree_st counts the cells (x, y) at which squares s and t hold the same
symbol.  Those agreement counts are also recorded, since they say exactly why a
case misses r.

CLI
---
    python deterministic_latin_support_test.py
    python deterministic_latin_support_test.py --n-values 5 7 11 --random-trials 10
"""

import argparse
import json
from itertools import permutations
from typing import List, Tuple

import numpy as np


def latin_square(n: int, a: int, b: int) -> np.ndarray:
    """L[x, y] = (a*x + b*y) mod n."""
    x = np.arange(n)[:, None]
    y = np.arange(n)[None, :]
    return (a * x + b * y) % n


def is_latin(L: np.ndarray, n: int) -> bool:
    full = set(range(n))
    return (all(set(L[x, :].tolist()) == full for x in range(n)) and
            all(set(L[:, y].tolist()) == full for y in range(n)))


def perm_tensor(L: np.ndarray, n: int) -> np.ndarray:
    """0/1 permutation tensor of a Latin square (flat, length n^3)."""
    v = np.zeros(n ** 3, dtype=np.int64)
    x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    v[(x * n + y) * n + L] = 1
    return v


def coefficient_cases(n: int, random_trials: int, seed: int) -> List[Tuple]:
    """(label, a, b, c, d) quadruples of distinct nonzero coefficients mod n."""
    def usable(q) -> bool:
        return 0 not in q and max(q) < n and len(set(q)) == 4

    cases = [c for c in (("first", 1, 2, 3, 4), ("small", 2, 3, 4, 5),
                         ("spread", 1, 2, n - 2, n - 1)) if usable(c[1:])]

    # designed collision: (a-1, b-1) parallel to (c-1, d-1) mod n, which makes all
    # three squares agree along a whole line instead of at the origin only
    for a, b in ((2, 3), (3, 4), (2, 4), (4, 5)):
        hit = next((lam for lam in range(2, n)
                    if usable((a, b, (lam * (a - 1) + 1) % n, (lam * (b - 1) + 1) % n))), None)
        if hit is not None:
            cases.append(("parallel", a, b,
                          (hit * (a - 1) + 1) % n, (hit * (b - 1) + 1) % n))
            break

    rng = np.random.default_rng(seed + n)
    seen = set()
    cases = [q for q in cases if not (q[1:] in seen or seen.add(q[1:]))]
    tries = 0
    while len([q for q in cases if q[0] == "random"]) < random_trials and tries < 500:
        tries += 1
        q = tuple(int(v) for v in rng.choice(np.arange(1, n), size=4, replace=False))
        if q not in seen:
            seen.add(q)
            cases.append(("random",) + q)
    return cases


def run(n_values: List[int], random_trials: int, seed: int, out_path: str) -> None:
    rows = []
    print(f"{'n':>4} {'a':>3} {'b':>3} {'c':>3} {'d':>3} {'case':>9} "
          f"{'support':>8} {'r':>8}  match")
    for n in n_values:
        r = 3 * n * n - 3 * n + 1
        for label, a, b, c, d in coefficient_cases(n, random_trials, seed):
            L = [latin_square(n, 1, 1), latin_square(n, a, b), latin_square(n, c, d)]
            latin_ok = [bool(is_latin(Lt, n)) for Lt in L]
            if not all(latin_ok):
                row = {"n": n, "a": a, "b": b, "c": c, "d": d, "case": label,
                       "latin_ok": latin_ok, "support_size": None, "r": r, "match": False}
                rows.append(row)
                print(f"{n:>4} {a:>3} {b:>3} {c:>3} {d:>3} {label:>9} "
                      f"{'not Latin':>8} {r:>8}  False")
                continue

            X = sum(perm_tensor(Lt, n) for Lt in L)        # values in {0,1,2,3}
            support_size = int(np.count_nonzero(X))
            agree = {f"agree_{s+1}{t+1}": int(np.sum(L[s] == L[t]))
                     for s, t in ((0, 1), (0, 2), (1, 2))}
            agree_123 = int(np.sum((L[0] == L[1]) & (L[1] == L[2])))
            row = {"n": n, "a": a, "b": b, "c": c, "d": d, "case": label,
                   "latin_ok": latin_ok, "support_size": support_size, "r": r,
                   "match": support_size == r, "agree_123": agree_123, **agree}
            rows.append(row)
            print(f"{n:>4} {a:>3} {b:>3} {c:>3} {d:>3} {label:>9} "
                  f"{support_size:>8} {r:>8}  {support_size == r}")

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
        matched = [r_ for r_ in rows if r_["match"]]
        missed = [r_ for r_ in rows if not r_["match"]]
        summary = {
            "summary": True, "n_values": n_values, "cases": len(rows),
            "matched_r": len(matched), "missed_r": len(missed),
            "matched_n": sorted({r_["n"] for r_ in matched}),
            "missed_detail": [{k: r_.get(k) for k in ("n", "a", "b", "c", "d", "case",
                                                      "support_size", "r", "agree_123")}
                              for r_ in missed],
        }
        f.write(json.dumps(summary) + "\n")

    print("\n" + "=" * 62)
    print(f"SUMMARY: {len(matched)}/{len(rows)} cases gave |supp(X)| = r")
    print("=" * 62)
    for n in n_values:
        got = [r_ for r_ in rows if r_["n"] == n]
        ok = sum(r_["match"] for r_ in got)
        print(f"  n={n:<3} {ok}/{len(got)} matched r = {3*n*n-3*n+1}")
    if missed:
        print("  misses:")
        for r_ in missed:
            print(f"    n={r_['n']} (a,b,c,d)=({r_['a']},{r_['b']},{r_['c']},{r_['d']}) "
                  f"[{r_['case']}] support={r_['support_size']} vs r={r_['r']} "
                  f"(all three squares agree on {r_.get('agree_123')} cells)")
    else:
        print("  no misses")
    print(f"  results written to {out_path}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Deterministic 3-Latin-square support test.")
    p.add_argument("--n-values", type=int, nargs="+",
                   default=[5, 7, 11, 13, 17, 19, 23, 29])
    p.add_argument("--random-trials", type=int, default=5,
                   help="extra random coefficient quadruples per n")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="deterministic_latin_support_results.jsonl")
    args = p.parse_args(argv)
    run(args.n_values, args.random_trials, args.seed, args.out)


if __name__ == "__main__":
    main()
