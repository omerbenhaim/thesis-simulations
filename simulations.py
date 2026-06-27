import itertools
import math
import random
from typing import Dict, List, Optional, Tuple, Any

import networkx as nx

import time
import json
from datetime import datetime
import os
from typing import Dict, List, Any
# ------------------------------------------------------------
# A. Kneser graph KG(n,k)
# ------------------------------------------------------------

def kneser_graph(n: int, k: int) -> nx.Graph:
    """
    Build the Kneser graph KG(n,k).
    Vertices are k-subsets of {0,1,...,n-1}.
    Two vertices are adjacent iff the subsets are disjoint.

    Notes:
    - If n < k, this is invalid.
    - If k == 0, there is one vertex: the empty set.
    - If n < 2k, the graph has vertices but no edges.
    """
    if n < 0 or k < 0:
        raise ValueError("n and k must be nonnegative")
    if k > n:
        raise ValueError("Need k <= n")

    G = nx.Graph()
    vertices = list(itertools.combinations(range(n), k))

    for v in vertices:
        G.add_node(v)

    for i in range(len(vertices)):
        A = set(vertices[i])
        for j in range(i + 1, len(vertices)):
            B = set(vertices[j])
            if A.isdisjoint(B):
                G.add_edge(vertices[i], vertices[j])

    return G


# ------------------------------------------------------------
# B. Random r-lift of a base graph
# ------------------------------------------------------------

def random_lift(base_graph: nx.Graph, lift_size: int, seed: Optional[int] = None
                ) -> Tuple[nx.Graph, Dict[Tuple[Any, Any], List[int]]]:
    """
    Standard random lift:
    for each base edge (u,v), choose a uniform random permutation pi of [0..lift_size-1]
    and connect (u,i) to (v,pi[i]).

    Returns:
      H            : lifted graph
      permutations : dictionary storing the chosen permutation for each base edge
    """
    if lift_size <= 0:
        raise ValueError("lift_size must be positive")

    rng = random.Random(seed)
    H = nx.Graph()
    perms = {}

    # Add lifted vertices
    for u in base_graph.nodes():
        for i in range(lift_size):
            H.add_node((u, i), base_vertex=u, fiber_index=i)

    # Add lifted edges
    for u, v in base_graph.edges():
        perm = list(range(lift_size))
        rng.shuffle(perm)
        perms[(u, v)] = perm[:]

        for i, j in enumerate(perm):
            H.add_edge((u, i), (v, j), base_edge=(u, v))

    return H, perms


# ------------------------------------------------------------
# C. Chromatic number
# Exact, practical only for modest graph sizes
# ------------------------------------------------------------

def greedy_dsatur_upper_bound(G: nx.Graph) -> Tuple[int, Dict[Any, int]]:
    """
    Greedy DSATUR-style coloring.
    Returns:
      number of colors used,
      coloring dict
    This is an upper bound, not necessarily optimal.
    """
    if G.number_of_nodes() == 0:
        return 0, {}

    coloring: Dict[Any, int] = {}
    uncolored = set(G.nodes())

    def saturation(v):
        return len({coloring[u] for u in G.neighbors(v) if u in coloring})

    while uncolored:
        v = max(uncolored, key=lambda x: (saturation(x), G.degree[x]))
        used = {coloring[u] for u in G.neighbors(v) if u in coloring}

        c = 0
        while c in used:
            c += 1

        coloring[v] = c
        uncolored.remove(v)

    num_colors = max(coloring.values()) + 1
    return num_colors, coloring


def clique_lower_bound(G: nx.Graph) -> int:
    """
    Exact clique-number lower bound via NetworkX clique enumeration.
    Fine for small/moderate graphs.
    """
    if G.number_of_nodes() == 0:
        return 0
    max_clique_size = 0
    for clique in nx.find_cliques(G):
        max_clique_size = max(max_clique_size, len(clique))
    return max_clique_size


def is_k_colorable(G: nx.Graph, k: int) -> Tuple[bool, Optional[Dict[Any, int]]]:
    """
    Backtracking k-colorability test using a DSATUR-like branching rule.
    Returns (True, coloring) if successful, else (False, None).
    """
    nodes = list(G.nodes())
    if not nodes:
        return True, {}

    adjacency = {v: set(G.neighbors(v)) for v in nodes}
    coloring: Dict[Any, int] = {}
    uncolored = set(nodes)

    def saturation(v):
        return len({coloring[u] for u in adjacency[v] if u in coloring})

    def feasible_colors(v):
        forbidden = {coloring[u] for u in adjacency[v] if u in coloring}
        return [c for c in range(k) if c not in forbidden]

    def backtrack() -> bool:
        if not uncolored:
            return True

        # DSATUR choice: max saturation, tie-break by degree
        v = max(uncolored, key=lambda x: (saturation(x), len(adjacency[x])))
        choices = feasible_colors(v)
        if not choices:
            return False

        # Small heuristic: try smaller color numbers first
        for c in choices:
            coloring[v] = c
            uncolored.remove(v)

            if backtrack():
                return True

            del coloring[v]
            uncolored.add(v)

        return False

    ok = backtrack()
    return (ok, coloring.copy()) if ok else (False, None)


def chromatic_number_exact(G: nx.Graph) -> Tuple[int, Dict[Any, int]]:
    """
    Exact chromatic number by:
      1) lower bound = clique number
      2) upper bound = greedy DSATUR coloring
      3) test k-colorability for k = lower..upper

    Practical only for modest graph sizes.
    """
    if G.number_of_nodes() == 0:
        return 0, {}

    lb = max(1, clique_lower_bound(G))
    ub, ub_coloring = greedy_dsatur_upper_bound(G)

    for k in range(lb, ub + 1):
        ok, coloring = is_k_colorable(G, k)
        if ok:
            return k, coloring  # exact optimum

    # Should never happen because ub coloring exists
    return ub, ub_coloring


# ------------------------------------------------------------
# B2. Exhaustive enumeration of all lifts
# Iterate over EVERY possible lift of a base graph for a given
# lift size, instead of sampling random ones.
# ------------------------------------------------------------

def build_lift_from_perms(
    base_graph: nx.Graph,
    lift_size: int,
    perms: Dict[Tuple[Any, Any], List[int]],
) -> nx.Graph:
    """
    Deterministically build a lift from an explicit choice of permutation
    for every base edge (mirrors the construction in random_lift, but with
    the permutations supplied rather than chosen at random).

    perms maps each base edge (u, v) to a permutation of [0..lift_size-1]:
    vertex (u, i) is connected to (v, perms[(u, v)][i]).
    """
    if lift_size <= 0:
        raise ValueError("lift_size must be positive")

    H = nx.Graph()
    for u in base_graph.nodes():
        for i in range(lift_size):
            H.add_node((u, i), base_vertex=u, fiber_index=i)

    for u, v in base_graph.edges():
        perm = perms[(u, v)]
        for i, j in enumerate(perm):
            H.add_edge((u, i), (v, j), base_edge=(u, v))

    return H


def _spanning_forest_edge_indices(base_graph: nx.Graph,
                                   edges: List[Tuple[Any, Any]]) -> set:
    """
    Return the set of indices (into `edges`) that form a spanning forest of
    base_graph, using a union-find pass in the given edge order.
    """
    parent = {n: n for n in base_graph.nodes()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    forest = set()
    for idx, (u, v) in enumerate(edges):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            forest.add(idx)
    return forest


def count_all_lifts(base_graph: nx.Graph, lift_size: int,
                    gauge_fix: bool = True) -> int:
    """
    Number of lifts that all_lifts() will enumerate.

    - Without gauge fixing: (lift_size!)^(#edges).
    - With gauge fixing: spanning-forest edges are pinned to the identity
      permutation (which can always be achieved by relabeling fibers), so the
      count drops to (lift_size!)^(#edges - #forest_edges).

    Use this to gauge feasibility before launching an exhaustive run.
    """
    edges = list(base_graph.edges())
    free = len(edges)
    if gauge_fix:
        free -= len(_spanning_forest_edge_indices(base_graph, edges))
    return math.factorial(lift_size) ** free


def all_lifts(base_graph: nx.Graph, lift_size: int, gauge_fix: bool = True):
    """
    Generator over all lifts of base_graph for the given lift_size.

    Yields (H, perms) pairs, where H is the lifted graph and perms is the
    dict of per-edge permutations used to build it.

    If gauge_fix is True (default), edges of a spanning forest are pinned to
    the identity permutation. This removes the trivial symmetry of relabeling
    fibers and yields one representative per equivalence class, while still
    covering every distinct lift up to that relabeling. Set gauge_fix=False to
    enumerate the full (lift_size!)^(#edges) raw space.
    """
    if lift_size <= 0:
        raise ValueError("lift_size must be positive")

    edges = list(base_graph.edges())
    identity = list(range(lift_size))
    all_perms = list(itertools.permutations(range(lift_size)))

    forest = _spanning_forest_edge_indices(base_graph, edges) if gauge_fix else set()
    free_idx = [i for i in range(len(edges)) if i not in forest]

    for combo in itertools.product(all_perms, repeat=len(free_idx)):
        perms: Dict[Tuple[Any, Any], List[int]] = {
            (u, v): identity[:] for (u, v) in edges
        }
        for slot, i in enumerate(free_idx):
            u, v = edges[i]
            perms[(u, v)] = list(combo[slot])

        H = build_lift_from_perms(base_graph, lift_size, perms)
        yield H, perms


def exhaustive_chromatic_trials(
    kneser_n: int,
    kneser_k: int,
    lift_size: int,
    gauge_fix: bool = True,
    keep_examples: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Exhaustively check the chromatic number of ALL possible lifts of KG(n,k)
    for the given lift_size.

    Returns a summary dict with:
      - chi_base            : chromatic number of the base Kneser graph
      - num_lifts           : number of lifts examined
      - distribution        : {chi value -> count of lifts achieving it}
      - min_chi / max_chi   : extremes over all lifts
      - examples            : {chi value -> one perms dict realizing it}
                              (only if keep_examples is True)
      - parameters echoed back (kneser_n, kneser_k, lift_size, gauge_fix)

    WARNING: the number of lifts grows like (lift_size!)^(#free_edges).
    Call count_all_lifts() first to make sure the run is feasible.
    """
    base = kneser_graph(kneser_n, kneser_k)
    chi_base, _ = chromatic_number_exact(base)

    distribution: Dict[int, int] = {}
    examples: Dict[int, Dict[Tuple[Any, Any], List[int]]] = {}
    num_lifts = 0

    for H, perms in all_lifts(base, lift_size, gauge_fix=gauge_fix):
        chi, _ = chromatic_number_exact(H)
        distribution[chi] = distribution.get(chi, 0) + 1
        if keep_examples and chi not in examples:
            examples[chi] = {e: p[:] for e, p in perms.items()}
        num_lifts += 1

        if verbose and num_lifts % 1000 == 0:
            print(f"  ...examined {num_lifts} lifts")

    result: Dict[str, Any] = {
        "kneser_n": kneser_n,
        "kneser_k": kneser_k,
        "lift_size": lift_size,
        "gauge_fix": gauge_fix,
        "chi_base": chi_base,
        "num_lifts": num_lifts,
        "distribution": dict(sorted(distribution.items())),
        "min_chi": min(distribution) if distribution else None,
        "max_chi": max(distribution) if distribution else None,
    }
    if keep_examples:
        result["examples"] = examples
    return result


def run_exhaustive_sweep(
    configs: List[Tuple[int, int, int]],
    gauge_fix: bool = True,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run exhaustive_chromatic_trials over several (n, k, lift_size) configs.

    configs: list of (kneser_n, kneser_k, lift_size) tuples.
    Returns one summary dict per config (see exhaustive_chromatic_trials).
    """
    summaries = []
    for (n, k, r) in configs:
        if verbose:
            base = kneser_graph(n, k)
            total = count_all_lifts(base, r, gauge_fix=gauge_fix)
            print(f"KG(n={n}, k={k}), lift_size={r}: "
                  f"enumerating {total} lifts (gauge_fix={gauge_fix})")
        summary = exhaustive_chromatic_trials(
            kneser_n=n, kneser_k=k, lift_size=r,
            gauge_fix=gauge_fix, verbose=verbose,
        )
        if verbose:
            print(f"  chi(base)={summary['chi_base']}  "
                  f"chi(lift) distribution={summary['distribution']}")
        summaries.append(summary)
    return summaries


# ------------------------------------------------------------
# One full experiment
# ------------------------------------------------------------

def run_one_experiment(
    kneser_n: int,
    kneser_k: int,
    lift_size: int,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    1) build KG(n,k)
    2) generate random lift of size lift_size
    3) compute exact chromatic number of the lift
    """
    base = kneser_graph(kneser_n, kneser_k)
    lift, perms = random_lift(base, lift_size=lift_size, seed=seed)

    chi_base, base_col = chromatic_number_exact(base)
    chi_lift, lift_col = chromatic_number_exact(lift)

    return {
        "kneser_n": kneser_n,
        "kneser_k": kneser_k,
        "lift_size": lift_size,
        "base_graph": base,
        "lift_graph": lift,
        "permutations": perms,
        "chi_base": chi_base,
        "chi_lift": chi_lift,
        "base_coloring": base_col,
        "lift_coloring": lift_col,
        "base_num_vertices": base.number_of_nodes(),
        "base_num_edges": base.number_of_edges(),
        "lift_num_vertices": lift.number_of_nodes(),
        "lift_num_edges": lift.number_of_edges(),
    }


# ------------------------------------------------------------
# Repeated simulation
# ------------------------------------------------------------

def run_trials(
    kneser_n: int,
    kneser_k: int,
    lift_size: int,
    trials: int,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    out = []

    for t in range(trials):
        s = rng.randrange(10**9)
        result = run_one_experiment(
            kneser_n=kneser_n,
            kneser_k=kneser_k,
            lift_size=lift_size,
            seed=s,
        )
        result["trial"] = t
        result["seed"] = s
        out.append(result)

    return out

# ------------------------------------------------------------
# Save results to file
# ------------------------------------------------------------

def append_distribution_to_ordered_json(
    trials: List[Dict[str, Any]],
    filename: str = "kneser_lift_results.json",
) -> str:
    """
    Accumulate results into one JSON file organized as:
        data[n][k][r] = {
            "total_experiments": ...,
            "total_trials": ...,
            "distribution": {...}
        }

    Only stores aggregate information, not individual trial details.
    """
    if not trials:
        raise ValueError("No trials to save")

    params = trials[0]
    n = str(params["kneser_n"])
    k = str(params["kneser_k"])
    r = str(params["lift_size"])

    new_distribution = {}
    for trial in trials:
        chi = str(trial["chi_lift"])
        new_distribution[chi] = new_distribution.get(chi, 0) + 1

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    if n not in data:
        data[n] = {}
    if k not in data[n]:
        data[n][k] = {}
    if r not in data[n][k]:
        data[n][k][r] = {
            "total_experiments": 0,
            "total_trials": 0,
            "distribution": {}
        }

    entry = data[n][k][r]

    entry["total_experiments"] += 1
    entry["total_trials"] += len(trials)

    for chi, count in new_distribution.items():
        entry["distribution"][chi] = entry["distribution"].get(chi, 0) + count

    # sort outer keys numerically
    ordered_data = {
        nk: {
            kk: {
                rk: {
                    "total_experiments": data[nk][kk][rk]["total_experiments"],
                    "total_trials": data[nk][kk][rk]["total_trials"],
                    "distribution": dict(
                        sorted(
                            data[nk][kk][rk]["distribution"].items(),
                            key=lambda x: int(x[0])
                        )
                    ),
                }
                for rk in sorted(data[nk][kk], key=int)
            }
            for kk in sorted(data[nk], key=int)
        }
        for nk in sorted(data, key=int)
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(ordered_data, f, indent=4)

    return os.path.abspath(filename)


if __name__ == "__main__":
    # # Several trials
    master_seed = None
    start = time.perf_counter()

    trials = run_trials(
        kneser_n=10,
        kneser_k=2,
        lift_size=2,
        trials=4,
        seed=master_seed,
    )
    end = time.perf_counter()

    params = trials[0]
    print(f"Base graph : KG(n={params['kneser_n']}, k={params['kneser_k']})")
    print(f"chi(base)  : {params['chi_base']}")
    print(f"Lift size  : {params['lift_size']}")
    print(f"Trials     : {len(trials)}")

    print("\nTrial summary:")
    for r in trials:
        print(
            f"trial={r['trial']}, seed={r['seed']}, "
            f"chi(base)={r['chi_base']}, chi(lift)={r['chi_lift']}"
        )

    distribution = {}
    for r in trials:
        chi = r["chi_lift"]
        distribution[chi] = distribution.get(chi, 0) + 1

    print("\nChromatic number distribution:")
    print(distribution)
    print(f"\nTotal runtime: {end - start:.4f} seconds")
    if master_seed is None:
        saved_file = append_distribution_to_ordered_json(trials)

    # # Exhaustively check chromatic number of ALL lifts for each config.
    # configs = [
    #     (5, 2, 2),   # (kneser_n, kneser_k, lift_size)
    # ]

    # # Optional: print how many lifts each config will enumerate first,
    # # so you don't accidentally launch an intractable run.
    # for (n, k, r) in configs:
    #     base = kneser_graph(n, k)
    #     print(f"KG({n},{k}) lift_size={r}: {count_all_lifts(base, r)} lifts")

    # summaries = run_exhaustive_sweep(configs, gauge_fix=True, verbose=True)

    # print("\n=== Exhaustive results ===")
    # for s in summaries:
    #     print(
    #         f"KG(n={s['kneser_n']}, k={s['kneser_k']}), lift_size={s['lift_size']}: "
    #         f"chi(base)={s['chi_base']}, "
    #         f"chi(lift) distribution={s['distribution']}, "
    #         f"min={s['min_chi']}, max={s['max_chi']}"
    #     )


