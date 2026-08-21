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


# ------------------------------------------------------------
# D. 2-lift structural metrics
# Compare lifts with different chromatic numbers.
#
# Design (agreed):
#   - base parity metrics: enumerate base simple cycles (len 3..6) ONCE
#     per (n,k), cache them, then per trial just XOR the crossed-edge
#     indicator around each cycle. Exact and cheap.
#   - lift cycle counts: computed DIRECTLY on the lifted graph (not derived
#     from base-cycle parity, since a simple lift cycle can project to a
#     non-simple base walk, e.g. a 6-cycle from two odd-parity triangles
#     sharing a vertex).
#   - odd_girth_lift: computed DIRECTLY on the lift via BFS.
#
# Nothing in section C (chromatic number) is touched.
# ------------------------------------------------------------

from collections import deque

# cache of base simple cycles, keyed by (n, k, max_cycle_len)
_BASE_CYCLE_CACHE: Dict[Tuple[int, int, int], Dict[int, list]] = {}


def _walk_simple_cycles(G: nx.Graph, max_len: int, on_cycle) -> None:
    """
    Enumerate every simple cycle of length 3..max_len exactly once, calling
    on_cycle(path) with the list of vertices in cyclic order.

    Each cycle is anchored at its minimum-index vertex and emitted in a single
    canonical direction (path[1] < path[-1]), so no cycle is reported twice.
    """
    nodes = list(G.nodes())
    index = {v: i for i, v in enumerate(nodes)}
    adj = {v: list(G.neighbors(v)) for v in nodes}

    def dfs(start, current, path, visited):
        si = index[start]
        for nxt in adj[current]:
            if index[nxt] < si:
                continue
            if nxt == start:
                if len(path) >= 3 and index[path[1]] < index[path[-1]]:
                    on_cycle(path)
                continue
            if nxt in visited or len(path) >= max_len:
                continue
            visited.add(nxt)
            path.append(nxt)
            dfs(start, nxt, path, visited)
            path.pop()
            visited.discard(nxt)

    for start in nodes:
        dfs(start, start, [start], {start})


def _count_simple_cycles(G: nx.Graph, max_len: int) -> Dict[int, int]:
    """Count simple cycles of each length 3..max_len (no storage)."""
    counts = {L: 0 for L in range(3, max_len + 1)}

    def cb(path):
        counts[len(path)] += 1

    _walk_simple_cycles(G, max_len, cb)
    return counts


def _base_cycle_edge_sets(G: nx.Graph, max_len: int) -> Dict[int, list]:
    """
    Enumerate base simple cycles and store each as a list of edge keys
    (frozenset({u, v})), grouped by cycle length. Used for parity metrics.
    """
    cycles = {L: [] for L in range(3, max_len + 1)}

    def cb(path):
        L = len(path)
        edges = [frozenset((path[i], path[(i + 1) % L])) for i in range(L)]
        cycles[L].append(edges)

    _walk_simple_cycles(G, max_len, cb)
    return cycles


def _get_base_cycles(base_graph: nx.Graph, n: int, k: int,
                     max_cycle_len: int) -> Dict[int, list]:
    """Return cached base simple cycles for (n, k), computing them once."""
    key = (n, k, max_cycle_len)
    if key not in _BASE_CYCLE_CACHE:
        _BASE_CYCLE_CACHE[key] = _base_cycle_edge_sets(base_graph, max_cycle_len)
    return _BASE_CYCLE_CACHE[key]


def _odd_girth(G: nx.Graph) -> Optional[int]:
    """
    Length of the shortest odd cycle in G, or None if G is bipartite.

    Uses the bipartite double cover: the shortest odd closed walk through a
    vertex s is the distance from (s, 0) to (s, 1) in the cover, and the
    shortest such walk is an odd cycle. We BFS from each vertex and keep the
    minimum.
    """
    adj = {v: list(G.neighbors(v)) for v in G.nodes()}
    best: Optional[int] = None

    for s in G.nodes():
        # BFS in the double cover from (s, 0), looking for (s, 1)
        dist = {(s, 0): 0}
        dq = deque([(s, 0)])
        found = None
        while dq:
            v, p = dq.popleft()
            d = dist[(v, p)]
            if best is not None and d + 1 >= best:
                continue
            for w in adj[v]:
                if w == s and p == 0:
                    found = d + 1
                    dq.clear()
                    break
                state = (w, 1 - p)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
            if found is not None:
                break
        if found is not None:
            best = found if best is None else min(best, found)
            if best == 3:
                break  # cannot do better than a triangle

    return best


def compute_lift_metrics(result: Dict[str, Any],
                         max_cycle_len: int = 6,
                         include_crossed_edges: bool = True) -> Dict[str, Any]:
    """
    Structural metrics for one lift experiment (see section header).

    Crossed-edge and base-parity metrics are only defined for 2-lifts; for
    lift_size != 2 they are stored as None. Lift cycle counts and odd girth
    are computed directly on the lifted graph for any lift size.

    Cycle lengths greater than max_cycle_len are NOT computed and are stored
    as None (never 0), so "not measured" is never confused with a true zero.

    include_crossed_edges controls whether the (gauge-dependent, verbose)
    list of crossed base edges is stored. It is redundant for reproduction
    since each sample stores `seed`, and (n, k, r, seed) regenerates the exact
    lift; set it False for leaner files. num_crossed_edges is always stored.
    """
    n = result["kneser_n"]
    k = result["kneser_k"]
    r = result["lift_size"]
    base = result["base_graph"]
    lift = result["lift_graph"]
    perms = result["permutations"]

    metrics: Dict[str, Any] = {
        "kneser_n": n,
        "kneser_k": k,
        "lift_size": r,
        "chi_base": result["chi_base"],
        "chi_lift": result["chi_lift"],
        "trial": result.get("trial"),
        "seed": result.get("seed"),
    }

    # --- lift cycle counts (direct on the lift) ---
    lift_counts = _count_simple_cycles(lift, max_cycle_len)

    def lift_count(L: int):
        return lift_counts.get(L, 0) if L <= max_cycle_len else None

    metrics["triangle_count_lift"] = lift_count(3)
    metrics["num_4_cycles_lift"] = lift_count(4)
    metrics["num_5_cycles_lift"] = lift_count(5)
    metrics["num_6_cycles_lift"] = lift_count(6)
    metrics["odd_girth_lift"] = _odd_girth(lift)

    # --- crossed-edge + base parity metrics (2-lifts only) ---
    if r == 2:
        identity = list(range(r))
        crossed_map: Dict[frozenset, bool] = {}
        crossed_edges = []
        for e, p in perms.items():
            is_crossed = list(p) != identity
            crossed_map[frozenset(e)] = is_crossed
            if is_crossed:
                crossed_edges.append(list(e))

        metrics["crossed_edges"] = crossed_edges if include_crossed_edges else None
        metrics["num_crossed_edges"] = len(crossed_edges)

        base_cycles = _get_base_cycles(base, n, k, max_cycle_len)

        def odd_parity_count(L: int):
            if L > max_cycle_len:
                return None
            total = 0
            for edges in base_cycles.get(L, []):
                parity = 0
                for e in edges:
                    if crossed_map.get(e, False):
                        parity ^= 1
                total += parity
            return total

        metrics["odd_parity_base_triangles"] = odd_parity_count(3)
        metrics["odd_parity_base_4_cycles"] = odd_parity_count(4)
        metrics["odd_parity_base_5_cycles"] = odd_parity_count(5)
        metrics["odd_parity_base_6_cycles"] = odd_parity_count(6)
    else:
        metrics["crossed_edges"] = None
        metrics["num_crossed_edges"] = None
        metrics["odd_parity_base_triangles"] = None
        metrics["odd_parity_base_4_cycles"] = None
        metrics["odd_parity_base_5_cycles"] = None
        metrics["odd_parity_base_6_cycles"] = None

    return metrics


# numeric metrics worth summarizing per chi_lift group (identifiers like
# trial/seed and constants like kneser_n are intentionally excluded)
_SUMMARY_METRIC_KEYS = [
    "num_crossed_edges",
    "triangle_count_lift",
    "num_4_cycles_lift",
    "num_5_cycles_lift",
    "num_6_cycles_lift",
    "odd_girth_lift",
    "odd_parity_base_triangles",
    "odd_parity_base_4_cycles",
    "odd_parity_base_5_cycles",
    "odd_parity_base_6_cycles",
]


class _Inline(dict):
    """A dict marked to be serialized on a single line in the metrics file."""
    pass


def _dump_metrics_file(structure: Any, filename: str) -> str:
    """
    Write `structure` as JSON with indent=4, except dicts marked _Inline are
    kept on a single line. This lets each per-metric summary occupy one line
    (so chi_lift blocks stay short and comparable) while the outer nesting
    stays readable.
    """
    placeholders: Dict[str, str] = {}

    def encode(o):
        if isinstance(o, _Inline):
            token = f"@@INLINE{len(placeholders)}@@"
            placeholders[token] = json.dumps(o, separators=(", ", ": "))
            return token
        if isinstance(o, dict):
            return {key: encode(val) for key, val in o.items()}
        if isinstance(o, list):
            return [encode(val) for val in o]
        return o

    text = json.dumps(encode(structure), indent=4)
    for token, compact in placeholders.items():
        text = text.replace('"' + token + '"', compact)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)
    return os.path.abspath(filename)


# --- running aggregates: let us update mean/std incrementally without ever
# --- keeping the individual samples around.

def _new_agg() -> Dict[str, Any]:
    return {"n": 0, "sum": 0, "sumsq": 0, "min": None, "max": None}


def _agg_add(agg: Dict[str, Any], v) -> None:
    agg["n"] += 1
    agg["sum"] += v
    agg["sumsq"] += v * v
    agg["min"] = v if agg["min"] is None else min(agg["min"], v)
    agg["max"] = v if agg["max"] is None else max(agg["max"], v)


def _agg_to_leaf(agg: Optional[Dict[str, Any]]):
    """Render a running aggregate as a compact one-line summary, or None.

    mean/std/min/max are the human-readable values; n/sum/sumsq are kept so the
    aggregate can be merged on the next append without needing the samples.
    """
    if not agg or agg["n"] == 0:
        return None
    n = agg["n"]
    mean = agg["sum"] / n
    if n > 1:
        var = (agg["sumsq"] - agg["sum"] ** 2 / n) / (n - 1)
        std = max(var, 0.0) ** 0.5
    else:
        std = 0.0
    return _Inline({
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": agg["min"],
        "max": agg["max"],
        "n": n,
        "sum": agg["sum"],
        "sumsq": agg["sumsq"],
    })


def _load_group(block: Dict[str, Any]):
    """
    Recover (num_trials, aggregates, samples) from a stored chi_lift block,
    accepting either the new aggregate format (summary leaves carry sum/sumsq)
    or the older sample-based format (rebuild aggregates from samples).
    """
    num_trials = block.get("num_trials", 0)
    aggs: Dict[str, Dict[str, Any]] = {}
    samples = block.get("samples") or []
    summary = block.get("summary")

    if summary and any(isinstance(v, dict) and "sum" in v for v in summary.values()):
        for key, leaf in summary.items():
            if isinstance(leaf, dict) and "sum" in leaf and "sumsq" in leaf:
                aggs[key] = {
                    "n": leaf["n"], "sum": leaf["sum"], "sumsq": leaf["sumsq"],
                    "min": leaf["min"], "max": leaf["max"],
                }
    elif samples:
        for key in _SUMMARY_METRIC_KEYS:
            agg = _new_agg()
            for s in samples:
                v = s.get(key)
                if v is not None:
                    _agg_add(agg, v)
            if agg["n"] > 0:
                aggs[key] = agg

    return num_trials, aggs, samples


def _load_all_groups(data: Dict[str, Any]) -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for n in data:
        for k in data[n]:
            for r in data[n][k]:
                for chi in data[n][k][r]:
                    nt, aggs, samples = _load_group(data[n][k][r][chi])
                    groups[(n, k, r, chi)] = {
                        "num_trials": nt, "aggs": aggs, "samples": samples,
                    }
    return groups


def _write_groups(groups: Dict[Tuple[str, str, str, str], Dict[str, Any]],
                  filename: str, store_samples: bool) -> str:
    nested: Dict[str, Any] = {}
    for (n, k, r, chi), g in groups.items():
        summary = {key: _agg_to_leaf(g["aggs"].get(key))
                   for key in _SUMMARY_METRIC_KEYS}
        block: Dict[str, Any] = {"num_trials": g["num_trials"], "summary": summary}
        if store_samples and g.get("samples"):
            block["samples"] = g["samples"]
        nested.setdefault(n, {}).setdefault(k, {}).setdefault(r, {})[chi] = block

    ordered = {
        nn: {
            kk: {
                rr: {cc: nested[nn][kk][rr][cc]
                     for cc in sorted(nested[nn][kk][rr], key=int)}
                for rr in sorted(nested[nn][kk], key=int)
            }
            for kk in sorted(nested[nn], key=int)
        }
        for nn in sorted(nested, key=int)
    }
    return _dump_metrics_file(ordered, filename)


def append_metrics_to_ordered_json(
    trials: List[Dict[str, Any]],
    filename: str = "kneser_lift_metrics.json",
    max_cycle_len: int = 6,
    store_samples: bool = False,
    include_crossed_edges: bool = True,
) -> str:
    """
    Accumulate per-trial structural metrics into one JSON file organized as:
        data[n][k][r][chi_lift] = {
            "num_trials": ...,
            "summary": {metric -> {mean, std, min, max, n, sum, sumsq}},
            "samples": [...]   # only if store_samples=True
        }

    Each metric's summary is one line, so chi_lift blocks stay short and easy
    to compare without scrolling. Statistics are maintained as running
    aggregates (the sum/sumsq fields), so they update correctly across repeated
    appends even though the individual samples are NOT stored by default.

    store_samples=True additionally keeps the raw per-trial records at the end
    of each block (after the summary). include_crossed_edges applies only then.
    """
    if not trials:
        raise ValueError("No trials to save")

    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}

    groups = _load_all_groups(data)

    for result in trials:
        m = compute_lift_metrics(
            result,
            max_cycle_len=max_cycle_len,
            include_crossed_edges=include_crossed_edges,
        )
        key = (str(m["kneser_n"]), str(m["kneser_k"]),
               str(m["lift_size"]), str(m["chi_lift"]))
        g = groups.setdefault(
            key, {"num_trials": 0, "aggs": {}, "samples": []}
        )
        g["num_trials"] += 1
        for mk in _SUMMARY_METRIC_KEYS:
            v = m.get(mk)
            if v is None:
                continue
            g["aggs"].setdefault(mk, _new_agg())
            _agg_add(g["aggs"][mk], v)
        if store_samples:
            g["samples"].append(m)

    return _write_groups(groups, filename, store_samples=store_samples)


def print_metrics_summary(
    filename: str = "kneser_lift_metrics.json",
    columns: Optional[List[str]] = None,
) -> None:
    """
    Print one compact "mean ± std" table per (n, k, r), with one row per
    chi_lift. Reads the summary blocks written by append_metrics_to_ordered_json.

    By default it shows the most meaningful discriminators (base parity metrics
    + odd girth); pass `columns` to choose other metric keys.
    """
    if columns is None:
        columns = [
            "odd_parity_base_triangles",
            "odd_parity_base_4_cycles",
            "odd_parity_base_5_cycles",
            "odd_parity_base_6_cycles",
            "odd_girth_lift",
        ]

    short = {
        "odd_parity_base_triangles": "odd_tri",
        "odd_parity_base_4_cycles": "odd_4c",
        "odd_parity_base_5_cycles": "odd_5c",
        "odd_parity_base_6_cycles": "odd_6c",
        "odd_girth_lift": "odd_girth",
        "num_crossed_edges": "ncross",
        "triangle_count_lift": "tri",
        "num_4_cycles_lift": "4c",
        "num_5_cycles_lift": "5c",
        "num_6_cycles_lift": "6c",
    }

    def cell(stats) -> str:
        if stats is None:
            return "-"
        return f"{stats['mean']:.1f}+/-{stats['std']:.1f}"

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    for nn in sorted(data, key=int):
        for kk in sorted(data[nn], key=int):
            for rr in sorted(data[nn][kk], key=int):
                group = data[nn][kk][rr]
                header = ["chi", "trials"] + [short.get(c, c) for c in columns]
                widths = [max(len(h), 11) for h in header]
                print(f"\nKG(n={nn}, k={kk}), lift_size={rr}")
                print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
                for chi in sorted(group, key=int):
                    blk = group[chi]
                    # use stored summary, falling back to rebuilding from
                    # aggregates/samples for any older file layout
                    summary = blk.get("summary")
                    if not summary:
                        _, aggs, _ = _load_group(blk)
                        summary = {mk: _agg_to_leaf(aggs.get(mk))
                                   for mk in _SUMMARY_METRIC_KEYS}
                    row = [chi, str(blk["num_trials"])]
                    row += [cell(summary.get(c)) for c in columns]
                    print("  ".join(c.ljust(w) for c, w in zip(row, widths)))


def resummarize_metrics_file(
    filename: str = "kneser_lift_metrics.json",
    store_samples: bool = False,
) -> str:
    """
    Rewrite an existing metrics file into the current compact format: one
    line per metric, running aggregates, samples dropped by default.

    Works on any older layout (sample-based or earlier summary blocks); use it
    once to migrate a file, or to drop samples from a file saved with
    store_samples=True.
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    groups = _load_all_groups(data)
    return _write_groups(groups, filename, store_samples=store_samples)


if __name__ == "__main__":
    # # Several trials
    master_seed = None
    # Longest cycle length to measure in the metrics. Drop to 5 (or 4) if
    # 6-cycle counting on the lift is too slow at larger n; uncomputed
    # lengths are stored as None, not 0.
    max_cycle_len = 6
    start = time.perf_counter()

    trials = run_trials(
        kneser_n=7,
        kneser_k=2,
        lift_size=2,
        trials=100,
        seed=master_seed,
    )
    end_trials = time.perf_counter()

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
    print(f"\nChromatic runtime (trials only): {end_trials - start:.4f} seconds")
    if master_seed is None:
        saved_file = append_distribution_to_ordered_json(trials)
        metrics_file = append_metrics_to_ordered_json(
            trials, max_cycle_len=max_cycle_len,
            include_crossed_edges=False,
        )
        print(f"Saved metrics to: {metrics_file}")
    else:
        print("Used fixed seed, so results were not saved.")

    end = time.perf_counter()
    print(f"Total runtime (incl. metrics): {end - start:.4f} seconds")

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


