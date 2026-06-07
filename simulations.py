import itertools
import math
import random
from typing import Dict, List, Optional, Tuple, Any

import networkx as nx


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
        out.append(result)

    return out


if __name__ == "__main__":
    # Several trials
    trials = run_trials(
        kneser_n=8,
        kneser_k=2,
        lift_size=3,
        trials=5,
        seed=7,
    )

    params = trials[0]
    print(f"Base graph : KG(n={params['kneser_n']}, k={params['kneser_k']})")
    print(f"chi(base)  : {params['chi_base']}")
    print(f"Lift size  : {params['lift_size']}")
    print(f"Trials     : {len(trials)}")

    print("\nTrial summary:")
    for r in trials:
        print(f"trial={r['trial']}, chi(base)={r['chi_base']}, chi(lift)={r['chi_lift']}")