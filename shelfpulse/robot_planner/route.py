"""Order a mission's bays into a short walk on the store's aisle graph.

Nearest neighbour from the dock, then 2-opt until no swap shortens the walk (a mission has at
most 8 bays, so this is instant). Distances are walking distances from `store_map`.
"""

from __future__ import annotations

from functools import cache

from shelfpulse.layout.store_map import DOCK, StoreMap


def order(bays: list[str], smap: StoreMap) -> list[str]:
    if len(bays) <= 1:
        return list(bays)

    @cache
    def d(a: str, b: str) -> float:
        return smap.shortest_path_m(a, b)

    left, path, here = set(bays), [], DOCK
    while left:
        here = min(left, key=lambda b: (d(here, b), b))
        path.append(here)
        left.remove(here)

    def length(p: list[str]) -> float:
        stops = [DOCK, *p, DOCK]
        return sum(d(a, b) for a, b in zip(stops, stops[1:], strict=False))

    improved = True
    while improved:
        improved = False
        for i in range(len(path) - 1):
            for j in range(i + 1, len(path)):
                cand = path[:i] + path[i : j + 1][::-1] + path[j + 1 :]
                if length(cand) < length(path) - 1e-9:
                    path, improved = cand, True
    return path


def walk_m(bays: list[str], smap: StoreMap) -> float:
    """Dock -> bays in order -> dock, metres."""
    stops = [DOCK, *bays, DOCK]
    return sum(smap.shortest_path_m(a, b) for a, b in zip(stops, stops[1:], strict=False))
