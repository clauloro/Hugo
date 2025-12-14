from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
import math
import random
import heapq

Node = int

# =========================
# Datos del problema
# =========================

@dataclass
class ProblemData:
    # Grafo (no dirigido) con tiempos de viaje
    adj: Dict[Node, List[Tuple[Node, float]]]

    depot: Node
    repair_nodes: Set[Node]                 # nodos de reparación (carreteras dañadas transformadas)
    demand_nodes: Set[Node]                 # nodos de demanda
    repair_time: Dict[Node, float]          # s_i
    demand_service_time: Dict[Node, float]  # s'_i

@dataclass
class ACOParams:
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.2
    Q: float = 1.0
    num_ants: int = 40
    num_iters: int = 150
    seed: int = 7

@dataclass
class Solution:
    repair_route: List[Node]
    relief_route: List[Node]
    d_repair: Dict[Node, float]         # tiempos de reparación (fin)
    relief_finish_time: float           # objetivo: último tiempo de servicio

# =========================
# Utilidades de grafo
# =========================

def add_undirected_edge(adj: Dict[Node, List[Tuple[Node, float]]], u: Node, v: Node, w: float):
    adj.setdefault(u, []).append((v, w))
    adj.setdefault(v, []).append((u, w))

def dijkstra_static(adj: Dict[Node, List[Tuple[Node, float]]], source: Node) -> Dict[Node, float]:
    """Dijkstra estándar: distancias mínimas sin ventanas."""
    dist: Dict[Node, float] = {source: 0.0}
    pq = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist.get(u, math.inf):
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist

def earliest_arrival_dijkstra(
    adj: Dict[Node, List[Tuple[Node, float]]],
    source: Node,
    start_time: float,
    repair_done_time: Dict[Node, float],
    repair_nodes: Set[Node],
) -> Dict[Node, float]:
    """
    Dijkstra de 'tiempo más temprano de llegada' con esperas.
    Si se llega a un nodo de reparación v antes de repair_done_time[v], se espera hasta repair_done_time[v].
    Propiedad FIFO -> Dijkstra funciona.
    """
    dist: Dict[Node, float] = {source: start_time}
    pq = [(start_time, source)]

    def apply_wait(node: Node, t: float) -> float:
        if node in repair_nodes:
            return max(t, repair_done_time.get(node, 0.0))
        return t

    # Si source es de reparación y aún no está reparado, también se esperaría (raro, pero coherente)
    dist[source] = apply_wait(source, dist[source])

    while pq:
        t_u, u = heapq.heappop(pq)
        if t_u != dist.get(u, math.inf):
            continue
        for v, w in adj.get(u, []):
            t_v = t_u + w
            t_v = apply_wait(v, t_v)
            if t_v < dist.get(v, math.inf):
                dist[v] = t_v
                heapq.heappush(pq, (t_v, v))
    return dist

def roulette_choice(candidates: List[Node], weights: List[float]) -> Node:
    s = sum(weights)
    if s <= 0:
        return random.choice(candidates)
    r = random.random() * s
    acc = 0.0
    for c, w in zip(candidates, weights):
        acc += w
        if acc >= r:
            return c
    return candidates[-1]

# =========================
# ACO (adaptado al paper)
# =========================

class IOSRCRVD_ACO:
    def __init__(self, data: ProblemData, params: ACOParams):
        self.data = data
        self.p = params
        random.seed(self.p.seed)

        # Feromonas (simplificación razonable para informe):
        # sobre transiciones "siguiente nodo a visitar" en (i->j) para reparación y para demanda.
        self.tau_rep: Dict[Tuple[Node, Node], float] = {}
        self.tau_dem: Dict[Tuple[Node, Node], float] = {}
        self._init_pheromones()

        # Precomputo distancias estáticas para heurística rápida (equipo reparación)
        self.static_dist_cache: Dict[Node, Dict[Node, float]] = {}

    def _init_pheromones(self):
        init = 1.0
        rep = list(self.data.repair_nodes) + [self.data.depot]
        dem = list(self.data.demand_nodes) + [self.data.depot]
        for i in rep:
            for j in rep:
                if i != j:
                    self.tau_rep[(i, j)] = init
        for i in dem:
            for j in dem:
                if i != j:
                    self.tau_dem[(i, j)] = init

    def _heuristic(self, cost: float) -> float:
        return 1.0 / max(cost, 1e-6)

    def _static_dist(self, src: Node) -> Dict[Node, float]:
        if src not in self.static_dist_cache:
            self.static_dist_cache[src] = dijkstra_static(self.data.adj, src)
        return self.static_dist_cache[src]

    # ---- 1) Construir ruta del equipo de reparación y tiempos d_i ----
    def build_repair_route_and_times(self) -> Tuple[List[Node], Dict[Node, float]]:
        route = [self.data.depot]
        remaining = set(self.data.repair_nodes)

        current = self.data.depot
        t = 0.0
        d_repair: Dict[Node, float] = {}

        while remaining:
            dist_cur = self._static_dist(current)
            candidates = [v for v in remaining if v in dist_cur and math.isfinite(dist_cur[v])]
            if not candidates:
                # no debería ocurrir si el grafo está conectado
                return route, {v: math.inf for v in self.data.repair_nodes}

            weights = []
            for v in candidates:
                tau = (self.tau_rep.get((current, v), 1.0) ** self.p.alpha)
                eta = (self._heuristic(dist_cur[v]) ** self.p.beta)
                weights.append(tau * eta)

            nxt = roulette_choice(candidates, weights)

            # viajar y reparar
            t += dist_cur[nxt]
            t += self.data.repair_time.get(nxt, 0.0)
            d_repair[nxt] = t  # tiempo de reparación completada (ventana)
            route.append(nxt)

            remaining.remove(nxt)
            current = nxt

        return route, d_repair

    # ---- 2) Construir ruta del vehículo de ayuda con ventanas (esperas) ----
    def build_relief_route_and_finish_time(self, d_repair: Dict[Node, float]) -> Tuple[List[Node], float]:
        route = [self.data.depot]
        remaining = set(self.data.demand_nodes)

        current = self.data.depot
        t = 0.0  # tiempo actual del vehículo de ayuda
        finish_time_last = 0.0

        while remaining:
            # distancias con ventanas desde el estado actual
            dist_time = earliest_arrival_dijkstra(
                self.data.adj, current, t, d_repair, self.data.repair_nodes
            )

            candidates = [v for v in remaining if v in dist_time and math.isfinite(dist_time[v])]
            if not candidates:
                return route, math.inf

            weights = []
            for v in candidates:
                # coste "efectivo": llegada considerando esperas
                arrival = dist_time[v]
                travel_effective = max(arrival - t, 0.0)

                tau = (self.tau_dem.get((current, v), 1.0) ** self.p.alpha)
                eta = (self._heuristic(travel_effective) ** self.p.beta)
                weights.append(tau * eta)

            nxt = roulette_choice(candidates, weights)

            # actualizar tiempo: llegar (con esperas) + servicio
            arrival_time = dist_time[nxt]
            t = arrival_time + self.data.demand_service_time.get(nxt, 0.0)
            finish_time_last = t

            route.append(nxt)
            remaining.remove(nxt)
            current = nxt

        # objetivo: tiempo del último servicio completado
        return route, finish_time_last

    # ---- Feromonas ----
    def evaporate(self):
        for k in list(self.tau_rep.keys()):
            self.tau_rep[k] *= (1.0 - self.p.rho)
        for k in list(self.tau_dem.keys()):
            self.tau_dem[k] *= (1.0 - self.p.rho)

    def reinforce_global_best(self, sol: Solution):
        if not math.isfinite(sol.relief_finish_time) or sol.relief_finish_time <= 0:
            return
        delta = self.p.Q / sol.relief_finish_time

        for i in range(len(sol.repair_route) - 1):
            a, b = sol.repair_route[i], sol.repair_route[i + 1]
            self.tau_rep[(a, b)] = self.tau_rep.get((a, b), 0.0) + delta

        for i in range(len(sol.relief_route) - 1):
            a, b = sol.relief_route[i], sol.relief_route[i + 1]
            self.tau_dem[(a, b)] = self.tau_dem.get((a, b), 0.0) + delta

    def solve(self) -> Solution:
        best = Solution([], [], {}, math.inf)

        for _ in range(self.p.num_iters):
            iter_best = Solution([], [], {}, math.inf)

            for _k in range(self.p.num_ants):
                rep_route, d_rep = self.build_repair_route_and_times()
                rel_route, finish = self.build_relief_route_and_finish_time(d_rep)

                sol = Solution(rep_route, rel_route, d_rep, finish)
                if sol.relief_finish_time < iter_best.relief_finish_time:
                    iter_best = sol

            # update
            self.evaporate()
            self.reinforce_global_best(iter_best)

            if iter_best.relief_finish_time < best.relief_finish_time:
                best = iter_best

        return best

# =========================
# EJEMPLO PEQUEÑO (informe)
# =========================

def build_example() -> ProblemData:
    # Nodos: 0 (depósito), 1,2,4 (tránsito),
    # 6 y 7 (reparación: arcos dañados transformados),
    # 3 y 5 (demanda).
    adj: Dict[Node, List[Tuple[Node, float]]] = {}

    # Tiempos de viaje del ejemplo transformado:
    # 0-1:4, 1-6:2, 6-2:2, 2-3:3, 2-7:2, 7-4:2, 4-5:3
    add_undirected_edge(adj, 0, 1, 4)
    add_undirected_edge(adj, 1, 6, 2)
    add_undirected_edge(adj, 6, 2, 2)
    add_undirected_edge(adj, 2, 3, 3)
    add_undirected_edge(adj, 2, 7, 2)
    add_undirected_edge(adj, 7, 4, 2)
    add_undirected_edge(adj, 4, 5, 3)

    repair_nodes = {6, 7}
    demand_nodes = {3, 5}

    # Tiempos de reparación (s_i)
    repair_time = {6: 6.0, 7: 5.0}

    # Tiempos de servicio de demanda (s'_i)
    demand_service_time = {3: 4.0, 5: 3.0}

    return ProblemData(
        adj=adj,
        depot=0,
        repair_nodes=repair_nodes,
        demand_nodes=demand_nodes,
        repair_time=repair_time,
        demand_service_time=demand_service_time,
    )

if __name__ == "__main__":
    data = build_example()
    params = ACOParams(alpha=1.0, beta=2.0, rho=0.2, Q=1.0, num_ants=50, num_iters=200, seed=7)
    aco = IOSRCRVD_ACO(data, params)
    best = aco.solve()

    print("=== Mejor solución encontrada (ACO) ===")
    print("Ruta equipo de reparación:", best.repair_route)
    print("Tiempos de reparación d_i:", {k: round(v, 2) for k, v in best.d_repair.items()})
    print("Ruta vehículo de ayuda:", best.relief_route)
    print("Tiempo final del último servicio (objetivo):", round(best.relief_finish_time, 2))
