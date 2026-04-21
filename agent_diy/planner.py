#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import heapq
import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from agent_diy.conf.conf import Config


DIRS = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]

DIR_TO_ACTION = {
    (1, 0): 0,
    (1, -1): 1,
    (0, -1): 2,
    (-1, -1): 3,
    (-1, 0): 4,
    (-1, 1): 5,
    (0, 1): 6,
    (1, 1): 7,
}


def _clip_pos(x: int, z: int) -> Tuple[int, int]:
    return int(np.clip(x, 0, Config.MAP_W - 1)), int(np.clip(z, 0, Config.MAP_H - 1))


def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    dx = a[0] - b[0]
    dz = a[1] - b[1]
    return float(math.hypot(dx, dz))


class TraditionalPlanner:
    def __init__(self):
        self.static_blocked = np.zeros((Config.MAP_H, Config.MAP_W), dtype=np.bool_)
        self.dynamic_risk = np.zeros((Config.MAP_H, Config.MAP_W), dtype=np.bool_)
        self.soft_risk = np.zeros((Config.MAP_H, Config.MAP_W), dtype=np.float32)
        self.seen = np.zeros((Config.MAP_H, Config.MAP_W), dtype=np.bool_)

        self.cur_pos = (0, 0)
        self.battery = 100
        self.battery_max = 100
        self.packages: List[int] = []

        self.npcs: List[Tuple[int, int]] = []
        self.chargers: List[Tuple[int, int]] = []
        self.warehouse: Optional[Tuple[int, int]] = None
        self.stations: Dict[int, Tuple[int, int]] = {}

        self.low_battery_lock = False
        self.target_kind = "NONE"
        self.target_pos: Optional[Tuple[int, int]] = None
        self.target_lock_steps = 0

        self.last_action = 0
        self.last_path_len = 0
        self.replan_count = 0
        self.last_path_cost = 0.0
        self.path_cost_sum = 0.0
        self.path_cost_count = 0

        self.step_count = 0
        self.charge_enter_count = 0
        self.risk_degrade_count = 0
        self.stuck_recovery_count = 0
        self.target_switch_intervals = deque(maxlen=100)
        self.last_target_switch_step = 0
        self.recent_positions = deque(maxlen=max(int(Config.STUCK_WINDOW), 4))
        self.recent_targets = deque(maxlen=8)
        self.risk_degrade_until_step = 0

        # D* Lite state
        self.g: Dict[Tuple[int, int], float] = {}
        self.rhs: Dict[Tuple[int, int], float] = {}
        self.open_heap: List[Tuple[float, float, int, int]] = []
        self.km = 0.0
        self.s_start: Optional[Tuple[int, int]] = None
        self.s_goal: Optional[Tuple[int, int]] = None
        self.s_last: Optional[Tuple[int, int]] = None
        self.prev_traversable: Optional[np.ndarray] = None

    def reset(self):
        self.dynamic_risk.fill(False)
        self.soft_risk.fill(0.0)
        self.cur_pos = (0, 0)
        self.battery = 100
        self.battery_max = 100
        self.packages = []
        self.npcs = []
        self.chargers = []
        self.warehouse = None
        self.stations = {}
        self.low_battery_lock = False
        self.target_kind = "NONE"
        self.target_pos = None
        self.target_lock_steps = 0
        self.last_action = 0
        self.last_path_len = 0
        self.replan_count = 0
        self.last_path_cost = 0.0
        self.path_cost_sum = 0.0
        self.path_cost_count = 0

        self.step_count = 0
        self.charge_enter_count = 0
        self.risk_degrade_count = 0
        self.stuck_recovery_count = 0
        self.target_switch_intervals.clear()
        self.last_target_switch_step = 0
        self.recent_positions.clear()
        self.recent_targets.clear()
        self.risk_degrade_until_step = 0

        self.g = {}
        self.rhs = {}
        self.open_heap = []
        self.km = 0.0
        self.s_start = None
        self.s_goal = None
        self.s_last = None
        self.prev_traversable = None

    def update_from_env(self, env_obs: dict):
        self.step_count += 1
        obs = env_obs.get("observation", {})
        frame_state = obs.get("frame_state", {})

        hero = frame_state.get("heroes", {})
        hero_pos = hero.get("pos", {})
        self.cur_pos = _clip_pos(int(hero_pos.get("x", 0)), int(hero_pos.get("z", 0)))
        self.battery = int(hero.get("battery", self.battery))
        self.battery_max = int(hero.get("battery_max", self.battery_max))
        self.packages = list(hero.get("packages", []))

        self.npcs = []
        for npc in frame_state.get("npcs", []):
            p = npc.get("pos", {})
            self.npcs.append(_clip_pos(int(p.get("x", 0)), int(p.get("z", 0))))

        self.chargers = []
        self.stations = {}
        self.warehouse = None
        for organ in frame_state.get("organs", []):
            st = int(organ.get("sub_type", 0))
            pos = organ.get("pos", {})
            p = _clip_pos(int(pos.get("x", 0)), int(pos.get("z", 0)))
            if st == 1 and self.warehouse is None:
                self.warehouse = p
            elif st == 2:
                self.chargers.append(p)
            elif st == 3:
                sid = int(organ.get("config_id", 0))
                self.stations[sid] = p

        self._update_local_map(obs.get("map_info", {}))
        self._update_dynamic_risk()
        self.recent_positions.append(self.cur_pos)

    def _update_local_map(self, map_info):
        if isinstance(map_info, dict):
            local = map_info.get("map_info", None)
        else:
            local = map_info

        if not isinstance(local, list) or len(local) < 21 or not isinstance(local[0], list) or len(local[0]) < 21:
            return

        cx, cz = self.cur_pos
        for lz in range(21):
            for lx in range(21):
                gx, gz = _clip_pos(cx + (lx - 10), cz + (lz - 10))
                val = int(local[lz][lx])
                self.static_blocked[gz, gx] = val == 0
                self.seen[gz, gx] = True

    def _update_dynamic_risk(self):
        self.dynamic_risk.fill(False)
        self.soft_risk.fill(0.0)

        hard_rad = max(int(Config.NPC_DANGER_RADIUS), 0)
        buffer_rad = max(int(Config.NPC_BUFFER_RADIUS), 0)
        if self.step_count < self.risk_degrade_until_step:
            buffer_rad = max(0, buffer_rad - 1)
        hard_total = hard_rad + buffer_rad
        soft_outer = hard_total + max(int(Config.SOFT_RISK_RADIUS), 0)

        for nx, nz in self.npcs:
            for dz in range(-soft_outer, soft_outer + 1):
                for dx in range(-soft_outer, soft_outer + 1):
                    gx, gz = _clip_pos(nx + dx, nz + dz)
                    d = max(abs(dx), abs(dz))
                    if d <= hard_total:
                        self.dynamic_risk[gz, gx] = True
                        self.soft_risk[gz, gx] = 1.0
                    elif d <= soft_outer and soft_outer > hard_total:
                        risk = float(soft_outer - d) / float(max(soft_outer - hard_total, 1))
                        if risk > float(self.soft_risk[gz, gx]):
                            self.soft_risk[gz, gx] = risk

    def _traversable(self, x: int, z: int) -> bool:
        if x < 0 or x >= Config.MAP_W or z < 0 or z >= Config.MAP_H:
            return False
        return not (self.static_blocked[z, x] or self.dynamic_risk[z, x])

    def _traversable_grid(self) -> np.ndarray:
        return np.logical_not(np.logical_or(self.static_blocked, self.dynamic_risk))

    def _move_valid(self, x: int, z: int, dx: int, dz: int) -> bool:
        nx, nz = x + dx, z + dz
        if not self._traversable(nx, nz):
            return False
        if dx != 0 and dz != 0:
            if (not self._traversable(x + dx, z)) and (not self._traversable(x, z + dz)):
                return False
        return True

    def _neighbors(self, s: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, z = s
        out = []
        for dx, dz in DIRS:
            if self._move_valid(x, z, dx, dz):
                out.append((x + dx, z + dz))
        return out

    def _predecessors(self, s: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, z = s
        out = []
        for dx, dz in DIRS:
            px, pz = x - dx, z - dz
            if px < 0 or px >= Config.MAP_W or pz < 0 or pz >= Config.MAP_H:
                continue
            if self._move_valid(px, pz, dx, dz):
                out.append((px, pz))
        return out

    def _cost(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        dx = b[0] - a[0]
        dz = b[1] - a[1]
        if not self._move_valid(a[0], a[1], dx, dz):
            return float("inf")
        move_cost = 1.4142 if (dx != 0 and dz != 0) else 1.0
        risk_cost = Config.SOFT_RISK_WEIGHT * float(self.soft_risk[b[1], b[0]])
        backtrack_cost = 0.0
        if b in self.recent_positions:
            backtrack_cost = Config.COST_W_BACKTRACK
        return move_cost + risk_cost + backtrack_cost

    def _g(self, s: Tuple[int, int]) -> float:
        return self.g.get(s, float("inf"))

    def _rhs(self, s: Tuple[int, int]) -> float:
        return self.rhs.get(s, float("inf"))

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return _dist(a, b)

    def _calculate_key(self, s: Tuple[int, int]) -> Tuple[float, float]:
        g_rhs = min(self._g(s), self._rhs(s))
        return g_rhs + self._heuristic(self.s_start, s) + self.km, g_rhs

    def _push_open(self, s: Tuple[int, int]):
        k1, k2 = self._calculate_key(s)
        heapq.heappush(self.open_heap, (k1, k2, s[0], s[1]))

    def _top_key(self) -> Tuple[float, float]:
        while self.open_heap:
            k1, k2, x, z = self.open_heap[0]
            s = (x, z)
            ck1, ck2 = self._calculate_key(s)
            if abs(k1 - ck1) < 1e-9 and abs(k2 - ck2) < 1e-9:
                return k1, k2
            heapq.heappop(self.open_heap)
        return float("inf"), float("inf")

    def _update_vertex(self, u: Tuple[int, int]):
        if u != self.s_goal:
            min_rhs = float("inf")
            for s in self._neighbors(u):
                min_rhs = min(min_rhs, self._cost(u, s) + self._g(s))
            self.rhs[u] = min_rhs
        if abs(self._g(u) - self._rhs(u)) > 1e-9:
            self._push_open(u)

    def _init_dstar(self, start: Tuple[int, int], goal: Tuple[int, int]):
        self.g = {}
        self.rhs = {}
        self.open_heap = []
        self.km = 0.0
        self.s_start = start
        self.s_goal = goal
        self.s_last = start
        self.rhs[goal] = 0.0
        self._push_open(goal)

    def _compute_shortest_path(self):
        expansions = 0
        max_expand = max(int(Config.MAX_REPLAN_EXPANSIONS), 1000)
        while expansions < max_expand:
            top = self._top_key()
            start_key = self._calculate_key(self.s_start)
            if not (
                (top[0] < start_key[0])
                or (abs(top[0] - start_key[0]) < 1e-9 and top[1] < start_key[1])
                or (abs(self._rhs(self.s_start) - self._g(self.s_start)) > 1e-9)
            ):
                break

            if not self.open_heap:
                break

            k1, k2, x, z = heapq.heappop(self.open_heap)
            u = (x, z)
            new_k1, new_k2 = self._calculate_key(u)
            if (k1, k2) < (new_k1, new_k2):
                self._push_open(u)
            elif self._g(u) > self._rhs(u):
                self.g[u] = self._rhs(u)
                for p in self._predecessors(u):
                    self._update_vertex(p)
            else:
                self.g[u] = float("inf")
                self._update_vertex(u)
                for p in self._predecessors(u):
                    self._update_vertex(p)

            expansions += 1

    def _update_changed_cells(self, changed_cells: List[Tuple[int, int]]):
        if not changed_cells:
            return
        touched = set()
        for c in changed_cells:
            touched.add(c)
            for p in self._predecessors(c):
                touched.add(p)
        for u in touched:
            self._update_vertex(u)

    def _extract_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        if start == goal:
            return [start]
        if math.isinf(self._g(start)):
            return [start]

        path = [start]
        cur = start
        max_len = Config.MAP_W * Config.MAP_H
        for _ in range(max_len):
            if cur == goal:
                break
            cands = []
            for s in self._neighbors(cur):
                val = self._cost(cur, s) + self._g(s)
                cands.append((val, s))
            if not cands:
                break
            cands.sort(key=lambda x: x[0])
            if math.isinf(cands[0][0]):
                break
            nxt = cands[0][1]
            if nxt == cur:
                break
            path.append(nxt)
            cur = nxt

        return path

    def _nearest(self, points: List[Tuple[int, int]]) -> Tuple[Optional[Tuple[int, int]], float]:
        if not points:
            return None, 1e9
        p = min(points, key=lambda q: _dist(q, self.cur_pos))
        return p, _dist(p, self.cur_pos)

    def _energy_cost(self, target: Tuple[int, int]) -> float:
        dist = _dist(self.cur_pos, target)
        reserve = 0.2 * float(max(self.battery_max, 1))
        remain = float(self.battery) - dist - reserve
        return 0.0 if remain >= 0 else float(-remain) / float(max(self.battery_max, 1))

    def _risk_cost(self, target: Tuple[int, int]) -> float:
        tx, tz = target
        base = float(self.soft_risk[tz, tx])
        around = []
        for dx, dz in DIRS:
            x, z = tx + dx, tz + dz
            if 0 <= x < Config.MAP_W and 0 <= z < Config.MAP_H:
                around.append(float(self.soft_risk[z, x]))
        if around:
            return 0.5 * base + 0.5 * float(np.mean(around))
        return base

    def _target_cost(self, kind: str, target: Tuple[int, int]) -> float:
        dist = _dist(self.cur_pos, target)
        risk = self._risk_cost(target)
        energy = self._energy_cost(target)
        backtrack = 1.0 if target in self.recent_targets else 0.0

        if kind == "CHARGE":
            energy *= 0.3
        elif kind == "RETURN_WAREHOUSE":
            backtrack *= 0.5

        return (
            Config.COST_W_DIST * dist
            + Config.COST_W_RISK * risk
            + Config.COST_W_ENERGY * energy
            + Config.COST_W_BACKTRACK * backtrack
        )

    def _is_stuck(self) -> bool:
        if len(self.recent_positions) < max(int(Config.STUCK_WINDOW), 4):
            return False
        if self.target_pos is None:
            return False
        first = self.recent_positions[0]
        last = self.recent_positions[-1]
        p = float(_dist(first, last))
        target_progress = float(_dist(first, self.target_pos) - _dist(last, self.target_pos))
        return (p < float(Config.STUCK_MIN_PROGRESS)) or (target_progress < 0.5)

    def _set_target(self, kind: str, pos: Optional[Tuple[int, int]]):
        prev_kind = self.target_kind
        prev_pos = self.target_pos
        changed = (prev_kind != kind) or (prev_pos != pos)

        self.target_kind = kind
        self.target_pos = pos
        self.target_lock_steps = int(getattr(Config, "TARGET_LOCK_STEPS", Config.TARGET_LOCK_MIN_STEPS))

        if changed:
            interval = int(self.step_count - self.last_target_switch_step)
            if self.last_target_switch_step > 0 and interval > 0:
                self.target_switch_intervals.append(interval)
            self.last_target_switch_step = self.step_count
            if pos is not None:
                self.recent_targets.append(pos)

    def _choose_best_target(self, candidates: List[Tuple[str, Tuple[int, int]]]) -> Tuple[str, Tuple[int, int]]:
        scored = []
        for kind, pos in candidates:
            scored.append((self._target_cost(kind, pos), kind, pos))
        scored.sort(key=lambda x: x[0])
        _, kind, pos = scored[0]
        return kind, pos

    def _select_target(self):
        batt_ratio = self.battery / max(self.battery_max, 1)
        prev_low_lock = self.low_battery_lock
        if self.low_battery_lock:
            if batt_ratio >= Config.CHARGE_EXIT_RATIO:
                self.low_battery_lock = False
        elif batt_ratio < Config.CHARGE_ENTER_RATIO:
            self.low_battery_lock = True

        if (not prev_low_lock) and self.low_battery_lock:
            self.charge_enter_count += 1

        stuck = self._is_stuck()
        if stuck:
            self.stuck_recovery_count += 1
            self.risk_degrade_count += 1
            self.risk_degrade_until_step = self.step_count + int(max(Config.STUCK_WINDOW // 2, 4))
            self.target_lock_steps = 0

        if self.target_lock_steps > 0 and self.target_pos is not None:
            self.target_lock_steps -= 1
            return

        candidates: List[Tuple[str, Tuple[int, int]]] = []

        if self.low_battery_lock:
            charger, d1 = self._nearest(self.chargers)
            wh_d = _dist(self.warehouse, self.cur_pos) if self.warehouse is not None else 1e9
            if charger is not None and d1 <= wh_d:
                candidates.append(("CHARGE", charger))
            elif self.warehouse is not None:
                candidates.append(("CHARGE", self.warehouse))
            else:
                self._set_target("NONE", self.cur_pos)
                return
            kind, pos = self._choose_best_target(candidates)
            self._set_target(kind, pos)
            return

        if len(self.packages) == 0:
            if self.warehouse is not None:
                self._set_target("RETURN_WAREHOUSE", self.warehouse)
            else:
                self._set_target("NONE", self.cur_pos)
            return

        deliver_targets: List[Tuple[int, Tuple[int, int]]] = []
        for pid in self.packages:
            sid = int(pid)
            if sid in self.stations:
                deliver_targets.append((sid, self.stations[sid]))

        candidates = []
        for _, pos in deliver_targets:
            candidates.append(("DELIVER", pos))
        if self.warehouse is not None:
            candidates.append(("RETURN_WAREHOUSE", self.warehouse))
        for c in self.chargers:
            candidates.append(("CHARGE", c))

        if candidates:
            kind, pos = self._choose_best_target(candidates)
            self._set_target(kind, pos)
        elif self.warehouse is not None:
            self._set_target("RETURN_WAREHOUSE", self.warehouse)
        else:
            self._set_target("NONE", self.cur_pos)

    def _dstar_plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        if start == goal:
            return [start]

        traversable = self._traversable_grid()
        changed_cells: List[Tuple[int, int]] = []
        if self.prev_traversable is None:
            self.prev_traversable = traversable.copy()
        else:
            diff = np.logical_xor(traversable, self.prev_traversable)
            ys, xs = np.where(diff)
            changed_cells = [(int(x), int(y)) for y, x in zip(ys, xs)]
            self.prev_traversable = traversable.copy()

        need_init = (self.s_goal != goal) or (self.s_start is None)
        if need_init:
            self._init_dstar(start, goal)
        else:
            self.km += self._heuristic(self.s_last, start)
            self.s_last = start
            self.s_start = start
            self._update_changed_cells(changed_cells)

        self._compute_shortest_path()
        return self._extract_path(start, goal)

    def _fallback_action(self, legal_act: List[int]) -> int:
        tx, tz = self.target_pos if self.target_pos is not None else self.cur_pos
        cx, cz = self.cur_pos

        cand = []
        for a, (dx, dz) in enumerate(DIRS):
            if a >= len(legal_act) or int(legal_act[a]) <= 0:
                continue
            if not self._move_valid(cx, cz, dx, dz):
                continue
            nx, nz = cx + dx, cz + dz
            cand.append((_dist((nx, nz), (tx, tz)), a))

        if cand:
            cand.sort(key=lambda x: x[0])
            return int(cand[0][1])

        return int(self.last_action)

    def next_action(self, env_obs: dict, legal_act: List[int]) -> int:
        self.update_from_env(env_obs)
        self._select_target()

        start = self.cur_pos
        goal = self.target_pos if self.target_pos is not None else self.cur_pos
        path = self._dstar_plan(start, goal)
        self.replan_count += 1
        self.last_path_len = len(path)
        self.last_path_cost = 0.0
        if len(path) >= 2:
            c = 0.0
            for i in range(len(path) - 1):
                c += self._cost(path[i], path[i + 1])
            self.last_path_cost = float(c)
            self.path_cost_sum += self.last_path_cost
            self.path_cost_count += 1

        if len(path) >= 2:
            nx, nz = path[1]
            dx, dz = nx - start[0], nz - start[1]
            if (dx, dz) in DIR_TO_ACTION:
                a = DIR_TO_ACTION[(dx, dz)]
                if a < len(legal_act) and int(legal_act[a]) > 0:
                    self.last_action = int(a)
                    return self.last_action

        self.last_action = self._fallback_action(legal_act)
        return self.last_action

    def planner_stats(self) -> dict:
        avg_switch = 0.0
        if len(self.target_switch_intervals) > 0:
            avg_switch = float(np.mean(self.target_switch_intervals))
        mean_path_cost = 0.0
        if self.path_cost_count > 0:
            mean_path_cost = float(self.path_cost_sum / self.path_cost_count)

        return {
            "target_kind": self.target_kind,
            "target_pos": self.target_pos,
            "last_path_len": int(self.last_path_len),
            "last_path_cost": float(self.last_path_cost),
            "mean_path_cost": float(mean_path_cost),
            "replan_count": int(self.replan_count),
            "low_battery_lock": int(self.low_battery_lock),
            "avg_target_switch_interval": float(avg_switch),
            "stuck_recovery_count": int(self.stuck_recovery_count),
            "risk_degrade_count": int(self.risk_degrade_count),
            "charge_enter_count": int(self.charge_enter_count),
        }
