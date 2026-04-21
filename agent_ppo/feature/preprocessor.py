#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Dual-branch Drone Delivery feature preprocessor.
双分支智运无人机特征预处理器（image/vector）。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from agent_ppo.conf.conf import Config


ACTION_TO_DELTA = {
    0: (1, 0),
    1: (1, -1),
    2: (0, -1),
    3: (-1, -1),
    4: (-1, 0),
    5: (-1, 1),
    6: (0, 1),
    7: (1, 1),
}
OPPOSITE_ACTION = {0: 4, 1: 5, 2: 6, 3: 7, 4: 0, 5: 1, 6: 2, 7: 3}


def norm(v, max_v, min_v=0.0):
    v = np.clip(v, min_v, max_v)
    return (v - min_v) / max(max_v - min_v, 1e-6)


def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def project_to_local(cur_pos: Tuple[float, float], target_pos: Tuple[float, float]) -> Tuple[int, int, bool]:
    dx = int(round(target_pos[0] - cur_pos[0]))
    dz = int(round(target_pos[1] - cur_pos[1]))
    x = int(np.clip(10 + dx, 0, Config.MAP_SIZE - 1))
    z = int(np.clip(10 + dz, 0, Config.MAP_SIZE - 1))
    visible = abs(dx) <= 10 and abs(dz) <= 10
    return x, z, visible


def add_gaussian(plane: np.ndarray, cx: int, cz: int, sigma: float, peak: float = 1.0):
    h, w = plane.shape
    radius = max(int(3 * sigma), 1)
    for z in range(max(0, cz - radius), min(h, cz + radius + 1)):
        for x in range(max(0, cx - radius), min(w, cx + radius + 1)):
            d2 = (x - cx) ** 2 + (z - cz) ** 2
            val = peak * np.exp(-d2 / max(2.0 * sigma * sigma, 1e-6))
            if val > plane[z, x]:
                plane[z, x] = float(val)


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_pos = (0.0, 0.0)
        self.battery = 100
        self.battery_max = 100
        self.packages: List[int] = []
        self.delivered = 0
        self.step_no = 0
        self.max_step = 1000
        self.legal_act = [1] * 8
        self.local_map: Optional[List[List[int]]] = None

        self.stations: List[Dict] = []
        self.chargers: List[Dict] = []
        self.warehouse: Optional[Dict] = None
        self.npcs: List[Dict] = []

        self.on_warehouse = False
        self.on_charger = False

        self.prev_pos = None
        self.prev_delivered = 0
        self.prev_battery = self.battery
        self.prev_packages: List[int] = []
        self.prev_target_dist = None
        self.prev_supply_dist = None
        self.prev_npc_dist = None
        self.prev_target_kind = "none"

        self.last_action = -1
        self.prev_action = -1
        self.prev_prev_action = -1

        self.target_dist = None
        self.supply_dist = None
        self.npc_dist = None
        self.primary_target_kind = "none"
        self.primary_target_pos = self.cur_pos

    def _update_action_history(self, last_action):
        self.prev_prev_action = self.prev_action
        self.prev_action = int(last_action) if last_action is not None else -1
        self.last_action = self.prev_action

    def _parse_obs(self, env_obs):
        self.prev_pos = self.cur_pos
        self.prev_delivered = self.delivered
        self.prev_battery = self.battery
        self.prev_packages = list(self.packages)
        self.prev_target_dist = self.target_dist
        self.prev_supply_dist = self.supply_dist
        self.prev_npc_dist = self.npc_dist
        self.prev_target_kind = self.primary_target_kind

        obs = env_obs["observation"]
        frame_state = obs["frame_state"]
        env_info = obs.get("env_info", {})

        hero = frame_state["heroes"]
        self.cur_pos = (float(hero["pos"]["x"]), float(hero["pos"]["z"]))
        self.battery = int(hero.get("battery", self.battery_max))
        self.battery_max = int(hero.get("battery_max", self.battery_max))
        self.packages = list(hero.get("packages", []))
        self.delivered = int(hero.get("delivered", self.delivered))
        self.step_no = int(obs.get("step_no", env_info.get("step_no", 0)))
        self.max_step = int(env_info.get("max_step", self.max_step))

        self.stations = []
        self.chargers = []
        self.warehouse = None
        for organ in frame_state.get("organs", []):
            st = organ.get("sub_type", 0)
            if st == 1:
                self.warehouse = organ
            elif st == 2:
                self.chargers.append(organ)
            elif st == 3:
                self.stations.append(organ)

        self.npcs = list(frame_state.get("npcs", []))
        map_info_obj = obs.get("map_info", None)
        if isinstance(map_info_obj, dict):
            self.local_map = map_info_obj.get("map_info", None)
        elif isinstance(map_info_obj, list):
            self.local_map = map_info_obj
        else:
            self.local_map = None

        self.legal_act = obs.get("legal_action", obs.get("legal_act", [1] * 8))

        self.on_warehouse = self._is_on_warehouse()
        self.on_charger = self._is_on_charger()
        self.target_dist, self.primary_target_kind, self.primary_target_pos = self._select_primary_target()
        self.supply_dist = self._get_supply_dist()
        self.npc_dist = self._get_nearest_npc_dist()

    def _is_on_warehouse(self) -> bool:
        if not self.warehouse:
            return False
        x = self.warehouse["pos"]["x"]
        z = self.warehouse["pos"]["z"]
        w = int(self.warehouse.get("w", 1))
        h = int(self.warehouse.get("h", 1))
        return x <= self.cur_pos[0] < x + w and z <= self.cur_pos[1] < z + h

    def _is_on_charger(self) -> bool:
        for c in self.chargers:
            c_pos = (float(c["pos"]["x"]), float(c["pos"]["z"]))
            c_range = max(float(c.get("range", 0.0)), 0.5)
            if euclidean(self.cur_pos, c_pos) <= c_range:
                return True
        return False

    def _get_supply_dist(self) -> Optional[float]:
        dists = []
        if self.warehouse:
            w_pos = (float(self.warehouse["pos"]["x"]), float(self.warehouse["pos"]["z"]))
            dists.append(euclidean(self.cur_pos, w_pos))
        for c in self.chargers:
            c_pos = (float(c["pos"]["x"]), float(c["pos"]["z"]))
            dists.append(euclidean(self.cur_pos, c_pos))
        return min(dists) if dists else None

    def _get_nearest_npc_dist(self) -> Optional[float]:
        if not self.npcs:
            return None
        return min(euclidean(self.cur_pos, (float(n["pos"]["x"]), float(n["pos"]["z"]))) for n in self.npcs)

    def _iter_target_stations(self):
        target_ids = set(self.packages)
        for s in self.stations:
            if s.get("config_id", 0) in target_ids:
                yield s

    def _select_primary_target(self):
        battery_ratio = self.battery / max(self.battery_max, 1)
        target_stations = list(self._iter_target_stations())

        if len(self.packages) == 0:
            if self.warehouse:
                pos = (float(self.warehouse["pos"]["x"]), float(self.warehouse["pos"]["z"]))
                return euclidean(self.cur_pos, pos), "warehouse", pos
            return None, "none", self.cur_pos

        if battery_ratio < 0.25:
            best_pos = None
            best_dist = None
            kind = "charger"
            if self.warehouse:
                w_pos = (float(self.warehouse["pos"]["x"]), float(self.warehouse["pos"]["z"]))
                best_pos = w_pos
                best_dist = euclidean(self.cur_pos, w_pos)
                kind = "warehouse"
            for c in self.chargers:
                c_pos = (float(c["pos"]["x"]), float(c["pos"]["z"]))
                d = euclidean(self.cur_pos, c_pos)
                if best_dist is None or d < best_dist:
                    best_pos, best_dist, kind = c_pos, d, "charger"
            if best_pos is not None:
                return best_dist, kind, best_pos

        if target_stations:
            target_stations.sort(key=lambda s: euclidean(self.cur_pos, (float(s["pos"]["x"]), float(s["pos"]["z"]))))
            s = target_stations[0]
            pos = (float(s["pos"]["x"]), float(s["pos"]["z"]))
            return euclidean(self.cur_pos, pos), "station", pos

        return None, "none", self.cur_pos

    def _get_local_passable(self, dx: int, dz: int) -> bool:
        if not self.local_map:
            return True
        cz = cx = 10
        nz, nx = cz + dz, cx + dx
        if nz < 0 or nz >= Config.MAP_SIZE or nx < 0 or nx >= Config.MAP_SIZE:
            return False
        if self.local_map[nz][nx] != 1:
            return False
        if dx != 0 and dz != 0:
            side1_ok = self.local_map[cz][cx + dx] == 1 if 0 <= cx + dx < Config.MAP_SIZE else False
            side2_ok = self.local_map[cz + dz][cx] == 1 if 0 <= cz + dz < Config.MAP_SIZE else False
            return side1_ok or side2_ok
        return True

    def _get_legal_action(self):
        env_legal = [int(x) for x in (self.legal_act[:8] if self.legal_act else [1] * 8)]
        local_legal = []
        for a in range(8):
            dx, dz = ACTION_TO_DELTA[a]
            local_legal.append(1 if self._get_local_passable(dx, dz) else 0)
        final_legal = [int(e and l) for e, l in zip(env_legal, local_legal)]
        return final_legal if sum(final_legal) > 0 else env_legal if sum(env_legal) > 0 else [1] * 8

    def _build_dir_score_feature(self, legal_action) -> np.ndarray:
        scores = []
        tgt_vec = (self.primary_target_pos[0] - self.cur_pos[0], self.primary_target_pos[1] - self.cur_pos[1])
        tgt_norm = math.sqrt(tgt_vec[0] ** 2 + tgt_vec[1] ** 2)
        low_battery = self.battery / max(self.battery_max, 1) < 0.25

        for a in range(8):
            dx, dz = ACTION_TO_DELTA[a]
            move_ok = float(legal_action[a])
            if tgt_norm > 1e-6:
                align = (dx * tgt_vec[0] + dz * tgt_vec[1]) / (math.sqrt(dx * dx + dz * dz) * tgt_norm)
                align = float(norm(align, 1.0, -1.0))
            else:
                align = 0.5

            next_pos = (self.cur_pos[0] + dx, self.cur_pos[1] + dz)
            if self.npcs:
                nearest_next_npc = min(euclidean(next_pos, (float(n["pos"]["x"]), float(n["pos"]["z"]))) for n in self.npcs)
                npc_safe = norm(min(nearest_next_npc, 6.0), 6.0)
            else:
                npc_safe = 1.0

            reverse_penalty = 0.12 if self.prev_action != -1 and OPPOSITE_ACTION.get(self.prev_action, -99) == a else 0.0
            score = 0.50 * align + 0.24 * move_ok + 0.20 * npc_safe - reverse_penalty
            if low_battery and self.primary_target_kind in {"charger", "warehouse"}:
                score += 0.06 * align
            scores.append(score)
        return np.array(scores, dtype=np.float32)

    def _build_image_obs(self) -> np.ndarray:
        passable = np.ones((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32)
        if self.local_map is not None:
            map_np = np.asarray(self.local_map, dtype=np.float32)
            if map_np.shape == (Config.MAP_SIZE, Config.MAP_SIZE):
                passable = map_np
            else:
                fixed = np.ones((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32)
                h = min(map_np.shape[0], Config.MAP_SIZE)
                w = min(map_np.shape[1], Config.MAP_SIZE)
                fixed[:h, :w] = map_np[:h, :w]
                passable = fixed

        target_field = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32)
        if self.primary_target_kind != "none":
            tx, tz, visible = project_to_local(self.cur_pos, self.primary_target_pos)
            add_gaussian(target_field, tx, tz, sigma=1.8, peak=1.0 if visible else 0.6)

        npc_risk = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32)
        for n in self.npcs:
            pos = (float(n["pos"]["x"]), float(n["pos"]["z"]))
            nx, nz, visible = project_to_local(self.cur_pos, pos)
            if visible:
                add_gaussian(npc_risk, nx, nz, sigma=1.6, peak=1.0)

        return np.stack([passable, target_field, npc_risk], axis=0).astype(np.float32)

    def _build_action_prior_feature(self) -> np.ndarray:
        last_action_oh = np.zeros((Config.LAST_ACTION_DIM,), dtype=np.float32)
        if 0 <= self.prev_action < Config.ACTION_NUM:
            last_action_oh[self.prev_action] = 1.0
        reverse_flag = 0.0
        repeat_flag = 0.0
        if self.prev_action != -1 and self.prev_prev_action != -1:
            reverse_flag = 1.0 if OPPOSITE_ACTION.get(self.prev_prev_action, -99) == self.prev_action else 0.0
            repeat_flag = 1.0 if self.prev_prev_action == self.prev_action else 0.0
        return np.concatenate([last_action_oh, np.array([reverse_flag, repeat_flag], dtype=np.float32)], dtype=np.float32)

    def _make_topk_entity_feature(self, entities: List[Tuple[float, float]], k: int, with_target: bool = False) -> np.ndarray:
        rows = []
        for e in entities[:k]:
            dist = euclidean(self.cur_pos, e)
            rel_x = norm((e[0] - self.cur_pos[0]) / max(dist, 1e-4), 1.0, -1.0)
            rel_z = norm((e[1] - self.cur_pos[1]) / max(dist, 1e-4), 1.0, -1.0)
            dist_norm = norm(dist, 1.41 * 128)
            if with_target:
                rows.extend([rel_x, rel_z, dist_norm, 1.0])
            else:
                rows.extend([rel_x, rel_z, dist_norm])
        cell = 4 if with_target else 3
        while len(rows) < k * cell:
            rows.append(0.0)
        return np.asarray(rows, dtype=np.float32)

    def _build_vector_obs(self, legal_action: List[int]) -> np.ndarray:
        battery_ratio = norm(self.battery, max(self.battery_max, 1))
        can_deliver_now = 0.0
        target_ids = set(self.packages)
        target_station_pos = []
        for s in self.stations:
            sid = s.get("config_id", 0)
            pos = (float(s["pos"]["x"]), float(s["pos"]["z"]))
            if sid in target_ids:
                target_station_pos.append(pos)
                if euclidean(self.cur_pos, pos) <= 1.5:
                    can_deliver_now = 1.0

        need_supply = 1.0 if (len(self.packages) == 0 or battery_ratio < 0.25) else 0.0
        task_stage = np.array(
            [
                1.0 if len(self.packages) > 0 else 0.0,
                can_deliver_now,
                need_supply,
                1.0 if self.on_warehouse else 0.0,
                1.0 if self.on_charger else 0.0,
            ],
            dtype=np.float32,
        )

        target_dist_norm = norm(self.target_dist if self.target_dist is not None else 200.0, 200.0)
        target_delta = 0.0
        if self.prev_target_dist is not None and self.target_dist is not None:
            target_delta = float(np.clip(self.prev_target_dist - self.target_dist, -2.0, 2.0) / 2.0)

        supply_dist_norm = norm(self.supply_dist if self.supply_dist is not None else 200.0, 200.0)
        supply_delta = 0.0
        if self.prev_supply_dist is not None and self.supply_dist is not None:
            supply_delta = float(np.clip(self.prev_supply_dist - self.supply_dist, -2.0, 2.0) / 2.0)

        progress_feat = np.array([target_dist_norm, target_delta, supply_dist_norm, supply_delta], dtype=np.float32)

        npc_min_dist_norm = norm(self.npc_dist if self.npc_dist is not None else 12.0, 12.0)
        near_npc = 1.0 if (self.npc_dist is not None and self.npc_dist <= Config.NPC_DANGER_RADIUS) else 0.0
        legal_count = sum(int(v > 0) for v in legal_action)
        legal_count_norm = norm(legal_count, 8)
        dir_risk_mean = 1.0 - float(np.mean(self._build_dir_score_feature(legal_action)))
        safety_feat = np.array([npc_min_dist_norm, near_npc, legal_count_norm, dir_risk_mean], dtype=np.float32)

        action_prior = self._build_action_prior_feature()

        can_reach_supply = 0.0
        if self.supply_dist is not None:
            can_reach_supply = 1.0 if self.battery >= (self.supply_dist + 10.0) else 0.0
        resource_feat = np.array([battery_ratio, can_reach_supply, norm(self.step_no, max(self.max_step, 1))], dtype=np.float32)

        dir_guide = self._build_dir_score_feature(legal_action)

        target_station_pos = sorted(target_station_pos, key=lambda p: euclidean(self.cur_pos, p))
        station_topk = self._make_topk_entity_feature(target_station_pos, Config.TOPK_STATION_K, with_target=True)

        charger_pos = sorted(
            [(float(c["pos"]["x"]), float(c["pos"]["z"])) for c in self.chargers],
            key=lambda p: euclidean(self.cur_pos, p),
        )
        charger_topk = self._make_topk_entity_feature(charger_pos, Config.TOPK_CHARGER_K, with_target=False)

        npc_pos = sorted(
            [(float(n["pos"]["x"]), float(n["pos"]["z"])) for n in self.npcs],
            key=lambda p: euclidean(self.cur_pos, p),
        )
        npc_topk = self._make_topk_entity_feature(npc_pos, Config.TOPK_NPC_K, with_target=False)

        count_summary = np.array(
            [norm(len(target_station_pos), Config.TOPK_STATION_K), norm(len(self.chargers), Config.MAX_CHARGER_COUNT), norm(len(self.npcs), Config.MAX_NPC_COUNT)],
            dtype=np.float32,
        )

        vector = np.concatenate(
            [
                task_stage,
                progress_feat,
                safety_feat,
                action_prior,
                resource_feat,
                dir_guide,
                station_topk,
                charger_topk,
                npc_topk,
                count_summary,
            ],
            dtype=np.float32,
        )

        if vector.shape[0] != Config.VECTOR_OBS_DIM:
            out = np.zeros((Config.VECTOR_OBS_DIM,), dtype=np.float32)
            keep = min(vector.shape[0], Config.VECTOR_OBS_DIM)
            out[:keep] = vector[:keep]
            return out
        return vector

    def _build_privileged_feature(self) -> np.ndarray:
        feat = []
        battery_ratio = norm(self.battery, max(self.battery_max, 1))
        feat.extend(
            [
                norm(self.cur_pos[0], 128, -128),
                norm(self.cur_pos[1], 128, -128),
                battery_ratio,
                norm(len(self.packages), 3),
                norm(self.delivered, 20),
                norm(self.step_no, max(self.max_step, 1)),
            ]
        )

        stations_sorted = sorted(self.stations, key=lambda s: int(s.get("config_id", 0)))
        for i in range(Config.MAX_STATION_COUNT):
            if i < len(stations_sorted):
                pos = stations_sorted[i]["pos"]
                feat.extend([norm(float(pos.get("x", 0.0)), 128, -128), norm(float(pos.get("z", 0.0)), 128, -128)])
            else:
                feat.extend([0.0, 0.0])

        chargers_sorted = sorted(
            self.chargers,
            key=lambda c: euclidean(self.cur_pos, (float(c["pos"]["x"]), float(c["pos"]["z"]))),
        )
        for i in range(Config.MAX_CHARGER_COUNT):
            if i < len(chargers_sorted):
                pos = chargers_sorted[i]["pos"]
                feat.extend([norm(float(pos.get("x", 0.0)), 128, -128), norm(float(pos.get("z", 0.0)), 128, -128)])
            else:
                feat.extend([0.0, 0.0])

        npcs_sorted = sorted(
            self.npcs,
            key=lambda n: euclidean(self.cur_pos, (float(n["pos"]["x"]), float(n["pos"]["z"]))),
        )
        for i in range(Config.MAX_NPC_COUNT):
            if i < len(npcs_sorted):
                pos = npcs_sorted[i]["pos"]
                feat.extend([norm(float(pos.get("x", 0.0)), 128, -128), norm(float(pos.get("z", 0.0)), 128, -128)])
            else:
                feat.extend([0.0, 0.0])

        if self.warehouse:
            w_pos = self.warehouse["pos"]
            feat.extend([norm(float(w_pos.get("x", 0.0)), 128, -128), norm(float(w_pos.get("z", 0.0)), 128, -128)])
        else:
            feat.extend([0.0, 0.0])

        feat.extend(
            [
                norm(self.max_step - self.step_no, max(self.max_step, 1)),
                1.0 if self.primary_target_kind == "station" else 0.0,
                1.0 if self.primary_target_kind in {"charger", "warehouse"} else 0.0,
                norm(self.npc_dist if self.npc_dist is not None else 12.0, 12.0),
            ]
        )

        arr = np.asarray(feat, dtype=np.float32)
        if arr.shape[0] < Config.PRIVILEGED_DIM:
            out = np.zeros((Config.PRIVILEGED_DIM,), dtype=np.float32)
            out[: arr.shape[0]] = arr
            return out
        return arr[: Config.PRIVILEGED_DIM]

    def feature_process(self, env_obs, last_action):
        self._update_action_history(last_action)
        self._parse_obs(env_obs)

        legal_action = self._get_legal_action()
        image_obs = self._build_image_obs()
        vector_obs = self._build_vector_obs(legal_action)
        privileged_feature = self._build_privileged_feature()
        reward = self._reward_process()

        return vector_obs, image_obs, privileged_feature, legal_action, reward

    def _reward_process(self):
        reward = Config.STEP_PENALTY

        newly_delivered = max(0, self.delivered - self.prev_delivered)
        if newly_delivered > 0:
            reward += Config.DELIVERY_REWARD * newly_delivered

        picked_count = max(0, len(self.packages) - len(self.prev_packages))
        if picked_count > 0 and self.on_warehouse:
            reward += Config.PICKUP_REWARD * picked_count

        battery_gain = max(0, self.battery - self.prev_battery)
        if battery_gain > 0 and self.on_charger:
            reward += Config.CHARGE_GAIN_SCALE * battery_gain

        if self.prev_target_dist is not None and self.target_dist is not None:
            progress = np.clip(self.prev_target_dist - self.target_dist, -2.0, 2.0)
            if self.primary_target_kind == "station":
                reward += Config.STATION_PROGRESS_SCALE * progress
            else:
                reward += 0.5 * Config.STATION_PROGRESS_SCALE * progress

        battery_ratio = self.battery / max(self.battery_max, 1)
        if self.prev_supply_dist is not None and self.supply_dist is not None:
            supply_progress = np.clip(self.prev_supply_dist - self.supply_dist, -2.0, 2.0)
            if battery_ratio < 0.25:
                reward += Config.CHARGER_PROGRESS_SCALE * supply_progress
                if supply_progress < 0:
                    reward += Config.LOW_BATTERY_MOVE_AWAY_PENALTY
            if len(self.packages) == 0:
                reward += Config.WAREHOUSE_PROGRESS_SCALE * supply_progress
                if supply_progress < 0:
                    reward += Config.EMPTY_LOAD_MOVE_AWAY_PENALTY

        if self.prev_pos is not None and self.last_action != -1 and self.prev_pos == self.cur_pos:
            reward += Config.INVALID_MOVE_PENALTY

        if self.prev_action != -1 and self.prev_prev_action != -1:
            if OPPOSITE_ACTION.get(self.prev_prev_action, -99) == self.prev_action:
                reward += Config.OSCILLATION_PENALTY
            if self.prev_prev_action == self.prev_action:
                reward += Config.REPEAT_MOVE_PENALTY

        if self.npc_dist is not None and self.npc_dist < Config.NPC_DANGER_RADIUS:
            reward -= Config.NPC_DANGER_PENALTY_SCALE * (Config.NPC_DANGER_RADIUS - self.npc_dist)

        if len(self.prev_packages) == 0 and self.on_warehouse:
            reward += Config.SUPPLY_BONUS
        if self.prev_battery / max(self.battery_max, 1) < 0.25 and self.on_charger:
            reward += Config.SUPPLY_BONUS

        if len(self.packages) > 0 and self.primary_target_kind == "station" and self.target_dist is not None:
            if self.target_dist <= 1.5:
                reward += Config.CAN_DELIVER_BONUS

        return [float(reward)]
