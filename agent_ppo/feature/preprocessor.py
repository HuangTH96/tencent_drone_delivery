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
    # 局部地图原点在左上角，中心坐标必须是(10, 10)，否则x, z作为索引会出现负数得情况
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

        self.prev_need_recharge = False
        self.need_recharge_flag = False
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

    """
    更新内容：

    - 记录历史。包括位置、投送数量、电量、剩余包裹、目标距离、离仓库距离、离npc距离、目标类型
    """
    def _parse_obs(self, env_obs):
        self.prev_need_recharge = self._need_recharge() 
        self.prev_pos = self.cur_pos
        self.prev_delivered = self.delivered
        self.prev_battery = self.battery
        self.prev_packages = list(self.packages)
        self.prev_target_dist = self.target_dist
        self.prev_supply_dist = self.supply_dist
        self.prev_npc_dist = self.npc_dist
        self.prev_target_kind = self.primary_target_kind

        obs = env_obs["observation"]
        frame_state = obs["frame_state"]        # 包含hero状态，npc状态，物件列表
        env_info = obs.get("env_info", {})      

        # ========= 提取hero信息 =========
        # 除 【score】外，都用到了
        hero = frame_state["heroes"]
        self.cur_pos = (float(hero["pos"]["x"]), float(hero["pos"]["z"]))
        self.battery = int(hero.get("battery", self.battery_max))
        self.battery_max = int(hero.get("battery_max", self.battery_max))
        self.packages = list(hero.get("packages", []))
        self.delivered = int(hero.get("delivered", self.delivered))

        # ========= 观测信息 =========
        ## 除 【环境信息】 和 【frame_no】 都用到了
        self.step_no = int(obs.get("step_no", env_info.get("step_no", 0)))
        self.legal_act = obs.get("legal_action", obs.get("legal_act", [1] * 8))

        # ========= 【极少】环境信息 =========
        self.max_step = int(env_info.get("max_step", self.max_step))

        # ========= 【全部】物件信息 =========
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

        # ========= 【全部】NPCs 和 地图信息 =========
        self.npcs = list(frame_state.get("npcs", []))
        map_info_obj = obs.get("map_info", None)
        if isinstance(map_info_obj, dict):
            self.local_map = map_info_obj.get("map_info", None)
        elif isinstance(map_info_obj, list):
            self.local_map = map_info_obj
        else:
            self.local_map = None

        # ========= 计算额外特征 ========
        self.on_warehouse = self._is_on_warehouse()
        self.on_charger = self._is_on_charger()
        self.target_dist, self.primary_target_kind, self.primary_target_pos = self._select_primary_target()
        self.supply_dist = self._get_supply_dist()
        self.npc_dist = self._get_nearest_npc_dist()
        self.need_recharge_flag = self._need_recharge()

    """
    判断是否进入仓库范围内。

    TODO：是否需要区分装货区域和非装货区域
    """
    def _is_on_warehouse(self) -> bool:
        if not self.warehouse:
            return False
        x = self.warehouse["pos"]["x"]
        z = self.warehouse["pos"]["z"]
        w = int(self.warehouse.get("w", 1))
        h = int(self.warehouse.get("h", 1))
        # 地图左上角为原点，x向右增加，z向下增加；假设仓库坐标在正中心
        # return x <= self.cur_pos[0] < x + w and z <= self.cur_pos[1] < z + h
        return x - (w - 1) / 2 <= self.cur_pos[0] < x + (w - 1) / 2 and z - (h - 1) / 2 <= self.cur_pos[1] < z + (h - 1) / 2

    """
    判断是否进入充电桩的可充电范围内
    """
    def _is_on_charger(self) -> bool:
        for c in self.chargers:
            # c_pos = (float(c["pos"]["x"]), float(c["pos"]["z"]))
            c_pos = (c["pos"]["x"], c["pos"]["z"])
            # c_range = max(float(c.get("range", 0.0)), 0.5)
            c_range = c.get("range", 1.0) 
            if euclidean(self.cur_pos, c_pos) <= c_range:
                return True
        return False

    """
    计算离**所有**可充电区域中最近的距离
    """
    def _get_supply_dist(self) -> Optional[float]:
        dists = []
        if self.warehouse:
            # w_pos = (float(self.warehouse["pos"]["x"]), float(self.warehouse["pos"]["z"]))
            w_pos = (self.warehouse["pos"]["x"], self.warehouse["pos"]["z"])
            dists.append(euclidean(self.cur_pos, w_pos))
        for c in self.chargers:
            # c_pos = (float(c["pos"]["x"]), float(c["pos"]["z"]))
            c_pos = (c["pos"]["x"], c["pos"]["z"])
            dists.append(euclidean(self.cur_pos, c_pos))
        return min(dists) if dists else None

    """
    计算离最近的npc的**直线距离**
    """
    def _get_nearest_npc_dist(self) -> Optional[float]:
        if not self.npcs:
            return None
        # return min(euclidean(self.cur_pos, (float(n["pos"]["x"]), float(n["pos"]["z"]))) for n in self.npcs)
        return min(euclidean(self.cur_pos, (n["pos"]["x"], n["pos"]["z"])) for n in self.npcs)
    
    """
    选择需要前往的驿站IDs

    TODO：是否需要考虑同一个驿站有多个包裹？
    """
    def _iter_target_stations(self):
        target_ids = set(self.packages)
        for s in self.stations:
            if s.get("config_id", 0) in target_ids:
                yield s

    """
    选择主要目标 

    - 没有包裹时，直接回仓库；
    - 电量低于25%时，优先去充电；
    - 其他情况下，去最近的目标驿站。
    """
    # def _select_primary_target(self):
    #     battery_ratio = self.battery / max(self.battery_max, 1)
    #     target_stations = list(self._iter_target_stations())

    #     # 如果没有包裹，计算离**仓库正中心**的距离
    #     # TODO：考虑电量是否能够支撑到仓库
    #     if len(self.packages) == 0:
    #         if self.warehouse:
    #         # if _is_reachable(self.warehouse, self.battery):
    #             # pos = (float(self.warehouse["pos"]["x"]), float(self.warehouse["pos"]["z"]))
    #             pos = (self.warehouse["pos"]["x"], self.warehouse["pos"]["z"])
    #             return euclidean(self.cur_pos, pos), "warehouse", pos
    #         # else:
    #         #     return nearest_charger_dist, "charger", nearest_charger_pos
    #         return None, "none", self.cur_pos
        
    #     # 如果电量低于25%，计算离**最近的可充电区域**的距离
    #     # TODO：应该让模型自己判断是否需要去充电，而不是简单地以电量硬阈值进行切换，可以考虑将电量和离可充电区域的距离都作为输入特征，让模型自己学会权衡；或者用宋的路径感知代替阈值
    #     if battery_ratio < 0.25:
    #         best_pos = None
    #         best_dist = None
    #         kind = "charger"
    #         if self.warehouse:
    #             # w_pos = (float(self.warehouse["pos"]["x"]), float(self.warehouse["pos"]["z"]))
    #             w_pos = self.warehouse["pos"]["x"], self.warehouse["pos"]["z"]
    #             best_pos = w_pos
    #             best_dist = euclidean(self.cur_pos, w_pos)
    #             kind = "warehouse"
    #         for c in self.chargers:
    #             # c_pos = (float(c["pos"]["x"]), float(c["pos"]["z"]))
    #             c_pos = (c["pos"]["x"], c["pos"]["z"])
    #             d = euclidean(self.cur_pos, c_pos)
    #             # TODO：把 _get_local_passable 已有的地图信息用来过滤掉明显被障碍物阻挡的方向，直线距离作为 fallback
    #             if best_dist is None or d < best_dist:
    #                 best_pos, best_dist, kind = c_pos, d, "charger"
    #         if best_pos is not None:
    #             return best_dist, kind, best_pos

    #     # 有包裹，且电量多于25%时，返回最近的目标驿站距离
    #     # TODO：26%的电量也不够去驿站，但是不会以充电站或者仓库为目标
    #     if target_stations:
    #         # target_stations.sort(key=lambda s: euclidean(self.cur_pos, (float(s["pos"]["x"]), float(s["pos"]["z"]))))
    #         target_stations.sort(key=lambda s: euclidean(self.cur_pos, (s["pos"]["x"], s["pos"]["z"])))
    #         s = target_stations[0]
    #         # pos = (float(s["pos"]["x"]), float(s["pos"]["z"]))
    #         pos = s["pos"]["x"], s["pos"]["z"]
    #         return euclidean(self.cur_pos, pos), "station", pos
        
    #     # 防御性代码，永远不会被执行
    #     return None, "none", self.cur_pos

    def _select_primary_target(self):
        target_stations = list(self._iter_target_stations())

        # 无包裹时
        if len(self.packages) == 0:
            if self.warehouse:
                w_pos = self.warehouse["pos"]["x"], self.warehouse["pos"]["z"]
                w_dist = euclidean(self.cur_pos, w_pos)

                # 电量足够到仓库，优先回仓库
                if self.battery >= w_dist + Config.RECHARGE_MARGIN:
                    return w_dist, "warehouse", w_pos
                
                # 电量不足到仓库，但足够到充电站，优先去充电
                best_pos, best_dist, kind = self._find_nearest_supply()
                if best_pos is not None:
                    return best_dist, kind, best_pos
                
                # 防御性fallback：没有补给点，硬着头皮去仓库
                return w_dist, "warehouse", w_pos
            return None, "none", self.cur_pos
        
        # 有包裹时，判断是否需要充电
        if self._need_recharge():
            best_pos, best_dist, kind = self._find_nearest_supply()
            if best_pos is not None:
                return best_dist, kind, best_pos   
            
        # 电量足够，去最近的目标驿站
        if target_stations:
            target_stations.sort(key=lambda s: euclidean(self.cur_pos, (s["pos"]["x"], s["pos"]["z"])))
            s = target_stations[0]
            pos = s["pos"]["x"], s["pos"]["z"]
            return euclidean(self.cur_pos, pos), "station", pos
        
        # 防御性代码
        return None, "none", self.cur_pos
        
    """
    判断从当前位置（dx, dz）移动一步是否可行
    """
    def _get_local_passable(self, dx: int, dz: int) -> bool:
        if not self.local_map:
            return True # 没有地图信息，默认可通行
        
        # TODO：检查local_map坐标原点位置，以下假设地图左上角为原点，x向右增加，z向下增加，当前格子为(10, 10)
        cz = cx = 10    
        nz, nx = cz + dz, cx + dx
        if nz < 0 or nz >= Config.MAP_SIZE or nx < 0 or nx >= Config.MAP_SIZE:
            return False    # 目标格子超过视野边界，不可通行
        
        if self.local_map[nz][nx] != 1:
            return False    # 目标格子不可通行
        
        if dx != 0 and dz != 0: # 斜向运动，防止穿角
            side1_ok = self.local_map[cz][cx + dx] == 1 if 0 <= cx + dx < Config.MAP_SIZE else False
            side2_ok = self.local_map[cz + dz][cx] == 1 if 0 <= cz + dz < Config.MAP_SIZE else False
            return side1_ok or side2_ok
        return True

    """
    判断8个方向的运动是否合法，综合环境提供的legal_act和地图信息进行判断

    TODO： 环境给出的合法动作不就是已经综合了地图信息和其他限制条件的吗？为什么还需要自己再判断一遍地图信息？
    """
    def _get_legal_action(self):
        env_legal = [int(x) for x in (self.legal_act[:8] if self.legal_act else [1] * 8)]
        local_legal = []
        for a in range(8):
            dx, dz = ACTION_TO_DELTA[a]
            local_legal.append(1 if self._get_local_passable(dx, dz) else 0)
        final_legal = [int(e and l) for e, l in zip(env_legal, local_legal)]
        return final_legal if sum(final_legal) > 0 else env_legal if sum(env_legal) > 0 else [1] * 8

    """
    构建方向得分特征

    - 对齐得分：动作方向和目标方向的夹角余弦值，范围[-1, 1]，越接近1表示越对齐；如果目标距离过近（小于1e-6），则默认得分为0.5
    - 可行性得分：动作在环境中是否合法（0或1）
    - NPC安全得分：如果动作后的位置离最近的NPC越远，得分越高，最高得1分；TODO：移动到NPC身边，并被抓获时，仍然给的正得分
    """
    def _build_dir_score_feature(self, legal_action) -> np.ndarray:
        scores = []
        tgt_vec = (self.primary_target_pos[0] - self.cur_pos[0], self.primary_target_pos[1] - self.cur_pos[1])
        tgt_norm = math.sqrt(tgt_vec[0] ** 2 + tgt_vec[1] ** 2)
        # TODO：低电量加成只在电量低于25%时有效，应该优化充电决策逻辑
        # low_battery = self.battery / max(self.battery_max, 1) < 0.25

        for a in range(8):
            dx, dz = ACTION_TO_DELTA[a]
            move_ok = float(legal_action[a])    # 动作可行性得分
            if tgt_norm > 1e-6:
                # 动作方向和目标方向是否对齐
                align = (dx * tgt_vec[0] + dz * tgt_vec[1]) / (math.sqrt(dx * dx + dz * dz) * tgt_norm)
                align = float(norm(align, 1.0, -1.0))
            else:
                align = 0.5

            next_pos = (self.cur_pos[0] + dx, self.cur_pos[1] + dz)
            if self.npcs:
                # nearest_next_npc = min(euclidean(next_pos, (float(n["pos"]["x"]), float(n["pos"]["z"]))) for n in self.npcs)
                nearest_next_npc = min(euclidean(next_pos, (n["pos"]["x"], n["pos"]["z"])) for n in self.npcs)
                npc_safe = norm(min(nearest_next_npc, 6.0), 6.0)
            else:
                npc_safe = 1.0

            # 惩罚动作来回切换
            reverse_penalty = 0.12 if self.prev_action != -1 and OPPOSITE_ACTION.get(self.prev_action, -99) == a else 0.0

            # 综合得分
            score = 0.50 * align + 0.24 * move_ok + 0.20 * npc_safe - reverse_penalty

            # 低电量时，加强目标对齐的得分
            # if low_battery and self.primary_target_kind in {"charger", "warehouse"}:
            if self.need_recharge_flag and self.primary_target_kind in {"charger", "warehouse"}:    
                score += 0.06 * align
            scores.append(score)
        return np.array(scores, dtype=np.float32)

    """
    构建21*21*3的图像特征，包含可通行性、目标场和NPC风险三个通道
    """
    def _build_image_obs(self) -> np.ndarray:
        passable = np.ones((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32)
        if self.local_map is not None:
            map_np = np.asarray(self.local_map, dtype=np.float32)
            if map_np.shape == (Config.MAP_SIZE, Config.MAP_SIZE):
                passable = map_np   # 正常情况下，直接将self.local_map转换为numpy数组作为可通行性图层
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

    """
    构建10D动作历史特征

    - 8D上一步动作的one-hot编码
    - 判断上上步和上一步之间是否发生振荡或者重复

    TODO：是否有必要引入别的历史相关的特征，或者拉长历史记录的长度？
    """
    def _build_action_prior_feature(self) -> np.ndarray:
        last_action_oh = np.zeros((Config.LAST_ACTION_DIM,), dtype=np.float32)
        # 上一步动作的one-hot编码
        if 0 <= self.prev_action < Config.ACTION_NUM:
            last_action_oh[self.prev_action] = 1.0

        reverse_flag = 0.0
        repeat_flag = 0.0
        if self.prev_action != -1 and self.prev_prev_action != -1:
            # 上上步和上一步完全相反，发生了振荡
            reverse_flag = 1.0 if OPPOSITE_ACTION.get(self.prev_prev_action, -99) == self.prev_action else 0.0
            # 上上步和上一步完全相同，发生了重复
            repeat_flag = 1.0 if self.prev_prev_action == self.prev_action else 0.0
        return np.concatenate([last_action_oh, np.array([reverse_flag, repeat_flag], dtype=np.float32)], dtype=np.float32)

    """
    构建周围K个实体

    TODO：entities的坐标应该是绝对格子，是整数
    TODO：充电站有时候也是目标
    TODO：with_target参数是否有必要？驿站、充电站和NPC的特征在观测向量中的位置不是固定的么？
    
    """
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

        # 不足K个实体时，补0填充
        while len(rows) < k * cell:
            rows.append(0.0)
        return np.asarray(rows, dtype=np.float32)

    def _build_vector_obs(self, legal_action: List[int]) -> np.ndarray:
        # 状态指示器：是否有包裹、是否可以立即投送、是否需要补给、是否在仓库、是否在充电桩
        battery_ratio = norm(self.battery, max(self.battery_max, 1))
        can_deliver_now = 0.0
        target_ids = set(self.packages)
        target_station_pos = []
        for s in self.stations:
            sid = s.get("config_id", 0)
            # pos = (float(s["pos"]["x"]), float(s["pos"]["z"]))
            pos = (s["pos"]["x"], s["pos"]["z"])
            if sid in target_ids:
                target_station_pos.append(pos)
                if euclidean(self.cur_pos, pos) <= 1.5:
                    can_deliver_now = 1.0

        # need_supply = 1.0 if (len(self.packages) == 0 or battery_ratio < 0.25) else 0.0
        need_supply = 1.0 if (len(self.packages) == 0 or self.need_recharge_flag) else 0.0
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

        # 进度差分特征
        ## 本步相对于上一步，到目标的距离变化
        target_dist_norm = norm(self.target_dist if self.target_dist is not None else 200.0, 200.0)
        target_delta = 0.0
        if self.prev_target_dist is not None and self.target_dist is not None:
            target_delta = float(np.clip(self.prev_target_dist - self.target_dist, -2.0, 2.0) / 2.0)

        ## 本步相对于上一步，到可充电区域的距离变化
        supply_dist_norm = norm(self.supply_dist if self.supply_dist is not None else 200.0, 200.0)
        supply_delta = 0.0
        if self.prev_supply_dist is not None and self.supply_dist is not None:
            supply_delta = float(np.clip(self.prev_supply_dist - self.supply_dist, -2.0, 2.0) / 2.0)

        progress_feat = np.array([target_dist_norm, target_delta, supply_dist_norm, supply_delta], dtype=np.float32)

        # 安全特征
        ## 最近的npc的归一化距离 + 是否在 npc 3个单元内
        npc_min_dist_norm = norm(self.npc_dist if self.npc_dist is not None else 12.0, 12.0)
        near_npc = 1.0 if (self.npc_dist is not None and self.npc_dist <= Config.NPC_DANGER_RADIUS) else 0.0
        ## 可行动作越少，说明周围障碍物越多
        legal_count = sum(int(v > 0) for v in legal_action)
        legal_count_norm = norm(legal_count, 8)
        ## 8个动作方向的整体风险
        dir_scores = self._build_dir_score_feature(legal_action)
        dir_risk_mean = 1.0 - float(np.mean(dir_scores))
        safety_feat = np.array([npc_min_dist_norm, near_npc, legal_count_norm, dir_risk_mean], dtype=np.float32)

        # 历史动作特征
        action_prior = self._build_action_prior_feature()

        # 资源特征
        can_reach_supply = 0.0
        if self.supply_dist is not None:
            # TODO：10的余量够么？应该让模型自己学会判断是否需要充电，还是之前说到的充电决策逻辑
            can_reach_supply = 1.0 if self.battery >= (self.supply_dist + 10.0) else 0.0
        ## 当前电量 + 是否需要充电 + 剩余总步数
        resource_feat = np.array([battery_ratio, can_reach_supply, norm(self.step_no, max(self.max_step, 1))], dtype=np.float32)

        # 方向引导
        dir_guide = dir_scores

        # TOP-K 实体特征
        ## 按欧式距离排序，选择全部3个驿站
        target_station_pos = sorted(target_station_pos, key=lambda p: euclidean(self.cur_pos, p))
        station_topk = self._make_topk_entity_feature(target_station_pos, Config.TOPK_STATION_K, with_target=True)

        ## 按欧氏距离排序，关注最近的2个充电站
        charger_pos = sorted(
            # [(float(c["pos"]["x"]), float(c["pos"]["z"])) for c in self.chargers],
            [(c["pos"]["x"], c["pos"]["z"]) for c in self.chargers],
            key=lambda p: euclidean(self.cur_pos, p),
        )
        charger_topk = self._make_topk_entity_feature(charger_pos, Config.TOPK_CHARGER_K, with_target=False)

        ## 按欧式距离排序，关注最近的3个NPC
        npc_pos = sorted(
            # [(float(n["pos"]["x"]), float(n["pos"]["z"])) for n in self.npcs],
            [(n["pos"]["x"], n["pos"]["z"]) for n in self.npcs],
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

        # 防御性编程，防止特征维度不匹配
        if vector.shape[0] != Config.VECTOR_OBS_DIM:
            out = np.zeros((Config.VECTOR_OBS_DIM,), dtype=np.float32)
            keep = min(vector.shape[0], Config.VECTOR_OBS_DIM)
            out[:keep] = vector[:keep]
            return out
        return vector

    """
    只喂给Critic的48D特权观测
    """
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

        # 按ID给所有驿站排序
        stations_sorted = sorted(self.stations, key=lambda s: int(s.get("config_id", 0)))
        for i in range(Config.MAX_STATION_COUNT):
            if i < len(stations_sorted):
                pos = stations_sorted[i]["pos"]
                # feat.extend([norm(float(pos.get("x", 0.0)), 128, -128), norm(float(pos.get("z", 0.0)), 128, -128)])
                feat.extend([norm(pos.get("x", 0.0), 128, -128), norm(pos.get("z", 0.0), 128, -128)])
            else:
                feat.extend([0.0, 0.0])

        # 按离agent的距离给充电站排序
        # TODO：不同时刻下，同一个充电站的特征在特征向量中的位置会发生变化，无法对应学习，应该固定每个充电站在特征向量中的位置
        chargers_sorted = sorted(
            self.chargers,
            # key=lambda c: euclidean(self.cur_pos, (float(c["pos"]["x"]), float(c["pos"]["z"]))),
            key=lambda c: euclidean(self.cur_pos, (c["pos"]["x"], c["pos"]["z"])),
        )
        for i in range(Config.MAX_CHARGER_COUNT):
            if i < len(chargers_sorted):
                pos = chargers_sorted[i]["pos"]
                # feat.extend([norm(float(pos.get("x", 0.0)), 128, -128), norm(float(pos.get("z", 0.0)), 128, -128)])
                feat.extend([norm(pos.get("x", 0.0), 128, -128), norm(pos.get("z", 0.0), 128, -128)])
            else:
                feat.extend([0.0, 0.0])

        # 按离agent的距离给NPC排序
        # TODO：不同时刻下，同一个NPC的特征在特征向量中的位置会发生变化，无法对应学习，应该固定每个NPC在特征向量中的位置
        npcs_sorted = sorted(
            self.npcs,
            # key=lambda n: euclidean(self.cur_pos, (float(n["pos"]["x"]), float(n["pos"]["z"]))),
            key=lambda n: euclidean(self.cur_pos, (n["pos"]["x"], n["pos"]["z"])),
        )
        for i in range(Config.MAX_NPC_COUNT):
            if i < len(npcs_sorted):
                pos = npcs_sorted[i]["pos"]
                # feat.extend([norm(float(pos.get("x", 0.0)), 128, -128), norm(float(pos.get("z", 0.0)), 128, -128)])
                feat.extend([norm(pos.get("x", 0.0), 128, -128), norm(pos.get("z", 0.0), 128, -128)])
            else:
                feat.extend([0.0, 0.0])

        # 仓库位置
        if self.warehouse:
            w_pos = self.warehouse["pos"]
            # feat.extend([norm(float(w_pos.get("x", 0.0)), 128, -128), norm(float(w_pos.get("z", 0.0)), 128, -128)])
            feat.extend([norm(w_pos.get("x", 0.0), 128, -128), norm(w_pos.get("z", 0.0), 128, -128)])
        else:
            feat.extend([0.0, 0.0])

        # 环境信息
        feat.extend(
            [
                norm(self.max_step - self.step_no, max(self.max_step, 1)),              # 剩余步数
                1.0 if self.primary_target_kind == "station" else 0.0,                  # 当前目标是驿站
                1.0 if self.primary_target_kind in {"charger", "warehouse"} else 0.0,   # 当前目标是充电站、仓库 TODO：为什么不分开？
                norm(self.npc_dist if self.npc_dist is not None else 12.0, 12.0),       # 最近的NPC距离
            ]
        )

        # 防御性编程
        arr = np.asarray(feat, dtype=np.float32)
        if arr.shape[0] < Config.PRIVILEGED_DIM:
            out = np.zeros((Config.PRIVILEGED_DIM,), dtype=np.float32)
            out[: arr.shape[0]] = arr
            return out
        return arr[: Config.PRIVILEGED_DIM]

    """
    入口函数
    """
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
        # =========== 基础步惩罚 =========== 
        reward = Config.STEP_PENALTY

        # =========== 投递奖励 =========== 
        newly_delivered = max(0, self.delivered - self.prev_delivered)
        if newly_delivered > 0:
            reward += Config.DELIVERY_REWARD * newly_delivered

        # =========== 捡包奖励 =========== 
        picked_count = max(0, len(self.packages) - len(self.prev_packages))
        if picked_count > 0 and self.on_warehouse:
            reward += Config.PICKUP_REWARD * picked_count

        # ===========  充电奖励 =========== 
        battery_gain = max(0, self.battery - self.prev_battery)
        if battery_gain > 0 and self.on_charger:
            reward += Config.CHARGE_GAIN_SCALE * battery_gain

        # =========== 近目标奖励 =========== 
        if self.prev_target_dist is not None and self.target_dist is not None:
            progress = np.clip(self.prev_target_dist - self.target_dist, -2.0, 2.0)
            # 投递优先于补给
            if self.primary_target_kind == "station":
                reward += Config.STATION_PROGRESS_SCALE * progress
            else:
                reward += 0.5 * Config.STATION_PROGRESS_SCALE * progress

        # =========== 补给进度奖励 =========== 
        # battery_ratio = self.battery / max(self.battery_max, 1)
        # if self.prev_supply_dist is not None and self.supply_dist is not None:
        #     # TODO：supply包括了充电站和仓库，无包裹时应该鼓励前往仓库，而非充电站
        #     supply_progress = np.clip(self.prev_supply_dist - self.supply_dist, -2.0, 2.0)
        #     # 低电量时鼓励靠近充电站
        #     # if battery_ratio < 0.25:
        #     if self.prev_need_recharge:
        #         reward += Config.CHARGER_PROGRESS_SCALE * supply_progress
        #         if supply_progress < 0:
        #             reward += Config.LOW_BATTERY_MOVE_AWAY_PENALTY
        #     # 无包裹时鼓励靠近仓库
        #     if len(self.packages) == 0:
        #         reward += Config.WAREHOUSE_PROGRESS_SCALE * supply_progress
        #         if supply_progress < 0:
        #             reward += Config.EMPTY_LOAD_MOVE_AWAY_PENALTY

        if self.prev_supply_dist is not None and self.supply_dist is not None:
            supply_progress = np.clip(self.prev_supply_dist - self.supply_dist, -2.0, 2.0)
            if self.prev_need_recharge:
                reward += Config.CHARGER_PROGRESS_SCALE * supply_progress
                if supply_progress < 0:
                    reward += Config.LOW_BATTERY_MOVE_AWAY_PENALTY
        
        if len(self.packages) == 0 and self.warehouse and self.prev_pos is not None:
            w_pos = (self.warehouse["pos"]["x"], self.warehouse["pos"]["z"])
            warehouse_dist = euclidean(self.cur_pos, w_pos)
            prev_warehouse_dist = euclidean(self.prev_pos, w_pos)
            warehouse_progress = np.clip(prev_warehouse_dist - warehouse_dist, -2.0, 2.0) 
            reward += Config.WAREHOUSE_PROGRESS_SCALE * warehouse_progress
            if warehouse_progress < 0:
                reward += Config.EMPTY_LOAD_MOVE_AWAY_PENALTY

        # =========== 惩罚无效移动，防止卡死 =========== 
        if self.prev_pos is not None and self.last_action != -1 and self.prev_pos == self.cur_pos:
            reward += Config.INVALID_MOVE_PENALTY

        # =========== 振荡惩罚 =========== 
        if self.prev_action != -1 and self.prev_prev_action != -1:
            if OPPOSITE_ACTION.get(self.prev_prev_action, -99) == self.prev_action:
                reward += Config.OSCILLATION_PENALTY
            # if self.prev_prev_action == self.prev_action:
            #     reward += Config.REPEAT_MOVE_PENALTY

        # =========== NPC危险惩罚 ===========
        # TODO：惩罚的是“现在离npc近”，而不是“动作导致离npc更近”
        if self.npc_dist is not None and self.npc_dist < Config.NPC_DANGER_RADIUS:
            reward -= Config.NPC_DANGER_PENALTY_SCALE * (Config.NPC_DANGER_RADIUS - self.npc_dist)

        # =========== 到达补给地一次性奖励 =========== 
        # TODO：仓库既能充电又能补充包裹，他的奖励是不是应该和给充电站的不一样？
        if len(self.prev_packages) == 0 and self.on_warehouse:
            reward += Config.SUPPLY_BONUS

        # =========== 低电量时，仅鼓励去充电站充电，不鼓励去仓库 =========== 
        if self.prev_need_recharge and self.on_charger:
            reward += Config.SUPPLY_BONUS

        # TODO：有必要么？会不会导致在目标周围徘徊，刷分？
        # if len(self.packages) > 0 and self.primary_target_kind == "station" and self.target_dist is not None:
        #     if self.target_dist <= 1.5:
        #         reward += Config.CAN_DELIVER_BONUS

        

        return [float(reward)]


    """
    判断当前电量是否能够完成剩余任务，有两个场景：

    - 有包裹，需要去驿站；
    - 无包裹，需要回仓库。
    """
    def _need_recharge(self) -> bool:
        supply_dist = self.supply_dist if self.supply_dist is not None else 999.0
        margin = Config.RECHARGE_MARGIN # 绕路的冗余距离

        # 如果所剩步数不可能再补货或者充电，则应该殊死一搏
        steps_remain = self.max_step - self.step_no
        if steps_remain < supply_dist + margin and self.primary_target_kind == "station":
            return False
        
        # 如果电量不足以支持到最近的可充电区域，则立刻需要充电
        if self.battery <= supply_dist + margin:
            return True

        # 有包裹，且正在前往投递的路上时，电量不足以支持到目标驿站或者补给地，则需要充电
        if len(self.packages) > 0 and self.target_dist is not None and self.primary_target_kind == "station":
            if self.battery < self.target_dist + supply_dist + margin:
                return True
        
        # 无包裹，但电量不足以支持到仓库，则需要充电
        if len(self.packages) == 0 and self.warehouse:
            w_pos = (self.warehouse["pos"]["x"], self.warehouse["pos"]["z"])
            warehouse_dist = euclidean(self.cur_pos, w_pos)
            if self.battery < warehouse_dist + margin:
                return True

        return False
    
    """
    找最近的补给点（仓库或充电桩），返回 (pos, dist, kind)
    """
    def _find_nearest_supply(self):
        best_pos = None
        best_dist = None
        kind = None

        if self.warehouse:
            w_pos = (self.warehouse["pos"]["x"], self.warehouse["pos"]["z"])
            w_dist = euclidean(self.cur_pos, w_pos)
            best_pos, best_dist, kind = w_pos, w_dist, "warehouse"

        for c in self.chargers:
            c_pos = (c["pos"]["x"], c["pos"]["z"])
            d = euclidean(self.cur_pos, c_pos)
            if best_dist is None or d < best_dist:
                best_pos, best_dist, kind = c_pos, d, "charger"

        return best_pos, best_dist, kind