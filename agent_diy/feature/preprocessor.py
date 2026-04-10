#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Drone Delivery feature preprocessor.
智运无人机特征预处理器。
"""


import numpy as np
from agent_ppo.conf.conf import Config


def norm(v, max_v, min_v=0):
    """Normalize v to [0, 1].

    将 v 归一化到 [0, 1]。
    """
    v = np.clip(v, min_v, max_v)
    return (v - min_v) / (max_v - min_v)

def _get_target_feature(hero_pos, target_pos):
    relative_pos = (target_pos[0] - hero_pos[0], target_pos[1] - hero_pos[1])
    dist = np.sqrt(relative_pos[0] ** 2 + relative_pos[1] ** 2)
    dir_x = relative_pos[0] / max(dist, 1e-4)   # [-1, 1]
    dir_z = relative_pos[1] / max(dist, 1e-4)   # [-1, 1]
    norm_dist = norm(dist, 1.41 * 128)           # [0, 1]

    return dir_x, dir_z, norm_dist

class Preprocessor:
    """feature preprocessor for Drone Delivery.

    智运无人机预处理器，仅保留最少信息。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all internal state.

        重置所有状态。
        """
        self.cur_pos = (0, 0)

        # Game state / 游戏状态
        self.battery = 100
        self.battery_max = 100
        self.packages = []
        self.delivered = 0
        self.last_delivered = 0
        self.step_no = 0

        # Entities / 实体
        self.stations = []

    # TODO：修改可以获得的观测信息
    def _parse_obs(self, env_obs):
        """Parse essential fields from observation dict.

        从 observation 字典中解析必要字段。
        """
        obs = env_obs["observation"]
        frame_state = obs["frame_state"]

        hero = frame_state["heroes"]
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])     # ego位置

        self.battery = hero.get("battery", self.battery_max)    # ego当前电量
        self.battery_max = hero.get("battery_max", 100)         # ego最大电量
        self.packages = hero.get("packages", [])                # 目前需要去投递的包裹（驿站）编号

        self.last_delivered = self.delivered                    # 从开始到前一步的累计投递数目
        self.delivered = hero.get("delivered", 0)               # 从开始到当前这一步已经投递的包裹数目
        self.step_no = obs.get("step_no", 0)                    # 当前已经走了几步  

        self.stations = []                                      # 记录全局驿站
        for organ in frame_state.get("organs", []):
            st = organ.get("sub_type", 0)
            if st == 3:
                self.stations.append(organ)

        self.legal_act = obs.get("legal_action", [1] * 8)       # 合法动作

    # TODO
    def feature_process(self, env_obs, last_action):
        """Core feature extraction. Returns (feature_22d, legal_action, reward).

        核心特征提取方法，返回 22 维特征向量、合法动作掩码和奖励。
        """
        self._parse_obs(env_obs)

        # 1. Hero state features (4D) / 英雄状态特征（4D）
        battery_ratio = norm(self.battery, self.battery_max)
        package_count_norm = norm(len(self.packages), 3)
        cur_pos_norm = norm(np.array(self.cur_pos, dtype=float), 128, -128)
        hero_feat = np.array(
            [
                battery_ratio,
                package_count_norm,
                cur_pos_norm[0],
                cur_pos_norm[1],
            ]
        )

        # 更新驿站特征
        # == 语义信息：1. 与hero之间的相对距离、hero携带的，与该驿站相关的包裹数
        # == exist, dir_x, dir_z, norm_center_dist, norm_pkg_num,  # 驿站1-10
        # == 其他说明：
        # 1. 在特征向量中固定槽位，明确“哪个特征属于哪个实体”；
        # 2. 按config_id排序填充每个驿站的特征
        # 3. 投递后会自动删除该包裹，比如一开始是[3,6,9]，当投递6后，self.packages会变成 [3,9]；也就是说，驿站的个数会变化
        
        TOTAL_STATIONS = 10
        STATION_FEAT_DIM = 5  

        # 统计每个目标驿站对应的包裹数
        # self.packages 是 list[int]，元素为驿站编号，同一驿站可能出现多次；
        pkg_count = {}  # config_id -> 包裹数
        for pkg_id in self.packages:
            pkg_count[pkg_id] = pkg_count.get(pkg_id, 0) + 1

        # 提取所有驿站的信息
        station_map = {s.get("config_id"): s for s in self.stations} # -> dict{int32:Organstate}

        # 遍历所有驿站（按config_id固定排序）
        station_feat_list = []
        for station in sorted(self.stations, key=lambda x:x.get("config_id", 0)):
            sid = station.get("config_id", 0)   # -> OrganState
            pkg_num = pkg_count.get(sid, 0)
            exist = 1.0 if pkg_num > 0 else 0.0
            if exist:
                target_pos = (s["pos"]["x"], s["pos"]["z"])
                dir_x, dir_z, norm_dist = _get_target_feature(self.cur_pos, target_pos)
                norm_pkg_num = pkg_count[sid] / 3.0          # [0, 1]
            else:
                dir_x, dir_z, norm_dist, norm_pkg_num = 0.0, 0.0, 0.0, 0.0

            station_feat_list.append(np.array([exist, dir_x, dir_z, norm_dist, norm_pkg_num]))

        station_feat = np.concatenate(station_feat_list)  # 50D
        assert len(station_feat) == MAX_TARGET_STATIONS * STATION_FEAT_DIM, \
            f"station_feat dim error: expected {MAX_TARGET_STATIONS * STATION_FEAT_DIM}, got {len(station_feat)}"

        # TODO： 添加充电站特征
        # == 语义信息：1. 添加所有充电桩（4个），记录中心点与hero方向信息，计算边界到hero的距离信息
        # == 其他说明：1. 在特征向量中固定槽位，明确“哪个特征属于哪个实体”；2. 按config_id排序填充每个驿站的特征
        # == exist, dir_x, dir_z, norm_boundary_dist,  # 充电桩1
        # == exist, dir_x, dir_z, norm_boundary_dist,  # 充电桩2
        # == exist, dir_x, dir_z, norm_boundary_dist,  # 充电桩3
        # == exist, dir_x, dir_z, norm_boundary_dist,  # 充电桩4
        # 充电桩尺寸 3 * 3


        charger_feat = None


        # TODO： 添加仓库信息
        # == 语义信息：1. 计算中心点与hero的相对位置，计算边界到hero的距离
        # == 其他说明：1. 在特征向量中固定槽位，明确“哪个特征属于哪个实体”；
        # == warehouse_dir_x, warehouse_dir_z, norm_boundary_dist


        warehouse_feat = None



        # TODO：添加NPC信息
        # == 语义信息：1. 计算NPCs 和 hero之间的方向信息和以及归一化的距离信息
        # == 其他说明：1. 在特征向量中固定槽位，明确“哪个特征属于哪个实体”；2. 按config_id排序填充每个驿站的特征
        # == exist, dir_x, dir_z, norm_dist_scalar,  # NPC1
        # == exist, dir_x, dir_z, norm_dist_scalar,  # NPC2
        # == exist, dir_x, dir_z, norm_dist_scalar,  # NPC3
        # == exist, dir_x, dir_z, norm_dist_scalar,  # NPC4




        npc_feat = None

        # 3. Legal action mask (8D) / 合法动作掩码（8D）
        legal_action = self._get_legal_action()


        # Concatenate features (Total 22D / 合计 22D)
        feature = np.concatenate(
            [
                hero_feat,
                station_feat,
                charger_feat,
                warehouse_feat,
                npc_feat,
                np.array(legal_action, dtype=float),
            ]
        )

        reward = self._reward_process()

        return feature, legal_action, reward

    def _get_legal_action(self):
        """Get legal action mask.

        获取合法动作掩码。
        """
        if hasattr(self, "legal_act") and self.legal_act:
            legal_action = [int(x) for x in self.legal_act[:8]]
        else:
            legal_action = [1] * 8

        if sum(legal_action) == 0:
            return [1] * 8

        return legal_action

    # TODO：增加奖励/惩罚
    # == 1. 惩罚被NPC抓到
    # == 2. 奖励有包裹单低电量时成功充电
    # == 3. 奖励无包裹时返回仓库
    # == 4. 奖励有包裹时靠近驿站
    def _reward_process(self):
        """Reward function.

        奖励函数。
        """
        reward = 0.0

        # 1. Delivery reward / 投递奖励
        newly_delivered = max(0, self.delivered - self.last_delivered)
        if newly_delivered > 0:
            reward += 1.0 * newly_delivered

        # 2. Step penalty / 步数惩罚
        reward -= 0.001

        return [reward]
