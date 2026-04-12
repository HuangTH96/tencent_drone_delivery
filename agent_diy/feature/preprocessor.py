#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Huang Tianhao

Drone Delivery feature preprocessor.
智运无人机特征预处理器。
"""


import numpy as np
from agent_diy.conf.conf import Config


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

        # reward 相关
        self.prev_pos = (0, 0)
        self.prev_dist_to_target = None                         
        self.prev_battery = None
        self.prev_packages_count = -1

        # Game state / 游戏状态
        self.battery = None
        self.battery_max = None
        self.packages = []
        self.delivered = 0
        self.last_delivered = 0
        self.step_no = 0

        # Entities / 实体
        self.stations = []
        self.chargers = []
        self.warehouse = None
        self.npcs = []

    # 提取观测信息
    def _parse_obs(self, env_obs):
        """Parse essential fields from observation dict.

        从 observation 字典中解析必要字段。
        """
        obs = env_obs["observation"]
        frame_state = obs["frame_state"]

        hero = frame_state["heroes"]
        self.prev_pos = self.cur_pos                            # 记录前一刻ego位置， 为了惩罚原地不动
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])     # 当前ego位置
        
        self.prev_battery = self.battery                        # 记录前一刻电量，为了鼓励低电量时靠近充电站
        self.battery = hero.get("battery", self.battery_max)    # ego当前电量
        self.battery_max = hero.get("battery_max", 300)         # ego最大电量

        self.prev_packages_count = len(self.packages)           # 保存前一时刻代投的包裹数目
        self.packages = hero.get("packages", [])                # 目前需要去投递的包裹（驿站）

        self.last_delivered = self.delivered                    # 从开始到前一步的累计投递数目
        self.delivered = hero.get("delivered", 0)               # 从开始到当前这一步已经投递的包裹数目
        self.step_no = obs.get("step_no", 0)                    # 当前已经走了几步  

        self.stations = []                                      # 记录全局驿站
        self.chargers = []                                      # 记录全局充电站
        self.warehouse = None                                   # 记录仓库    
        for organ in frame_state.get("organs", []):
            st = organ.get("sub_type", 0)
            if st == 3:
                self.stations.append(organ)
            elif st == 2:
                self.chargers.append(organ)
            else:
                self.warehouse = organ

        self.npcs =frame_state["npcs"]                          # list of NPCs -> list[NpcState]
        self.legal_act = obs.get("legal_action", [1] * 8)       # 合法动作

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

        # 统计每个目标驿站对应的包裹数
        # self.packages 是 list[int]，元素为驿站编号，同一驿站可能出现多次；
        pkg_count = {}  # config_id -> 包裹数
        for pkg_id in self.packages:
            pkg_count[pkg_id] = pkg_count.get(pkg_id, 0) + 1

        # station_map = {s.get("config_id"): s for s in self.stations}

        # 遍历所有驿站（按config_id固定排序）
        # station_feat_list = []

        # for sid in range(1, Config.TOTAL_STATIONS + 1): 
        #     station = station_map.get(sid)   # -> OrganState

        #     if station is not None:
        #         pkg_num = pkg_count.get(sid, 0)
        #         # 根据有无包裹判断是否需要对该驿站的特征进行补零操作
        #         exist = 1.0 if pkg_num > 0 else 0.0
        #         if exist == 1.0:
        #             target_pos_s = (station["pos"]["x"], station["pos"]["z"])
        #             dir_x_s, dir_z_s, norm_dist_s = _get_target_feature(self.cur_pos, target_pos_s)
        #             norm_pkg_num = pkg_count[sid] / 3.0          # [0, 1]
        #         else:
        #             dir_x_s, dir_z_s, norm_dist_s, norm_pkg_num = 0.0, 0.0, 0.0, 0.0
        #     else: 
        #         # 对本局没有的驿站进行补零
        #         exist, dir_x_s, dir_z_s, norm_dist_s, norm_pkg_num = 0.0, 0.0, 0.0, 0.0, 0.0
            
        #     station_feat_list.append(np.array([exist, dir_x_s, dir_z_s, norm_dist_s, norm_pkg_num]))

        # station_feat = np.concatenate(station_feat_list)  # 50D
        # assert len(station_feat) == Config.TOTAL_STATIONS * Config.STATION_FEAT_DIM, \
        #     f"station_feat dim error: expected {Config.TOTAL_STATIONS * Config.STATION_FEAT_DIM}, got {len(station_feat)}"

        target_ids = set(self.packages)

        target_stations = [s for s in self.stations if s.get("config_id", 0) in target_ids]
        sorted_stations = sorted(target_stations, 
                                key=lambda s: np.sqrt((s["pos"]["x"] - self.cur_pos[0])**2 + 
                                                    (s["pos"]["z"] - self.cur_pos[1])**2))

        station_feat_list = []
        for i in range(Config.MAX_TARGET_STATIONS):
            if i < len(sorted_stations):
                s = sorted_stations[i]
                sid = s.get("config_id", 0)
                target_pos_s = (s["pos"]["x"], s["pos"]["z"])
                dir_x_s, dir_z_s, norm_dist_s = _get_target_feature(self.cur_pos, target_pos_s)
                norm_pkg_num = pkg_count.get(sid, 0) / 3.0
                station_feat_list.append(np.array([1.0, dir_x_s, dir_z_s, norm_dist_s, norm_pkg_num]))
            else:
                station_feat_list.append(np.zeros(Config.STATION_FEAT_DIM))

        station_feat = np.concatenate(station_feat_list)  # 15D

        # 添加充电站特征
        # == 语义信息：1. 添加所有充电桩（4个），记录中心点与hero方向信息，计算边界到hero的距离信息
        # == dir_x, dir_z, norm_boundary_dist,  # 充电桩1-4
        # == 其他说明：1. 在特征向量中固定槽位，明确“哪个特征属于哪个实体”；2. 按config_id排序填充每个驿站的特征; 3. TODO: 充电桩有充电范围，不应该以充电桩位置为target_pos_c
        charger_feat_list = []  
        sorted_chargers = sorted(self.chargers, key=lambda x:x.get("config_id", 0))
        for i in range(Config.TOTAL_CHARGERS):
            if i < len(sorted_chargers):
                c = sorted_chargers[i]
                target_pos_c = (c["pos"]["x"], c["pos"]["z"])
                dir_x_c, dir_z_c, norm_dist_c = _get_target_feature(self.cur_pos, target_pos_c)
                charger_feat_list.append(np.array([dir_x_c, dir_z_c, norm_dist_c]))
            else:
                charger_feat_list.append(np.zeros(Config.CHARGER_FEAT_DIM))

        charger_feat = np.concatenate(charger_feat_list)    # 12D
        assert len(charger_feat) == Config.TOTAL_CHARGERS * Config.CHARGER_FEAT_DIM, \
            f"charger_feat dim error: expected {Config.TOTAL_CHARGERS * Config.CHARGER_FEAT_DIM}, got {len(charger_feat)}"

        # 添加仓库信息
        # == 语义信息：1. 计算中心点与hero的相对位置，计算边界到hero的距离
        # == 其他说明：1. 在特征向量中固定槽位，明确“哪个特征属于哪个实体”；2. TODO：仓库分装货区域和非装货区域
        # == warehouse_dir_x, warehouse_dir_z, norm_boundary_dist
        target_pos_w = (self.warehouse["pos"]["x"], self.warehouse["pos"]["z"])
        dir_x_w, dir_z_w, norm_dist_w = _get_target_feature(self.cur_pos, target_pos_w)

        warehouse_feat = np.array([dir_x_w, dir_z_w, norm_dist_w])

        # 添加NPC信息
        # == 语义信息：1. 计算NPCs 和 hero之间的方向信息和以及归一化的距离信息
        # == 其他说明：1. 跨局槽位对应无所谓，只需要保证单个episode内，在特征向量中固定槽位，明确“哪个特征属于哪个实体”；2. 按config_id排序填充每个驿站的特征；3. 需要补零，保持特征个数
        # == exist, dir_x, dir_z, norm_dist_scalar,  # NPC
        npc_feat_list = []
        sorted_npcs = sorted(self.npcs, key=lambda x:x.get("npc_id", ""))
        for i in range(Config.TOTAL_NPCS):
            if i < len(sorted_npcs):
                npc =sorted_npcs[i]
                target_pos_npc = (npc["pos"]["x"], npc["pos"]["z"])
                exist = 1.0
                dir_x_npc, dir_z_npc, norm_dist_npc = _get_target_feature(self.cur_pos, target_pos_npc)
            else:
                exist = 0.0
                dir_x_npc, dir_z_npc, norm_dist_npc = 0.0, 0.0, 0.0
            
            npc_feat_list.append(np.array([exist, dir_x_npc, dir_z_npc, norm_dist_npc]))
        
        npc_feat = np.concatenate(npc_feat_list)    # 16D
        assert len(npc_feat) == Config.TOTAL_NPCS * Config.NPC_FEAT_DIM, \
            f"npc_feat dim error: expected {Config.TOTAL_NPCS * Config.NPC_FEAT_DIM}, got {len(npc_feat)}"

        # 3. Legal action mask (8D) / 合法动作掩码（8D）
        legal_action = self._get_legal_action()

        # Concatenate features (Total 93D)
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

    # 增加奖励/惩罚
    def _reward_process(self):
        """Reward function.

        - 成功投递一个包裹：1
        - 地图大小：128 * 128 个单位
        - 离开仓库，不充电的情况下，单趟投递行为限制步数：300，每步移动一个单位
        - 步数惩罚：0.001
        - 惩罚原地不动：0.01
        - 靠近npc： -0.1
        - 被NPC抓获：-5
        - 靠近最近的驿站： 0.002
        - 鼓励低电量去充电： 0.3
        - 奖励投完后补充包裹：0.3
        """
        reward = 0.0

        # 1. Delivery reward / 投递奖励
        newly_delivered = max(0, self.delivered - self.last_delivered)
        if newly_delivered > 0:
            reward += Config.DELIVERED * newly_delivered

        # 2. Step penalty / 步数惩罚
        reward -= Config.STEP_PENALTY

        # 3. 惩罚原地不动
        if self.cur_pos == self.prev_pos:
            reward -= Config.STATIONARY_PENALTY

        # 4. 惩罚接近NPC，以及被抓获
        # if self.npcs:
        #     min_npc_dist = min(
        #         np.sqrt((npc["pos"]["x"] - self.cur_pos[0])**2 + 
        #                 (npc["pos"]["z"] - self.cur_pos[1])**2)
        #         for npc in self.npcs
        #     )
            
        #     # 被抓获（距离<=1格，任务终止）
        #     if min_npc_dist <= 1:
        #         reward -= 1.0
        #     # 接近NPC预警（距离<=5格）
        #     elif min_npc_dist <= 3:
        #         reward -= 0.01 # * (3 - min_npc_dist)  # 越近惩罚越大，范围[0.1, 0.4]

        # 5. 奖励有包裹时，靠近当前携带包裹对应的驿站中距离最近的那个
        if self.packages and self.stations:
            # 找当前携带包裹对应的驿站中距离最近的
            station_map = {s.get("config_id"): s for s in self.stations}
            pkg_ids = set(self.packages)
            
            min_dist = float('inf')
            for sid in pkg_ids:
                s = station_map.get(sid)
                if s is not None:
                    dist = np.sqrt((s["pos"]["x"] - self.cur_pos[0])**2 +
                                (s["pos"]["z"] - self.cur_pos[1])**2)
                    if dist < min_dist:
                        min_dist = dist
            
            if self.prev_dist_to_target is not None:
                dist_delta = self.prev_dist_to_target - min_dist
                if dist_delta > 0:
                    reward += Config.APPROACH_STATION * dist_delta  # 靠近给奖励
                    # print(f"【奖励】靠近目标，当前奖励为:{reward}")
                else: 
                    reward += Config.LEAVE_STATION * dist_delta     # 远离给惩罚
                    # print(f"【惩罚】远离目标，当前奖励为：{reward}")
            self.prev_dist_to_target = min_dist
        else:
            self.prev_dist_to_target = None

        # 6. 鼓励低电量是及时充电
        # if self.prev_battery is not None:
        #     prev_battery_ratio = self.prev_battery / max(self.battery_max, 1)
        #     if self.battery == self.battery_max and prev_battery_ratio < 0.3:
        #         reward += 0.00001

        # # 7. 仓库奖励，奖励无包裹时返回仓库补充
        # if self.prev_packages_count >= 0:    # 跳过第一步，此时prev_package_count为-1
        #     # 奖励在没有包裹的情况下，回到仓库又装好了
        #     if self.prev_packages_count == 0 and len(self.packages) == 3:
        #         reward += 0.00001
        return [reward]
