from collections import deque
from typing import Optional, List, Tuple
import math

ACTION_TO_DELTA = {
    0: (1, 0), 1: (1, -1), 2: (0, -1), 3: (-1, -1),
    4: (-1, 0), 5: (-1, 1), 6: (0, 1),  7: (1, 1),
}
DELTA_TO_ACTION = {v: k for k, v in ACTION_TO_DELTA.items()}

def get_first_step_action(
    local_map: List[List[int]],
    target_pos: Tuple[int, int],
    cur_pos: Tuple[int, int],
    npc_positions: List[Tuple[int, int]],
    npc_danger_radius: float = 2.0,
) -> Optional[int]:
    """
    在21×21局部地图上BFS，返回朝目标的第一步动作（0-7），无路可走返回None。
    npc_positions：当前帧NPC的局部地图坐标，BFS会绕开NPC周围danger_radius格。
    """
    MAP_SIZE = 21
    center = 10  # hero始终在(10,10)
    
    # 目标转局部坐标
    dx = int(round(target_pos[0] - cur_pos[0]))
    dz = int(round(target_pos[1] - cur_pos[1]))
    goal_lx = center + dx
    goal_lz = center + dz
    
    # 目标不在视野内，clip到边界
    goal_lx = max(0, min(MAP_SIZE - 1, goal_lx))
    goal_lz = max(0, min(MAP_SIZE - 1, goal_lz))
    
    # 构建NPC危险区（局部坐标）
    npc_blocked = set()
    for npc_lx, npc_lz in npc_positions:
        for ddx in range(-int(npc_danger_radius) - 1, int(npc_danger_radius) + 2):
            for ddz in range(-int(npc_danger_radius) - 1, int(npc_danger_radius) + 2):
                if math.sqrt(ddx**2 + ddz**2) <= npc_danger_radius:
                    bx, bz = npc_lx + ddx, npc_lz + ddz
                    if 0 <= bx < MAP_SIZE and 0 <= bz < MAP_SIZE:
                        npc_blocked.add((bx, bz))
    
    start = (center, center)
    goal = (goal_lx, goal_lz)
    
    if start == goal:
        return None
    
    # BFS
    visited = {start: None}  # node -> (parent, action)
    queue = deque([start])
    
    while queue:
        cx, cz = queue.popleft()
        
        for action, (adx, adz) in ACTION_TO_DELTA.items():
            nx, nz = cx + adx, cz + adz
            
            if not (0 <= nx < MAP_SIZE and 0 <= nz < MAP_SIZE):
                continue
            if local_map[nz][nx] != 1:
                continue
            if (nx, nz) in npc_blocked:
                continue
            # 斜向穿角检查
            if adx != 0 and adz != 0:
                if local_map[cz][cx + adx] != 1 and local_map[cz + adz][cx] != 1:
                    continue
            if (nx, nz) in visited:
                continue
            
            visited[(nx, nz)] = (cx, cz), action
            
            if (nx, nz) == goal:
                # 回溯找第一步
                node = (nx, nz)
                while visited[node][0] != start:
                    node = visited[node][0]
                return visited[node][1]
            
            queue.append((nx, nz))
    
    return None  # 无路可走