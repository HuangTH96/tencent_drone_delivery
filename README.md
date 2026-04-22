**特征工程TODO list**

高优先级（直接影响决策质量）

1. `_select_primary_target`：无包裹+低电量时应就近充电
已经讨论过，加入可达性判断，能到仓库就去仓库，不能到则就近充电桩。【完成】

2. 充电决策逻辑：用 `need_recharge()` 替代25%硬阈值。同时影响 `_select_primary_target`、`_build_dir_score_feature` 里的低电量加成、`_reward_process` 里的充电奖励触发条件，需要同步修改三处。【完成】

3. `_reward_process`：去掉 REPEAT_MOVE_PENALTY, 直线运动会被错误惩罚。【完成】

4. `_reward_process`：补给进度奖励中 supply 应区分仓库和充电桩。无包裹时应该鼓励朝仓库靠近，而不是最近补给点（可能是充电桩）。【完成】


中优先级（特征质量提升）

5. `_build_dir_score_feature`：NPC安全得分改为惩罚项，且距离≤1时给强负分
当前即将被抓时仍给正分的问题。

6. `_build_privileged_feature`：充电站改为按固定ID排序
防止同一充电桩在不同时刻槽位漂移，与驿站的处理方式保持一致。

7. `_build_privileged_feature`：charger 和 warehouse 的目标标志位分开
把 {"charger", "warehouse"} 拆成两个独立 flag，PRIVILEGED_DIM 从48改成49。

8. `_make_topk_entity_feature`：去掉冗余的 with_target=True 的固定1.0
TOPK_STATION_K 从4改成3，VECTOR_OBS_DIM 相应减少3。


低优先级（防御性/边界情况）

9. `_is_on_warehouse`：确认仓库坐标语义后确定正确的范围判断，需要查官网文档。【完成】

10. `_reward_process`：去掉 CAN_DELIVER_BONUS。进度奖励已经覆盖这个场景，且存在刷分风险。【完成】

11. `_get_legal_action`：确认环境 legal_act 是否已包含地图信息。如果是，自己的地图 mask 是冗余的，可以简化。

12. `TOPK_CHARGER_K` 和 `TOPK_NPC_K` 改为4，与 MAX_CHARGER_COUNT 和 MAX_NPC_COUNT 保持一致，避免遗漏信息。