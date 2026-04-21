#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Stronger Drone Delivery PPO configuration.
更强版智运无人机 PPO 配置。
"""


class Config:
    # ---------------- Observation dimensions ----------------
    MAP_SIZE = 21
    IMAGE_CHANNELS = 3  # passable map + target field + npc risk
    IMAGE_FLAT_DIM = IMAGE_CHANNELS * MAP_SIZE * MAP_SIZE

    VECTOR_TASK_STAGE_DIM = 5
    VECTOR_PROGRESS_DIM = 4
    VECTOR_SAFETY_DIM = 4
    VECTOR_ACTION_PRIOR_DIM = 10
    VECTOR_RESOURCE_DIM = 3
    VECTOR_DIR_GUIDE_DIM = 8

    TOPK_STATION_K = 3
    TOPK_CHARGER_K = 2
    TOPK_NPC_K = 3
    VECTOR_TOPK_STATION_DIM = TOPK_STATION_K * 4
    VECTOR_TOPK_CHARGER_DIM = TOPK_CHARGER_K * 3
    VECTOR_TOPK_NPC_DIM = TOPK_NPC_K * 3
    VECTOR_COUNT_SUMMARY_DIM = 3

    VECTOR_OBS_DIM = (
        VECTOR_TASK_STAGE_DIM
        + VECTOR_PROGRESS_DIM
        + VECTOR_SAFETY_DIM
        + VECTOR_ACTION_PRIOR_DIM
        + VECTOR_RESOURCE_DIM
        + VECTOR_DIR_GUIDE_DIM
        + VECTOR_TOPK_STATION_DIM
        + VECTOR_TOPK_CHARGER_DIM
        + VECTOR_TOPK_NPC_DIM
        + VECTOR_COUNT_SUMMARY_DIM
    )

    # Legacy constants retained for compatibility with existing logic.
    LAST_ACTION_DIM = 8

    MAX_STATION_COUNT = 10
    MAX_CHARGER_COUNT = 4
    MAX_NPC_COUNT = 4

    # hero(6) + station(10*2) + charger(4*2) + npc(4*2) + warehouse(2) + env(4)
    PRIVILEGED_DIM = 48

    DIM_OF_OBSERVATION = VECTOR_OBS_DIM

    # ---------------- Action / value ----------------
    ACTION_NUM = 8
    LABEL_SIZE_LIST = [ACTION_NUM]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    VALUE_NUM = 1

    # ---------------- PPO hyperparameters ----------------
    GAMMA = 0.995
    LAMDA = 0.95

    INIT_LEARNING_RATE_START = 3e-4
    MIN_LEARNING_RATE = 8e-5
    LR_DECAY = 0.9995

    BETA_START = 0.010
    BETA_END = 0.0015
    ENTROPY_DECAY = 0.9993

    CLIP_PARAM = 0.18
    VF_COEF = 0.65
    HUBER_DELTA = 1.0
    MAX_KL = 0.03

    GRAD_CLIP_RANGE = 0.5
    USE_GRAD_CLIP = True

    # ---------------- Observation normalization ----------------
    OBS_NORM_EPS = 1e-4
    OBS_NORM_CLIP = 5.0

    # ---------------- Model ----------------
    IMAGE_EMBED_DIM = 128
    VECTOR_EMBED_DIM = 128
    FUSION_HIDDEN_DIM = 192
    CRITIC_PRIV_HIDDEN_DIM = 128

    HIDDEN_DIM = 192
    ACTOR_HIDDEN_DIM = 128
    CRITIC_HIDDEN_DIM = 192
    NUMB_HEAD = 1

    # ---------------- Reward shaping ----------------
    DELIVERY_REWARD = 2.2
    PICKUP_REWARD = 0.18
    CHARGE_GAIN_SCALE = 0.010
    STEP_PENALTY = -0.0012

    STATION_PROGRESS_SCALE = 0.050
    CHARGER_PROGRESS_SCALE = 0.030
    WAREHOUSE_PROGRESS_SCALE = 0.030

    INVALID_MOVE_PENALTY = -0.035
    OSCILLATION_PENALTY = -0.018
    REPEAT_MOVE_PENALTY = -0.006

    NPC_DANGER_PENALTY_SCALE = 0.048
    NPC_DANGER_RADIUS = 3.0

    CAN_DELIVER_BONUS = 0.06
    SUPPLY_BONUS = 0.06
    LOW_BATTERY_MOVE_AWAY_PENALTY = -0.020
    EMPTY_LOAD_MOVE_AWAY_PENALTY = -0.010

    # ---------------- Misc ----------------
    SAVE_OPTIMIZER_STATE = False
