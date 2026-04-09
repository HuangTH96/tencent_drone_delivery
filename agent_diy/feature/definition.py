#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from common_python.utils.common_func import create_cls
import numpy as np
from agent_diy.conf.conf import Config

# # The create_cls function is used to dynamically create a class. The first parameter of the function is the type name,
# # and the remaining parameters are the attributes of the class, which should have a default value of None.
# # create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
# ObsData = create_cls(
#     "ObsData",
#     feature=None,
#     legal_act=None,
# )


# ActData = create_cls(
#     "ActData",
#     act=None,
# )


# # SampleData is used to transfer training samples between aisrv and learner.
# # SampleData用于在aisrv和learner之间传递训练样本
# SampleData = create_cls(
#     "SampleData",
#     obs=153,  # Observation dimension / 观测维度
#     legal_actions=8,  # Legal action dimension / 合法动作维度
#     actions=1,  # Action dimension / 动作维度
#     probs=8,  # Action probability distribution dimension / 动作概率分布维度
#     rewards=1,  # Reward / 奖励
#     advantages=1,  # Advantage function / 优势函数
#     values=1,  # Value function / 价值函数
#     dones=1,  # Whether terminated / 是否结束
# )


# def reward_shaping(frame_no, score, terminated, truncated, remain_info, _remain_info, obs, _obs):
#     """Reward shaping function.

#     奖励塑形函数。
#     """
#     pass


# def sample_process(list_game_data):
#     """Sample processing function.

#     样本处理函数。
#     """
#     pass



# ===================================== #
# ObsData: feature vector + legal action mask
# 观测数据：feature 为特征向量，legal_action 为合法动作掩码
ObsData = create_cls("ObsData", feature=None, legal_action=None)

# ActData: sampled action, greedy action, action probabilities, state value
# 动作数据：action 为采样动作，d_action 为贪心动作，prob 为动作概率，value 为状态价值
ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)

# SampleData: int values are treated as dimensions by the framework
# 样本数据：字段值为 int 时框架自动按维度处理
# TODO：根据feature更改SampleData
SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    act=1,
    reward=Config.VALUE_NUM,
    done=1,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,
    reward_sum=Config.VALUE_NUM,
)

def sample_process(list_sample_data):
    """Sample post-processing: fill next_value and compute GAE advantage.

    填充 next_value 并用 GAE 计算 advantage / reward_sum。
    """
    # Fill next_value: each step's next_value = next step's value
    # 填充 next_value：每步的 next_value = 下一步的 value
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    """Generalized Advantage Estimation (GAE).

    广义优势估计。
    """
    gae = 0.0
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    for sample in reversed(list_sample_data):
        delta = sample.reward - sample.value + gamma * sample.next_value
        gae = gae * gamma * lamda + delta
        sample.advantage = gae
        sample.reward_sum = gae + sample.value