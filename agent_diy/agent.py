#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Drone Delivery DIY Agent class based on kaiwudrl BaseAgent interface.
智运无人机 DIY Agent 主类，基于 kaiwudrl BaseAgent 接口。
"""


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_diy.algorithm.algorithm import Algorithm 
from agent_diy.model.model import Model 
from agent_diy.feature.definition import ActData, ObsData
from agent_diy.feature.preprocessor import Preprocessor 
from agent_diy.conf.conf import Config

class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        """Reset per-episode state.

        每局开始时重置状态。
        """
        self.preprocessor.reset()
        self.last_action = -1

    def _forward(self, feature, legal_action):
        """Gradient-free forward pass, returns (logits, value).

        无梯度前向推理，返回 (logits, value)。
        """
        self.model.set_eval_mode()
        obs_t = (
            torch.tensor(np.array([feature]), dtype=torch.float32).view(1, Config.DIM_OF_OBSERVATION).to(self.device)
        )
        with torch.no_grad():
            logits, value = self.model(obs_t, inference=True)
        return logits.cpu().numpy()[0], value.cpu().numpy()[0]

    def predict(self, list_obs_data):
        """Training inference: sample action from probability distribution.

        训练时推理：按概率采样动作（探索）。
        """
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action

        logits, value = self._forward(feature, legal_action)
        legal_np = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits, legal_np)
        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)

        return [ActData(action=[action], d_action=[d_action], prob=list(prob), value=value)]

    def exploit(self, env_obs):
        """Evaluation inference: greedy action selection.

        评估时推理：贪心选最大概率动作（利用）。
        """
        obs_data, _ = self.observation_process(env_obs)
        if obs_data is None:
            return 0
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    def learn(self, list_sample_data):
        """Delegate to Algorithm for PPO update.

        委托给 Algorithm 执行 PPO 更新。
        """
        return self.algorithm.learn(list_sample_data)

    def observation_process(self, env_obs):
        """Convert raw env observation to ObsData + remain_info.

        将原始环境观测转换为 ObsData + remain_info。
        """

        # TODO： 重建feature和reward
        feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
        remain_info = {"reward": reward}
        return (
            ObsData(feature=list(feature), legal_action=legal_action),
            remain_info,
        )

    def action_process(self, act_data, is_stochastic=True):
        """Extract int action from ActData and update last_action.

        从 ActData 提取整数动作，并更新 last_action。
        """
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return self.last_action

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self.logger.info(f"load model {model_file_path} successfully")

    def _legal_soft_max(self, logits, legal_action):
        """Apply legal action mask and compute normalized probabilities.

        对 logits 应用合法动作掩码并计算归一化概率。
        """
        _w, _e = 1e20, 1e-5
        tmp = logits - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        return tmp / (np.sum(tmp, keepdims=True) * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        """Sample from probability distribution (or argmax).

        从概率分布中采样（或取最大值）。
        """
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))


# ========================================= #
# class Agent(BaseAgent):
#     def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
#         """Initialize the agent.

#         初始化 Agent。
#         """
#         super().__init__(agent_type, device, logger, monitor)

#     def predict(self, list_obs_data):
#         """Predict action from observation data.

#         根据观测数据推理动作。
#         """
#         pass

#     def exploit(self, list_obs_data):
#         """Evaluation mode inference (greedy).

#         评估模式推理（贪心）。
#         """
#         pass

#     def learn(self, list_sample_data):
#         """Train the model.

#         训练模型。
#         """
#         pass

#     def save_model(self, path=None, id="1"):
#         """Save model checkpoint.

#         保存模型检查点。
#         """
#         pass

#     def load_model(self, path=None, id="1"):
#         """Load model checkpoint.

#         加载模型检查点。
#         """
#         pass

#     def observation_process(self, obs, preprocessor, extra_info=None):
#         """This function is an important feature processing function, mainly responsible for:
#             - Parsing information in the raw data
#             - Parsing preprocessed feature data
#             - Processing the features and returning the processed feature vector
#             - Concatenation of features
#             - Annotation of legal actions
#         Function inputs:
#             - obs: Local observation information returned by the environment
#             - preprocessor: Preprocessor
#             - extra_info: Global information returned by the environment
#         Function outputs:
#             - ObsData: Observation data for model inference
#             - remain_info: Other data for reward calculation

#         该函数是特征处理的重要函数, 主要负责：
#             - 解析原始数据里的信息
#             - 解析预处理后的特征数据
#             - 对特征进行处理, 并返回处理后的特征向量
#             - 特征的拼接
#             - 合法动作的标注
#         函数的输入：
#             - obs: 环境返回的局部观测信息
#             - preprocessor: 预处理器
#             - extra_info: 环境返回的全局状态信息
#         函数的输出：
#             - ObsData: 用于模型推理的观测数据
#             - remain_info: 用于奖励计算的其他数据
#         """
#         pass

#     def action_process(self, act_data):
#         """Process action data.

#         处理动作数据。
#         """
#         pass

# ===================================== #