#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Stronger Drone Delivery Agent.
更强版智运无人机 Agent。
"""

import os
from pathlib import Path

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model


class RunningMeanStd:
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = float(eps)

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            batch_mean = x
            batch_var = np.zeros_like(x, dtype=np.float32)
            batch_count = 1.0
        else:
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            batch_count = float(x.shape[0])
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta * delta * self.count * batch_count / total_count
        new_var = m_2 / max(total_count, 1e-6)
        self.mean = new_mean.astype(np.float32)
        self.var = np.maximum(new_var, Config.OBS_NORM_EPS).astype(np.float32)
        self.count = float(total_count)


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        np.random.seed(0)
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
        self.obs_rms = RunningMeanStd(shape=(Config.VECTOR_OBS_DIM,))
        self.last_action = -1
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        self.preprocessor.reset()
        self.last_action = -1

    def _normalize_obs(self, vector_obs: np.ndarray, update: bool = True) -> np.ndarray:
        vector_obs = np.asarray(vector_obs, dtype=np.float32)
        if update:
            self.obs_rms.update(vector_obs)
        normed = (vector_obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + Config.OBS_NORM_EPS)
        return np.clip(normed, -Config.OBS_NORM_CLIP, Config.OBS_NORM_CLIP).astype(np.float32)

    def _forward(self, vector_obs, image_obs, privileged_obs, legal_action):
        self.model.set_eval_mode()
        vector_t = torch.tensor(np.array([vector_obs]), dtype=torch.float32).view(1, Config.VECTOR_OBS_DIM).to(self.device)
        image_t = (
            torch.tensor(np.array([image_obs]), dtype=torch.float32)
            .view(1, Config.IMAGE_CHANNELS, Config.MAP_SIZE, Config.MAP_SIZE)
            .to(self.device)
        )
        priv_t = torch.tensor(np.array([privileged_obs]), dtype=torch.float32).view(1, Config.PRIVILEGED_DIM).to(self.device)
        with torch.no_grad():
            logits, value = self.model(image_t, vector_t, priv_t, inference=True)
        return logits.cpu().numpy()[0], value.cpu().numpy()[0]

    def predict(self, list_obs_data):
        vector_obs = list_obs_data[0].vector_obs
        image_obs = list_obs_data[0].image_obs
        privileged_obs = list_obs_data[0].privileged_obs
        legal_action = list_obs_data[0].legal_action

        logits, value = self._forward(vector_obs, image_obs, privileged_obs, legal_action)
        legal_np = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits, legal_np)
        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)

        return [ActData(action=[action], d_action=[d_action], prob=list(prob), value=value)]

    def exploit(self, env_obs):
        obs_data, _ = self.observation_process(env_obs)
        if obs_data is None:
            return 0
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def observation_process(self, env_obs):
        vector_obs, image_obs, privileged_obs, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
        vector_obs = self._normalize_obs(vector_obs, update=True)
        remain_info = {"reward": reward}
        return (
            ObsData(
                vector_obs=list(vector_obs),
                image_obs=image_obs,
                privileged_obs=list(privileged_obs),
                legal_action=legal_action,
            ),
            remain_info,
        )

    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return self.last_action

    def _resolve_model_file(self, path=None, id="1"):
        base_dir = path
        if base_dir is None:
            env_dir = os.environ.get('MODEL_DIR') or os.environ.get('KAIWU_MODEL_DIR')
            base_dir = env_dir if env_dir else str(Path.cwd())
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        return str(Path(base_dir) / f"model.ckpt-{str(id)}.pkl")

    def save_model(self, path=None, id="1"):
        model_file_path = self._resolve_model_file(path, id)
        payload = {
            'model_state_dict': {k: v.clone().cpu() for k, v in self.model.state_dict().items()},
            'obs_rms_mean': self.obs_rms.mean,
            'obs_rms_var': self.obs_rms.var,
            'obs_rms_count': self.obs_rms.count,
        }
        if Config.SAVE_OPTIMIZER_STATE:
            payload['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(payload, model_file_path)
        if self.logger:
            self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        model_file_path = self._resolve_model_file(path, id)
        if not os.path.exists(model_file_path):
            if self.logger:
                self.logger.warning(f"load model skipped, file not found: {model_file_path}")
            return
        payload = torch.load(model_file_path, map_location=self.device)
        try:
            if isinstance(payload, dict) and 'model_state_dict' in payload:
                self.model.load_state_dict(payload['model_state_dict'])
                mean = np.asarray(payload.get('obs_rms_mean', self.obs_rms.mean), dtype=np.float32)
                var = np.asarray(payload.get('obs_rms_var', self.obs_rms.var), dtype=np.float32)
                if mean.shape == self.obs_rms.mean.shape:
                    self.obs_rms.mean = mean
                if var.shape == self.obs_rms.var.shape:
                    self.obs_rms.var = var
                self.obs_rms.count = float(payload.get('obs_rms_count', self.obs_rms.count))
                if Config.SAVE_OPTIMIZER_STATE and 'optimizer_state_dict' in payload:
                    self.optimizer.load_state_dict(payload['optimizer_state_dict'])
            else:
                self.model.load_state_dict(payload)
        except RuntimeError as ex:
            if self.logger:
                self.logger.warning(f"load model skipped due to state_dict mismatch: {ex}")
            return
        if self.logger:
            self.logger.info(f"load model {model_file_path} successfully")

    def _legal_soft_max(self, logits, legal_action):
        _w, _e = 1e20, 1e-5
        tmp = logits - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        return tmp / (np.sum(tmp, keepdims=True) * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))
