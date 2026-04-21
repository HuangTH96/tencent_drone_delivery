#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Drone Delivery training workflow.
智运无人机训练工作流。
"""


import os
import time

import numpy as np
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """Training entry point called by the platform.

    训练入口，平台调用。
    """
    last_save_model_time = time.time()
    env = envs[0]
    agent = agents[0]

    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, check agent_ppo/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    """Single-episode runner.

    单局运行器。
    """

    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_training_metrics_time = 0
        self.rng = np.random.default_rng()

    @staticmethod
    def _normalize_range(value, lo, hi):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return (lo, hi)
        a, b = int(value[0]), int(value[1])
        if a > b:
            a, b = b, a
        return (max(lo, a), min(hi, b))

    def _build_episode_conf(self):
        env_conf = dict(self.usr_conf.get("env_conf", {}))
        if not env_conf.get("episode_randomize", False):
            return self.usr_conf

        d_lo, d_hi = self._normalize_range(env_conf.get("drone_count_range", [1, 4]), 1, 4)
        c_lo, c_hi = self._normalize_range(env_conf.get("charger_count_range", [1, 4]), 1, 4)
        s_lo, s_hi = self._normalize_range(env_conf.get("station_count_range", [3, 10]), 3, 10)
        b_lo, b_hi = self._normalize_range(env_conf.get("battery_max_range", [100, 999]), 100, 999)

        env_conf["drone_count"] = int(self.rng.integers(d_lo, d_hi + 1))
        env_conf["charger_count"] = int(self.rng.integers(c_lo, c_hi + 1))
        env_conf["station_count"] = int(self.rng.integers(s_lo, s_hi + 1))
        env_conf["battery_max"] = int(self.rng.integers(b_lo, b_hi + 1))

        return {"env_conf": env_conf}

    def run_episodes(self):
        """Run one episode and yield the sample list.

        运行一局并 yield 样本列表。
        """
        while True:
            # Print training metrics periodically / 定期打印训练指标
            now = time.time()
            if now - self.last_training_metrics_time >= 60:
                metrics = get_training_metrics()
                self.last_training_metrics_time = now
                if metrics:
                    self.logger.info(f"training_metrics: {metrics}")

            # Reset environment / 重置环境
            episode_conf = self._build_episode_conf()
            env_obs = self.env.reset(episode_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            obs_data, remain_info = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_delivered = 0
            total_reward_sum = 0.0

            self.logger.info(f"Episode {self.episode_cnt} start, conf={episode_conf}")

            # Main loop / 主循环
            while not done:
                # Inference / 推理
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data)

                # Environment step / 环境交互
                env_reward, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                step += 1
                done = terminated or truncated

                # Next step observation / 下一步观测
                _obs_data, _remain_info = self.agent.observation_process(env_obs)

                # Reward from observation_process remain_info (avoid double computation).
                # 从 observation_process 的 remain_info 取奖励，避免重复计算。
                reward = np.array(_remain_info.get("reward", [0.0]), dtype=np.float32)

                total_reward_sum += float(reward.sum())

                # End-of-episode additional reward / 局末额外奖励
                final_reward = np.zeros(Config.VALUE_NUM, dtype=np.float32)
                if done:
                    total_delivered = self.agent.preprocessor.delivered
                    total_score = env_obs["observation"]["env_info"].get("total_score", 0)

                    if terminated:
                        # Abnormal termination (collision or energy depleted): small penalty
                        # 异常终止（碰撞 or 电量耗尽）：给小惩罚
                        final_reward[0] = -1.0
                        result_str = "FAIL"
                    else:
                        # Normal end: reached max steps
                        # 正常到达最大步数
                        final_reward[0] = 0.0
                        result_str = "WIN"

                    self.logger.info(
                        f"[GAMEOVER] ep:{self.episode_cnt} steps:{step} "
                        f"result:{result_str} delivered:{total_delivered} "
                        f"score:{total_score} total_reward:{total_reward_sum:.2f}"
                    )

                # Build sample frame / 构造样本帧
                frame = SampleData(
                    vector_obs=np.array(obs_data.vector_obs, dtype=np.float32),
                    image_obs=np.array(obs_data.image_obs, dtype=np.float32).reshape(-1),
                    priv_obs=np.array(obs_data.privileged_obs, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    reward=reward,
                    done=np.array([float(done)], dtype=np.float32),
                    value=act_data.value.flatten()[: Config.VALUE_NUM],
                    next_value=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    advantage=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                    reward_sum=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                )
                collector.append(frame)

                # End of episode: merge final_reward, compute GAE, yield
                # 局末：合并 final_reward、GAE、yield
                if done:
                    if len(collector) > 0:
                        collector[-1].reward = collector[-1].reward + final_reward

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        self.monitor.put_data(
                            {
                                os.getpid(): {
                                    "reward": round(total_reward_sum + final_reward[0], 4),
                                    "episode_cnt": self.episode_cnt,
                                    "delivered": total_delivered,
                                }
                            }
                        )
                        self.last_report_monitor_time = now

                    if len(collector) > 0:
                        collector = sample_process(collector)
                        yield collector
                    break

                obs_data = _obs_data
                remain_info = _remain_info
