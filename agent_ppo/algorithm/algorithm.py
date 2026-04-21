#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Stronger Drone Delivery PPO algorithm implementation.
更强版智运无人机 PPO 算法实现。
"""

import os
import time

import torch
import torch.nn.functional as F
from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.label_size = Config.ACTION_NUM
        self.value_num = Config.VALUE_NUM
        self.clip_param = Config.CLIP_PARAM
        self.vf_coef = Config.VF_COEF
        self.entropy_coef = Config.BETA_START
        self.last_report_monitor_time = 0
        self.learn_step = 0

    def learn(self, list_sample_data):
        vector_obs = torch.stack([f.vector_obs for f in list_sample_data]).to(self.device)
        image_obs = (
            torch.stack([f.image_obs for f in list_sample_data])
            .to(self.device)
            .view(-1, Config.IMAGE_CHANNELS, Config.MAP_SIZE, Config.MAP_SIZE)
            .to(torch.float32)
        )
        priv_obs = torch.stack([f.priv_obs for f in list_sample_data]).to(self.device)
        legal_action = torch.stack([f.legal_action for f in list_sample_data]).to(self.device)
        act = torch.stack([f.act for f in list_sample_data]).to(self.device).view(-1, 1)
        old_prob = torch.stack([f.prob for f in list_sample_data]).to(self.device)
        advantage = torch.stack([f.advantage for f in list_sample_data]).to(self.device)
        old_value = torch.stack([f.value for f in list_sample_data]).to(self.device)
        reward_sum = torch.stack([f.reward_sum for f in list_sample_data]).to(self.device)
        reward = torch.stack([f.reward for f in list_sample_data]).to(self.device)

        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        self.model.set_train_mode()
        self.optimizer.zero_grad()
        logits, value_pred = self.model(image_obs, vector_obs, priv_obs)

        total_loss, info = self._compute_loss(
            logits=logits,
            value_pred=value_pred,
            legal_action=legal_action,
            old_action=act,
            old_prob=old_prob,
            advantage=advantage,
            old_value=old_value,
            reward_sum=reward_sum,
            reward=reward,
        )

        if info['approx_kl'] > Config.MAX_KL:
            if self.logger and self.learn_step % 50 == 0:
                self.logger.info(f"[LEARN] skip update because approx_kl={info['approx_kl']:.6f} > {Config.MAX_KL}")
            return {"total_loss": float(total_loss.detach().cpu().item()), "skipped": True}

        total_loss.backward()
        grad_norm = 0.0
        if Config.USE_GRAD_CLIP:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP_RANGE))
        self.optimizer.step()
        self.learn_step += 1
        self._update_schedules()

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            results = {
                "total_loss": round(total_loss.item(), 4),
                "value_loss": round(info["value_loss"], 4),
                "policy_loss": round(info["policy_loss"], 4),
                "entropy_loss": round(info["entropy_loss"], 4),
                "reward": round(info["reward_mean"], 4),
                "approx_kl": round(info["approx_kl"], 6),
                "clip_frac": round(info["clip_frac"], 4),
                "grad_norm": round(grad_norm, 4),
                "lr": round(self.optimizer.param_groups[0]["lr"], 8),
                "entropy_coef": round(self.entropy_coef, 6),
            }
            if self.logger:
                self.logger.info(
                    f"[LEARN] policy_loss={results['policy_loss']} "
                    f"value_loss={results['value_loss']} "
                    f"entropy={results['entropy_loss']} "
                    f"reward={results['reward']} "
                    f"kl={results['approx_kl']} clip_frac={results['clip_frac']} "
                    f"lr={results['lr']} ent_coef={results['entropy_coef']}"
                )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_monitor_time = now

        return {"total_loss": total_loss.item()}

    def _update_schedules(self):
        lr = max(Config.MIN_LEARNING_RATE, Config.INIT_LEARNING_RATE_START * (Config.LR_DECAY ** self.learn_step))
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.entropy_coef = max(Config.BETA_END, Config.BETA_START * (Config.ENTROPY_DECAY ** self.learn_step))

    def _compute_loss(self, logits, value_pred, legal_action, old_action, old_prob, advantage, old_value, reward_sum, reward):
        masked_logits = self._masked_logits(logits, legal_action)
        log_prob_dist = F.log_softmax(masked_logits, dim=1)
        prob_dist = log_prob_dist.exp()
        entropy_loss = -(prob_dist * log_prob_dist).sum(1).mean()

        act_index = old_action[:, 0].long().unsqueeze(1)
        new_log_prob = log_prob_dist.gather(1, act_index)
        old_act_prob = old_prob.gather(1, act_index).clamp(1e-9)
        old_log_prob = torch.log(old_act_prob)
        ratio = torch.exp(new_log_prob - old_log_prob)

        adv = advantage.squeeze(-1) if advantage.dim() > 1 else advantage
        adv = adv.unsqueeze(-1)
        policy_loss1 = -ratio * adv
        policy_loss2 = -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv
        policy_loss = torch.maximum(policy_loss1, policy_loss2).mean()

        v = value_pred.squeeze(-1) if value_pred.dim() > 1 else value_pred
        ov = old_value.squeeze(-1) if old_value.dim() > 1 else old_value
        tgt = reward_sum.squeeze(-1) if reward_sum.dim() > 1 else reward_sum
        v_clip = ov + (v - ov).clamp(-self.clip_param, self.clip_param)
        value_loss_unclipped = F.huber_loss(v, tgt, reduction='none', delta=Config.HUBER_DELTA)
        value_loss_clipped = F.huber_loss(v_clip, tgt, reduction='none', delta=Config.HUBER_DELTA)
        value_loss = torch.maximum(value_loss_unclipped, value_loss_clipped).mean()

        total_loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy_loss

        approx_kl = (old_log_prob - new_log_prob).mean().abs().detach().cpu().item()
        clip_frac = ((ratio - 1.0).abs() > self.clip_param).float().mean().detach().cpu().item()
        info = {
            'value_loss': value_loss.detach().cpu().item(),
            'policy_loss': policy_loss.detach().cpu().item(),
            'entropy_loss': entropy_loss.detach().cpu().item(),
            'reward_mean': reward.mean().detach().cpu().item(),
            'approx_kl': approx_kl,
            'clip_frac': clip_frac,
        }
        return total_loss, info

    def _masked_logits(self, logits, legal_action):
        return logits + (legal_action - 1.0) * 1e9
