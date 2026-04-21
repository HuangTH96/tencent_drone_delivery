#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Dual-branch Drone Delivery policy network.
双分支智运无人机策略网络：Image + Vector。
"""

import torch
import torch.nn as nn
from agent_ppo.conf.conf import Config


def make_fc_layer(in_features: int, out_features: int, gain: float = 1.0):
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight, gain=gain)
    nn.init.zeros_(fc.bias)
    return fc


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = make_fc_layer(dim, dim)
        self.fc2 = make_fc_layer(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.act(self.norm1(self.fc1(x)))
        h = self.norm2(self.fc2(h))
        return self.act(x + h)


class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(Config.IMAGE_CHANNELS, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            make_fc_layer(64 * 10 * 10, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.proj(self.conv(x))


class VectorEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            make_fc_layer(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            make_fc_layer(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class Tower(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, out_gain: float):
        super().__init__()
        self.net = nn.Sequential(
            make_fc_layer(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            ResidualMLPBlock(hidden_dim),
            make_fc_layer(hidden_dim, out_dim, gain=out_gain),
        )

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "drone_delivery_dual_branch"
        self.device = device

        # Actor encoders
        self.actor_img_encoder = ImageEncoder(Config.IMAGE_EMBED_DIM)
        self.actor_vec_encoder = VectorEncoder(Config.VECTOR_OBS_DIM, Config.VECTOR_EMBED_DIM)
        self.actor_fusion = nn.Sequential(
            make_fc_layer(Config.IMAGE_EMBED_DIM + Config.VECTOR_EMBED_DIM, Config.FUSION_HIDDEN_DIM),
            nn.LayerNorm(Config.FUSION_HIDDEN_DIM),
            nn.SiLU(),
        )
        self.actor_backbone = nn.Sequential(
            ResidualMLPBlock(Config.FUSION_HIDDEN_DIM),
            ResidualMLPBlock(Config.FUSION_HIDDEN_DIM),
        )
        self.actor_head = Tower(Config.FUSION_HIDDEN_DIM, Config.ACTOR_HIDDEN_DIM, Config.ACTION_NUM, out_gain=0.01)

        # Critic encoders (independent from actor)
        self.critic_img_encoder = ImageEncoder(Config.IMAGE_EMBED_DIM)
        self.critic_vec_encoder = VectorEncoder(Config.VECTOR_OBS_DIM, Config.VECTOR_EMBED_DIM)
        self.critic_priv_encoder = nn.Sequential(
            make_fc_layer(Config.PRIVILEGED_DIM, Config.CRITIC_PRIV_HIDDEN_DIM),
            nn.LayerNorm(Config.CRITIC_PRIV_HIDDEN_DIM),
            nn.SiLU(),
            make_fc_layer(Config.CRITIC_PRIV_HIDDEN_DIM, Config.CRITIC_PRIV_HIDDEN_DIM),
            nn.LayerNorm(Config.CRITIC_PRIV_HIDDEN_DIM),
            nn.SiLU(),
        )
        self.critic_fusion = nn.Sequential(
            make_fc_layer(
                Config.IMAGE_EMBED_DIM + Config.VECTOR_EMBED_DIM + Config.CRITIC_PRIV_HIDDEN_DIM,
                Config.HIDDEN_DIM,
            ),
            nn.LayerNorm(Config.HIDDEN_DIM),
            nn.SiLU(),
        )
        self.critic_backbone = nn.Sequential(
            ResidualMLPBlock(Config.HIDDEN_DIM),
            ResidualMLPBlock(Config.HIDDEN_DIM),
        )
        self.critic_head = Tower(Config.HIDDEN_DIM, Config.CRITIC_HIDDEN_DIM, Config.VALUE_NUM, out_gain=0.1)

    def forward(self, image_s, vector_s, priv_s=None, inference=False):
        image = image_s.to(torch.float32)
        vector = vector_s.to(torch.float32)

        actor_img = self.actor_img_encoder(image)
        actor_vec = self.actor_vec_encoder(vector)
        actor_h = self.actor_backbone(self.actor_fusion(torch.cat([actor_img, actor_vec], dim=1)))
        logits = self.actor_head(actor_h)

        critic_img = self.critic_img_encoder(image)
        critic_vec = self.critic_vec_encoder(vector)
        if priv_s is None:
            priv = torch.zeros((vector.size(0), Config.PRIVILEGED_DIM), dtype=torch.float32, device=vector.device)
        else:
            priv = priv_s.to(torch.float32)
        critic_priv = self.critic_priv_encoder(priv)

        critic_h = self.critic_backbone(self.critic_fusion(torch.cat([critic_img, critic_vec, critic_priv], dim=1)))
        value = self.critic_head(critic_h)
        return [logits, value]

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
