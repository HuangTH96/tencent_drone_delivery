#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Improved monitor panel configuration builder for Drone Delivery.
增强版智运无人机监控面板配置构建器。
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    monitor = MonitorConfigBuilder()
    config_dict = (
        monitor.title("智运无人机")
        .add_group(group_name="算法指标", group_name_en="algorithm")
        .add_panel(name="累积回报", name_en="reward", type="line")
        .add_metric(metrics_name="reward", expr="avg(reward{})")
        .end_panel()
        .add_panel(name="总损失", name_en="total_loss", type="line")
        .add_metric(metrics_name="total_loss", expr="avg(total_loss{})")
        .end_panel()
        .add_panel(name="价值损失", name_en="value_loss", type="line")
        .add_metric(metrics_name="value_loss", expr="avg(value_loss{})")
        .end_panel()
        .add_panel(name="策略损失", name_en="policy_loss", type="line")
        .add_metric(metrics_name="policy_loss", expr="avg(policy_loss{})")
        .end_panel()
        .add_panel(name="熵损失", name_en="entropy_loss", type="line")
        .add_metric(metrics_name="entropy_loss", expr="avg(entropy_loss{})")
        .end_panel()
        .add_panel(name="投递数量", name_en="delivered", type="line")
        .add_metric(metrics_name="delivered", expr="avg(delivered{})")
        .end_panel()
        .add_panel(name="完成局数", name_en="episode_cnt", type="line")
        .add_metric(metrics_name="episode_cnt", expr="avg(episode_cnt{})")
        .end_panel()
        .end_group()
        .build()
    )
    return config_dict
