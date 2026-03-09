#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
现代信号处理课程设计：GPR埋设目标检测仿真
主题：基于LFM脉冲压缩与背景去除的单通道GPR目标检测
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SimConfig:
    c: float = 3e8
    eps_r: float = 9.0                      # 相对介电常数（干土近似）
    fc: float = 500e6                       # 中心频率 500 MHz
    bw: float = 300e6                       # 带宽 300 MHz
    pulse_dur: float = 30e-9                # 脉冲时长 30 ns
    fs: float = 5e9                         # 采样率 5 GHz
    t_max: float = 180e-9                   # 观测窗口 180 ns
    num_traces: int = 121                   # B-scan横向采样点
    x_min: float = -1.2                     # m
    x_max: float = 1.2                      # m
    noise_std: float = 0.06
    clutter_scale: float = 0.16


@dataclass
class Target:
    x0: float       # 目标横向位置（m）
    z: float        # 埋深（m）
    amp: float      # 反射系数幅度


def lfm_pulse(cfg: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    """生成基带LFM发射信号（有限时窗）。"""
    n = int(cfg.pulse_dur * cfg.fs)
    t = np.arange(n) / cfg.fs
    mu = cfg.bw / cfg.pulse_dur
    # 基带chirp：exp(j*pi*mu*t^2)
    s = np.exp(1j * np.pi * mu * (t - cfg.pulse_dur / 2) ** 2)
    # 汉宁窗减旁瓣
    s *= np.hanning(n)
    return t, s


def fractional_delay(signal: np.ndarray, delay_s: float, fs: float, out_len: int) -> np.ndarray:
    """频域实现分数延迟并截断到out_len。"""
    n_sig = len(signal)
    n_fft = 1
    n_need = n_sig + out_len
    while n_fft < n_need:
        n_fft *= 2

    s_fft = np.fft.fft(signal, n_fft)
    freqs = np.fft.fftfreq(n_fft, d=1 / fs)
    phase = np.exp(-1j * 2 * np.pi * freqs * delay_s)
    delayed = np.fft.ifft(s_fft * phase)

    # 取前out_len样本作为接收窗口
    return delayed[:out_len]


def build_bscan(cfg: SimConfig, targets: list[Target]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v = cfg.c / np.sqrt(cfg.eps_r)
    t_axis = np.arange(int(cfg.t_max * cfg.fs)) / cfg.fs
    x_axis = np.linspace(cfg.x_min, cfg.x_max, cfg.num_traces)
    _, tx = lfm_pulse(cfg)

    rx = np.zeros((len(t_axis), len(x_axis)), dtype=np.complex128)

    # 目标回波
    for ix, x in enumerate(x_axis):
        for tgt in targets:
            r = np.sqrt((x - tgt.x0) ** 2 + tgt.z**2)
            tau = 2 * r / v
            # 距离衰减（简化模型）
            atten = tgt.amp / (r**2 + 1e-6)
            rx[:, ix] += atten * fractional_delay(tx, tau, cfg.fs, len(t_axis))

    # 地表直达波 + 缓变杂波
    direct_tau = 5e-9
    direct = 0.8 * fractional_delay(tx, direct_tau, cfg.fs, len(t_axis))[:, None]
    clutter = cfg.clutter_scale * (
        np.exp(-t_axis[:, None] / (55e-9)) * (1 + 0.2 * np.sin(2 * np.pi * x_axis[None, :] / 1.4))
    )

    # 复噪声
    noise = cfg.noise_std * (
        np.random.randn(*rx.shape) + 1j * np.random.randn(*rx.shape)
    )

    rx_total = rx + direct + clutter + noise
    return t_axis, x_axis, tx, rx_total


def matched_filter(rx: np.ndarray, tx: np.ndarray) -> np.ndarray:
    """按列做匹配滤波（脉冲压缩）。"""
    n_t, n_x = rx.shape
    n_fft = 1
    while n_fft < (n_t + len(tx) - 1):
        n_fft *= 2

    h = np.conj(tx[::-1])
    H = np.fft.fft(h, n_fft)

    out = np.zeros((n_t, n_x), dtype=np.complex128)
    for ix in range(n_x):
        R = np.fft.fft(rx[:, ix], n_fft)
        y = np.fft.ifft(R * H)
        out[:, ix] = y[:n_t]
    return out


def analytic_envelope(x: np.ndarray) -> np.ndarray:
    """FFT法Hilbert变换获取包络（逐列）。"""
    n, m = x.shape
    X = np.fft.fft(x, axis=0)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2
    z = np.fft.ifft(X * h[:, None], axis=0)
    return np.abs(z)


def estimate_depths(center_trace_env: np.ndarray, t_axis: np.ndarray, cfg: SimConfig, topk: int = 3) -> list[dict]:
    """在中心A-scan上做简单峰值检出并估计埋深。"""
    v = cfg.c / np.sqrt(cfg.eps_r)
    # 排除直达波区间
    valid = t_axis > 12e-9
    sig = center_trace_env.copy()
    sig[~valid] = 0

    # 简单非极大值抑制
    idx_sorted = np.argsort(sig)[::-1]
    picks = []
    guard = int(2e-9 * cfg.fs)
    used = np.zeros_like(sig, dtype=bool)

    for idx in idx_sorted:
        if sig[idx] <= 0:
            break
        if used[idx]:
            continue
        t = t_axis[idx]
        z_est = v * t / 2
        picks.append({
            "sample": int(idx),
            "time_ns": float(t * 1e9),
            "depth_m": float(z_est),
            "amplitude": float(sig[idx]),
        })
        lo = max(0, idx - guard)
        hi = min(len(sig), idx + guard + 1)
        used[lo:hi] = True
        if len(picks) >= topk:
            break

    return picks


def main() -> None:
    np.random.seed(42)

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    cfg = SimConfig()
    targets = [
        Target(x0=-0.35, z=0.48, amp=1.0),
        Target(x0=0.55, z=0.72, amp=1.2),
    ]

    t_axis, x_axis, tx, rx = build_bscan(cfg, targets)
    pc = matched_filter(rx, tx)

    # 匹配滤波结果存在群时延，做时间轴修正
    t_pc = t_axis - (len(tx) - 1) / cfg.fs

    # 背景去除
    pc_bg_removed = pc - np.mean(pc, axis=1, keepdims=True)

    env_raw = analytic_envelope(np.real(pc))
    env_bg = analytic_envelope(np.real(pc_bg_removed))

    # 中心A-scan峰值检测
    center_idx = len(x_axis) // 2
    picks = estimate_depths(env_bg[:, center_idx], t_pc, cfg, topk=4)

    # 可视化
    def db(x: np.ndarray) -> np.ndarray:
        return 20 * np.log10(np.maximum(x, 1e-9) / np.max(x))

    fig1, ax1 = plt.subplots(figsize=(8, 4.5), dpi=140)
    ax1.plot(np.real(tx), lw=1.2)
    ax1.set_title("Tx LFM Pulse (Real Part)")
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Amplitude")
    ax1.grid(alpha=0.25)
    fig1.tight_layout()
    fig1.savefig(os.path.join(out_dir, "01_tx_pulse.png"))
    plt.close(fig1)

    fig2, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), dpi=140, sharey=True)

    im0 = axes[0].imshow(
        db(np.abs(rx)),
        aspect="auto",
        cmap="turbo",
        extent=[x_axis[0], x_axis[-1], t_pc[-1] * 1e9, t_pc[0] * 1e9],
        vmin=-40,
        vmax=0,
    )
    axes[0].set_title("Raw B-scan (dB)")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("Time (ns)")

    im1 = axes[1].imshow(
        db(env_raw),
        aspect="auto",
        cmap="turbo",
        extent=[x_axis[0], x_axis[-1], t_pc[-1] * 1e9, t_pc[0] * 1e9],
        vmin=-40,
        vmax=0,
    )
    axes[1].set_title("After Matched Filter (dB)")
    axes[1].set_xlabel("x (m)")

    im2 = axes[2].imshow(
        db(env_bg),
        aspect="auto",
        cmap="turbo",
        extent=[x_axis[0], x_axis[-1], t_pc[-1] * 1e9, t_pc[0] * 1e9],
        vmin=-40,
        vmax=0,
    )
    axes[2].set_title("Matched Filter + BG Removal (dB)")
    axes[2].set_xlabel("x (m)")

    cbar = fig2.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("Relative Amplitude (dB)")
    fig2.tight_layout()
    fig2.savefig(os.path.join(out_dir, "02_bscan_compare.png"))
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 4.5), dpi=140)
    tr = env_bg[:, center_idx]
    tr_db = 20 * np.log10(np.maximum(tr, 1e-9) / np.max(tr))
    ax3.plot(t_pc * 1e9, tr_db, lw=1.2, label="Center trace envelope")
    for p in picks:
        ax3.axvline(p["time_ns"], color="r", ls="--", alpha=0.6)
        ax3.text(p["time_ns"] + 0.4, -8 - 6 * picks.index(p), f"z≈{p['depth_m']:.2f} m", color="r")
    ax3.set_title("Center A-scan Peak Picks")
    ax3.set_xlabel("Time (ns)")
    ax3.set_ylabel("Relative amplitude (dB)")
    ax3.set_ylim(-45, 2)
    ax3.grid(alpha=0.25)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(os.path.join(out_dir, "03_center_trace_peaks.png"))
    plt.close(fig3)

    summary = {
        "config": asdict(cfg),
        "targets_true": [asdict(t) for t in targets],
        "detected_peaks": picks,
        "files": [
            "results/01_tx_pulse.png",
            "results/02_bscan_compare.png",
            "results/03_center_trace_peaks.png",
        ],
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Simulation done.")
    print("Detected peaks (center trace):")
    for p in picks:
        print(
            f"  sample={p['sample']}, t={p['time_ns']:.2f} ns, "
            f"z_est={p['depth_m']:.3f} m, amp={p['amplitude']:.4f}"
        )
    print(f"Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
