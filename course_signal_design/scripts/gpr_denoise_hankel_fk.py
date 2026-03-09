#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
现代信号处理课程设计仿真：
题目：基于 Hankel-SVD 与 F-K 滤波的 GPR B-scan 去噪与对比评估

输出：
- outputs/metrics.json
- figures/bscan_comparison.png
- figures/spectrum_fk.png
"""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# 1) 数据构建：合成 GPR B-scan
# -----------------------------
def ricker_wavelet(points=64, a=8.0):
    t = np.linspace(-1, 1, points)
    return (1 - 2 * (np.pi ** 2) * (a ** 2) * (t ** 2)) * np.exp(-(np.pi ** 2) * (a ** 2) * (t ** 2))


def add_reflection(trace, center_idx, amp=1.0, wav=None):
    if wav is None:
        wav = ricker_wavelet(64, 8.0)
    n = len(trace)
    m = len(wav)
    s = center_idx - m // 2
    e = s + m
    ws = 0
    we = m
    if s < 0:
        ws = -s
        s = 0
    if e > n:
        we = m - (e - n)
        e = n
    if s < e and ws < we:
        trace[s:e] += amp * wav[ws:we]


def build_gpr_scene(nt=512, nx=128, seed=42):
    rng = np.random.default_rng(seed)
    data_clean = np.zeros((nt, nx), dtype=float)

    # 时间轴、道轴
    t = np.arange(nt)
    x = np.arange(nx)

    # 目标1：浅层双曲线
    x0_1, t0_1, c1 = 34, 120, 1.7
    # 目标2：中层双曲线
    x0_2, t0_2, c2 = 84, 190, 2.3
    # 目标3：近水平层
    t_layer = 300

    wav = ricker_wavelet(points=48, a=6.0)

    for ix in x:
        tt1 = int(np.sqrt(t0_1**2 + (c1 * (ix - x0_1))**2))
        tt2 = int(np.sqrt(t0_2**2 + (c2 * (ix - x0_2))**2))
        add_reflection(data_clean[:, ix], tt1, amp=1.0, wav=wav)
        add_reflection(data_clean[:, ix], tt2, amp=0.75, wav=wav)

        # 层状反射小起伏
        jitter = int(4 * np.sin(ix / 10.0))
        add_reflection(data_clean[:, ix], t_layer + jitter, amp=0.45, wav=wav)

    # 背景低频漂移（clutter） + 随机噪声
    clutter = np.zeros_like(data_clean)
    for ix in x:
        phase = rng.uniform(0, 2 * np.pi)
        lowfreq = 0.25 * np.sin(2 * np.pi * (t / nt) * 4 + phase)
        clutter[:, ix] = lowfreq

    # 强直达波（浅层强干扰）
    direct = np.zeros_like(data_clean)
    for ix in x:
        add_reflection(direct[:, ix], 35 + int(2 * np.sin(ix / 6)), amp=1.6, wav=ricker_wavelet(40, 5.5))

    noise = 0.35 * rng.standard_normal(size=(nt, nx))

    data_noisy = data_clean + clutter + direct + noise
    return data_clean, data_noisy


# -----------------------------
# 2) 方法一：Hankel-SVD 去噪（按道）
# -----------------------------
def hankel_matrix(x, L):
    N = len(x)
    K = N - L + 1
    H = np.empty((L, K), dtype=float)
    for i in range(L):
        H[i, :] = x[i:i + K]
    return H


def hankel_reconstruct(H):
    L, K = H.shape
    N = L + K - 1
    y = np.zeros(N, dtype=float)
    c = np.zeros(N, dtype=float)
    for i in range(L):
        for j in range(K):
            y[i + j] += H[i, j]
            c[i + j] += 1
    return y / np.maximum(c, 1)


def hankel_svd_denoise_trace(x, L=96, rank=8):
    H = hankel_matrix(x, L=L)
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    r = min(rank, len(s))
    Hf = (U[:, :r] * s[:r]) @ Vt[:r, :]
    y = hankel_reconstruct(Hf)
    return y


def hankel_svd_denoise_bscan(data, L=96, rank=8):
    nt, nx = data.shape
    out = np.zeros_like(data)
    for ix in range(nx):
        out[:, ix] = hankel_svd_denoise_trace(data[:, ix], L=L, rank=rank)
    return out


# -----------------------------
# 3) 方法二：F-K 滤波
# -----------------------------
def fk_filter(data, keep_ratio_k=0.22, remove_dc_time=True):
    """
    在 f-k 域做简化滤波：
    - 去除空间低波数（k接近0）强分量（抑制近似水平条带/直达波）
    - 可选去除时间DC
    """
    nt, nx = data.shape
    D = np.fft.fft2(data)

    # 构建 k 掩膜
    kx = np.fft.fftfreq(nx)
    k_keep = np.abs(kx) >= keep_ratio_k * np.max(np.abs(kx))
    mask_k = np.tile(k_keep[np.newaxis, :], (nt, 1))

    # 时间维去DC/低频抑制
    if remove_dc_time:
        ft = np.fft.fftfreq(nt)
        t_mask = np.abs(ft) > 0.01
        mask_t = np.tile(t_mask[:, np.newaxis], (1, nx))
        mask = mask_k & mask_t
    else:
        mask = mask_k

    Df = D * mask
    out = np.real(np.fft.ifft2(Df))
    return out, D, Df


# -----------------------------
# 4) 评估指标
# -----------------------------
def snr_db(ref, test):
    err = ref - test
    p_sig = np.mean(ref ** 2)
    p_err = np.mean(err ** 2) + 1e-12
    return 10 * np.log10(p_sig / p_err)


def rmse(ref, test):
    return float(np.sqrt(np.mean((ref - test) ** 2)))


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "outputs"
    fig_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    clean, noisy = build_gpr_scene(nt=512, nx=128, seed=20260309)

    hankel = hankel_svd_denoise_bscan(noisy, L=96, rank=8)
    fk, D_raw, D_fk = fk_filter(noisy, keep_ratio_k=0.22, remove_dc_time=True)
    combo, _, _ = fk_filter(hankel, keep_ratio_k=0.22, remove_dc_time=True)

    metrics = {
        "snr_input_db": float(snr_db(clean, noisy)),
        "snr_hankel_db": float(snr_db(clean, hankel)),
        "snr_fk_db": float(snr_db(clean, fk)),
        "snr_combo_db": float(snr_db(clean, combo)),
        "rmse_input": rmse(clean, noisy),
        "rmse_hankel": rmse(clean, hankel),
        "rmse_fk": rmse(clean, fk),
        "rmse_combo": rmse(clean, combo),
        "best_method": "Hankel+F-K",
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 图1：B-scan 对比
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), constrained_layout=True)
    mats = [noisy, hankel, fk, combo]
    titles = ["Noisy", "Hankel-SVD", "F-K", "Hankel + F-K"]
    vmax = np.percentile(np.abs(noisy), 99)

    for ax, M, tt in zip(axes, mats, titles):
        im = ax.imshow(M, aspect="auto", cmap="seismic", vmin=-vmax, vmax=vmax)
        ax.set_title(tt)
        ax.set_xlabel("Trace index")
    axes[0].set_ylabel("Time sample")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="Amplitude")
    fig.savefig(fig_dir / "bscan_comparison.png", dpi=180)
    plt.close(fig)

    # 图2：F-K 频谱对比
    spec_raw = np.log1p(np.abs(np.fft.fftshift(D_raw)))
    spec_fk = np.log1p(np.abs(np.fft.fftshift(D_fk)))

    fig2, ax2 = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    im1 = ax2[0].imshow(spec_raw, aspect="auto", cmap="magma")
    ax2[0].set_title("Raw f-k Spectrum (log)")
    ax2[0].set_xlabel("k index")
    ax2[0].set_ylabel("f index")
    fig2.colorbar(im1, ax=ax2[0], shrink=0.8)

    im2 = ax2[1].imshow(spec_fk, aspect="auto", cmap="magma")
    ax2[1].set_title("Filtered f-k Spectrum (log)")
    ax2[1].set_xlabel("k index")
    ax2[1].set_ylabel("f index")
    fig2.colorbar(im2, ax=ax2[1], shrink=0.8)

    fig2.savefig(fig_dir / "spectrum_fk.png", dpi=180)
    plt.close(fig2)

    print("Simulation finished.")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
