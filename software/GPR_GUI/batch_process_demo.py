#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理A8-NEW-1.csv，生成各方法对比图
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.ndimage import uniform_filter1d
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import os

# 读取数据
df = pd.read_csv('/home/baiiy1/.openclaw/workspace/repos/GPR_GUI/sample_data/A8-NEW-1.csv', header=None)
data_raw = df.iloc[:, 3].values
samples_per_trace = 801
n_traces = len(data_raw) // samples_per_trace
data = data_raw[:samples_per_trace * n_traces].reshape(samples_per_trace, n_traces)

# 处理NaN
col_means = np.nanmean(data, axis=0)
nan_mask = np.isnan(data)
for i in range(data.shape[1]):
    mask = nan_mask[:, i]
    data[mask, i] = col_means[i] if not np.isnan(col_means[i]) else 0

print(f"数据形状: {data.shape}")

# ============ 各处理方法 ============

def method_svd_bg(data, rank=1):
    U, S, Vt = svd(data, full_matrices=False)
    S_bg = np.zeros_like(S)
    S_bg[:rank] = S[:rank]
    background = (U * S_bg) @ Vt
    return data - background

def method_fk_filter(data, angle_low=10, angle_high=65, taper_width=5):
    F = fftshift(fft2(data))
    ny, nx = F.shape
    ky = np.fft.fftfreq(ny)
    kx = np.fft.fftfreq(nx)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    angle = np.degrees(np.arctan2(KY, KX))
    
    mask = np.ones_like(F)
    band_mask = (angle >= angle_low) & (angle <= angle_high)
    
    if taper_width > 0:
        sigma = taper_width
        for i in range(ny):
            for j in range(nx):
                if band_mask[i, j]:
                    dist_to_low = abs(angle[i, j] - angle_low)
                    dist_to_high = abs(angle[i, j] - angle_high)
                    dist = min(dist_to_low, dist_to_high)
                    if dist < taper_width:
                        mask[i, j] = 1 - np.exp(-(dist**2) / (2 * sigma**2))
                    else:
                        mask[i, j] = 0.05
    else:
        mask[band_mask] = 0.0
    
    F_filtered = F * mask
    return np.real(ifft2(ifftshift(F_filtered)))

def method_hankel_svd(data, window_length=200, rank=5):
    ny, nx = data.shape
    if window_length is None or window_length <= 0:
        window_length = ny // 4
    window_length = min(window_length, ny - 1)
    
    result = np.zeros_like(data)
    for col in range(nx):
        trace = data[:, col]
        m = ny - window_length + 1
        if m <= 0:
            result[:, col] = trace
            continue
        hankel = np.zeros((m, window_length))
        for i in range(window_length):
            hankel[:, i] = trace[i:i+m]
        U, S, Vt = svd(hankel, full_matrices=False)
        S_filtered = np.zeros_like(S)
        S_filtered[:rank] = S[:rank]
        hankel_filtered = (U * S_filtered) @ Vt
        trace_filtered = np.zeros(ny)
        trace_filtered[:m] = hankel_filtered[:, 0]
        trace_filtered[m:] = hankel_filtered[-1, 1:]
        result[:, col] = trace_filtered
    return result

def method_sliding_avg(data, window_size=10):
    return data - uniform_filter1d(data, size=window_size, axis=1, mode='nearest')

# ============ 执行处理 ============
print("处理中...")
results = {
    "原始数据": data,
    "SVD背景抑制 (rank=1)": method_svd_bg(data, rank=1),
    "SVD背景抑制 (rank=2)": method_svd_bg(data, rank=2),
    "F-K域极角滤波 (10°-65°)": method_fk_filter(data, 10, 65, 5),
    "Hankel-SVD (窗=200,秩=5)": method_hankel_svd(data, 200, 5),
    "滑动平均 (窗=10)": method_sliding_avg(data, 10),
}

# ============ 生成对比图 ============
output_dir = '/home/baiiy1/.openclaw/workspace/repos/GPR_GUI/output/batch_comparison'
os.makedirs(output_dir, exist_ok=True)

# 大图: 2行3列
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

vmin, vmax = data.min(), data.max()

for idx, (name, result) in enumerate(results.items()):
    ax = axes[idx]
    im = ax.imshow(result, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Trace')
    ax.set_ylabel('Sample')
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
comparison_path = os.path.join(output_dir, 'all_methods_comparison.png')
plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
print(f"对比图已保存: {comparison_path}")

# 保存各方法单独图像
for name, result in results.items():
    plt.figure(figsize=(10, 6))
    plt.imshow(result, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    plt.title(name, fontsize=14, fontweight='bold')
    plt.xlabel('Trace')
    plt.ylabel('Sample')
    plt.colorbar(fraction=0.046)
    plt.tight_layout()
    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('°', 'deg')
    plt.savefig(os.path.join(output_dir, f'{safe_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {safe_name}.png")

print(f"\n所有结果保存在: {output_dir}")
