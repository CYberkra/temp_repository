#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPR GUI Enhanced - 集成调研后的处理方法
支持参数配置: 窗宽、时间、秩选择等
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import re

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.linalg import svd
from scipy.ndimage import uniform_filter1d
from scipy.fft import fft2, ifft2, fftshift, ifftshift

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from read_file_data import readcsv, savecsv, save_image


# ============ 处理方法实现 ============

def method_svd_background(data, rank=1, **kwargs):
    """SVD背景抑制 - 去除前rank个奇异值"""
    U, S, Vt = svd(data, full_matrices=False)
    S_bg = np.zeros_like(S)
    S_bg[:rank] = S[:rank]
    background = (U * S_bg) @ Vt
    return data - background, background


def method_fk_filter(data, angle_low=10, angle_high=65, taper_width=5, **kwargs):
    """F-K域极角滤波 - 基于2024 MDPI论文"""
    # 2D FFT
    F = fftshift(fft2(data))
    
    # 获取频率坐标
    ny, nx = F.shape
    ky = np.fft.fftfreq(ny)
    kx = np.fft.fftfreq(nx)
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    
    # 极角
    angle = np.degrees(np.arctan2(KY, KX))
    
    # 构建高斯锥形滤波器 (阻带: angle_low ~ angle_high)
    mask = np.ones_like(F)
    
    # 阻带区域
    band_mask = (angle >= angle_low) & (angle <= angle_high)
    
    # 高斯边缘过渡
    if taper_width > 0:
        sigma = taper_width
        # 在阻带边界添加渐变
        for i in range(ny):
            for j in range(nx):
                if band_mask[i, j]:
                    # 计算到最近边界的距离
                    dist_to_low = abs(angle[i, j] - angle_low)
                    dist_to_high = abs(angle[i, j] - angle_high)
                    dist = min(dist_to_low, dist_to_high)
                    if dist < taper_width:
                        mask[i, j] = 1 - np.exp(-(dist**2) / (2 * sigma**2))
                    else:
                        mask[i, j] = 0.05  # 小值而非0，保留部分信息
    else:
        mask[band_mask] = 0.0
    
    # 应用滤波器
    F_filtered = F * mask
    
    # 逆变换
    result = np.real(ifft2(ifftshift(F_filtered)))
    return result, mask


def method_hankel_svd(data, window_length=None, rank=None, **kwargs):
    """Hankel矩阵SVD去噪 - 基于2019 Sensors论文"""
    ny, nx = data.shape
    
    # 自动窗口长度
    if window_length is None or window_length <= 0:
        window_length = ny // 4
    window_length = min(window_length, ny - 1)
    
    result = np.zeros_like(data)
    
    # 逐道处理
    for col in range(nx):
        trace = data[:, col]
        
        # 构建Hankel矩阵
        m = ny - window_length + 1
        if m <= 0:
            result[:, col] = trace
            continue
            
        hankel = np.zeros((m, window_length))
        for i in range(window_length):
            hankel[:, i] = trace[i:i+m]
        
        # SVD
        U, S, Vt = svd(hankel, full_matrices=False)
        
        # 自动秩选择 (如未指定)
        if rank is None or rank <= 0:
            # 差分谱法
            diff_spec = np.diff(S)
            threshold = np.mean(np.abs(diff_spec))
            rank = 1
            for i in range(len(diff_spec) - 2):
                if (abs(diff_spec[i]) < threshold and 
                    abs(diff_spec[i+1]) < threshold):
                    rank = i + 1
                    break
            rank = max(rank, 1)
        
        # 重构
        S_filtered = np.zeros_like(S)
        S_filtered[:rank] = S[:rank]
        hankel_filtered = (U * S_filtered) @ Vt
        
        # 恢复信号
        trace_filtered = np.zeros(ny)
        trace_filtered[:m] = hankel_filtered[:, 0]
        trace_filtered[m:] = hankel_filtered[-1, 1:]
        
        result[:, col] = trace_filtered
    
    return result, None


def method_sliding_average(data, window_size=10, axis=1, **kwargs):
    """滑动平均背景抑制"""
    background = uniform_filter1d(data, size=window_size, axis=axis, mode='nearest')
    return data - background, background


# ============ 方法注册表 ============

PROCESSING_METHODS = {
    "original": {
        "name": "原始数据 (无处理)",
        "func": lambda data, **kwargs: (data, None),
        "params": []
    },
    "svd_bg": {
        "name": "SVD背景抑制 (低秩去除)",
        "func": method_svd_background,
        "params": [
            {"name": "rank", "label": "秩 (去除前rank个)", "type": "int", "default": 1, "min": 1, "max": 10}
        ]
    },
    "fk_filter": {
        "name": "F-K域极角滤波",
        "func": method_fk_filter,
        "params": [
            {"name": "angle_low", "label": "阻带起始角度 (°)", "type": "int", "default": 10, "min": 0, "max": 90},
            {"name": "angle_high", "label": "阻带结束角度 (°)", "type": "int", "default": 65, "min": 0, "max": 90},
            {"name": "taper_width", "label": "过渡带宽度 (°)", "type": "int", "default": 5, "min": 0, "max": 20}
        ]
    },
    "hankel_svd": {
        "name": "Hankel矩阵SVD去噪",
        "func": method_hankel_svd,
        "params": [
            {"name": "window_length", "label": "窗口长度 (0=自动)", "type": "int", "default": 0, "min": 0, "max": 500},
            {"name": "rank", "label": "保留秩 (0=自动)", "type": "int", "default": 0, "min": 0, "max": 50}
        ]
    },
    "sliding_avg": {
        "name": "滑动平均背景抑制",
        "func": method_sliding_average,
        "params": [
            {"name": "window_size", "label": "滑动窗口大小", "type": "int", "default": 10, "min": 1, "max": 100}
        ]
    },
    # 原有方法兼容
    "subtracting_average": {
        "name": "全局均值减法 (原有)",
        "func": lambda data, **kwargs: (data - np.mean(data, axis=1, keepdims=True), None),
        "params": []
    },
}


class GPRGuiEnhanced(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GPR GUI Enhanced - 集成调研算法")
        self.geometry("1300x800")
        
        self.data = None
        self.data_path = None
        self.header_info = None
        self.processed_data = None
        
        self._build_ui()
        
    def _build_ui(self):
        # 主分割面板
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧面板
        left_frame = ttk.Frame(paned, width=350)
        paned.add(left_frame, weight=1)
        
        # 右侧面板
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=3)
        
        # ========== 左侧面板 ==========
        # 文件操作区
        file_frame = ttk.LabelFrame(left_frame, text="文件操作", padding=5)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_frame, text="导入 CSV", command=self.load_csv).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="导出处理结果", command=self.export_result).pack(fill=tk.X, pady=2)
        
        # 方法选择区
        method_frame = ttk.LabelFrame(left_frame, text="处理方法选择", padding=5)
        method_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.method_var = tk.StringVar(value="original")
        method_names = [(k, v["name"]) for k, v in PROCESSING_METHODS.items()]
        
        self.method_combo = ttk.Combobox(method_frame, values=[n for _, n in method_names], 
                                         state="readonly", width=30)
        self.method_combo.set(method_names[0][1])
        self.method_combo.pack(fill=tk.X, pady=5)
        self.method_combo.bind("<<ComboboxSelected>>", self.on_method_change)
        
        # 参数配置区
        self.params_frame = ttk.LabelFrame(left_frame, text="参数配置", padding=5)
        self.params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.param_vars = {}
        self.param_widgets = {}
        
        # 应用按钮
        ttk.Button(left_frame, text="应用处理方法", command=self.apply_method).pack(fill=tk.X, padx=5, pady=10)
        
        # 信息显示区
        info_frame = ttk.LabelFrame(left_frame, text="信息", padding=5)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.info_text = tk.Text(info_frame, height=10, width=40)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # ========== 右侧面板 ==========
        # 工具栏
        toolbar = ttk.Frame(right_frame)
        toolbar.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(toolbar, text="原始数据", command=lambda: self.plot_data(self.data, "原始数据")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="处理后数据", command=lambda: self.plot_data(self.processed_data, "处理后数据")).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="差异图", command=self.plot_difference).pack(side=tk.LEFT, padx=2)
        
        # 绘图区
        plot_frame = ttk.Frame(right_frame)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig = Figure(figsize=(9, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("B-scan")
        self.ax.set_xlabel("Distance (trace index)")
        self.ax.set_ylabel("Time (sample index)")
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self._log("欢迎使用GPR GUI增强版。请导入CSV文件。")
        self._build_param_widgets("original")
        
    def _log(self, msg: str):
        self.info_text.insert(tk.END, msg + "\n")
        self.info_text.see(tk.END)
        
    def _build_param_widgets(self, method_key):
        """为选定方法构建参数控件"""
        # 清除旧控件
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        self.param_vars.clear()
        self.param_widgets.clear()
        
        method_info = PROCESSING_METHODS.get(method_key, {})
        params = method_info.get("params", [])
        
        if not params:
            ttk.Label(self.params_frame, text="该方法无需配置参数").pack(pady=10)
            return
            
        for param in params:
            frame = ttk.Frame(self.params_frame)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=param["label"], width=20).pack(side=tk.LEFT)
            
            var = tk.IntVar(value=param.get("default", 0))
            self.param_vars[param["name"]] = var
            
            spin = ttk.Spinbox(frame, from_=param.get("min", 0), to=param.get("max", 100),
                              textvariable=var, width=10)
            spin.pack(side=tk.LEFT, padx=5)
            
            self.param_widgets[param["name"]] = spin
            
    def on_method_change(self, event=None):
        """处理方法改变时更新参数控件"""
        selected_name = self.method_combo.get()
        method_key = None
        for k, v in PROCESSING_METHODS.items():
            if v["name"] == selected_name:
                method_key = k
                break
        if method_key:
            self._build_param_widgets(method_key)
            
    def get_current_method_key(self):
        """获取当前选中的方法key"""
        selected_name = self.method_combo.get()
        for k, v in PROCESSING_METHODS.items():
            if v["name"] == selected_name:
                return k
        return "original"
        
    def load_csv(self):
        path = filedialog.askopenfilename(
            title="选择CSV文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*")],
        )
        if not path:
            return
        try:
            self.data = readcsv(path)
            self.data_path = path
            
            # 检测header
            self.header_info = self._detect_header(path)
            
            self._log(f"已加载: {path}")
            self._log(f"数据形状: {self.data.shape}")
            
            if self.header_info:
                self._log(f"Header: {self.header_info}")
                
            self.processed_data = self.data.copy()
            self.plot_data(self.data, "原始数据")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载CSV失败: {e}")
            self._log(f"加载失败: {e}")
            
    def _detect_header(self, path: str):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [f.readline().strip() for _ in range(4)]
            return self._parse_header_lines(lines)
        except:
            return None
            
    def _parse_header_lines(self, lines):
        keys = ["Number of Samples", "Time windows", "Number of Traces", "Trace interval"]
        info = {}
        for line in lines:
            if "=" in line:
                left, right = line.split("=", 1)
                key = left.strip()
                m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", right)
                if m:
                    info[key] = float(m.group(0))
        if all(k in info for k in keys):
            return {
                "samples": int(info["Number of Samples"]),
                "time_ns": float(info["Time windows"]),
                "traces": int(info["Number of Traces"]),
                "interval_m": float(info["Trace interval"]),
            }
        return None
        
    def plot_data(self, data, title="B-scan"):
        if data is None:
            messagebox.showwarning("警告", "请先加载数据")
            return
            
        self.ax.clear()
        
        extent = None
        if self.header_info:
            extent = [
                0, 
                self.header_info["interval_m"] * (self.header_info["traces"] - 1),
                self.header_info["time_ns"], 
                0
            ]
            self.ax.set_xlabel("Distance (m)")
            self.ax.set_ylabel("Time (ns)")
        else:
            self.ax.set_xlabel("Trace")
            self.ax.set_ylabel("Sample")
            
        im = self.ax.imshow(data, cmap="gray", aspect="auto", extent=extent)
        self.ax.set_title(title)
        
        # 添加colorbar
        if hasattr(self, '_colorbar'):
            self._colorbar.remove()
        self._colorbar = self.fig.colorbar(im, ax=self.ax, fraction=0.046)
        
        self.canvas.draw()
        
    def plot_difference(self):
        if self.data is None or self.processed_data is None:
            messagebox.showwarning("警告", "需要原始和处理后数据")
            return
        diff = self.processed_data - self.data
        self.plot_data(diff, "差异图 (处理后 - 原始)")
        
    def apply_method(self):
        if self.data is None:
            messagebox.showwarning("警告", "请先导入CSV文件")
            return
            
        method_key = self.get_current_method_key()
        method_info = PROCESSING_METHODS[method_key]
        
        self._log(f"应用方法: {method_info['name']}")
        
        # 收集参数
        params = {}
        for name, var in self.param_vars.items():
            params[name] = var.get()
            self._log(f"  参数 {name}: {params[name]}")
            
        try:
            func = method_info["func"]
            result, aux = func(self.data, **params)
            self.processed_data = result
            
            self.plot_data(result, f"{method_info['name']} 结果")
            self._log("处理完成")
            
            # 计算简单统计
            self._log(f"  输出范围: [{result.min():.6f}, {result.max():.6f}]")
            self._log(f"  输出均值: {result.mean():.6f}")
            
        except Exception as e:
            messagebox.showerror("错误", f"处理失败: {e}")
            self._log(f"处理失败: {e}")
            import traceback
            self._log(traceback.format_exc())
            
    def export_result(self):
        if self.processed_data is None:
            messagebox.showwarning("警告", "没有可导出的数据")
            return
            
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*")],
        )
        if path:
            try:
                savecsv(self.processed_data, path)
                self._log(f"已保存: {path}")
                
                # 同时保存图像
                png_path = path.replace(".csv", ".png")
                save_image(self.processed_data, png_path, "处理结果")
                self._log(f"已保存图像: {png_path}")
                
            except Exception as e:
                messagebox.showerror("错误", f"导出失败: {e}")


if __name__ == "__main__":
    app = GPRGuiEnhanced()
    app.mainloop()
