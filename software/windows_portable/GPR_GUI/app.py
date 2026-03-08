#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPR GUI (Enhanced)
- Load CSV
- Display B-scan
- Select processing method (original + researched)
- Configure method parameters (window width, time, rank, etc.)
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import re
from datetime import datetime

# matplotlib for B-scan
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from scipy.linalg import svd
from scipy.ndimage import uniform_filter1d
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# add core module path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "PythonModule_core"))
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)
# ensure local dir (for read_file_data.py)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from read_file_data import savecsv, save_image


# 匹配真实的带单位表头
_HEADER_KEYS = [
    "Number of Samples",
    "Time windows (ns)",   
    "Number of Traces",
    "Trace interval (m)", 
]


def _parse_header_lines(lines):
    if len(lines) < 4:
        return None
    info = {}
    for line in lines[:4]:
        if "=" not in line:
            return None
        left, right = line.split("=", 1)
        key = left.strip()
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", right)
        if not m:
            return None
        try:
            val = float(m.group(0))
        except ValueError:
            return None
        info[key] = val
    if not all(k in info for k in _HEADER_KEYS):
        return None
    return {
        "a_scan_length": int(info["Number of Samples"]),
        "total_time_ns": float(info["Time windows (ns)"]),
        "num_traces": int(info["Number of Traces"]),
        "trace_interval_m": float(info["Trace interval (m)"]),
    }


def detect_csv_header(path: str):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [f.readline().strip() for _ in range(4)]
    except OSError:
        return None
    return _parse_header_lines(lines)


# ============ Methods (research) ============

def method_svd_background(data, rank=1, **kwargs):
    """SVD background removal - remove top-r singular values"""
    U, S, Vt = svd(data, full_matrices=False)
    S_bg = np.zeros_like(S)
    S_bg[:rank] = S[:rank]
    background = (U * S_bg) @ Vt
    return data - background, background


def method_fk_filter(data, angle_low=10, angle_high=65, taper_width=5, **kwargs):
    """F-K cone filter (Corrected)"""
    F = fftshift(fft2(data))
    ny, nx = F.shape
    
    ky = fftshift(np.fft.fftfreq(ny))
    kx = fftshift(np.fft.fftfreq(nx))
    
    KY, KX = np.meshgrid(ky, kx, indexing='ij')
    
    angle = np.degrees(np.arctan2(np.abs(KY), np.abs(KX)))

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
    result = np.real(ifft2(ifftshift(F_filtered)))
    return result, mask


def method_hankel_svd(data, window_length=None, rank=None, **kwargs):
    """Hankel SVD denoising (Corrected with Diagonal Averaging)"""
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
        
        if rank is None or rank <= 0:
            diff_spec = np.diff(S)
            threshold = np.mean(np.abs(diff_spec))
            rank_val = 1
            for i in range(len(diff_spec) - 2):
                if (abs(diff_spec[i]) < threshold and
                    abs(diff_spec[i+1]) < threshold):
                    rank_val = i + 1
                    break
            rank_val = max(rank_val, 1)
        else:
            rank_val = max(rank, 1)
            
        S_filtered = np.zeros_like(S)
        S_filtered[:rank_val] = S[:rank_val]
        hankel_filtered = (U * S_filtered) @ Vt
        
        trace_filtered = np.zeros(ny)
        counts = np.zeros(ny)
        
        for i in range(m):
            for j in range(window_length):
                trace_filtered[i + j] += hankel_filtered[i, j]
                counts[i + j] += 1
                
        trace_filtered /= counts
        result[:, col] = trace_filtered
        
    return result, None


def method_sliding_average(data, window_size=10, axis=1, **kwargs):
    """Sliding-average background removal"""
    background = uniform_filter1d(data, size=window_size, axis=axis, mode='nearest')
    return data - background, background


# ============ Method registry ============

PROCESSING_METHODS = {
    # Original methods (PythonModule_core)
    "compensatingGain": {
        "name": "0 compensatingGain (manual gain compensation)",
        "type": "core",
        "module": "compensatingGain",
        "func": "compensatingGain",
        "params": [
            {"name": "gain_min", "label": "Gain min", "type": "float", "default": 1.0, "min": 0.1, "max": 20.0},
            {"name": "gain_max", "label": "Gain max", "type": "float", "default": 6.0, "min": 0.1, "max": 50.0},
        ],
    },
    "dewow": {
        "name": "1 dewow (low-frequency drift correction)",
        "type": "core",
        "module": "dewow",
        "func": "dewow",
        "params": [
            {"name": "window", "label": "Window (samples)", "type": "int", "default": 31, "min": 1, "max": 1000},
        ],
    },
    "set_zero_time": {
        "name": "2 set_zero_time (zero-time correction)",
        "type": "core",
        "module": "set_zero_time",
        "func": "set_zero_time",
        "params": [
            {"name": "new_zero_time", "label": "Zero-time (ns)", "type": "float", "default": 5.0, "min": 0.0, "max": 1000.0},
        ],
    },
    "agcGain": {
        "name": "3 agcGain (AGC correction)",
        "type": "core",
        "module": "agcGain",
        "func": "agcGain",
        "params": [
            {"name": "window", "label": "Window (samples)", "type": "int", "default": 31, "min": 1, "max": 1000},
        ],
    },
    "subtracting_average_2D": {
        "name": "4 subtracting_average_2D (background removal)",
        "type": "core",
        "module": "subtracting_average_2D",
        "func": "subtracting_average_2D",
        "params": [],
    },
    "running_average_2D": {
        "name": "5 running_average_2D (spike clutter suppression)",
        "type": "core",
        "module": "running_average_2D",
        "func": "running_average_2D",
        "params": [],
    },

    # Research methods (local)
    "svd_bg": {
        "name": "SVD background removal (low-rank)",
        "type": "local",
        "func": method_svd_background,
        "params": [
            {"name": "rank", "label": "Rank (remove top r)", "type": "int", "default": 1, "min": 1, "max": 20},
        ],
    },
    "fk_filter": {
        "name": "F-K cone filter",
        "type": "local",
        "func": method_fk_filter,
        "params": [
            {"name": "angle_low", "label": "Stopband start angle (°)", "type": "int", "default": 10, "min": 0, "max": 90},
            {"name": "angle_high", "label": "Stopband end angle (°)", "type": "int", "default": 65, "min": 0, "max": 90},
            {"name": "taper_width", "label": "Taper width (°)", "type": "int", "default": 5, "min": 0, "max": 20},
        ],
    },
    "hankel_svd": {
        "name": "Hankel SVD denoising",
        "type": "local",
        "func": method_hankel_svd,
        "params": [
            {"name": "window_length", "label": "Window length (0=auto)", "type": "int", "default": 0, "min": 0, "max": 2000},
            {"name": "rank", "label": "Rank kept (0=auto)", "type": "int", "default": 0, "min": 0, "max": 100},
        ],
    },
    "sliding_avg": {
        "name": "Sliding-average background removal",
        "type": "local",
        "func": method_sliding_average,
        "params": [
            {"name": "window_size", "label": "Window size", "type": "int", "default": 10, "min": 1, "max": 200},
            {"name": "axis", "label": "Axis (0/1)", "type": "int", "default": 1, "min": 0, "max": 1},
        ],
    },
}


class GPRGui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.colors = {
            "bg": "#F3F5F7",
            "panel": "#FFFFFF",
            "panel_alt": "#FAFBFC",
            "border": "#D7DEE5",
            "text": "#1F2937",
            "muted": "#6B7280",
            "accent": "#3B82F6",
            "button_hover": "#EEF2F7",
        }
        self.configure(bg=self.colors["bg"])
        self.option_add("*Font", ("Segoe UI", 10))
        self.option_add("*Background", self.colors["bg"])
        self.option_add("*Foreground", self.colors["text"])
        self.option_add("*Frame.Background", self.colors["panel"])
        self.option_add("*Button.Background", self.colors["panel"])
        self.option_add("*Button.Foreground", self.colors["text"])
        self.option_add("*Button.ActiveBackground", self.colors["button_hover"])
        self.option_add("*Button.ActiveForeground", self.colors["text"])
        self.option_add("*Checkbutton.Background", self.colors["panel"])
        self.option_add("*Checkbutton.Foreground", self.colors["text"])
        self.option_add("*Listbox.Background", self.colors["panel"])
        self.option_add("*Listbox.Foreground", self.colors["text"])
        self.option_add("*Entry.Background", self.colors["panel"])
        self.option_add("*Entry.Foreground", self.colors["text"])

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background=self.colors["panel"])
        style.configure("TLabel", background=self.colors["panel"], foreground=self.colors["text"])
        style.configure("TLabelframe", background=self.colors["panel"], foreground=self.colors["text"])
        style.configure("TLabelframe.Label", background=self.colors["panel"], foreground=self.colors["muted"])
        style.configure("TCombobox", fieldbackground=self.colors["panel"], background=self.colors["panel"])
        style.configure("TCheckbutton", background=self.colors["panel"], foreground=self.colors["text"])
        style.configure("TButton", background=self.colors["panel"], foreground=self.colors["text"], padding=6)
        style.map("TButton", background=[("active", self.colors["button_hover"])])

        self._load_icons()
        if self.icons.get("app"):
            self.iconphoto(True, self.icons["app"])

        self.title("GPR GUI - Enhanced")
        self.geometry("1200x760")

        self.data = None
        self.data_path = None
        self.header_info = None

        left = tk.Frame(
            self,
            bg=self.colors["panel"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=10, pady=10)
        right = tk.Frame(
            self,
            bg=self.colors["panel_alt"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        btn_style = dict(
            bg=self.colors["panel"],
            fg=self.colors["text"],
            activebackground=self.colors["button_hover"],
            activeforeground=self.colors["text"],
            bd=0,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            padx=10,
            pady=6,
        )
        icon_btn_style = dict(btn_style, compound="left", anchor="w", padx=12, pady=6)

        tk.Button(
            left,
            text="Import CSV",
            command=self.load_csv,
            image=self.icons.get("import"),
            **icon_btn_style,
        ).pack(pady=5)
        tk.Button(
            left,
            text="Apply Selected Method",
            command=self.apply_method,
            image=self.icons.get("apply"),
            **icon_btn_style,
        ).pack(pady=5)

        tk.Label(left, text="Methods").pack(pady=(15, 5))
        self.method_combo = ttk.Combobox(left, state="readonly", width=30)
        self.method_keys = list(PROCESSING_METHODS.keys())
        self.method_combo["values"] = [PROCESSING_METHODS[k]["name"] for k in self.method_keys]
        self.method_combo.current(0)
        self.method_combo.pack(pady=5)
        self.method_combo.bind("<<ComboboxSelected>>", self._on_method_change)

        tk.Label(left, text="Parameters").pack(pady=(15, 5))
        self.param_frame = tk.Frame(left)
        self.param_frame.pack(fill=tk.X, pady=5)
        self.param_vars = {}
        self._render_params(self.method_keys[0])

        tk.Label(left, text="Batch (multi-select)").pack(pady=(10, 2))
        self.batch_list = tk.Listbox(left, selectmode=tk.MULTIPLE, height=6, width=30)
        for name in self.method_combo["values"]:
            self.batch_list.insert(tk.END, name)
        self.batch_list.pack(pady=2)
        tk.Button(
            left,
            text="Run Batch Compare",
            command=self.run_batch,
            image=self.icons.get("batch"),
            **icon_btn_style,
        ).pack(pady=4)

        tk.Button(
            left,
            text="Generate Report",
            command=self.generate_report,
            image=self.icons.get("report"),
            **icon_btn_style,
        ).pack(pady=4)

        self.symmetric_var = tk.BooleanVar(value=False)
        self.symm_check = tk.Checkbutton(
            left,
            text="Symmetric gray stretch (vmin/vmax)",
            variable=self.symmetric_var,
            command=self._on_symmetric_toggle,
        )
        self.symm_check.pack(pady=5)

        # Display options
        tk.Label(left, text="Colormap").pack(pady=(10, 2))
        self.cmap_var = tk.StringVar(value="gray")
        self.cmap_combo = ttk.Combobox(left, state="readonly", width=28, textvariable=self.cmap_var)
        self.cmap_combo["values"] = ["gray", "viridis", "plasma", "inferno", "magma", "jet", "seismic"]
        self.cmap_combo.pack(pady=2)
        self.cmap_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_plot())

        self.cmap_invert_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            left,
            text="Invert colormap",
            variable=self.cmap_invert_var,
            command=self._refresh_plot,
        ).pack(pady=2)

        # Preprocess toggles
        tk.Label(left, text="Preprocess").pack(pady=(10, 2))
        self.normalize_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            left,
            text="Normalize (max abs)",
            variable=self.normalize_var,
            command=self._refresh_plot,
        ).pack(pady=2)
        self.demean_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            left,
            text="Demean (per trace)",
            variable=self.demean_var,
            command=self._refresh_plot,
        ).pack(pady=2)

        # Crop window
        tk.Label(left, text="Crop Window").pack(pady=(10, 2))
        self.crop_enable_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            left,
            text="Enable crop",
            variable=self.crop_enable_var,
            command=self._refresh_plot,
        ).pack(pady=2)
        crop_row1 = tk.Frame(left)
        crop_row1.pack(fill=tk.X, pady=1)
        tk.Label(crop_row1, text="Time start", width=10, anchor="w").pack(side=tk.LEFT)
        self.time_start_var = tk.StringVar(value="")
        tk.Entry(crop_row1, textvariable=self.time_start_var, width=10).pack(side=tk.LEFT)
        tk.Label(crop_row1, text="end", width=4).pack(side=tk.LEFT)
        self.time_end_var = tk.StringVar(value="")
        tk.Entry(crop_row1, textvariable=self.time_end_var, width=10).pack(side=tk.LEFT)

        crop_row2 = tk.Frame(left)
        crop_row2.pack(fill=tk.X, pady=1)
        tk.Label(crop_row2, text="Dist start", width=10, anchor="w").pack(side=tk.LEFT)
        self.dist_start_var = tk.StringVar(value="")
        tk.Entry(crop_row2, textvariable=self.dist_start_var, width=10).pack(side=tk.LEFT)
        tk.Label(crop_row2, text="end", width=4).pack(side=tk.LEFT)
        self.dist_end_var = tk.StringVar(value="")
        tk.Entry(crop_row2, textvariable=self.dist_end_var, width=10).pack(side=tk.LEFT)

        tk.Button(left, text="Apply Crop", command=self._refresh_plot, width=22).pack(pady=3)
        tk.Button(left, text="Reset Crop", command=self._reset_crop, width=22).pack(pady=3)

        # Fast preview options
        tk.Label(left, text="Fast Preview").pack(pady=(10, 2))
        self.fast_preview_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            left,
            text="Enable chunked preview",
            variable=self.fast_preview_var,
        ).pack(pady=2)
        prev_row = tk.Frame(left)
        prev_row.pack(fill=tk.X, pady=1)
        tk.Label(prev_row, text="Max samples", width=10, anchor="w").pack(side=tk.LEFT)
        self.max_samples_var = tk.StringVar(value="512")
        tk.Entry(prev_row, textvariable=self.max_samples_var, width=10).pack(side=tk.LEFT)
        prev_row2 = tk.Frame(left)
        prev_row2.pack(fill=tk.X, pady=1)
        tk.Label(prev_row2, text="Max traces", width=10, anchor="w").pack(side=tk.LEFT)
        self.max_traces_var = tk.StringVar(value="200")
        tk.Entry(prev_row2, textvariable=self.max_traces_var, width=10).pack(side=tk.LEFT)

        tk.Label(left, text="Info / Notes").pack(pady=(15, 5))
        self.info = tk.Text(left, height=18, width=35)
        self.info.pack(pady=5)
        self._log("Welcome. Please import a CSV to view B-scan.")

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("B-scan")
        self.ax.set_xlabel("Distance (trace index)")
        self.ax.set_ylabel("Time (sample index)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _log(self, msg: str):
        self.info.insert(tk.END, msg + "\n")
        self.info.see(tk.END)

    def _on_method_change(self, event=None):
        idx = self.method_combo.current()
        key = self.method_keys[idx]
        self._render_params(key)

    def _on_symmetric_toggle(self):
        if self.data is not None:
            self.plot_data(self.data)

    def _refresh_plot(self):
        if self.data is not None:
            self.plot_data(self.data)

    def _reset_crop(self):
        self.time_start_var.set("")
        self.time_end_var.set("")
        self.dist_start_var.set("")
        self.dist_end_var.set("")
        self.crop_enable_var.set(False)
        self._refresh_plot()

    def _get_colormap(self):
        cmap = (self.cmap_var.get() or "gray").strip()
        if self.cmap_invert_var.get():
            if cmap.endswith("_r"):
                cmap = cmap[:-2]
            else:
                cmap = cmap + "_r"
        return cmap

    def _apply_preprocess(self, data: np.ndarray) -> np.ndarray:
        out = data
        if self.demean_var.get():
            mean = np.mean(out, axis=0, keepdims=True)
            out = out - mean
        if self.normalize_var.get():
            maxv = np.max(np.abs(out))
            if maxv == 0:
                maxv = 1e-6
            out = out / maxv
        return out

    def _load_icons(self):
        self.icons = {}
        icon_dir = os.path.join(BASE_DIR, "assets")
        icon_files = {
            "app": "icon-app.png",
            "import": "icon-import.png",
            "apply": "icon-apply.png",
            "batch": "icon-batch.png",
            "report": "icon-report.png",
        }
        for key, filename in icon_files.items():
            path = os.path.join(icon_dir, filename)
            if os.path.exists(path):
                try:
                    self.icons[key] = tk.PhotoImage(file=path)
                except tk.TclError:
                    pass

    def _parse_float(self, text: str):
        try:
            return float(text)
        except Exception:
            return None

    def _get_crop_bounds(self, data: np.ndarray):
        if not self.crop_enable_var.get():
            return None

        n_time, n_dist = data.shape
        time_start = self._parse_float(self.time_start_var.get().strip())
        time_end = self._parse_float(self.time_end_var.get().strip())
        dist_start = self._parse_float(self.dist_start_var.get().strip())
        dist_end = self._parse_float(self.dist_end_var.get().strip())

        if self.header_info:
            total_time = float(self.header_info.get("total_time_ns", n_time))
            num_traces = max(1, int(self.header_info.get("num_traces", n_dist)))
            trace_interval = float(self.header_info.get("trace_interval_m", 1.0))
            dist_total = trace_interval * (num_traces - 1)

            if time_start is None:
                time_start = 0.0
            if time_end is None:
                time_end = total_time
            if dist_start is None:
                dist_start = 0.0
            if dist_end is None:
                dist_end = dist_total

            time_start = max(0.0, min(total_time, time_start))
            time_end = max(0.0, min(total_time, time_end))
            dist_start = max(0.0, min(dist_total, dist_start))
            dist_end = max(0.0, min(dist_total, dist_end))

            if time_end < time_start:
                time_start, time_end = time_end, time_start
            if dist_end < dist_start:
                dist_start, dist_end = dist_end, dist_start

            def time_to_idx(t):
                if total_time <= 0 or n_time <= 1:
                    return 0
                return int(round(t / total_time * (n_time - 1)))

            def dist_to_idx(d):
                if dist_total <= 0 or n_dist <= 1:
                    return 0
                return int(round(d / dist_total * (n_dist - 1)))

            t0 = max(0, min(n_time - 1, time_to_idx(time_start)))
            t1 = max(0, min(n_time - 1, time_to_idx(time_end)))
            d0 = max(0, min(n_dist - 1, dist_to_idx(dist_start)))
            d1 = max(0, min(n_dist - 1, dist_to_idx(dist_end)))
        else:
            if time_start is None:
                time_start = 0.0
            if time_end is None:
                time_end = float(n_time - 1)
            if dist_start is None:
                dist_start = 0.0
            if dist_end is None:
                dist_end = float(n_dist - 1)

            time_start = max(0.0, min(n_time - 1, time_start))
            time_end = max(0.0, min(n_time - 1, time_end))
            dist_start = max(0.0, min(n_dist - 1, dist_start))
            dist_end = max(0.0, min(n_dist - 1, dist_end))

            if time_end < time_start:
                time_start, time_end = time_end, time_start
            if dist_end < dist_start:
                dist_start, dist_end = dist_end, dist_start

            t0 = int(round(time_start))
            t1 = int(round(time_end))
            d0 = int(round(dist_start))
            d1 = int(round(dist_end))

        return {
            "t0": t0,
            "t1": t1,
            "d0": d0,
            "d1": d1,
            "time_start": time_start,
            "time_end": time_end,
            "dist_start": dist_start,
            "dist_end": dist_end,
        }

    def _apply_crop(self, data: np.ndarray):
        bounds = self._get_crop_bounds(data)
        if not bounds:
            return data, None
        t0, t1, d0, d1 = bounds["t0"], bounds["t1"], bounds["d0"], bounds["d1"]
        cropped = data[t0:t1 + 1, d0:d1 + 1]
        return cropped, bounds

    def _downsample_data(self, data: np.ndarray) -> np.ndarray:
        if not self.fast_preview_var.get():
            return data
        try:
            max_samples = int(float(self.max_samples_var.get() or 0))
            max_traces = int(float(self.max_traces_var.get() or 0))
        except Exception:
            return data
        n_time, n_dist = data.shape
        if max_samples > 0 and n_time > max_samples:
            idx = np.linspace(0, n_time - 1, max_samples).astype(int)
            data = data[idx, :]
        if max_traces > 0 and n_dist > max_traces:
            idx = np.linspace(0, n_dist - 1, max_traces).astype(int)
            data = data[:, idx]
        return data

    def _render_params(self, method_key: str):
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        self.param_vars = {}
        params = PROCESSING_METHODS[method_key].get("params", [])
        if not params:
            tk.Label(self.param_frame, text="(No parameters)").pack(anchor="w")
            return
        for p in params:
            row = tk.Frame(self.param_frame)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=p["label"], width=18, anchor="w").pack(side=tk.LEFT)
            var = tk.StringVar(value=str(p.get("default", "")))
            entry = tk.Entry(row, textvariable=var, width=12)
            entry.pack(side=tk.LEFT)
            self.param_vars[p["name"]] = (var, p)

    def _get_params(self):
        params = {}
        for name, (var, meta) in self.param_vars.items():
            raw = var.get().strip()
            if raw == "":
                raw = str(meta.get("default", ""))
            try:
                if meta["type"] == "int":
                    val = int(float(raw))
                elif meta["type"] == "float":
                    val = float(raw)
                else:
                    val = raw
            except ValueError:
                raise ValueError(f"Invalid value for {meta['label']}")
            params[name] = val
        return params

    def load_csv(self):
        path = filedialog.askopenfilename(
            title="Select CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*")],
        )
        if not path:
            return
        
        try:
            header_info = detect_csv_header(path)
            
            # 检测并跳过表头
            skip_lines = 0
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for i in range(10): 
                    line = f.readline()
                    if not line: break
                    if "=" in line or "Samples" in line or "Traces" in line:
                        skip_lines = i + 1

            # 使用 pandas 读取文件（可选分块/快速预览）
            if self.fast_preview_var.get():
                try:
                    max_samples = int(float(self.max_samples_var.get() or 0))
                    max_traces = int(float(self.max_traces_var.get() or 0))
                except Exception:
                    max_samples, max_traces = 0, 0
                target_rows = max_samples if max_samples > 0 else 200000
                if header_info and max_samples > 0 and max_traces > 0:
                    target_rows = max_samples * max_traces
                rows = []
                count = 0
                for chunk in pd.read_csv(path, header=None, skiprows=skip_lines, chunksize=200000):
                    rows.append(chunk)
                    count += len(chunk)
                    if count >= target_rows:
                        break
                df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            else:
                df = pd.read_csv(path, header=None, skiprows=skip_lines)
            raw_data = df.values

            # ================= 核心修复：自动识别并折叠一维顺序数据 =================
            if header_info:
                samples = header_info["a_scan_length"]
                traces = header_info["num_traces"]
                
                # 情况1：如果你读入的是像 A8-NEW-1.csv 这种带有 GPS 信息的 5 列原始数据
                if raw_data.shape[1] <= 10 and raw_data.shape[0] >= samples * traces:
                    # 提取第 4 列（索引 3）作为雷达信号
                    col_idx = 3 if raw_data.shape[1] > 3 else raw_data.shape[1] - 1
                    signal_1d = raw_data[:, col_idx]
                    
                    # 按照 道数×采样点 的顺序重新折叠并转置，恢复成标准的二维雷达图
                    data = signal_1d[:traces * samples].reshape((traces, samples)).T
                    
                # 情况2：处理完毕后已经变成了二维矩阵的情况，防止错误转置
                elif raw_data.shape[0] == traces and raw_data.shape[1] >= samples:
                    data = raw_data[:, :samples].T
                    
                # 情况3：标准二维矩阵
                elif raw_data.shape[0] >= samples and raw_data.shape[1] >= traces:
                    data = raw_data[:samples, :traces]
                else:
                    data = raw_data
            else:
                data = raw_data
            # =====================================================================

            if self.fast_preview_var.get():
                data = self._downsample_data(data)
                self._log("Fast preview: data downsampled.")
                
            # 处理残留的 NaN
            if np.isnan(data).any():
                data = np.nan_to_num(data, nan=np.nanmean(data))
                
            if data.ndim == 1:
                data = data.reshape(-1, 1)

            self.data = data
            self.data_path = path
            self.header_info = header_info
            
            self._log(f"Loaded CSV: {path}  shape={data.shape}")
            if header_info:
                self._log(
                    "Header detected: "
                    f"A-scan length={header_info['a_scan_length']} samples; "
                    f"Total time={header_info['total_time_ns']} ns; "
                    f"A-scan count={header_info['num_traces']}; "
                    f"Trace interval={header_info['trace_interval_m']} m"
                )
            else:
                self._log("No header metadata detected; using index axes.")
                
            self.plot_data(data)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {e}")
            self._log(f"Failed to load CSV: {e}")

    def plot_data(self, data: np.ndarray):
        self.ax.clear()
        extent = None
        if self.header_info:
            num_traces = max(1, int(self.header_info.get("num_traces", data.shape[1])))
            trace_interval = float(self.header_info.get("trace_interval_m", 1.0))
            total_time = float(self.header_info.get("total_time_ns", data.shape[0]))
            distance_end = trace_interval * (num_traces - 1)
            # Y轴向下
            extent = [0.0, distance_end, total_time, 0.0]
            self.ax.set_xlabel("Distance (m)")
            self.ax.set_ylabel("Time (ns)")
        else:
            self.ax.set_xlabel("Distance (trace index)")
            self.ax.set_ylabel("Time (sample index)")

        valid_data = np.nan_to_num(data)
        valid_data = self._apply_preprocess(valid_data)
        valid_data, bounds = self._apply_crop(valid_data)

        if self.header_info:
            total_time = float(self.header_info.get("total_time_ns", valid_data.shape[0]))
            num_traces = max(1, int(self.header_info.get("num_traces", valid_data.shape[1])))
            trace_interval = float(self.header_info.get("trace_interval_m", 1.0))
            distance_end = trace_interval * (num_traces - 1)
            time_start = 0.0
            time_end = total_time
            dist_start = 0.0
            dist_end = distance_end
            if bounds:
                time_start = bounds["time_start"]
                time_end = bounds["time_end"]
                dist_start = bounds["dist_start"]
                dist_end = bounds["dist_end"]
            extent = [dist_start, dist_end, time_end, time_start]
        else:
            if bounds:
                extent = [bounds["dist_start"], bounds["dist_end"], bounds["time_end"], bounds["time_start"]]
            else:
                extent = None

        cmap = self._get_colormap()

        if self.symmetric_var.get():
            # ==== 可选：对称拉伸（vmin/vmax） ====
            stdcont = np.nanmax(np.abs(valid_data))
            if stdcont == 0:
                stdcont = 1e-6
            vmin = -stdcont
            vmax = stdcont
            self.ax.imshow(
                valid_data,
                cmap=cmap,
                aspect="auto",
                extent=extent,
                vmin=vmin,
                vmax=vmax,
            )
        else:
            # 默认保持原版视觉（matplotlib 自动拉伸）
            self.ax.imshow(valid_data, cmap=cmap, aspect="auto", extent=extent)
        self.ax.set_title("B-scan")
        self.canvas.draw()

    def _save_outputs(self, data: np.ndarray, method_key: str):
        out_dir = os.path.join(BASE_DIR, "output")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"{method_key}_out.csv")
        out_png = os.path.join(out_dir, f"{method_key}_out.png")

        save_data = self._apply_preprocess(np.nan_to_num(data))
        save_data, bounds = self._apply_crop(save_data)
        savecsv(save_data, out_csv)

        time_range = None
        distance_range = None
        if self.header_info:
            total_time = float(self.header_info["total_time_ns"])
            num_traces = max(1, int(self.header_info["num_traces"]))
            trace_interval = float(self.header_info["trace_interval_m"])
            distance_end = trace_interval * (num_traces - 1)
            if bounds:
                time_range = (bounds["time_start"], bounds["time_end"])
                distance_range = (bounds["dist_start"], bounds["dist_end"])
            else:
                time_range = (0.0, total_time)
                distance_range = (0.0, distance_end)

        save_image(
            save_data,
            out_png,
            title=method_key,
            time_range=time_range,
            distance_range=distance_range,
            cmap=self._get_colormap(),
        )
        return out_csv, out_png

    def run_batch(self):
        if self.data is None or self.data_path is None:
            messagebox.showwarning("No data", "Please import a CSV first.")
            return
        selected = self.batch_list.curselection()
        if not selected:
            messagebox.showwarning("No selection", "Please select methods for batch processing.")
            return

        base_data = self.data
        last_data = None
        for idx in selected:
            method_key = self.method_keys[idx]
            method = PROCESSING_METHODS[method_key]
            self._log(f"Batch: {method['name']}")

            params = {}
            for p in method.get("params", []):
                params[p["name"]] = p.get("default")

            try:
                if method["type"] == "core":
                    mod = __import__(method["module"])
                    func = getattr(mod, method["func"])
                    length_trace = base_data.shape[0]
                    start_position = 0
                    end_position = base_data.shape[1]
                    scans_per_meter = 1

                    temp_in_csv = os.path.join(BASE_DIR, "output", "temp_in.csv")
                    savecsv(base_data, temp_in_csv)

                    out_dir = os.path.join(BASE_DIR, "output")
                    os.makedirs(out_dir, exist_ok=True)
                    out_csv = os.path.join(out_dir, f"{method_key}_out.csv")
                    out_png = os.path.join(out_dir, f"{method_key}_out.png")

                    if method_key == "compensatingGain":
                        gain_min = float(params.get("gain_min", 1.0))
                        gain_max = float(params.get("gain_max", 6.0))
                        gain_func = np.linspace(gain_min, gain_max, base_data.shape[0]).tolist()
                        func(temp_in_csv, out_csv, out_png, length_trace, start_position, end_position, gain_func)
                    elif method_key == "dewow":
                        window = int(params.get("window", max(1, length_trace // 4)))
                        func(temp_in_csv, out_csv, out_png, length_trace, start_position, scans_per_meter, window)
                    elif method_key == "set_zero_time":
                        new_zero_time = float(params.get("new_zero_time", 5.0))
                        func(temp_in_csv, out_csv, out_png, length_trace, start_position, scans_per_meter, new_zero_time)
                    elif method_key == "agcGain":
                        window = int(params.get("window", max(1, length_trace // 4)))
                        func(temp_in_csv, out_csv, out_png, length_trace, start_position, scans_per_meter, window)
                    elif method_key == "subtracting_average_2D":
                        func(temp_in_csv, out_csv, out_png, length_trace, start_position, scans_per_meter)
                    elif method_key == "running_average_2D":
                        func(temp_in_csv, out_csv, out_png, length_trace, start_position, scans_per_meter)
                    else:
                        self._log("Unknown core method; skipped.")
                        continue

                    if os.path.exists(out_csv):
                        newdata_df = pd.read_csv(out_csv, header=None)
                        newdata = newdata_df.values
                        if newdata.ndim == 1:
                            newdata = newdata.reshape(-1, 1)
                        last_data = newdata
                        self._save_outputs(newdata, method_key)
                        self._log(f"Batch saved: {out_csv}")
                else:
                    result = method["func"](base_data, **params)
                    newdata = result[0] if isinstance(result, tuple) else result
                    last_data = newdata
                    out_csv, _ = self._save_outputs(newdata, method_key)
                    self._log(f"Batch saved: {out_csv}")
            except Exception as e:
                self._log(f"Batch error ({method_key}): {e}")

        if last_data is not None:
            self.data = last_data
            self.plot_data(self.data)

    def generate_report(self):
        if self.data is None or self.data_path is None:
            messagebox.showwarning("No data", "Please import a CSV first.")
            return
        out_dir = os.path.join(BASE_DIR, "output")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(out_dir, f"report_{ts}.md")
        image_path = os.path.join(out_dir, f"report_{ts}.png")

        try:
            self.fig.savefig(image_path, dpi=150)
        except Exception as e:
            self._log(f"Report screenshot failed: {e}")

        bounds = None
        try:
            bounds = self._get_crop_bounds(self._apply_preprocess(np.nan_to_num(self.data)))
        except Exception:
            bounds = None

        method_key = self.method_keys[self.method_combo.current()]
        method_name = PROCESSING_METHODS[method_key]["name"]
        try:
            params = self._get_params()
        except Exception:
            params = {}

        lines = []
        lines.append(f"# GPR GUI Report ({ts})")
        lines.append("")
        lines.append(f"- Data file: {self.data_path}")
        lines.append(f"- Method: {method_name}")
        if params:
            lines.append(f"- Params: {params}")
        lines.append(f"- Colormap: {self._get_colormap()}")
        lines.append(f"- Symmetric stretch: {self.symmetric_var.get()}")
        lines.append(f"- Normalize: {self.normalize_var.get()}")
        lines.append(f"- Demean: {self.demean_var.get()}")
        if bounds:
            lines.append(
                f"- Crop: time {bounds['time_start']}~{bounds['time_end']} ; distance {bounds['dist_start']}~{bounds['dist_end']}"
            )
        else:
            lines.append("- Crop: disabled")
        lines.append("")
        lines.append(f"- Screenshot: {image_path}")
        lines.append("")
        lines.append("## Log")
        log_text = self.info.get("1.0", tk.END).strip()
        lines.append("```")
        lines.append(log_text)
        lines.append("```")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        self._log(f"Report saved: {report_path}")

    def apply_method(self):
        if self.data is None or self.data_path is None:
            messagebox.showwarning("No data", "Please import a CSV first.")
            return
        idx = self.method_combo.current()
        method_key = self.method_keys[idx]
        method = PROCESSING_METHODS[method_key]
        self._log(f"Applying: {method['name']}")

        try:
            params = self._get_params()
        except ValueError as e:
            messagebox.showerror("Invalid parameter", str(e))
            return

        out_dir = os.path.join(BASE_DIR, "output")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, f"{method_key}_out.csv")
        out_png = os.path.join(out_dir, f"{method_key}_out.png")

        try:
            if method["type"] == "core":
                mod = __import__(method["module"])
                func = getattr(mod, method["func"])
                length_trace = self.data.shape[0]
                start_position = 0
                end_position = self.data.shape[1]
                scans_per_meter = 1

                # ================= 核心修复：中间文件传递 =================
                # 把 GUI 当前纯净的二维数据先写入一个临时文件 temp_in.csv
                # 让外部的 core 方法去读这个只有雷达信号的二维矩阵，
                # 防止它们读到原始文件的 GPS 数据产生乱码！
                temp_in_csv = os.path.join(out_dir, "temp_in.csv")
                savecsv(self.data, temp_in_csv)
                # ==========================================================

                if method_key == "compensatingGain":
                    gain_min = float(params.get("gain_min", 1.0))
                    gain_max = float(params.get("gain_max", 6.0))
                    gain_func = np.linspace(gain_min, gain_max, self.data.shape[0]).tolist()
                    func(temp_in_csv, out_csv, out_png, length_trace, start_position, end_position, gain_func)
                elif method_key == "dewow":
                    window = int(params.get("window", max(1, length_trace // 4)))
                    func(temp_in_csv, out_csv, out_png, length_trace, start_position, scans_per_meter, window)
                elif method_key == "set_zero_time":
                    new_zero_time = float(params.get("new_zero_time", 5.0))
                    func(temp_in_csv, out_csv, out_png, length_trace, start_position, scans_per_meter, new_zero_time)
                elif method_key == "agcGain":
                    window = int(params.get("window", max(1, length_trace // 4)))
                    func(temp_in_csv, out_csv, out_png, length_trace, start_position, scans_per_meter, window)
                elif method_key == "subtracting_average_2D":
                    func(temp_in_csv, out_csv, out_png, length_trace, start_position, scans_per_meter)
                elif method_key == "running_average_2D":
                    func(temp_in_csv, out_csv, out_png, length_trace, start_position, scans_per_meter)
                else:
                    self._log("Unknown core method; no processing applied.")
                    self.plot_data(self.data)
                    return

                if os.path.exists(out_csv):
                    # 读取外部方法输出的 CSV 时，也使用 pandas 进行更好的容错
                    newdata_df = pd.read_csv(out_csv, header=None)
                    newdata = newdata_df.values
                    if newdata.ndim == 1:
                        newdata = newdata.reshape(-1, 1)
                    self.data = newdata
                    out_csv, out_png = self._save_outputs(newdata, method_key)
                    self.plot_data(newdata)
                    self._log(f"Processed data saved: {out_csv}")
                else:
                    self._log("Processing finished but output CSV not found.")
            else:
                # 运行内置的研究方法 (Research methods)
                result = method["func"](self.data, **params)
                if isinstance(result, tuple):
                    newdata = result[0]
                else:
                    newdata = result
                self.data = newdata
                self.plot_data(newdata)
                out_csv, out_png = self._save_outputs(newdata, method_key)
                self._log(f"Processed data saved: {out_csv}")
        except Exception as e:
            self._log(f"Processing error: {e}")
            messagebox.showerror("Error", f"Processing error: {e}")


if __name__ == "__main__":
    app = GPRGui()
    app.mainloop()