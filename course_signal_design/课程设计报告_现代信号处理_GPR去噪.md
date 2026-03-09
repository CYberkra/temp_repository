# 《现代信号处理》课程设计报告

## 基本信息（请替换）
- 课程名称：现代信号处理
- 题目：基于 Hankel-SVD 与 F-K 滤波的 GPR B-scan 去噪与对比评估
- 学生姓名：`[姓名]`
- 学号：`[学号]`
- 专业：`[专业]`
- 指导教师：`[教师姓名]`
- 日期：`[YYYY-MM-DD]`

---

## 1. 问题描述
地质雷达（GPR）B-scan 数据常受到随机噪声、低频杂波及直达波干扰，导致地下目标反射特征（如双曲线）被淹没。为提升目标可辨识性，本设计对比三种方法：

1. Hankel-SVD 去噪
2. F-K 滤波
3. Hankel-SVD + F-K 级联

目标是通过仿真数据验证各方法在噪声抑制与信号保真之间的表现差异。

## 2. 方法

### 2.1 合成数据建模
构建二维 B-scan（时间采样点 × 道数）：
- 目标反射：两个双曲线目标 + 一条弱层状反射
- 干扰：
  - 高斯白噪声
  - 低频杂波（每道随机相位正弦）
  - 浅层直达波强干扰

### 2.2 Hankel-SVD 去噪
对每条 A-scan 道信号：
1. 构造 Hankel 矩阵
2. 奇异值分解（SVD）
3. 保留前 r 个主奇异值重构
4. 反 Hankel 化得到去噪信号

该方法利用信号低秩结构抑制随机噪声。

### 2.3 F-K 滤波
对 B-scan 做二维 FFT 进入 f-k 域，构建掩膜：
- 抑制空间低波数（k≈0）分量，削弱近水平干扰
- 去除时间低频/DC 成分

再逆变换回时空域得到去噪结果。

### 2.4 级联方法
先做 Hankel-SVD，再进行 F-K 滤波，以同时利用：
- SVD 的低秩去噪能力
- F-K 对结构化干扰（直达波/条带）的抑制能力

## 3. 仿真设置
- 环境：Python 3
- 主要库：NumPy、Matplotlib
- 数据尺寸：`nt=512, nx=128`
- 随机种子：`20260309`
- Hankel 参数：`L=96, rank=8`
- F-K 参数：`keep_ratio_k=0.22, remove_dc_time=True`

脚本路径：
`/mnt/e/Openclaw/.openclaw/workspace/shared/course_signal_design/scripts/gpr_denoise_hankel_fk.py`

## 4. 仿真结果

### 4.1 定量指标（与理想 clean 场景对比）
- 输入噪声数据：
  - SNR = **-16.19 dB**
  - RMSE = **0.3963**
- Hankel-SVD：
  - SNR = **-11.07 dB**
  - RMSE = **0.2200**
- F-K：
  - SNR = **-13.92 dB**
  - RMSE = **0.3052**
- Hankel-SVD + F-K：
  - SNR = **-5.30 dB**
  - RMSE = **0.1132**

结论：本仿真下 **级联方法最优**，较输入数据 SNR 提升约 **10.89 dB**，RMSE 显著下降。

### 4.2 可视化结果
- B-scan 对比图：
  `figures/bscan_comparison.png`
- F-K 频谱图：
  `figures/spectrum_fk.png`

## 5. 结论
1. Hankel-SVD 对随机噪声抑制明显，但对结构化干扰仍有限。
2. F-K 滤波对低波数条带/直达波抑制有效，但单独使用时保真度一般。
3. Hankel-SVD + F-K 级联兼顾随机噪声与结构化干扰，综合性能最佳。
4. 该流程可直接迁移到实测 GPR 数据，后续可加入参数网格搜索与更多指标（如 SSIM、目标检测率）。

---

## 附录：运行方式
在终端执行：

```bash
cd /mnt/e/Openclaw/.openclaw/workspace/shared/course_signal_design
python3 scripts/gpr_denoise_hankel_fk.py
```

运行后将自动生成：
- `outputs/metrics.json`
- `figures/bscan_comparison.png`
- `figures/spectrum_fk.png`

## 附录：导出 PDF 操作步骤

### 方案 A（推荐，Pandoc）
1. 安装 Pandoc（若未安装）
2. 在项目目录执行：

```bash
cd /mnt/e/Openclaw/.openclaw/workspace/shared/course_signal_design
pandoc "课程设计报告_现代信号处理_GPR去噪.md" -o "课程设计报告_现代信号处理_GPR去噪.pdf" --pdf-engine=xelatex
```

> 若中文字体报错，可安装 TeX Live 并配置中文字体（如 SimSun / Noto Sans CJK）。

### 方案 B（Typora / VS Code）
1. 打开该 Markdown 报告
2. 选择“导出/打印为 PDF”
3. 检查页边距、标题层级与图片是否完整

### 方案 C（浏览器打印）
1. 将 Markdown 渲染为 HTML（如使用 Markdown Preview）
2. 浏览器 `Ctrl+P` 选择“另存为 PDF”

