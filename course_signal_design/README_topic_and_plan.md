# 《现代信号处理》课程设计题目与方案说明

## 1. 题目
**基于 LFM 脉冲压缩与背景去除的单通道 GPR 埋设目标检测仿真**

## 2. 选题理由（可在周四前完成）
- 与现代信号处理核心内容高度匹配：**匹配滤波（脉冲压缩）+ 杂波抑制（背景去除）+ 峰值检测**。
- 与 GPR 应用场景吻合：地下目标回波弱、直达波/背景杂波强。
- 纯 Python 可复现，计算量适中，能在普通笔记本快速得到图像结果。

## 3. 技术路线
1. 生成基带 LFM 发射脉冲（500 MHz 中心、300 MHz 带宽）。
2. 构建二维 B-scan 数据：
   - 地下两个点目标回波（时延随横向位置变化，形成双曲线特征）；
   - 叠加地表直达波、缓变杂波和复高斯噪声。
3. 处理流程：
   - 匹配滤波进行脉冲压缩，提升距离分辨率和信噪比；
   - 背景去除（逐时刻减均值）抑制静态杂波；
   - 包络检测与峰值提取，估计中心测线目标深度。
4. 输出：
   - 原始/处理后 B-scan 对比图；
   - 中心 A-scan 峰值检测图；
   - `summary.json` 给出关键参数与检测结果。

## 4. 可交付内容
- `simulate_gpr_course_design.py`：可运行仿真脚本。
- `report_course_design.md`：课程设计报告（含学号/姓名/专业占位）。
- `results/*.png` + `results/summary.json`：仿真结果。

## 5. 运行方式
```bash
cd /mnt/e/Openclaw/.openclaw/workspace/shared/course_signal_design
python3 simulate_gpr_course_design.py
```
