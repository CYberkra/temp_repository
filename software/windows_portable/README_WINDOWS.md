# GPR_GUI Windows 可用包

## 快速使用（推荐）
1. 安装 Python 3.10+（勾选 Add Python to PATH）
2. 双击 `start_qt.bat`

首次运行会自动安装依赖。

## 文件说明
- `start_qt.bat`：启动 Qt GUI（主界面）
- `start_basic.bat`：启动基础 Tk 界面
- `install_deps.bat`：手动安装依赖
- `GPR_GUI/`：程序源码

## 备注
- 这是“Windows 可运行包”（源码+启动脚本），无需在 Linux 端构建 exe。
- 若需纯 exe，可在 Windows 上运行 `pyinstaller` 进行打包。
