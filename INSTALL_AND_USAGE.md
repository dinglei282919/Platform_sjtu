# 平台安装与启动

平台提供 PySide6 桌面端和仅监听本机的 Web 端。运行四个风险管控模块需要 Python 环境，浏览器页面由 React 构建。异常行为检测使用 MATLAB Runtime R2024b。风险管控页面的设置、曲线和导出见[操作手册](RISK_CONTROL_MANUAL.md)。源码分层与新增模块约定见[架构指南](AGENTS.md)。

## 1. 前置条件

- Windows、Conda 和 Python 3.10。项目验证环境使用 Python 3.10.20。
- 需要构建 Web 页面时安装 Node.js 和 npm。已有 `web_frontend/dist` 时，可直接由 FastAPI 提供页面。
- 使用异常行为检测时安装对应 MATLAB Runtime，步骤见第 4 节。重新编译导出包需要完整 MATLAB 开发环境。

## 2. 创建 Python 环境

在仓库根目录打开 PowerShell，创建并激活环境。

```powershell
conda create -n Platform python=3.10.20 -y
conda activate Platform
python --version
```

## 3. 安装项目依赖

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
python -m pip check
```

Web 端从锁文件安装依赖并构建。使用启动脚本时，它也能在缺少构建产物时完成首次构建。

```powershell
Push-Location web_frontend
npm ci
npm run build
Pop-Location
```

## 4. 异常行为检测的 MATLAB Runtime

在已激活的平台环境中安装仓库内的导出包。将示例路径换成实际 Runtime 根目录，核对 DLL 并初始化一次。

```powershell
python -m pip install --no-build-isolation .\build_python
$env:MCR_ROOT = 'C:\path\to\MATLAB_Runtime\R2024b'
$env:Path = "$env:MCR_ROOT\runtime\win64;$env:MCR_ROOT\bin\win64;$env:MCR_ROOT\extern\bin\win64;$env:Path"
Test-Path "$env:MCR_ROOT\runtime\win64\mclmcrrt24_2.dll"
python -c "import gridattackpkg; h=gridattackpkg.initialize(); print('gridattackpkg ok'); h.terminate()"
```

当前 DNN-MPC 使用共享 Python 核心。旧 `dnnmpcpkg` 的环境、重新导出和独立验证保存在[历史编译包说明](dnn_mpc/DNN_MPC_PYTHON_PACKAGE_SETUP.md)。

## 5. 启动桌面或 Web

```powershell
python Framework.py
```

Web 用另一个已激活平台环境的终端启动。

```powershell
.\scripts\run_web_local.ps1
```

打开 [本机 Web 平台](http://127.0.0.1:8000/)。`-NoBrowser` 可跳过自动打开浏览器。前端改动后用 `-Rebuild` 重新构建，此选项会重启已有的平台后台，应先处理正在运行的任务。可用 `-PythonExe` 指定解释器。开发时另在 `web_frontend` 运行 `npm run dev`，Vite 将 `/api` 请求代理至 8000。

在“风险管控优化决策”菜单中选择单模块即可进入对应在线试验。[深度神经网络优化控制](RISK_CONTROL_MANUAL.md#dnn-mpc) · [多变量过程自愈控制](RISK_CONTROL_MANUAL.md#self-healing) · [FDI](RISK_CONTROL_MANUAL.md#fdi) · [重放](RISK_CONTROL_MANUAL.md#replay)

## 6. 使用说明

- 异常行为检测模块的 MATLAB Runtime 路径应填写：

```text
C:\path\to\MATLAB_Runtime\R2024b
```

### 异常检测的流程工业参数

- `随机攻击强度范围（%）`：每次随机攻击试验的强度区间，默认 `5%–10%`，可设范围 `5%–50%`。调用 Runtime 时会除以 100 转换为 `percent_range=[0.05, 0.10]`。
- `测量噪声强度（%）`：默认 `2%`。
- `过程扰动强度（%）`：默认 `5%`。
- 两项参数的可设范围均为 `1%–30%`。
- 当前实现是对既有 `gridattackpkg` 的兼容映射：测量噪声和过程扰动的百分比均除以 10 后传入 Runtime。该映射不改变已编译 MATLAB 包的底层数学模型。

- 历史 DNN-MPC 编译包的 `MCR_ROOT` 应填写实际安装目录；当前 Python 页面不需要该配置。路径示例：

```text
C:\path\to\MATLAB_Runtime\R2024b
```

- 如果界面路径与安装位置不符，请更新 Runtime 配置后重新运行。
- 如果提示缺少 `numpy`、`PySide6`、`torch` 等包，说明安装依赖和运行程序使用的不是同一个 Python 环境。

确认当前 Python 路径：

```cmd
python -c "import sys; print(sys.executable)"
```

应指向 Conda 环境 `Platform`。

## 7. 常见问题

### 找不到 `mclmcrrt24_2.dll`

检查 `MCR_ROOT` 是否正确：

```cmd
dir C:\path\to\MATLAB_Runtime\R2024b\runtime\win64\mclmcrrt24_2.dll
```

如果不存在，说明 Runtime 路径不对或安装的不是 R2024b。

### 仍然使用旧 Runtime 路径

确认启动目录：

```cmd
cd
```

检查页面中的 Runtime 路径、`MCR_ROOT` 与 PATH 是否均指向实际安装的 R2024b 目录。默认值不代表目标机器的安装位置，应按部署环境配置。

### Python 包安装到错误环境

不要混用多个 Python 路径。推荐始终使用：

```cmd
conda activate Platform
python -m pip install ...
python Framework.py
```

## 8. 如何增加新模块

新增 Qt 页面、注册菜单、接入 Web 服务以及回归测试的步骤见[架构指南第 5 节](AGENTS.md#new-module)。分支、提交与合并的约定见[项目管理规范](PROJECT_MANAGEMENT.md)。
