# 化工控制过程 DNN-MPC MATLAB Python Package 使用说明

> 路径约定：`C:\path\to\...` 是示例占位路径，执行命令前请替换为实际项目、环境或 Runtime 安装目录。

> 本文仅适用于历史编译包。当前四个风险管控页面使用 Python 共享核心；实现、部署与输出结构见 [风险管控算法实现与使用手册](../RISK_CONTROL_MANUAL.md)。

历史 DNN-MPC 算法通过 MATLAB Compiler SDK 导出为 Python Package，可由独立 Python 脚本在 MATLAB Runtime R2024b 下调用。当前平台的 DNN/MPC 页面使用 Python 共享核心，不调用该编译包。默认 Python 导入模块名为 `dnnmpcpkg`，默认导出目录为 `dnn_mpc/build_python`，pip 包名为 `dnnmpcpkg-R2024b`。

## 1. 环境要求

编译机器需要：

- MATLAB R2024b
- MATLAB Compiler SDK
- Deep Learning Toolbox；`fitnet` / `train` 必须可用
- Optimization Toolbox；`optimoptions` / `fmincon` 必须可用
- DNN-MPC 运行依赖的绘图等 MATLAB 能力

部署机器需要：

- MATLAB Runtime R2024b
- platform 的 Python/conda 环境
- 已导出的 `dnn_mpc/build_python` 目录，或已安装到当前 Python 环境中的 `dnnmpcpkg-R2024b`

目标系统采用 Runtime-only 模式，不需要完整 MATLAB，不需要 MATLAB Engine。

## 2. 编译前检查

在完整 MATLAB R2024b 中确认训练依赖存在：

```matlab
which fitnet -all
which train -all
which optimoptions -all
which fmincon -all
```

如果 `fitnet` 为空，说明当前 MATLAB 没有安装或没有授权 Deep Learning Toolbox，`DNNTrain` 会失败。

如果 `optimoptions` 或 `fmincon` 为空，说明当前 MATLAB 没有安装或没有授权 Optimization Toolbox，`MPC simulation` 会失败。

## 3. 生成 Python Package

在 MATLAB R2024b 命令行中运行：

```matlab
cd("C:\path\to\Platform_sjtu\dnn_mpc\matlab")
build_dnnmpcpkg_python_package()
```

也可以指定输出目录和包名：

```matlab
build_dnnmpcpkg_python_package( ...
    "C:\path\to\Platform_sjtu\dnn_mpc\build_python", ...
    "dnnmpcpkg")
```

构建脚本会显式检查 `fitnet`、`train`、`optimoptions` 和 `fmincon`。如果依赖缺失，会在编译阶段报 `MissingTrainingDependency`，而不是等到 platform 运行时才失败。

构建成功后，目录结构应包含：

```text
dnn_mpc/build_python/
|-- dnnmpcpkg/
|   |-- __init__.py
|   `-- dnnmpcpkg.ctf
|-- dnnmpcpkg_R2024b.egg-info/
|-- setup.py
|-- pyproject.toml
|-- requiredMCRProducts.txt
|-- unresolvedSymbols.txt
`-- readme.txt
```

建议检查 `unresolvedSymbols.txt`，正常情况下应只有表头或没有实际未解析符号。

## 4. 重新编译后的安装步骤

每次重新编译 `dnn_mpc/build_python` 后，都需要把新包重新安装到 platform 的 conda 环境。先关闭正在运行的 platform，再执行：

```powershell
conda activate platform
cd C:\path\to\Platform_sjtu\dnn_mpc\build_python
python -m pip install --upgrade --force-reinstall --no-cache-dir --no-build-isolation .
python -m pip show dnnmpcpkg-R2024b
```

独立脚本应使用安装了该导出包的 Python 环境；用 `python -m pip show dnnmpcpkg-R2024b` 检查实际安装位置。

如果重编译后仍看到旧错误，先关闭所有 Python/platform 进程，再清理 MATLAB Runtime 的 CTF 解包缓存：

```powershell
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\MathWorks\MatlabRuntimeCache\R2024b\dnnmpc*" -ErrorAction SilentlyContinue
```

然后重新启动 platform。

## 5. 最小验证

先配置当前 PowerShell 会话的 Runtime 路径：

```powershell
$env:MCR_ROOT = "C:\path\to\MATLAB_Runtime\R2024b"
$env:PATH = "$env:MCR_ROOT\runtime\win64;$env:MCR_ROOT\bin\win64;$env:MCR_ROOT\extern\bin\win64;$env:PATH"
```

验证导入和初始化：

```powershell
python -c "import dnnmpcpkg; print(dnnmpcpkg.__file__); h=dnnmpcpkg.initialize(); print('init ok'); h.terminate()"
```

验证 `DNNTrain` 对应的训练入口：

```powershell
python -c "import dnnmpcpkg; h=dnnmpcpkg.initialize(); print(h.run_process_control_training(r'C:\path\to\Platform_sjtu\dnn_mpc\output', 1000, 50, '64,64', '')); h.terminate()"
```

训练成功后会生成：

```text
dnn_mpc/output/process_control_nn_model.mat
dnn_mpc/output/training_summary.json
dnn_mpc/output/training_performance.png
dnn_mpc/output/prediction_error.png
```

再验证 `MPC simulation` 对应的仿真入口：

```powershell
python -c "import dnnmpcpkg; h=dnnmpcpkg.initialize(); print(h.run_process_control_mpc_validation(r'C:\path\to\Platform_sjtu\dnn_mpc\output', r'C:\path\to\Platform_sjtu\dnn_mpc\output\process_control_nn_model.mat', 5.0, 10)); h.terminate()"
```

仿真成功后会生成：

```text
dnn_mpc/output/process_control_mpc_result.mat
dnn_mpc/output/mpc_validation_summary.json
dnn_mpc/output/process_control_trajectory.png
dnn_mpc/output/control_input.png
dnn_mpc/output/tracking_error.png
dnn_mpc/output/cost_curve.png
```

## 6. 独立脚本完整调用示例

新建 `test_dnnmpcpkg.py`：

```python
import os
from pathlib import Path

MCR_ROOT = Path(r"<MCR_ROOT>")
OUTPUT_DIR = Path(r"C:\path\to\Platform_sjtu\dnn_mpc\output")

runtime_dir = MCR_ROOT / "runtime" / "win64"
bin_dir = MCR_ROOT / "bin" / "win64"
extern_dir = MCR_ROOT / "extern" / "bin" / "win64"
dll_path = runtime_dir / "mclmcrrt24_2.dll"

if not dll_path.exists():
    raise FileNotFoundError(f"MATLAB Runtime DLL not found: {dll_path}")

os.environ["PATH"] = os.pathsep.join([
    str(runtime_dir),
    str(bin_dir),
    str(extern_dir),
    os.environ.get("PATH", ""),
])

import dnnmpcpkg

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
handle = dnnmpcpkg.initialize()
try:
    result_json = handle.run_process_control_pipeline(
        str(OUTPUT_DIR),
        1000,
        50,
        1.0,
        5,
        "64,64",
        "",
    )
    print(result_json)
finally:
    handle.terminate()
```

运行：

```powershell
python .\test_dnnmpcpkg.py
```

成功后输出目录会生成：

```text
process_control_dataset.mat
process_control_nn_model.mat
process_control_mpc_result.mat
summary.json
training_summary.json
mpc_validation_summary.json
training_performance.png
prediction_error.png
process_control_trajectory.png
control_input.png
tracking_error.png
cost_curve.png
progress.json
```
