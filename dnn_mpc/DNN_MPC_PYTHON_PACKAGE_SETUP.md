# 化工控制过程 DNN-MPC MATLAB Python Package 使用说明

本模块参照 `gridattackpkg`，通过 MATLAB Compiler SDK 导出为 Python Package，再由 platform 在 MATLAB Runtime R2024b 下调用。默认 Python 导入模块名为 `dnnmpcpkg`，默认导出目录为 `dnn_mpc/build_python`，pip 包名为 `dnnmpcpkg-R2024b`。

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
cd("H:\repository\Platform_sjtu\dnn_mpc\matlab")
build_dnnmpcpkg_python_package()
```

也可以指定输出目录和包名：

```matlab
build_dnnmpcpkg_python_package( ...
    "H:\repository\Platform_sjtu\dnn_mpc\build_python", ...
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
cd H:\repository\Platform_sjtu\dnn_mpc\build_python
python -m pip install --upgrade --force-reinstall --no-cache-dir --no-build-isolation .
python -m pip show dnnmpcpkg-R2024b
```

如果 platform 页面里 `Python包目录` 填写的是 `H:\repository\Platform_sjtu\dnn_mpc\build_python`，平台也会优先从该导出目录加载包；但仍建议执行上面的 pip 安装，便于目标机部署和排查。

如果重编译后仍看到旧错误，先关闭所有 Python/platform 进程，再清理 MATLAB Runtime 的 CTF 解包缓存：

```powershell
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\MathWorks\MatlabRuntimeCache\R2024b\dnnmpc*" -ErrorAction SilentlyContinue
```

然后重新启动 platform。

## 5. 最小验证

先配置当前 PowerShell 会话的 Runtime 路径：

```powershell
$env:MCR_ROOT = "H:\software\matlab software\Matlab R2024b Runtime\R2024b"
$env:PATH = "$env:MCR_ROOT\runtime\win64;$env:MCR_ROOT\bin\win64;$env:MCR_ROOT\extern\bin\win64;$env:PATH"
```

验证导入和初始化：

```powershell
python -c "import dnnmpcpkg; print(dnnmpcpkg.__file__); h=dnnmpcpkg.initialize(); print('init ok'); h.terminate()"
```

验证 `DNNTrain` 对应的训练入口：

```powershell
python -c "import dnnmpcpkg; h=dnnmpcpkg.initialize(); print(h.run_process_control_training(r'H:\repository\Platform_sjtu\dnn_mpc\output', 1000, 50, '64,64', '')); h.terminate()"
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
python -c "import dnnmpcpkg; h=dnnmpcpkg.initialize(); print(h.run_process_control_mpc_validation(r'H:\repository\Platform_sjtu\dnn_mpc\output', r'H:\repository\Platform_sjtu\dnn_mpc\output\process_control_nn_model.mat', 5.0, 10)); h.terminate()"
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

## 6. platform 页面配置

进入 `化工控制过程 - DNNTrain` 或 `化工控制过程 - MPC simulation` 后设置：

- `Python包目录`: `H:\repository\Platform_sjtu\dnn_mpc\build_python`
- `Python包名`: `dnnmpcpkg`
- `MCR_ROOT`: MATLAB Runtime R2024b 根目录，例如 `H:\software\matlab software\Matlab R2024b Runtime\R2024b`
- `输出目录`: `H:\repository\Platform_sjtu\dnn_mpc\output` 或自定义目录
- `模型文件`: `MPC simulation` 页默认使用最近一次 `DNNTrain` 生成的 `process_control_nn_model.mat`

`DNNTrain` 成功后，platform 会记录最近一次训练的模型路径；进入或运行 `MPC simulation` 时，如果用户没有手动指定其他模型，会自动填入该模型文件。

`Python包目录` 可以留空；留空时平台使用当前 Python 环境中已 pip 安装的 `dnnmpcpkg`。为便于排查，部署时建议填写 `dnn_mpc/build_python`。
