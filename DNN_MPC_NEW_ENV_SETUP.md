# Dnn-Mpc 新环境部署教程

本文面向“目标系统只安装 MATLAB Runtime R2024b，不安装完整 MATLAB，也不使用 MATLAB Engine”的部署场景。`dnnmpcpkg` 是化工控制过程 DNN-MPC 模块通过 MATLAB Compiler SDK 导出的 Python Package，平台通过 MATLAB Runtime 直接调用它。

## 0.部署完成之后的使用

Step1. 设置好Runtime路径&超参数之后，点击`运行训练模块`进行训练.

Step2. 再设置一次Runtime的路径&超参数之后&模型文件(默认为Step1生成的模型路径文件路径)之后，点击`运行MPC仿真`按钮运行simulation.

## 1. 适用范围

- 操作系统: Windows
- MATLAB 运行时版本: `R2024b`
- Python 版本: `3.9` / `3.10` / `3.11` / `3.12`
- Python 导入模块名: `dnnmpcpkg`
- pip 包名: `dnnmpcpkg-R2024b`
- 平台页面: `化工控制过程 - DNNTrain` / `化工控制过程 - MPC simulation`

> 说明:
> - 目标机只需要 MATLAB Runtime R2024b。
> - 目标机不需要完整 MATLAB，不需要 MATLAB Compiler SDK，不需要 MATLAB Engine。
> - 重新编译 `.m` 源码时才需要完整 MATLAB R2024b、MATLAB Compiler SDK，以及 DNNTrain 依赖的 Deep Learning Toolbox、MPC simulation 依赖的 Optimization Toolbox。

## 2. 需要准备什么

在目标系统里至少需要下面几项：

1. `platform` Python/conda 环境
2. MATLAB Runtime R2024b
3. 已导出的 `dnn_mpc/build_python` 目录
4. platform 项目代码，尤其是 `process_control_dnn_mpc.py`

导出目录结构通常是：

```text
dnn_mpc/build_python/
|-- dnnmpcpkg/
|   |-- __init__.py
|   `-- dnnmpcpkg.ctf
|-- dnnmpcpkg_R2024b.egg-info/
|-- pyproject.toml
|-- setup.py
`-- readme.txt
```

后文把这个目录记作：

```text
<EXPORT_DIR>
```

例如：

```text
H:\repository\Platform_sjtu\dnn_mpc\build_python
```

## 3. 确认 MATLAB Runtime R2024b

假设 Runtime 安装目录是：`<MCR_ROOT>`

必须确认下面这个 DLL 存在：

```powershell
Test-Path "<MCR_ROOT>\runtime\win64\mclmcrrt24_2.dll"
```

如果返回 `False`，说明 Runtime 路径不对，或者安装的不是 R2024b。

## 4. 安装 dnnmpcpkg 到 platform 环境

每次拿到重新编译后的 `dnn_mpc\build_python`，都需要关闭正在运行的 platform，并在 platform conda 环境中重新安装：

```powershell
conda activate platform
cd H:\repository\Platform_sjtu\dnn_mpc\build_python
python -m pip install --upgrade --force-reinstall --no-cache-dir --no-build-isolation .
```

安装完成后检查包信息：

```powershell
python -m pip show dnnmpcpkg-R2024b
```

## 5. 配置 Runtime PATH 并测试导入

MATLAB 生成的 Python Package 在 `import dnnmpcpkg` 时会立即查找 Runtime DLL，所以必须先把 Runtime 目录加入 `PATH`。

```powershell
$env:MCR_ROOT = "<MCR_ROOT>"
$env:PATH = "$env:MCR_ROOT\runtime\win64;$env:MCR_ROOT\bin\win64;$env:MCR_ROOT\extern\bin\win64;$env:PATH"

python -c "import dnnmpcpkg; print(dnnmpcpkg.__file__)"
```

如果输出类似下面路径，说明 Python 包已经可导入：

```text
..\lib\site-packages\dnnmpcpkg\__init__.py
```

再测试 Runtime 初始化：

```powershell
python -c "import dnnmpcpkg; h=dnnmpcpkg.initialize(); print('init ok'); h.terminate()"
```

如果输出：

```text
init ok
```

说明 Python 环境、`dnnmpcpkg` 和 MATLAB Runtime 已经打通。

如果重新编译并重新安装后仍看到旧的 MATLAB 报错，先关闭所有 Python/platform 进程，再清理 Runtime 解包缓存：

```powershell
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\MathWorks\MatlabRuntimeCache\R2024b\dnnmpc*" -ErrorAction SilentlyContinue
```

然后重新打开 platform。

## 6. 最小完整运行示例

新建 `test_dnnmpcpkg.py`：

```python
import os
from pathlib import Path

MCR_ROOT = Path(r"<MCR_ROOT>")
OUTPUT_DIR = Path(r"H:\repository\Platform_sjtu\dnn_mpc\output")

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

## 7. platform 页面配置

打开 platform 后进入 `化工控制过程` 栏目，下面有两个独立页面：

```text
化工控制过程 - DNNTrain
化工控制过程 - MPC simulation
```

两个页面使用同一套 Runtime 包配置，建议填写：

```text
Python包目录: ..\dnn_mpc\build_python
Python包名: dnnmpcpkg
MCR_ROOT: ..\R2024b
输出目录: ..\dnn_mpc\output
模型文件: ..\dnn_mpc\output\process_control_nn_model.mat
```

`Python包目录` 可以留空；留空时平台会使用当前 Python 环境中已 pip 安装的 `dnnmpcpkg`。为便于排查，建议部署时仍填写 `<EXPORT_DIR>`。

页面功能拆分如下：

- `DNNTrain`: 运行训练模块，支持外部数据集输入，展示训练曲线和预测误差图。
- `MPC simulation`: 默认读取最近一次 `DNNTrain` 生成的模型文件，运行 MPC 仿真，展示状态轨迹、控制输入、跟踪误差和代价曲线。
- 两个页面都通过 `progress.json` 做进度轮询。

现在所有计算都通过 `dnnmpcpkg` + MATLAB Runtime 执行，不再依赖 MATLAB Engine、旧本地示例页面或本地 `.m` 源码。

## 8. 标准部署顺序

1. 安装 MATLAB Runtime R2024b
2. 准备 platform conda 环境
3. 拷贝 platform 项目和最新的 `dnn_mpc/build_python`
4. 关闭正在运行的 platform
5. 用 platform Python 重新安装 `<EXPORT_DIR>`
6. 配置并测试 Runtime PATH
7. 运行 `init ok` 最小测试
8. 打开 platform，在 `化工控制过程 - DNNTrain` 页面运行训练
9. 进入 `化工控制过程 - MPC simulation` 页面；模型文件会默认填入最近一次 DNNTrain 生成的 `process_control_nn_model.mat`，确认后运行仿真

## 9. 重新编译后的更新检查

重新编译并安装新包后，建议按下面顺序确认：

```powershell
conda activate platform
python -m pip show dnnmpcpkg-R2024b
python -c "import dnnmpcpkg; print(dnnmpcpkg.__file__)"
```

`dnnmpcpkg.__file__` 应指向当前 platform 环境的 `site-packages`，或页面中 `Python包目录` 指定的 `dnn_mpc\build_python`。如果仍然执行旧逻辑，关闭 platform 并清理 `MatlabRuntimeCache\R2024b\dnnmpc*` 后再试。

## 10. 常见问题

### 10.1 点击 DNNTrain 时提示 `fitnet` 未定义

这说明 `MCR_ROOT` 已经生效，MATLAB Runtime 已经进入 `dnnmpcpkg`，但当前包里的训练依赖不可用。需要在完整 MATLAB R2024b 编译环境中确认：

```matlab
which fitnet -all
which train -all
```

确认可用后重新运行：

```matlab
cd("H:\repository\Platform_sjtu\dnn_mpc\matlab")
build_dnnmpcpkg_python_package()
```

之后回到 platform conda 环境，按第 4 节重新安装。


### 10.2 点击 MPC simulation 时提示 `optimoptions` 未定义

这同样说明 `MCR_ROOT` 已经生效，MATLAB Runtime 已经进入 `dnnmpcpkg`，但当前包里的优化求解依赖不可用。`optimoptions` 和 `fmincon` 来自 Optimization Toolbox，需要在完整 MATLAB R2024b 编译环境中确认：

```matlab
which optimoptions -all
which fmincon -all
```

确认可用后重新运行：

```matlab
cd("H:\repository\Platform_sjtu\dnn_mpc\matlab")
build_dnnmpcpkg_python_package()
```

之后回到 platform conda 环境，按第 4 节重新安装，并关闭 platform 后清理 `MatlabRuntimeCache\R2024b\dnnmpc*` 再重启。
