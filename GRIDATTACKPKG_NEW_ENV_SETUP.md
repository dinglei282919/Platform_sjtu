# gridattackpkg 新环境使用教程

本文面向“全新环境从零开始使用 `gridattackpkg` 宏包”的场景，基于当前项目已经验证通过的 `R2024b` 导出包整理。

## 1. 适用范围

- 操作系统: Windows
- MATLAB 运行时版本: `R2024b`
- Python 版本: `3.9` / `3.10` / `3.11` / `3.12`
- 宏包名称: `gridattackpkg`

> 说明:
> - 这里的“宏包”指 MATLAB Compiler SDK 导出的 Python 包。
> - 你可以使用完整 MATLAB `R2024b`，也可以只安装 MATLAB Runtime `R2024b`。

## 2. 你需要准备什么

在新环境里，至少需要下面三样东西:

1. 一个可用的 Python 环境
2. MATLAB `R2024b` 或 MATLAB Runtime `R2024b`
3. 导出的 `gridattackpkg` 安装目录

导出的目录通常长这样:

```text
build_python/
|-- gridattackpkg/
|   |-- __init__.py
|   `-- gridattackpkg.ctf
|-- pyproject.toml
|-- setup.py
`-- readme.txt
```

后文把这个目录记作:

```text
<EXPORT_DIR>
```

例如:

```text
D:\temp_save\APPdesign1\build_python
```

## 3. 创建一个全新的 Python 环境

推荐用 `venv`。

```powershell
python -m venv D:\venvs\gridattackpkg_env
```

激活环境:

```powershell
D:\venvs\gridattackpkg_env\Scripts\Activate.ps1
```

升级基础打包工具:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

## 4. 安装 MATLAB Runtime 或确认 MATLAB 安装

### 方案 A: 已安装完整 MATLAB R2024b

例如:

```text
E:\MATLAB2024
```

你需要确认下面这个 DLL 存在:

```text
E:\MATLAB2024\runtime\win64\mclmcrrt24_2.dll
```

### 方案 B: 只安装 MATLAB Runtime R2024b

例如:

```text
C:\Program Files\MATLAB\MATLAB Runtime\R2024b
```

同样确认这个 DLL 存在:

```text
C:\Program Files\MATLAB\MATLAB Runtime\R2024b\runtime\win64\mclmcrrt24_2.dll
```

后文把 MATLAB 根目录记作:

```text
<MCR_ROOT>
```

## 5. 安装 gridattackpkg 宏包

进入导出目录:

```powershell
cd <EXPORT_DIR>
```

安装:

```powershell
python -m pip install --no-build-isolation .
```

如果你想强制覆盖旧版本:

```powershell
python -m pip install --upgrade --force-reinstall --no-build-isolation .
```

安装完成后可以检查:

```powershell
python -m pip show gridattackpkg-R2024b
```

## 6. 最小可运行示例

新建一个文件，例如 `test_gridattackpkg.py`:

```python
import os
from pathlib import Path

MCR_ROOT = Path(r"E:\MATLAB2024")

runtime_dir = MCR_ROOT / "runtime" / "win64"
bin_dir = MCR_ROOT / "bin" / "win64"
extern_dir = MCR_ROOT / "extern" / "bin" / "win64"
dll_path = runtime_dir / "mclmcrrt24_2.dll"

if not dll_path.exists():
    raise FileNotFoundError(f"MATLAB Runtime DLL not found: {dll_path}")

prefix = os.pathsep.join([str(runtime_dir), str(bin_dir), str(extern_dir)])
os.environ["PATH"] = prefix + os.pathsep + os.environ.get("PATH", "")

import gridattackpkg
import matlab

handle = gridattackpkg.initialize()
try:
    overrides = {
        "percent_range": matlab.double([0.05, 0.10]),
        "sigma1": 0.2,
        "sigma2": 0.1,
    }

    output_dir = Path.cwd() / "gridattackpkg_core_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    params = handle.make_default_params(overrides)
    core_out = handle.run_core(params, str(output_dir))
    handle.plot_results_from_core(core_out, nargout=0)

    print("run_core success")
    print(f"output dir: {output_dir}")
finally:
    handle.terminate()
```

运行:

```powershell
python .\test_gridattackpkg.py
```

## 7. 只测试“能不能加载宏包”

如果你还不想执行完整算法，只想验证环境是否通了，可以先跑最小初始化测试:

```powershell
$env:PATH = "E:\MATLAB2024\runtime\win64;E:\MATLAB2024\bin\win64;E:\MATLAB2024\extern\bin\win64;$env:PATH"
python -c "import gridattackpkg, matlab; h=gridattackpkg.initialize(); print('ok'); h.terminate()"
```

如果输出:

```text
ok
```

说明下面几件事已经没问题:

- Python 环境正常
- `gridattackpkg` 已安装
- MATLAB Runtime 路径已配置
- `mclmcrrt24_2.dll` 能被加载

## 8. 如何调用导出的函数

### 8.1 初始化包

```python
handle = gridattackpkg.initialize()
```

### 8.2 调用有返回值的函数

```python
params = handle.make_default_params(overrides)
core_out = handle.run_core(params, str(output_dir))
```

### 8.3 调用无返回值的函数

MATLAB 导出的 Python 接口里，无返回值函数通常这样写:

```python
handle.plot_results_from_core(core_out, nargout=0)
```

### 8.4 结束运行

```python
handle.terminate()
```

## 9. 当前项目中可参考的调用范例

如果你已经拿到了当前项目源码，可以直接参考这两个文件:

- `run_core_plot_with_three_params.py`
- `anomaly_detection.py`

其中 `run_core_plot_with_three_params.py` 是最适合拿来做独立调用模板的。

## 10. 常见问题

### 10.1 `ModuleNotFoundError: No module named 'gridattackpkg'`

原因:

- 没有安装导出的宏包
- 安装到了别的 Python 环境

排查:

```powershell
python -m pip show gridattackpkg-R2024b
python -c "import sys; print(sys.executable)"
```

### 10.2 找不到 `mclmcrrt24_2.dll`

原因:

- 没安装 `R2024b`
- `PATH` 没加对
- `MCR_ROOT` 写错

排查:

确认下面路径存在:

```text
<MCR_ROOT>\runtime\win64\mclmcrrt24_2.dll
```

### 10.3 `initialize()` 失败

先检查:

1. Python 位数是否是 64 位
2. MATLAB Runtime 是否是 `R2024b`
3. `PATH` 是否已经在导入 `gridattackpkg` 之前设置

建议先运行第 7 节的最小初始化测试。

### 10.4 旧版本缓存干扰

如果你刚重新导出了 `.ctf`，但运行结果像是还在用旧代码，可以删除 Runtime 缓存后再测:

```powershell
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\MathWorks\MatlabRuntimeCache\R2024b\gridat*"
```

### 10.5 路径里包含特殊字符或中文后初始化异常

在当前项目中，曾出现过 `.ctf` 从特殊路径加载时初始化异常的问题。稳妥起见，建议:

- 项目目录使用纯英文路径
- 虚拟环境路径使用纯英文路径
- 导出包路径使用纯英文路径

## 11. 推荐的标准流程

建议每次在新机器上按这个顺序操作:

1. 安装 Python
2. 创建虚拟环境
3. 安装 `pip/setuptools/wheel`
4. 安装 MATLAB `R2024b` 或 MATLAB Runtime `R2024b`
5. 用导出的 `build_python` 安装 `gridattackpkg`
6. 先做一次 `initialize()` 最小测试
7. 再跑完整 `run_core()` 验证

## 12. 一套可直接复用的命令示例

假设:

- Python 环境: `D:\venvs\gridattackpkg_env`
- 导出目录: `D:\temp_save\APPdesign1\build_python`
- MATLAB 根目录: `E:\MATLAB2024`

那么命令可以直接写成:

```powershell
python -m venv D:\venvs\gridattackpkg_env

D:\venvs\gridattackpkg_env\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel

cd D:\temp_save\APPdesign1\build_python
python -m pip install --upgrade --force-reinstall --no-build-isolation .

$env:PATH = "E:\MATLAB2024\runtime\win64;E:\MATLAB2024\bin\win64;E:\MATLAB2024\extern\bin\win64;$env:PATH"
python -c "import gridattackpkg, matlab; h=gridattackpkg.initialize(); print('ok'); h.terminate()"
```

如果最后输出 `ok`，说明这个新环境已经可以开始正常调用 `gridattackpkg`。
