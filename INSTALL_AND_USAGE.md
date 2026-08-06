# 本机 Web 版（B/S）

桌面端（CS）和本机 Web 端（BS）并行保留。本机 Web 版只绑定 `127.0.0.1:8000`，不向局域网开放。

项目架构、源码边界、BS/CS 模块映射和新增模块规范以根目录 [`AGENTS.md`](AGENTS.md) 为权威说明；本文保留面向人工使用的安装、启动和 CS 模块教程。

首次启动前安装 Node.js LTS，并在项目根目录执行：

```powershell
.\scripts\run_web_local.ps1
```

脚本会在首次运行时构建 React 前端、启动 FastAPI 服务并打开浏览器。访问地址固定为：

```text
http://127.0.0.1:8000
```

前端源码变更后可执行 `.\scripts\run_web_local.ps1 -Rebuild`。当前 BS 端已包含多评估准则评分、SDG-HAZOP、SIL 验证、风险场景适配方案生成、安全威胁分类、异常行为检测以及 DNN-MPC 的训练评估和优化控制仿真；CS 桌面端仍保留全部对应功能。

# 安装与使用教程

本文档说明如何为平台创建 Python 3.10.20 环境、安装依赖、配置 MATLAB Runtime R2024b，并启动程序。

## 1. 前置条件

- Windows 系统
- 已安装 Conda
- 已安装 MATLAB R2024b 与 MATLAB Runtime R2024b
- 本机 Runtime 根路径：

```text
E:\MATLAB2024
```

确认关键 DLL 存在：

```cmd
dir E:\MATLAB2024\runtime\win64\mclmcrrt24_2.dll
```

## 2. 创建 Python 环境

在 `cmd` 中进入项目目录：

```cmd
cd /d D:\ADRL\Platform_sjtu
```

创建并激活环境：

```cmd
conda create -n Platform python=3.10.20 -y
conda activate Platform
```

确认版本：

```cmd
python --version
```

应输出：

```text
Python 3.10.20
```

## 3. 安装依赖

安装基础依赖：

```cmd
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade --force-reinstall -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

安装 MATLAB 导出的本地 Python 包：

```cmd
python -m pip install --upgrade --force-reinstall --no-build-isolation .\build_python
python -m pip install --upgrade --force-reinstall --no-build-isolation .\dnn_mpc\build_python
```

检查安装结果：

```cmd
python -m pip check
python -m pip show gridattackpkg-R2024b dnnmpcpkg-R2024b
```

## 4. 配置 Runtime

临时配置当前 `cmd` 会话的 MATLAB Runtime 路径：

```cmd
set MCR_ROOT=E:\MATLAB2024
set PATH=%MCR_ROOT%\runtime\win64;%MCR_ROOT%\bin\win64;%MCR_ROOT%\extern\bin\win64;%PATH%
```

验证 `gridattackpkg`：

```cmd
python -c "import gridattackpkg, matlab; h=gridattackpkg.initialize(); print('gridattackpkg ok'); h.terminate()"
```

验证 `dnnmpcpkg`：

```cmd
python -c "import dnnmpcpkg; h=dnnmpcpkg.initialize(); print('dnnmpcpkg ok'); h.terminate()"
```

## 5. 启动平台

每次启动前先激活环境：

```cmd
conda activate Platform
cd /d D:\ADRL\Platform_sjtu
```

启动程序：

```cmd
python Framework.py
```

## 6. 使用说明

- 异常行为检测模块的 MATLAB Runtime 路径应填写：

```text
E:\MATLAB2024
```

### 异常检测的流程工业参数

- `随机攻击强度范围（%）`：每次随机攻击试验的强度区间，默认 `5%–10%`，可设范围 `5%–50%`。调用 Runtime 时会除以 100 转换为 `percent_range=[0.05, 0.10]`。
- `测量噪声强度（%）`：默认 `2%`。
- `过程扰动强度（%）`：默认 `5%`。
- 两项参数的可设范围均为 `1%–30%`。
- 当前实现是对既有 `gridattackpkg` 的兼容映射：测量噪声和过程扰动的百分比均除以 10 后传入 Runtime。该映射不改变已编译 MATLAB 包的底层数学模型。

- DNN-MPC 模块的 `MCR_ROOT` 也应填写：

```text
E:\MATLAB2024
```

- 如果界面仍显示旧路径，说明运行的不是当前项目副本，或源码没有改到实际运行目录。
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
dir E:\MATLAB2024\runtime\win64\mclmcrrt24_2.dll
```

如果不存在，说明 Runtime 路径不对或安装的不是 R2024b。

### 仍然使用旧 Runtime 路径

确认启动目录：

```cmd
cd
```

确认源码中的默认路径：

```cmd
powershell -NoProfile -Command "Select-String -SimpleMatch -Path 'anomaly_detection.py','process_control_dnn_mpc.py' -Pattern 'E:\MATLAB2024','D:\MATLAB','mclmcrrt24_2.dll','mclmcrrt9_14.dll'"
```

期望只看到 `E:\MATLAB2024` 和 `mclmcrrt24_2.dll`。

### Python 包安装到错误环境

不要混用多个 Python 路径。推荐始终使用：

```cmd
conda activate Platform
python -m pip install ...
python Framework.py
```

## 8. 如何增加新模块

平台主界面由 `main_interface.py` 管理，功能页面通常是一个继承 `QWidget` 的 PySide6 组件。新增模块时，建议按下面流程操作。

### 8.1 新建模块页面文件

在项目根目录新建一个 Python 文件，例如：

```text
new_module.py
```

最小页面模板：

```python
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class NewModuleWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("新模块页面"))
```

如果模块需要后台计算，建议使用 `QThread` 或 `QObject + QThread`，避免长时间任务阻塞界面。

### 8.2 在主界面增加容器变量

打开：

```text
main_interface.py
```

在 `MainWindow.__init__` 中增加两个成员变量：

```python
self._new_module_content_widget = None
self._new_module_widget = None
```

其中：

- `_new_module_content_widget` 是页面容器
- `_new_module_widget` 是真实功能页面，建议懒加载

### 8.3 在内容区创建页面容器

在 `_build_content()` 中，仿照已有模块增加一个 `QFrame` 容器：

```python
self._new_module_content_widget = QFrame()
new_module_layout = QVBoxLayout(self._new_module_content_widget)
new_module_layout.setContentsMargins(0, 0, 0, 0)
new_module_layout.setSpacing(0)
self._new_module_content_widget.hide()

body_layout.addWidget(self._new_module_content_widget, 1)
```

如果 `_on_submodule_clicked()` 里有统一隐藏页面的循环，也要把新容器加入循环：

```python
for widget in (
    self._anomaly_content_widget,
    self._new_module_content_widget,
):
    if widget is not None:
        widget.hide()
```

实际代码中应保留原有容器，只把新容器补进去，不要删除已有模块。

### 8.4 在导航栏增加菜单入口

在 `_build_function_bar()` 的 `items` 列表里增加新模块入口。

如果要加到已有一级菜单，例如“风险动态分析”：

```python
("图标", "风险动态分析", False, [
    "已有子模块",
    "新模块名称",
])
```

如果要新增一级菜单：

```python
("图标", "新一级菜单", False, ["新模块名称"])
```

注意：`_on_submodule_clicked()` 中判断使用的是子模块名称字符串，所以菜单里的名称必须和后面的判断完全一致。

### 8.5 在点击逻辑中加载页面

在 `_on_submodule_clicked()` 中增加分支：

```python
elif submodule_title == "新模块名称":
    from new_module import NewModuleWidget

    self._ensure_lazy_page(
        "_new_module_widget",
        self._new_module_content_widget,
        NewModuleWidget,
    )
    self._new_module_content_widget.show()
```

推荐使用这种懒加载方式：只有用户第一次点击模块时才创建页面，启动速度更快，也能减少模块依赖错误对主程序启动的影响。

### 8.6 验证新模块

先做语法检查：

```cmd
python -m py_compile new_module.py main_interface.py
```

启动平台：

```cmd
conda activate Platform
cd /d D:\ADRL\Platform_sjtu
python Framework.py
```

检查：

- 一级菜单能正常展开
- 子模块点击后标题正确变化
- 新页面能显示
- 切换到其他模块后，新页面会隐藏
- 如果新模块有后台任务，关闭窗口时不会残留线程

### 8.7 增加依赖

如果新模块引入了新的第三方库，需要更新：

```text
requirements.txt
```

然后重新安装：

```cmd
conda activate Platform
cd /d D:\ADRL\Platform_sjtu
python -m pip install -r requirements.txt
```

不要只在当前机器手动 `pip install` 后不更新 `requirements.txt`，否则换环境后模块会缺依赖。
