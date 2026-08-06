# 项目 BS/CS 架构与开发指南

本文档是本项目给 Codex 和开发者使用的结构说明。第一次进入项目时，先阅读本文，再根据任务范围选择 BS 或 CS 代码。不要把一个架构的页面修改方式套用到另一个架构。

## 1. 项目边界

### 1.1 CS 桌面端

CS（桌面端）使用 PySide6，主入口链路为：

```text
Framework.py -> main_interface.py -> 各功能 QWidget 页面
```

当前桌面端页面源码位于项目根目录，主要包括：

| 功能 | CS 页面文件 |
| --- | --- |
| 基于移动目标防御的异常检测 | `anomaly_detection.py` |
| 潜在安全威胁识别与自动分类 | `error_classification.py` |
| 多评估准则融合的风险学习分析 | `auto_score.py` |
| 风险场景动态匹配与适配方案生成算法 | `cdq_risk_matching.py` |
| 控制模型训练评估、优化控制仿真验证 | `process_control_dnn_mpc.py` |
| SDG-HAZOP | `sdg_hazop.py` |
| 基于 GSPN-MC 模型的动态化 SIL 验证方法 | `sil_validation.py` |

导航菜单、页面容器、页面切换和懒加载逻辑集中在 `main_interface.py`。启动 CS 时运行：

```powershell
conda activate Platform
python Framework.py
```

### 1.2 BS 本机 Web 端

BS（浏览器/服务端）由 React 前端和 FastAPI 后端组成：

| 层 | 目录/文件 | 技术 |
| --- | --- | --- |
| 页面和导航 | `web_frontend/src/App.tsx` | React + TypeScript |
| 页面样式 | `web_frontend/src/styles.css` | CSS |
| 前端构建和开发服务器 | `web_frontend/package.json`、`web_frontend/vite.config.ts` | Vite |
| API、请求模型、任务和结果接口 | `web_backend/app.py` | FastAPI |
| 长任务协调 | `web_backend/task_manager.py` | 单工作线程任务管理 |
| 算法/Runtime 适配 | `web_backend/services/` | BS 后端服务层 |
| 本机启动脚本 | `scripts/run_web_local.ps1` | PowerShell + Uvicorn |

BS 启动入口是：

```powershell
.\scripts\run_web_local.ps1
```

默认地址为 `http://127.0.0.1:8000`，只绑定本机，不应将启动脚本改成对局域网开放，除非用户明确提出部署需求。

前端开发模式使用 Vite 的 `5173` 端口，并把 `/api` 代理到 `http://127.0.0.1:8000`。开发模式只负责前端热更新，FastAPI 仍需单独运行。

`web_backend/services` 是 BS 的适配层，不是 CS 页面目录。它可能复用根目录的算法模块、`platform_core` 或 MATLAB Runtime 导出包，因此新增 BS 功能时应在这里做服务适配，不要把 PySide6 页面直接导入 FastAPI。

### 1.3 共享层和运行生成物

- `platform_core/`：保存不依赖 PySide6、React 或 FastAPI 的公共算法逻辑。CS 和 BS 都需要使用的纯 Python 数学逻辑优先放在这里。
- `input_data/`：CSV、Excel、MAT 等输入数据。当前模块依赖的文件名和相对路径是接口约定，不要随意重命名或移动。
- `output_figures/`：通用算法生成的图片和结果。
- `gridattackpkg_core_output/`：异常检测 Runtime 相关输出。
- `dnn_mpc/output/`：DNN/MPC 训练和仿真输出，例如 `training_performance.png`、`prediction_error.png`、`progress.json`。
- `web_runtime/`：BS 本机任务运行期间的临时任务和结果目录。

输出目录中的文件通常是运行生成物。除非任务明确要求更新样例结果，否则不要把临时输出当作源码修改。

### 1.4 MATLAB Runtime 导出包

以下目录是 MATLAB 编译导出的 Python 包，不属于 BS 或 CS 页面源码：

- `build_python/`：`gridattackpkg`，供异常行为检测使用。
- `dnn_mpc/build_python/`：`dnnmpcpkg`，供 DNNTrain 和 MPC 使用。

不要直接修改这些目录中的生成文件，尤其是导出的 `__init__.py`、`.ctf` 和其他编译产物。需要改变底层 MATLAB 算法时，应修改 MATLAB 源工程并重新导出，再同步安装说明和版本信息。

## 2. 当前模块映射

页面名称、算法来源和 BS 接口的对应关系如下。修改模块时先确认自己修改的是正确架构。

| 模块 | CS 页面/入口 | BS 页面 | BS 后端服务/API | 依赖或数据 |
| --- | --- | --- | --- | --- |
| 基于移动目标防御的异常检测 | `anomaly_detection.py` | `App.tsx` 的 `AnomalyPage` | `web_backend/services/anomaly.py`；`/api/anomaly/*` | `gridattackpkg` |
| 潜在安全威胁识别与自动分类 | `error_classification.py` | `ClassificationPage` | `web_backend/services/classification.py`；`/api/classification/*` | `input_data/error_classification_*.csv` |
| 多评估准则融合的风险学习分析 | `auto_score.py` | `ScorePage` | `platform_core/scoring.py`、`web_backend/services/scoring.py`；`/api/score/*` | 公共评分逻辑 |
| 风险场景动态匹配与适配方案生成算法 | `cdq_risk_matching.py` | `CdqPageReplica` | `web_backend/services/cdq.py`；`/api/cdq/*` | `input_data/cdq_data.xlsx` |
| 控制模型训练评估 | `process_control_dnn_mpc.py` 的 `page_mode="training"` | `TrainingPage` | `web_backend/services/training.py`；`/api/training/*` | `dnnmpcpkg`、DNN 训练输出 |
| 优化控制仿真验证 | `process_control_dnn_mpc.py` 的 `page_mode="mpc"` | `MpcPage` | `web_backend/services/mpc.py`；`/api/mpc/*` | `dnnmpcpkg`、MPC 输出 |
| SDG-HAZOP | `sdg_hazop.py` | `SdgPage` | `web_backend/services/sdg.py`；`/api/sdg/*` | SDG 示例/分析数据 |
| 基于 GSPN-MC 模型的动态化 SIL 验证方法 | `sil_validation.py` | `SilPage` | `web_backend/services/sil.py`；`/api/sil/*` | GSPN-MC 仿真参数和结果 |

CS 菜单名称和 BS 菜单名称应保持业务含义一致。当前 CS 菜单定义在 `main_interface.py`，BS 菜单和 `ModuleId` 定义在 `web_frontend/src/App.tsx`，后端模块列表在 `web_backend/app.py` 的 `/api/modules` 接口中。

## 3. 从零安装依赖

### 3.1 基础环境

当前项目以 Windows、Conda、Python 3.10.20 为基准。推荐使用名为 `Platform` 的环境：

```powershell
conda create -n Platform python=3.10.20 -y
conda activate Platform
python -m pip install --upgrade pip setuptools wheel
```

Python 依赖以根目录 `requirements.txt` 为准，安装命令为：

```powershell
python -m pip install --upgrade --force-reinstall -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

该文件同时包含 PySide6、Matplotlib、NumPy、Pandas、PyTorch、FastAPI、Uvicorn 和测试所需的 Python 包。不要只在本机手动安装新依赖而不更新 `requirements.txt`。

### 3.2 BS 前端依赖

BS 前端需要 Node.js LTS 和 npm。依赖声明在 `web_frontend/package.json`，锁定版本在 `web_frontend/package-lock.json`。安装和构建：

```powershell
cd web_frontend
npm install
npm run build
cd ..
```

如果只是运行已经存在的 `web_frontend/dist`，不需要启动 Vite 开发服务器；修改前端源码后必须重新构建，或使用启动脚本的 `-Rebuild` 参数。

### 3.3 MATLAB Runtime 和导出包

要跑通依赖 MATLAB 的完整功能，需要 MATLAB Runtime R2024b。典型安装根目录为：

```text
E:\MATLAB2024
```

必须确认以下 DLL 存在：

```text
E:\MATLAB2024\runtime\win64\mclmcrrt24_2.dll
```

在同一个 `Platform` Python 环境中安装两个本地导出包：

```powershell
python -m pip install --upgrade --force-reinstall --no-build-isolation .\build_python
python -m pip install --upgrade --force-reinstall --no-build-isolation .\dnn_mpc\build_python
python -m pip check
```

`build_python` 和 `dnn_mpc/build_python` 不在 PyPI 上，不能用普通 `pip install gridattackpkg` 或 `pip install dnnmpcpkg` 替代。也可以在对应页面填写本机包目录，但部署和调试时应确保服务实际使用的 Python 环境与页面/脚本使用的环境一致。

运行时可在当前 PowerShell 会话配置：

```powershell
$env:MCR_ROOT = 'E:\MATLAB2024'
$env:Path = "$env:MCR_ROOT\runtime\win64;$env:MCR_ROOT\bin\win64;$env:MCR_ROOT\extern\bin\win64;$env:Path"
```

完整 MATLAB 只在重新编译导出包时需要，目标运行机器不需要安装完整 MATLAB：

- DNNTrain 重新编译需要 Deep Learning Toolbox。
- MPC 重新编译需要 Optimization Toolbox。
- 具体导出流程参考 `GRIDATTACKPKG_NEW_ENV_SETUP.md` 和 `DNN_MPC_NEW_ENV_SETUP.md`。

### 3.4 数据和输出约定

以下文件是当前功能的输入或样例数据，缺失时不要用空文件静默替代，应提示用户补齐：

- `input_data/cdq_data.xlsx`
- `input_data/error_classification_train.csv`
- `input_data/error_classification_test.csv`
- `input_data/error_classification_easy_train.csv`
- `input_data/error_classification_easy_test.csv`
- `input_data/error_classification_hard_train.csv`
- `input_data/error_classification_hard_test.csv`

常用结果目录是 `output_figures/`、`gridattackpkg_core_output/` 和 `dnn_mpc/output/`。结果图片、JSON、进度文件的名称需要与现有前端和后端约定保持一致。

### 3.5 本机启动脚本的环境路径

当前 `scripts/run_web_local.ps1` 内部固定使用：

```text
D:\ana3\envs\Platform\python.exe
```

并默认查找：

```text
C:\Program Files\nodejs\npm.cmd
```

换机器或 Conda 环境路径不同，必须修改该脚本中的 `$python`/`$npm`，或直接使用当前环境手动启动：

```powershell
conda activate Platform
python -m uvicorn web_backend.app:app --host 127.0.0.1 --port 8000
```

手动启动前仍需先构建 `web_frontend`，并在项目根目录执行命令。

## 4. 启动、验证和排错

### 4.1 CS 启动

```powershell
conda activate Platform
python Framework.py
```

### 4.2 BS 启动

普通启动（首次运行会安装前端依赖、构建前端并启动 FastAPI）：

```powershell
.\scripts\run_web_local.ps1
```

前端源码发生变化后强制重建：

```powershell
.\scripts\run_web_local.ps1 -Rebuild
```

重建但不自动打开浏览器：

```powershell
.\scripts\run_web_local.ps1 -Rebuild -NoBrowser
```

访问：

```text
http://127.0.0.1:8000
```

前端开发模式：

```powershell
cd web_frontend
npm install
npm run dev
```

开发模式访问 Vite 地址（通常为 `http://127.0.0.1:5173`），API 请求会通过 `vite.config.ts` 代理到 `http://127.0.0.1:8000`。

### 4.3 基础验证

在项目根目录、已激活的 `Platform` 环境中执行：

```powershell
python -m pip check
python -m unittest discover -s tests -p "test_*.py"
python -m py_compile Framework.py main_interface.py
```

BS 前端构建验证：

```powershell
Push-Location web_frontend
npm run build
Pop-Location
```

Runtime 最小初始化验证：

```powershell
python -c "import gridattackpkg, matlab; h=gridattackpkg.initialize(); print('gridattackpkg ok'); h.terminate()"
python -c "import dnnmpcpkg; h=dnnmpcpkg.initialize(); print('dnnmpcpkg ok'); h.terminate()"
```

### 4.4 常见问题

- 报 `mclmcrrt24_2.dll` 找不到：检查 MATLAB Runtime 是否为 R2024b，以及 `MCR_ROOT`、页面 Runtime 路径和系统 `PATH` 是否指向同一安装目录。
- `gridattackpkg` 或 `dnnmpcpkg` 不可导入：确认两个本地包安装在当前 `python` 对应的 `Platform` 环境中，执行 `python -m pip show ...` 和 `python -c "import sys; print(sys.executable)"`。
- BS 页面仍是旧版本：确认前端已执行 `npm run build`，必要时使用 `.\scripts\run_web_local.ps1 -Rebuild`，并清理/重新生成过期的 `web_frontend/dist`。
- `fitnet`、`optimoptions`、`fmincon` 等 MATLAB 函数缺失：这是导出包或 MATLAB 工具箱不完整的问题，不是通过安装普通 Python 包解决的；重新检查 MATLAB 导出环境和对应 Toolbox。
- 分类模块报数据缺失：检查 `input_data/error_classification_*.csv` 文件是否存在且未被改名。
- CDQ 模块报数据缺失：检查 `input_data/cdq_data.xlsx` 是否存在，且没有改变工作目录导致相对路径失效。
- Runtime 包路径和输出目录不一致：统一使用当前项目中的 `build_python`、`dnn_mpc/build_python` 和 `dnn_mpc/output`，不要混用其他项目副本的路径。

只查看 BS 静态界面时，可以先安装 Python/FastAPI 和 Node.js 并构建前端，不必立即安装 MATLAB Runtime；要执行异常检测、DNNTrain、MPC 等 Runtime 功能时，仍必须完成 R2024b Runtime 和对应导出包配置。

## 5. 新增模块规范

### 5.1 新增 CS 模块

1. 在项目根目录新建继承 `QWidget` 的页面文件，例如 `new_module.py`。
2. 在 `main_interface.py` 中增加页面导入（推荐在懒加载分支内导入）、页面容器成员和统一隐藏逻辑。
3. 在导航菜单列表中增加一级菜单或子模块名称。
4. 在 `_on_submodule_clicked()` 中增加与菜单名称完全一致的显示分支。
5. 长时间计算必须使用 `QThread` 或 `QObject + QThread`，不能在 GUI 线程中直接执行耗时 Runtime、训练或仿真。
6. 使用 MATLAB Runtime 时，补充包目录、Runtime 根目录检查、输出文件检查和可读的错误提示。
7. 保持页面切换、窗口关闭和线程清理逻辑完整，不要只让新页面能打开而忽略旧页面隐藏或后台线程退出。
8. 新增第三方依赖时同步更新根目录 `requirements.txt`。
9. 至少执行：

   ```powershell
   python -m py_compile new_module.py main_interface.py
   python Framework.py
   ```

   并验证菜单展开、页面显示、切换隐藏、任务结束和错误提示。

### 5.2 新增 BS 模块

1. 在 `web_frontend/src/App.tsx` 中增加 `ModuleId`、导航项和页面选择分支。
2. 新建页面组件，沿用现有 `Page`、`panel`、结果图、状态区和错误提示结构。
3. 在 `web_frontend/src/styles.css` 中补充样式，重点检查父级高度、`overflow`、图片完整显示、左右栏对齐和窄窗口响应式布局。
4. 需要后端计算时，在 `web_backend/services/` 中新建适配器；不要在 React 组件中复制 Runtime 或长时间算法逻辑。
5. 在 `web_backend/app.py` 中增加必要的 Pydantic 请求模型、默认配置接口、任务提交接口、任务查询/进度接口，以及图片、JSON 或其他结果接口。
6. 长时间训练、仿真和 Runtime 调用必须使用 `TaskManager`，不能阻塞 FastAPI 请求线程。
7. 进度文件、结果文件和图片只能通过任务 ID 或受控输出目录访问，避免任意路径读取。
8. 如果 BS 需要复用 CS 的算法逻辑，应抽取无 UI 依赖的部分到 `platform_core` 或独立 service；不要导入 PySide6 页面。
9. 在 `tests/test_web_core.py` 增加 API、任务状态、进度、结果文件和错误场景测试。
10. 修改前端依赖时同步更新 `web_frontend/package.json` 和 `web_frontend/package-lock.json`；修改 Python 依赖时同步更新 `requirements.txt`。
11. 构建并测试：

    ```powershell
    Push-Location web_frontend
    npm run build
    Pop-Location
    python -m unittest discover -s tests -p "test_*.py"
    ```

### 5.3 共享算法模块

- 不把 PySide6 UI 代码直接导入 BS 后端。
- CS 和 BS 共用的数学逻辑优先放入 `platform_core`，并让页面层只负责参数收集、任务调度和结果展示。
- CS、BS 必须保持默认参数、参数校验、结果字段、图片名称和输出目录约定一致；复刻页面时先比较现有 CS 行为，再调整 BS 展示。
- 不直接修改 MATLAB 导出的 `__init__.py`、`.ctf` 或其他生成文件。
- 新增 MATLAB Runtime 导出包时，必须同时补充安装说明、Runtime 版本/依赖检查、包导入检查和最小调用示例。
- 数据文件、输出文件和 API 字段一旦被前端或后端使用，就视为模块接口的一部分；修改时要同步修改另一端和测试。

## 6. 维护规则

- 先用 `rg` 定位实际入口、模块名和接口，再修改文件；不要凭截图或旧文档猜测当前结构。
- 页面问题优先改对应架构的页面层：CS 改 PySide6 页面/`main_interface.py`，BS 改 `web_frontend`/`web_backend`，不要跨架构复制 UI 代码。
- 涉及训练、仿真或 Runtime 的改动，必须同时检查进度、结果图、结果 JSON 和错误状态。
- 不要删除或覆盖用户已有的运行结果、输入数据和未相关改动。
- 修改完成后至少运行与改动范围匹配的语法检查、前端构建和测试；涉及 Runtime 时再执行对应初始化检查。
