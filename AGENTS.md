# 项目 BS/CS 架构与开发指南

> 路径约定：`C:\path\to\...` 是示例占位路径，执行命令前请替换为实际项目、环境或 Runtime 安装目录。

本文档说明项目的 BS/CS 架构、模块职责与开发约定。修改功能时应区分浏览器端与桌面端，遵循各自的页面结构和共享计算层接口。

## 1. 项目边界

### 1.1 CS 桌面端

CS（桌面端）使用 PySide6，主入口链路为：

```text
Framework.py -> main_interface.py -> 各功能 QWidget 页面
```

| 功能 | CS 页面文件 |
| --- | --- |
| 基于移动目标防御的异常检测 | `anomaly_detection.py` |
| 潜在安全威胁识别与自动分类 | `error_classification.py` |
| 多评估准则融合的风险学习分析 | `auto_score.py` |
| 风险场景动态匹配与适配方案生成算法 | `cdq_risk_matching.py` |
| 深度神经网络优化控制 | `desktop/risk_control/workspace.py` 的模型训练与在线控制视图 |
| 虚假数据注入下的自适应控制 | `desktop/risk_control/fdi_page.py` |
| 重放攻击下的自适应控制 | `desktop/risk_control/replay_page.py` |
| 多变量过程自愈控制 | `desktop/risk_control/healing_page.py` |
| 综合仿真展示 | `desktop/risk_control/workspace.py` 的综合页占位 |
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
| 风险管控页面、组件与样式 | `web_frontend/src/`、`styles.css` | React + TypeScript + CSS |
| 页面样式 | `web_frontend/src/styles.css` | CSS |
| 前端构建和开发服务器 | `web_frontend/package.json`、`web_frontend/vite.config.ts` | Vite |
| API、请求模型、任务和结果接口 | `web_backend/app.py` | FastAPI |
| 长任务协调 | `platform_core/task_manager.py`，`web_backend/task_manager.py` 提供兼容导入 | 任务、持久化与取消 |
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
- `platform_core/risk_control/` 实现共享 CSTR、各控制算法、会话和日志。四个模块独立运行，桌面直接调用服务，Web 通过 FastAPI 调用同一套服务。
- `input_data/`：CSV、Excel、MAT 等输入数据。当前模块依赖的文件名和相对路径是接口约定，不要随意重命名或移动。
- `output_figures/`：通用算法生成的图片和结果。
- `gridattackpkg_core_output/`：异常检测 Runtime 相关输出。
- `PLATFORM_CSTR_RUNTIME` 可指定 CSTR 任务、模型和各模块采样日志的位置。默认运行目录与回看方式见[操作手册](RISK_CONTROL_MANUAL.md#development)。`dnn_mpc/python_output/` 中已有历史结果应保留。
- `web_runtime/`：BS 本机任务运行期间的临时任务和结果目录。

输出目录中的文件通常是运行生成物。除非任务明确要求更新样例结果，否则不要把临时输出当作源码修改。

### 1.4 MATLAB Runtime 导出包

当前异常行为检测依赖 MATLAB 编译导出的 Python 包。历史 DNN-MPC 编译包可按其原工程单独安装。

- `build_python/`：`gridattackpkg`，供异常行为检测使用。
- 历史 `dnnmpcpkg` 的重新导出与独立调用见[旧 DNN-MPC 包说明](dnn_mpc/DNN_MPC_PYTHON_PACKAGE_SETUP.md)。当前四个 CSTR 模块使用 Python 共享核心。

不要直接修改这些目录中的生成文件，尤其是导出的 `__init__.py`、`.ctf` 和其他编译产物。需要改变底层 MATLAB 算法时，应修改 MATLAB 源工程并重新导出，再同步安装说明和版本信息。

## 2. 当前模块映射

页面名称、算法来源和 BS 接口的对应关系如下。修改模块时先确认自己修改的是正确架构。

| 模块 | CS 页面/入口 | BS 页面 | BS 后端服务/API | 依赖或数据 |
| --- | --- | --- | --- | --- |
| 基于移动目标防御的异常检测 | `anomaly_detection.py` | `App.tsx` 的 `AnomalyPage` | `web_backend/services/anomaly.py`；`/api/anomaly/*` | `gridattackpkg` |
| 潜在安全威胁识别与自动分类 | `error_classification.py` | `ClassificationPage` | `web_backend/services/classification.py`；`/api/classification/*` | `input_data/error_classification_*.csv` |
| 多评估准则融合的风险学习分析 | `auto_score.py` | `ScorePage` | `platform_core/scoring.py`、`web_backend/services/scoring.py`；`/api/score/*` | 公共评分逻辑 |
| 风险场景动态匹配与适配方案生成算法 | `cdq_risk_matching.py` | `CdqPageReplica` | `web_backend/services/cdq.py`；`/api/cdq/*` | `input_data/cdq_data.xlsx` |
| 深度神经网络优化控制 | `desktop/risk_control/workspace.py` | `CstrWorkspace.tsx` | `web_backend/app.py`，`/api/cstr/config`、`/api/cstr/training`、`/api/cstr/sessions` | `platform_core/risk_control/dnn_mpc/`、`service.py`、`session.py`、`assets/cstr/default_model.npz` |
| 虚假数据注入下的自适应控制 | `desktop/risk_control/fdi_page.py` | `FdiPage.tsx` | `web_backend/services/fdi.py`、`/api/cstr/fdi/*` | `platform_core/risk_control/fdi.py` 与同目录的控制和会话实现。三张趋势图与演示样本共用 `assets/cstr/fdi-demo.json` |
| 重放攻击下的自适应控制 | `desktop/risk_control/replay_page.py` | `ReplayPage.tsx` | `web_backend/services/replay.py`，`/api/cstr/replay/*` | `platform_core/risk_control/{replay,replay_session,replay_display}.py` |
| 多变量过程自愈控制 | `desktop/risk_control/healing_page.py` | `HealingPage.tsx` | `web_backend/services/healing.py`，`/api/cstr/healing/*` | `platform_core/risk_control/{healing,healing_display,healing_session}.py` |
| 综合仿真展示 | `desktop/risk_control/workspace.py` | `CstrWorkspace.tsx` | 共享导航，综合运行器待接入 | 各模块的独立结果见[操作手册](RISK_CONTROL_MANUAL.md#scope) |
| SDG-HAZOP | `sdg_hazop.py` | `SdgPage` | `web_backend/services/sdg.py`；`/api/sdg/*` | SDG 示例/分析数据 |
| 基于 GSPN-MC 模型的动态化 SIL 验证方法 | `sil_validation.py` | `SilPage` | `web_backend/services/sil.py`；`/api/sil/*` | GSPN-MC 仿真参数和结果 |

CS 菜单名称和 BS 菜单名称应保持业务含义一致。当前 CS 菜单定义在 `main_interface.py`，BS 菜单和 `ModuleId` 定义在 `web_frontend/src/App.tsx`，后端模块列表在 `web_backend/app.py` 的 `/api/modules` 接口中。

### 2.1 风险管控的调用边界

| 职责 | 文件 | 修改时核对 |
| --- | --- | --- |
| 模块入口 | `platform_core/risk_control/__init__.py`、`desktop/risk_control/workspace.py`、`web_frontend/src/CstrWorkspace.tsx`、`web_backend/app.py` | 菜单名称、可用状态及页面选择 |
| 数值参数输入 | Web 的 `PlatformUI.tsx`、`ErrorDialog.tsx` 与 Qt 的 `desktop/risk_control/controls.py` | 使用各字段已有的范围与精度，在确认无效输入时弹窗、恢复上次确认值，并将焦点交还原字段 |
| 对象与会话 | `platform_core/risk_control/cstr.py`、`thermal.py`、`service.py`、`comparison_service.py`、`journal.py` | 有效采样、运行标识、控制权、历史记录。算法状态由各分支持有 |
| 对照试验的 HTTP 生命周期 | `web_backend/services/comparison.py` | 自愈、FDI、重放复用创建、在线连接、控制权限和释放。各模块的适配器声明自己的指令集、快照、回看和导出能力 |
| CSTR 动态工艺图 | `assets/cstr/process.svg`、`platform_core/risk_control/presentation.py`、Qt 的 `ProcessDiagram` 和 Web 的 `CstrProcessDiagram` | 从同一运行、分支和采样点取真实状态与实际冷却流量。模块 4 在 `replay_display.diagram_sample` 映射读数，历史测温只显示在观测支路。暂停和历史回看时停下管线动画 |
| FDI 与重放曲线 | Web 的 `ComparisonTrend.tsx`、`trendGestures.ts`、`SampleChart.tsx` 和 Qt 的 `desktop/risk_control/comparison_trends.py`、`workspace.Trend` | Web 与自愈共用绘图区手势，Qt 使用在线趋势图类。同一运行的图共用时间窗口，纵轴逐图调整。按窗口内最后一个已发布采样点读取工艺图与详情，保留实际注入区间 |

## 3. 环境、数据与输出约定

依赖版本以 `requirements.txt` 与 `web_frontend/package-lock.json` 为准。桌面和 API 使用同一个 Python 环境。修改依赖时同时更新相应清单及安装说明。开发前端使用 Node.js 和 `npm ci`，启动已构建的本机服务时直接读取 `web_frontend/dist`。异常行为检测的编译包及 Runtime DLL 路径需单独核对。[安装与环境验证](INSTALL_AND_USAGE.md) · [异常检测 Runtime](GRIDATTACKPKG_NEW_ENV_SETUP.md)

### 3.1 数据和输出约定

以下文件是当前功能的输入或样例数据，缺失时不要用空文件静默替代，应提示用户补齐：

- `input_data/cdq_data.xlsx`
- `input_data/error_classification_train.csv`
- `input_data/error_classification_test.csv`
- `input_data/error_classification_easy_train.csv`
- `input_data/error_classification_easy_test.csv`
- `input_data/error_classification_hard_train.csv`
- `input_data/error_classification_hard_test.csv`

通用算法与 Runtime 结果位于 `output_figures/`、`gridattackpkg_core_output/`。在线 CSTR 使用第 1.3 节的运行目录。结果图片、JSON、进度文件的名称需要与现有前端和后端约定保持一致。

### 3.2 本机启动脚本的环境路径

`scripts/run_web_local.ps1` 依次从 `-PythonExe` 参数或 `PLATFORM_PYTHON` 环境变量、已激活的 Conda 环境、系统 PATH 选择 Python，并从 PATH 查找 `npm.cmd`。无需修改脚本中的机器路径；也可激活环境后手动启动：

```powershell
conda activate Platform
python -m uvicorn web_backend.app:app --host 127.0.0.1 --port 8000
```

手动启动前仍需先构建 `web_frontend`，并在项目根目录执行命令。

## 4. 启动、验证和排错

在仓库根目录运行与改动相应的检查。使用平台 Conda 环境，纯数值及接口回归可设置 `QT_QPA_PLATFORM=offscreen`。

```powershell
python -B -m unittest discover -s tests -p "test_*.py"
node --test tests/*.cjs
npm --prefix web_frontend run build
```

桌面从 `Framework.py` 进入实际菜单，Web 检查 `/api/health`、`/api/modules` 以及模块配置。会话改动要核对控制权限、重复指令、复位后的历史读取和释放。涉及编译包时再执行对应 Runtime 初始化检查。[本机启动与故障排查](INSTALL_AND_USAGE.md) · [CSTR 接口与记录](RISK_CONTROL_MANUAL.md#development)

运行 HTTP 任务的测试在临时目录创建 `TaskManager`。测试结束时先等待工作线程退出，再释放临时目录。`web_runtime/tasks/` 保存本机平台的真实任务历史。

<a id="new-module"></a>

## 5. 新增模块规范

### 5.1 新增 CS 模块

1. 沿用现有模块结构新建继承 `QWidget` 的页面；风险管控页面及共用 Qt 控件统一放在 `desktop/risk_control/`，其他模块保持现有目录。优先复用已有控件，不为单个模块另建前端目录。
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
5. 在 `web_backend/app.py` 注册 API。风险管控对照模块的请求模型和专用接口随各自的 `web_backend/services/` 适配器维护，通用生命周期由第 2.1 节的共享路由提供。有限后台任务复用公共 TaskManager。
6. 有限训练与 Runtime 任务交给 `TaskManager`，实时 CSTR 由会话运行器逐拍推进。HTTP 请求只提交命令或读取快照。
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
- 保留用户已有的运行结果、输入数据以及与本次工作无关的改动。
- 修改完成后至少运行与改动范围匹配的语法检查、前端构建和测试；涉及 Runtime 时再执行对应初始化检查。

## 7. 实现记录与历史版本

新增或调整功能时，在同次交付中核对操作手册、实际入口和接口。研究设计、控制假设、逐拍例子和验收结果写入课题目录对应的中英文实现记录。实验保留配置、输入来源、单位、评价窗口、对照结果和失效条件，并链接可复核的脚本或数据。历史版本的数字和截图继续按原版本引用。

旧版独立化学反应器、倒立摆及各自页面的设计可在 `ed33ff8` 查看。当时的实现、操作和验证范围见 `git show ed33ff8:RISK_CONTROL_MANUAL.md`。当前分支的菜单以第 2 节为准。
