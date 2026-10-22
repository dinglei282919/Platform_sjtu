# 化学反应器模块

分支：feat/chemical-reactor。起点：2026-10-22 成功 fetch 后的 origin/main（0728dde）。本分支不包含 self-healing 模块，原工作目录保持独立。

## 启动与检查

在本 worktree 根目录激活项目环境后运行：

~~~powershell
python -B Framework.py
~~~

进入“风险管控优化决策 → 化学反应器自适应控制仿真”。

Web 版使用现有启动脚本。与另一个 worktree 同时运行时，指定不同端口，避免启动脚本复用或重启另一个分支的服务：

~~~powershell
$env:PLATFORM_WEB_PORT = '8012'
./scripts/run_web_local.ps1 -PythonExe 'H:/conda/envs/project_sjtu/python.exe' -Rebuild
~~~

解释器路径只是本机示例，也可先激活自己的环境，再省略 -PythonExe。浏览器地址为 http://127.0.0.1:8012。

1. 框图使用原 diagram 的 GIF/PNG；可暂停动画。
2. 默认参数点击“运行仿真”，两端都在后台计算，可取消；计算时暂停 GIF。
3. 完成后查看六组曲线，或使用框图下方滑块和回放按钮查看 t、σ、x1、x2、u。
4. 桌面可导出或载入 CSV/NPZ/JSON/PNG 结果。Web 提供下载，任务 ID 在当前浏览器标签页中保存，切换模块或刷新后返回本模块即可恢复任务。
5. 参数修改后再次运行会清空当前显示，历史 Web 结果保留在独立任务目录。

输出间隔控制采样，不是求解器步长。核心保留 20 分钟上限和原 BDF 求解器容差。方程求值使用 Numba 编译内核，第一次运行包含编译开销，后续复用磁盘编译缓存；仍实时计算本次轨迹。

## 文件与依赖

- platform_core/reactor/core.py：从 I 盘原样迁入的 37 维模型、参数校验与批量 Jacobian。
- platform_core/reactor/reporting.py：同一套六图绘制及 CSV/NPZ/JSON/PNG 导出。
- reactor_page.py：QWidget 页面、QThread、取消、框图和回放。
- web_backend/services/reactor.py：TaskManager 服务适配、进度和结果。
- web_backend/app.py：参数请求、任务、受控下载与框图 API。
- web_frontend/src/ReactorPage.tsx：Web 界面和任务恢复。
- assets/reactor：共享原始框图资源。
- tests/fixtures/reactor_jacobian.npz：从 I 盘默认结果提取的 t=0、0.2、1.5、2.3 四个状态，仅用于 Jacobian 回归，不是运行缓存。

无需 MATLAB Runtime；使用 NumPy、SciPy、Matplotlib、PySide6，并新增 Numba 0.68.0（已写入 requirements.txt）。现有环境需运行 `python -m pip install numba==0.68.0`。页面路径按当前 main 的结构放置；后续合入已重构目录的主线时，按新目录移动入口并调整引用。

## 验证

快速回归：
~~~powershell
python -B -m unittest discover -s tests -p 'test_*.py'
cd web_frontend
npm run build
~~~

完整桌面平台回归（约数分钟，不由单元测试自动执行）：
~~~powershell
python -B scripts/check_reactor_integration.py
~~~

该脚本通过 MainWindow 页面按钮启动完整 3 秒仿真，检查模块切换、线程结束、动画恢复、结果与导出；输出在 web_runtime/reactor_validation。移植验证记录见 migration_validation.json。

## 数值边界

框图为固定工艺示意，液位、流速和双路进料不是数值状态驱动；不得把模式切换理解为进料开关。状态和控制输入未推定实际物理单位。现有实现尚未完成 MATLAB 逐点轨迹比对，采样控制峰值尚未通过严格容差收敛检查。原模型映射见 model_mapping.md，I 盘阶段历史验证见 validation.md；历史记录中的相对路径指原研究目录。
2026-10-22 本次验证：平台 109 项 Python 测试通过；曲线布局调整后 21 项 reactor 测试再次通过；前端生产构建通过。浏览器按钮启动的默认完整仿真为 202.969 秒，平台 MainWindow 按钮/QThread 路径为 208.235 秒（离屏）；两端输出与 I 盘 performance_fix 基准的全部轨迹数组完全相同。真实 Windows 窗口已检查平台启动、菜单和框图；浏览器已检查刷新恢复、曲线、回放、取消及取消后重试。

## 功能与结构复核（2026-10-22）

| 检查项 | I 盘独立模块 | 平台桌面 | 平台 Web |
| --- | --- | --- | --- |
| 默认模型、容差、37 维状态、批量 Jacobian | 基准 | 同一共享核心 | 同一共享核心 |
| 八项可编辑参数、范围和跨参数校验 | 支持 | 支持 | 支持 |
| 后台仿真、进度、取消、失败重试 | QThread | QThread | TaskManager |
| GIF/PNG 框图、计算时暂停动画 | 支持 | 支持 | 支持 |
| 六组曲线、缩放、数据滑块与时间回放 | 支持 | 支持 | 支持 |
| CSV、NPZ、JSON、PNG 导出 | 本地导出 | 本地导出 | 任务文件下载 |
| 读取已有结果 | 读取本地目录 | 读取本地目录 | 恢复本标签页已保存的服务器任务；不提供本地 NPZ 上传 |
| 打开时自动加载 I 盘示例 | 有附带示例时加载 | 不依赖研究目录或附带预计算结果 | 不依赖研究目录或附带预计算结果 |
| 命令行研究实验、严格容差对照 | 研究脚本 | 保留在 I 盘，不作为平台界面功能 | 同左 |

结构遵循本分支 main 起点的 AGENTS.md：桌面 QWidget 在根目录、数学与报告在 platform_core/reactor、Web 适配器在 web_backend/services、路由在 app.py、页面在既有 web_frontend/src，复用 TaskManager、ControlTaskStatus、DnnPlotViewer 和 taskSession。不提前搬入 self-healing 分支尚未合并的目录重构。

运行输出统一为 dnn_mpc/python_output/reactor；Web 每次计算在 runs/唯一编号 下保存，桌面导出对话框默认定位此处（目录尚未生成时定位项目根目录），仍允许用户选择导出位置。web_runtime/tasks 仅保留平台任务元数据。历史验证用的 web_runtime/reactor 是旧适配器目录，复核后删除测试生成物。

## 界面对齐与计算优化（2026-10-22）

图表交互补齐：桌面曲线由 Matplotlib 原生工具栏切换为公共 `DnnMpcPlotView`，通过内存 SVG 显示六图，支持滚轮缩放、左键拖动、双击适应窗口，以及缩小/放大/适应窗口/适应宽度按钮。Web 沿用公共 `DnnPlotViewer`，两端图表宽高比统一为 13:7。新计算清空旧图并禁用工具，结果生成后恢复；CSV/NPZ/JSON/PNG 导出不变。公共查看器新增内存 SVG 加载入口，原文件加载接口保持兼容。

交互验证：114 项 Python 测试和前端构建通过，包含 reactor 实际 Qt 鼠标事件及公共查看器回归；浏览器实测滚轮 100% → 194%、拖动位置变化、双击恢复 100%、适应宽度与窗口。桌面显示检查使用前次已验证的完整结果，无需为显示交互重新求解。

Web 复用 `control-module` 样式，桌面复用 `control_page_style`、`ControlScrollArea`、`ControlTaskStatus`。参数栏、结果面板、按钮、字号和进度区与 FDI/replay 页面一致，去掉页面内重复标题，保留窄窗口纵向布局及所有原有参数、回放、取消和导出功能。

默认 BDF 完整 3 秒计算从 220.031 秒降至 95.343 秒（约 2.31 倍）；两次测试均为实际求解，环境负载会影响时间。新增 `platform_core/reactor/kernel.py`，只编译高频方程求值，不开启 fastmath、不更改模型、求解器和容差、不裁剪 Nussbaum 函数。原 NumPy `derivatives` 和 Jacobian 保留；研究对照可用 `simulate(compiled=False)`。

编译计算与原版不是逐位相等：默认整段 x1/x2 最大绝对差为 8.42e-10 / 3.00e-6，观测状态最大差为 3.16e-6，控制输入最大绝对差约 0.292，采样控制峰值均为 131.4456333。37 维状态相对 L2 差约 0.001414；高增益阶段的内部变量差异详见 `performance_validation.json`。这项优化未解决既有的严格容差收敛和 MATLAB 逐点复现边界。LSODA/Radau 在原容差下的采样峰值有明显差异，因此未替换默认 BDF。

回归：113 项 Python 测试通过，包含两种模式、多个增益与切换时刻的编译/参考方程对照，以及两套初始条件下的严格容差短轨迹对照；前端生产构建、pip check、语法检查和 diff check 通过。

完整平台按钮验收：桌面 MainWindow 路径计算 112.281 秒，浏览器路径约 116.3 秒（与其他验收并行时的实测）。两端终态指标一致，已检查完整曲线、数据回放、模块切换、刷新后返回模块恢复结果，以及 720px 窄窗口无页面横向溢出。验收截图不纳入文档目录；数值验证记录保留为历史。
