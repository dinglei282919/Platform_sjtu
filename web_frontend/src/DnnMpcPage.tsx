import { useEffect, useRef, useState, type FormEvent } from 'react'
import { ApiError, api, post } from './api'
import { DnnPlotViewer } from './DnnPlotViewer'
import { ControlTaskStatus } from './ControlTaskStatus'
import { readSession, writeSession } from './taskSession'

type Mode = 'training' | 'mpc'
type Form = { output_dir: string; model_path: string; dataset_path?: string; sample_count?: number; epochs?: number; lr?: number; hidden_layers?: string; sim_time?: number; prediction_horizon?: number }
type Payload = { json_path: string; model_path: string; image_revision: number; result: { model_sha256: string; metrics: Record<string, number>; warning?: string; elapsed_seconds: number } }
type Task = { id: string; status: string; error?: string; logs: string[]; result?: Payload; progress?: { percent?: number; message?: string; revision?: number; image_revision?: number | null; outputs?: Record<string, string> } }
type Session = { form: Form; taskId: string | null; manualModel: boolean }
const mpcPlots = [
  ['dnn_mpc', '总览', 'simulation_figure', '四个状态偏差与控制输入的整体响应'],
  ['mpc_states', '状态响应', 'state_figure', '位置、速度、角度与角速度相对参考值的偏差'],
  ['mpc_control', '控制输入', 'control_figure', '控制力随仿真时间的变化'],
] as const
let lastMpcPlot: typeof mpcPlots[number][0] = 'dnn_mpc'
const initial: Form = { output_dir: '', model_path: '' }
const sessions: Partial<Record<Mode, Session>> = {
  training: readSession<Session>('training.session', { form: initial, taskId: null, manualModel: false }),
  mpc: readSession<Session>('mpc.session', { form: initial, taskId: null, manualModel: false }),
}
for (const mode of ['training', 'mpc'] as const) {
  if (!sessions[mode]?.form.output_dir || !sessions[mode]?.form.model_path) delete sessions[mode]
}
const absolute = (value: string) => /^[A-Za-z]:[\\/]/.test(value.trim()) || /^\\\\/.test(value.trim()) || value.trim().startsWith('/')

function validate(mode: Mode, form: Form): string | null {
  if (!absolute(form.output_dir) || !absolute(form.model_path)) return '输出目录和模型文件必须使用本机绝对路径。'
  if (!form.model_path.trim().toLowerCase().endsWith('.npz')) return '模型必须是 Python 训练导出的 .npz 文件。'
  if (mode === 'training') {
    if (!Number.isInteger(form.sample_count) || form.sample_count! < 100 || form.sample_count! > 100000) return '生成样本数必须为 100～100000 的整数。'
    if (!Number.isInteger(form.epochs) || form.epochs! < 1 || form.epochs! > 5000) return '训练轮数必须为 1～5000 的整数。'
    if (!Number.isFinite(form.lr) || form.lr! < 0.000001 || form.lr! > 0.1) return '学习率必须在 0.000001～0.1 之间。'
    if (!/^\d+\s*,\s*\d+$/.test(form.hidden_layers?.trim() ?? '') || form.hidden_layers!.split(',').some(x => Number(x) < 1 || Number(x) > 512)) return '必须设置两个隐藏层，每层 1～512 个节点，例如 64,64。'
    if (form.dataset_path?.trim() && (!absolute(form.dataset_path) || !form.dataset_path.trim().toLowerCase().endsWith('.mat'))) return '训练数据必须使用本机绝对路径和 .mat 扩展名。'
  } else {
    if (!Number.isFinite(form.sim_time) || form.sim_time! < 0.2 || form.sim_time! > 20) return '仿真时长必须在 0.2～20 秒之间。'
    if (!Number.isInteger(form.prediction_horizon) || form.prediction_horizon! < 1 || form.prediction_horizon! > 60) return '预测步长必须为 1～60 的整数。'
  }
  return null
}

function DnnMpcStage({ mode }: { mode: Mode }) {
  const [form, setForm] = useState<Form>(sessions[mode]?.form ?? initial)
  const [taskId, setTaskId] = useState<string | null>(sessions[mode]?.taskId ?? null)
  const [task, setTask] = useState<Task | null>(null)
  const [error, setError] = useState('')
  const [ready, setReady] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [selecting, setSelecting] = useState(false)
  const [imageError, setImageError] = useState(false)
  const [plotErrors, setPlotErrors] = useState<Record<string, boolean>>({})
  const [activePlot, setActivePlot] = useState(lastMpcPlot)
  const tabButtons = useRef<(HTMLButtonElement | null)[]>([])
  const training = mode === 'training'
  const running = submitting || (!!taskId && (!task || ['queued', 'running'].includes(task.status)))
  const payload = task?.result
  const metrics = payload?.result.metrics
  const percent = task?.status === 'succeeded' ? 100 : Math.max(0, Math.min(100, task?.progress?.percent ?? 0))
  const imageName = training ? 'training_rmse.svg' : 'dnn_mpc.svg'
  const hasImage = !!payload || !!task?.progress?.outputs?.[training ? 'training_figure' : 'simulation_figure']
  const imageUrl = `/api/${mode}/tasks/${taskId}/images/${imageName}?v=${task?.progress?.image_revision ?? task?.progress?.revision ?? payload?.image_revision ?? 0}`

  useEffect(() => {
    if (sessions[mode]) writeSession(`${mode}.session`, sessions[mode])
  }, [form, taskId, mode])

  useEffect(() => {
    let active = true
    void api<Form>(`/${mode}/defaults`).then(defaults => {
      if (!active) return
      const saved = sessions[mode]
      const next = saved ? { ...defaults, ...saved.form } : defaults
      if (!training && !saved?.manualModel) next.model_path = defaults.model_path
      setForm(next)
      sessions[mode] = { form: next, taskId: saved?.taskId ?? null, manualModel: saved?.manualModel ?? false }
      setReady(true)
    }).catch(e => { if (active) setError(String(e.message ?? e)) })
    return () => { active = false }
  }, [mode, training])

  useEffect(() => {
    if (!taskId) { setTask(null); return }
    let active = true
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      try {
        const value = await api<Task>(`/tasks/${taskId}`)
        if (!active) return
        setTask(value)
        setError('')
        if (['succeeded', 'failed', 'cancelled'].includes(value.status)) return
      } catch (e) {
        if (!active) return
        if (e instanceof ApiError && e.status === 404) {
          setTask(null)
          setTaskId(null)
          if (sessions[mode]?.taskId === taskId) sessions[mode]!.taskId = null
          setError('任务不存在，可能服务已重启或任务已被清理，请重新运行。')
          return
        }
        setError(`任务状态读取失败，将自动重试：${e instanceof Error ? e.message : String(e)}`)
      }
      if (active) timer = setTimeout(poll, training ? 1000 : 500)
    }
    void poll()
    return () => { active = false; clearTimeout(timer) }
  }, [taskId, training, mode])
  useEffect(() => { setImageError(false); setPlotErrors({}) }, [imageUrl])

  function selectPlot(index: number) {
    lastMpcPlot = mpcPlots[index][0]
    setActivePlot(lastMpcPlot)
  }

  useEffect(() => {
    if (training || task?.status !== 'succeeded' || !payload) return
    let active = true
    const verify = async () => {
      try {
        const current = await post<{ sha256: string | null }>('/mpc/model-info', { model_path: form.model_path })
        if (active && current.sha256 !== payload.result.model_sha256) {
          setTaskId(null); setTask(null)
          if (sessions.mpc) sessions.mpc.taskId = null
          setError('模型文件已改变，请重新运行仿真。')
        }
      } catch (e) { if (active) setError(`模型检查失败：${e instanceof Error ? e.message : String(e)}`) }
    }
    void verify()
    const timer = setInterval(() => void verify(), 2000)
    return () => { active = false; clearInterval(timer) }
  }, [training, task?.status, payload, form.model_path])

  function update(key: keyof Form, value: string | number) {
    const next = { ...form, [key]: value }
    const manualModel = (sessions[mode]?.manualModel ?? false) || key === 'model_path'
    if (training && key === 'output_dir' && !manualModel) next.model_path = `${String(value).replace(/[\\/]+$/, '')}/trained_model.npz`
    setForm(next)
    // Changing a selected model invalidates the old simulation and its exports.
    const nextId = key === 'model_path' && !training ? null : taskId
    if (nextId !== taskId) { setTaskId(nextId); setTask(null) }
    sessions[mode] = { form: next, taskId: nextId, manualModel }
  }

  async function choose(key: 'output_dir' | 'model_path' | 'dataset_path') {
    setSelecting(true)
    try {
      const kind = key === 'output_dir' ? 'directory' : key === 'dataset_path' ? 'open-mat' : training ? 'save-npz' : 'open-npz'
      const { path } = await post<{ path: string }>('/local-paths/select', { kind, initial_path: form[key] ?? '' })
      if (path) update(key, path)
    } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
    finally { setSelecting(false) }
  }

  async function useLatest() {
    try {
      const defaults = await api<Form>('/mpc/defaults')
      const next = { ...form, model_path: defaults.model_path }
      setForm(next); setTaskId(null); setTask(null)
      sessions.mpc = { form: next, taskId: null, manualModel: false }
    } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
  }

  async function submit(event: FormEvent) {
    event.preventDefault()
    const issue = validate(mode, form)
    if (issue) { setError(issue); return }
    setError(''); setSubmitting(true); setTask(null)
    try {
      const created = await post<{ task_id: string }>(`/${mode}/tasks`, form)
      setTaskId(created.task_id)
      sessions[mode] = { form, taskId: created.task_id, manualModel: sessions[mode]?.manualModel ?? false }
    } catch (e) { setTaskId(null); setError(e instanceof Error ? e.message : String(e)) }
    finally { setSubmitting(false) }
  }

  const pathField = (label: string, key: 'output_dir' | 'model_path' | 'dataset_path') => <label className="training-path-field">
    <span>{label}</span><div className="training-path-control"><input aria-label={label} value={form[key] ?? ''} placeholder={key === 'dataset_path' ? '留空则按原状态方程生成数据集' : undefined} onChange={e => update(key, e.target.value)} disabled={running || !ready}/><button type="button" onClick={() => void choose(key)} disabled={running || selecting || !ready}>选择</button>{key === 'dataset_path' && <button type="button" onClick={() => update(key, '')} disabled={running || selecting || !ready}>清空</button>}</div>
  </label>
  const downloads = training ? [['SVG', 'training_rmse.svg'], ['PDF', 'training_rmse.pdf'], ['模型 NPZ', 'trained_model.npz'], ['训练记录', 'training_history.csv'], ['数据集 MAT', 'training_dataset.mat']] : [['当前图 SVG', `${activePlot}.svg`], ['当前图 PDF', `${activePlot}.pdf`], ['结果 ZIP', 'dnn_mpc_results.zip']]
  return <main className="training-page dnn-python-page control-module" aria-label={training ? '控制模型训练评估' : '优化控制仿真验证'}>
    <div className="training-layout">
      <section className="training-panel training-config-panel">
        <h3>{training ? '训练设置' : '仿真设置'}</h3>
        <form className="training-config-scroll" onSubmit={e => void submit(e)}>
          <section className="training-subpanel"><h4>路径设置</h4>
            {pathField('输出目录', 'output_dir')}{pathField('模型文件', 'model_path')}
            {!training && <button type="button" onClick={() => void useLatest()} disabled={running || !ready}>使用最近训练模型</button>}
          </section>
          <section className="training-subpanel"><h4>{training ? '训练参数' : '仿真参数'}</h4>
            {training ? <>
              <label className="training-number-field">生成样本数<span className="dnn-parameter-control">
                <input aria-label="生成样本数" aria-describedby="dnn-samples-range" type="number" min={100} max={100000} step={1} value={form.sample_count ?? 1000} onChange={e => update('sample_count', Number(e.target.value))} disabled={running || !!form.dataset_path?.trim()} required/>
                <small id="dnn-samples-range">范围：100～100000（整数）；仅自动生成时生效。</small>
              </span></label>
              <label className="training-number-field">训练轮数<span className="dnn-parameter-control">
                <input aria-label="训练轮数" aria-describedby="dnn-epochs-range" type="number" min={1} max={5000} step={1} value={form.epochs ?? 200} onChange={e => update('epochs', Number(e.target.value))} disabled={running} required/>
                <small id="dnn-epochs-range">范围：1～5000（整数）。</small>
              </span></label>
              <label className="training-number-field">学习率<span className="dnn-parameter-control">
                <input aria-label="学习率" aria-describedby="dnn-lr-range" type="number" min={0.000001} max={0.1} step="any" value={form.lr ?? 0.001} onChange={e => update('lr', Number(e.target.value))} disabled={running} required/>
                <small id="dnn-lr-range">范围：0.000001～0.1。</small>
              </span></label>
              <label className="training-number-field">两个隐藏层<span className="dnn-parameter-control">
                <input aria-label="两个隐藏层" aria-describedby="dnn-hidden-range" value={form.hidden_layers ?? '64,64'} onChange={e => update('hidden_layers', e.target.value)} disabled={running} required/>
                <small id="dnn-hidden-range">每层 1～512 个节点；用英文逗号分隔，例如 64,64。</small>
              </span></label>
              {pathField('外部数据集', 'dataset_path')}
              <p>{form.dataset_path?.trim() ? '使用完整 MAT 数据；清空路径可启用自动生成。' : '按原状态方程生成仿真数据后训练；dt=0.02 s，随机种子 42。'}</p>
              <p>tanh / Adam。训练、验证、测试为 70% / 15% / 15%。</p>
            </> : <>
              <label className="training-number-field">仿真时长 (s)<span className="dnn-parameter-control">
                <input aria-label="仿真时长" aria-describedby="dnn-duration-range" type="number" min={0.2} max={20} step="any" value={form.sim_time ?? 5} onChange={e => update('sim_time', Number(e.target.value))} disabled={running} required/>
                <small id="dnn-duration-range">范围：0.2～20 s。</small>
              </span></label>
              <label className="training-number-field">预测步长<span className="dnn-parameter-control">
                <input aria-label="预测步长" aria-describedby="dnn-horizon-range" type="number" min={1} max={60} step={1} value={form.prediction_horizon ?? 10} onChange={e => update('prediction_horizon', Number(e.target.value))} disabled={running} required/>
                <small id="dnn-horizon-range">范围：1～60（整数）；每步 0.02 s。</small>
              </span></label>
              <p>Δt = 0.02 s；Q = diag(300, 1, 300, 1)，R = 0.01；P 由模型计算；控制输入 ±10 N。</p>
              <p>状态由同一预测网络更新，属于模型内闭环验证。</p>
            </>}
            <div className="training-run-action"><button type="submit" disabled={running || !ready || selecting}>{running ? '执行中…' : training ? '运行训练模块' : '运行 MPC 仿真'}</button></div>
          </section>
          <section className="training-subpanel"><h4>结果文件</h4><div className="training-result-path">{payload?.json_path ?? '运行后显示本次结果路径。'}</div></section>
        </form>
      </section>
      <section className="training-results" aria-label={training ? '训练结果' : 'MPC 仿真结果'}>
        <section className="training-panel training-result-panel">
          <h3>结果图示</h3>
          <ControlTaskStatus state={running ? 'running' : task?.status === 'succeeded' ? 'succeeded' : task?.status === 'failed' ? 'failed' : task?.status === 'cancelled' ? 'cancelled' : 'idle'}
            message={task?.status === 'failed' ? '执行失败，请查看错误详情' : task?.status === 'cancelled' ? '任务已取消，可重新运行' : task?.status === 'succeeded' ? `执行完成 · 用时 ${payload?.result.elapsed_seconds.toFixed(2)} 秒` : task?.progress?.message ?? (running ? '等待执行' : '设置参数后开始运行')}
            percent={percent} label={training ? '训练进度' : 'MPC 仿真进度'}/>
          {training ? <>
            <header className="dnn-output-heading"><h4>训练 / 验证 RMSE</h4><p className="muted">训练集与验证集的标准化误差随轮次变化</p></header>
            <DnnPlotViewer key={`${mode}:${taskId}`} src={hasImage && !imageError ? imageUrl : undefined}
              alt="训练和验证 RMSE 曲线"
              placeholder={running ? '计算中，等待图像更新…' : '运行后显示本次计算结果'} onError={() => setImageError(true)}/>
            </>
            : <div className="mpc-output-plots">
              <div className="mpc-plot-tabs" role="tablist" aria-label="MPC 输出图像">{mpcPlots.map(([name, title], index) =>
                <button type="button" key={name} role="tab" id={`mpc-tab-${name}`} aria-controls={`mpc-panel-${name}`}
                  aria-selected={activePlot === name} tabIndex={activePlot === name ? 0 : -1}
                  ref={node => { tabButtons.current[index] = node }} onClick={() => selectPlot(index)}
                  onKeyDown={event => {
                    let next: number
                    if (event.key === 'ArrowRight') next = (index + 1) % mpcPlots.length
                    else if (event.key === 'ArrowLeft') next = (index + mpcPlots.length - 1) % mpcPlots.length
                    else if (event.key === 'Home') next = 0
                    else if (event.key === 'End') next = mpcPlots.length - 1
                    else return
                    event.preventDefault(); selectPlot(next); tabButtons.current[next]?.focus()
                  }}>{title}</button>)}</div>
              {mpcPlots.map(([name, title, outputKey, description]) => <section key={name} className="mpc-plot-panel" role="tabpanel"
                id={`mpc-panel-${name}`} aria-labelledby={`mpc-tab-${name}`} hidden={activePlot !== name}>
                <header className="dnn-output-heading"><h4>{title}</h4><p className="muted">{description}</p></header>
                <DnnPlotViewer key={`${taskId}:${name}`}
                  src={(!!payload || !!task?.progress?.outputs?.[outputKey]) && !plotErrors[name] ? imageUrl.replace(imageName, `${name}.svg`) : undefined}
                  alt={`${title}曲线`} placeholder={plotErrors[name] ? '图片加载失败，请重试。' : running ? '计算中，等待图像更新…' : '运行后显示本次计算结果'}
                  onError={() => setPlotErrors(previous => ({ ...previous, [name]: true }))}/>
                {plotErrors[name] && <button type="button" onClick={() => setPlotErrors(previous => ({ ...previous, [name]: false }))}>重新加载图片</button>}
              </section>)}
            </div>}
          <div className="dnn-result-summary">
            {metrics && <p className="dnn-metrics">{training ? `测试 RMSE（原始量纲综合）：${metrics.test_rmse.toFixed(6)}；最佳轮次：${metrics.best_epoch}` : `跟踪 RMSE：${metrics.tracking_rmse.toFixed(4)}；最大 |u|：${metrics.max_abs_control.toFixed(2)} N；输入越界：${metrics.input_violations}；累计代价：${metrics.total_cost.toFixed(2)}`}</p>}
            <p className="dnn-caption">{training ? '曲线为标准化 RMSE；参数修改后需重新训练。' : 'x₁：位置(m)，x₂：速度(m/s)，x₃=θ−π(rad)，x₄：角速度(rad/s)，u：控制力(N)。图示相对参考值的偏差；综合 RMSE 混合量纲，累计代价不代表实际能耗。'}</p>
            {payload?.result.warning && <p role="alert">{payload.result.warning}</p>}
            {task?.status === 'succeeded' && <div className="dnn-downloads">{downloads.map(([label, name]) => <a key={name} href={`/api/dnn/${mode}/tasks/${taskId}/files/${name}`} download>下载 {label}</a>)}</div>}
          </div>
        </section>
        {task?.logs?.length ? <details className="dnn-logs"><summary>运行日志</summary><pre>{task.logs.join('\n')}</pre></details> : null}
      </section>
    </div>
    {(error || task?.error) && <div role="alert" className="error-box">{error || task?.error}</div>}
  </main>
}


export function DnnMpcPage() {
  const [mode, setMode] = useState<Mode>(() => readSession<Mode>('dnn-mpc.stage', 'training') === 'mpc' ? 'mpc' : 'training')
  function selectMode(next: Mode) {
    setMode(next)
    writeSession('dnn-mpc.stage', next)
  }
  return <section className="dnn-workflow" aria-label="DNN/MPC 控制训练与优化">
    <div className="mpc-plot-tabs dnn-stage-tabs" role="tablist" aria-label="DNN/MPC 工作流程">
      {(['training', 'mpc'] as const).map((stage, index) => <button type="button" key={stage}
        role="tab" id={`dnn-stage-${stage}`} aria-controls="dnn-stage-panel" aria-selected={mode === stage}
        tabIndex={mode === stage ? 0 : -1} onClick={() => selectMode(stage)} onKeyDown={event => {
          if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return
          event.preventDefault()
          const next = event.key === 'Home' ? 'training' : event.key === 'End' ? 'mpc' : stage === 'training' ? 'mpc' : 'training'
          selectMode(next)
          document.getElementById(`dnn-stage-${next}`)?.focus()
        }}>{index + 1}. {stage === 'training' ? '模型训练' : 'MPC 仿真'}</button>)}
    </div>
    <p className="dnn-workflow-hint">训练模型后进入 MPC 仿真，也可直接选择已有模型。切换阶段保留参数与任务记录。</p>
    <div id="dnn-stage-panel" role="tabpanel" aria-labelledby={`dnn-stage-${mode}`}>
      <DnnMpcStage key={mode} mode={mode}/>
    </div>
  </section>
}
