import { scrollControlParameters } from './controlInteraction'
import { useEffect, useRef, useState } from 'react'
import { ApiError, api, post } from './api'
import { DnnPlotViewer } from './DnnPlotViewer'
import { ControlTaskStatus } from './ControlTaskStatus'
import { readSession, writeSession } from './taskSession'

type Config = Record<string, string | number>
type Parameter = { key: string; label: string; min?: number; max?: number; step?: number; integer?: boolean; choices?: Record<string, string> }
type Catalog = { scenarios: Record<string, string>; defaults: Record<string, Config>; parameters: Parameter[]; notice: string }
type Result = { module: 'self_healing'; config: { duration: number; scenario: string }; elapsed_seconds: number; output_names: string[]; metrics: Record<string, Record<string, { rmse: number[] }>> }
type Task = { id: string; status: string; error?: string; logs: string[]; cancellable?: boolean; cancel_requested?: boolean; progress?: { percent?: number; message?: string; image_revision?: number; outputs?: Record<string, string> }; result?: Result }
const groups = [
  ['outputs', '输出对比', 'NORMAL 为正常工况，FAULT 为故障下仅底层控制；SCSC、CCSC 为两种独立自愈对照。竖虚线表示故障起点。'],
  ['inputs', '输入设定值', '四组对照实际施加的输入设定值；修正在下一批生效。'],
] as const
// Preserve the running task when switching to another module and back.
type PlotName = typeof groups[number][0]
const session = readSession<{ config: Config; taskId: string | null; activePlot: PlotName }>('self-healing.session', { config: {}, taskId: null, activePlot: 'outputs' })
if (!groups.some(([name]) => name === session.activePlot)) session.activePlot = 'outputs'
if (!session.config || typeof session.config !== 'object' || Array.isArray(session.config)) session.config = {}

export function SelfHealingPage() {
  const [config, setConfig] = useState<Config>(session.config)
  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [defaultsError, setDefaultsError] = useState('')
  const [defaultsRevision, setDefaultsRevision] = useState(0)
  const [taskId, setTaskId] = useState(session.taskId)
  const [task, setTask] = useState<Task | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [error, setError] = useState('')
  const [pollingError, setPollingError] = useState('')
  const [imageErrors, setImageErrors] = useState<Record<string, boolean>>({})
  const [activePlot, setActivePlot] = useState(session.activePlot)
  useEffect(() => {
    Object.assign(session, { config, taskId, activePlot })
    writeSession('self-healing.session', session)
  }, [config, taskId, activePlot])
  const tabButtons = useRef<(HTMLButtonElement | null)[]>([])
  const scenarioButtons = useRef<(HTMLButtonElement | null)[]>([])
  const scenarios = Object.entries(catalog?.scenarios ?? {})
  const running = submitting || (!!taskId && (!task || ['queued', 'running'].includes(task.status)))
  const result = task?.status === 'succeeded' ? task.result : undefined
  const imageRevision = task?.progress?.image_revision ?? 0
  useEffect(() => { setImageErrors({}) }, [taskId, imageRevision])
  const percent = task?.status === 'succeeded' ? 100 : Math.min(99, Math.max(0, task?.progress?.percent ?? 0))

  useEffect(() => {
    let active = true
    void api<Catalog>('/self-healing/defaults').then(next => {
      if (!active) return
      setCatalog(next); setDefaultsError('')
      setConfig(previous => Object.keys(previous).length ? previous : next.defaults.water)
    }).catch(reason => { if (active) setDefaultsError(String(reason)) })
    return () => { active = false }
  }, [defaultsRevision])

  function selectPlot(index: number) {
    const name = groups[index][0]
    setActivePlot(name)
    session.activePlot = name
  }

  function selectScenario(key: string) {
    if (!catalog || running || key === config.scenario) return
    setConfig({ ...catalog.defaults[key] })
    setTaskId(null); setTask(null); setError(''); setPollingError(''); setImageErrors({})
    setCancelling(false)
  }

  useEffect(() => {
    if (!taskId) return
    let active = true
    let timer: number | undefined
    let retryDelay = 1000
    async function load() {
      try {
        const next = await api<Task>(`/tasks/${taskId}`)
        if (!active) return
        setTask(next)
        setPollingError('')
        retryDelay = 1000
        if (['queued', 'running'].includes(next.status)) timer = window.setTimeout(load, 500)
      } catch (reason) {
        if (!active) return
        if (reason instanceof ApiError && reason.status === 404) {
          setError('任务不存在，可能已被清理，请重新运行。')
          setPollingError('')
          setTask(null)
          setTaskId(null)
          setCancelling(false)
          session.taskId = null
          return
        }
        const detail = reason instanceof Error ? reason.message : String(reason)
        setPollingError(`状态查询暂时失败，${retryDelay / 1000} 秒后自动重试：${detail}`)
        timer = window.setTimeout(load, retryDelay)
        retryDelay = Math.min(retryDelay * 2, 5000)
      }
    }
    void load()
    return () => { active = false; window.clearTimeout(timer) }
  }, [taskId])

  async function start() {
    setCancelling(false)
    if (!catalog) return
    const parsed: Config = { scenario: config.scenario }
    for (const parameter of catalog.parameters) {
      const value = config[parameter.key]
      if (parameter.choices) parsed[parameter.key] = value
      else {
        const numeric = Number(value)
        if (value === undefined || !String(value).trim() || !Number.isFinite(numeric) ||
            numeric < parameter.min! || numeric > parameter.max! || (parameter.integer && !Number.isInteger(numeric))) {
          setError(`${parameter.label}的数值无效。`)
          return
        }
        parsed[parameter.key] = numeric
      }
    }
    setError(''); setPollingError(''); setTask(null); setTaskId(null); setImageErrors({}); setSubmitting(true)
    session.taskId = null
    try {
      const created = await post<{ task_id: string }>('/self-healing/tasks', { config: parsed })
      session.taskId = created.task_id
      writeSession('self-healing.session', { config: parsed, taskId: created.task_id, activePlot })
      setTaskId(created.task_id)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
    } finally { setSubmitting(false) }
  }

  async function cancel() {
    if (!taskId || !running || cancelling) return
    setCancelling(true)
    try {
      setTask(await post<Task>(`/tasks/${taskId}/cancel`, {}))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason))
      setCancelling(false)
    }
  }

  const status = task?.status === 'cancelled' ? '仿真已取消。' : task?.status === 'failed' ? '执行失败' : result
    ? `仿真完成 · ${catalog?.scenarios[result.config.scenario] ?? result.config.scenario} · ${result.config.duration} 模型时间 · 用时 ${result.elapsed_seconds.toFixed(2)} 秒`
    : task?.progress?.message ?? (running ? '等待执行' : '设置参数后开始运行')
  return <main onWheelCapture={scrollControlParameters} className="self-healing-page control-module" aria-label="多变量过程自愈控制仿真">
    <div className="self-healing-scenarios" role="tablist" aria-label="自愈控制场景">
      {scenarios.map(([key, label], index) => <button type="button" role="tab" key={key}
        id={`self-healing-scenario-${key}`} aria-controls="self-healing-scenario-panel"
        aria-selected={config.scenario === key} tabIndex={config.scenario === key ? 0 : -1} disabled={running}
        ref={node => { scenarioButtons.current[index] = node }} onClick={() => selectScenario(key)}
        onKeyDown={event => {
          let next: number
          if (event.key === 'ArrowRight') next = (index + 1) % scenarios.length
          else if (event.key === 'ArrowLeft') next = (index + scenarios.length - 1) % scenarios.length
          else if (event.key === 'Home') next = 0
          else if (event.key === 'End') next = scenarios.length - 1
          else return
          event.preventDefault()
          selectScenario(scenarios[next][0])
          scenarioButtons.current[next]?.focus()
        }}>{label}</button>)}
    </div>
    <div className="fdi-page" role="tabpanel" id="self-healing-scenario-panel"
      aria-labelledby={catalog ? `self-healing-scenario-${config.scenario}` : undefined}>
    <section className="panel fdi-controls">
      <h3>仿真设置</h3>
      {defaultsError && <div className="error" role="alert">{defaultsError}<button onClick={() => setDefaultsRevision(value => value + 1)}>重新加载配置</button></div>}
      {catalog && <>
        {catalog.parameters.map(parameter => <div key={parameter.key}>
          <label htmlFor={`self-healing-${parameter.key}`}>{parameter.label}</label>
          {parameter.choices ? <select id={`self-healing-${parameter.key}`} value={config[parameter.key] ?? ''} disabled={running}
            onChange={event => setConfig(previous => ({ ...previous, [parameter.key]: event.target.value }))}>
            {Object.entries(parameter.choices).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select> : <input id={`self-healing-${parameter.key}`} type="number" min={parameter.min} max={parameter.max} step={parameter.step}
            value={config[parameter.key] ?? ''} disabled={running} onChange={event => setConfig(previous => ({ ...previous, [parameter.key]: event.target.value }))}/>}
        </div>)}
      </>}
      <button type="button" className="primary" disabled={running || !catalog} onClick={() => void start()}>{running ? '仿真运行中…' : '开始自愈控制仿真'}</button>
      <button type="button" disabled={!running || !task?.cancellable || cancelling || task.cancel_requested} onClick={() => void cancel()}>{running && (cancelling || task?.cancel_requested) ? '正在取消…' : '取消仿真'}</button>
      <p className="muted">{catalog?.notice}</p>
    </section>
    <section className="fdi-results" aria-label="自愈控制仿真结果">
      {(error || pollingError || task?.error) && <div className="error" role="alert">{error || pollingError || task?.error}</div>}
      {result && <section className="panel self-healing-metrics">
        <h3>最后 20% 时域 RMSE</h3><p className="muted">各输出自身单位；FAULT 为仅底层控制。完整评价窗口和 MAPE、ATAE 见下载文件。</p>
        <div className="self-healing-table"><table><thead><tr><th>输出</th>{['normal', 'fault', 'scsc', 'ccsc'].map(method => <th key={method}>{method.toUpperCase()}</th>)}</tr></thead>
        <tbody>{result.output_names.map((label, j) => <tr key={label}><th>{label}</th>{['normal', 'fault', 'scsc', 'ccsc'].map(method =>
          <td key={method}>{result.metrics[method].tail_20_percent.rmse[j].toPrecision(4)}</td>)}</tr>)}</tbody></table></div>
        <div className="self-healing-downloads">{['normal.csv', 'fault.csv', 'scsc.csv', 'ccsc.csv', 'metrics.csv', 'trajectories.npz', 'summary.json', 'result.json'].map(name =>
          <a key={name} href={`/api/self-healing/tasks/${taskId}/files/${name}`} download>{name}</a>)}</div>
      </section>}
      <section className="panel fdi-output" aria-label="结果图示">
        <h3>结果图示</h3>
        <ControlTaskStatus state={running ? 'running' : task?.status === 'succeeded' ? 'succeeded' : task?.status === 'failed' ? 'failed' : task?.status === 'cancelled' ? 'cancelled' : 'idle'}
          message={status} percent={percent} label="自愈控制仿真进度"/>
        <div className="fdi-tabs" role="tablist" aria-label="选择输出图像">{groups.map(([name, title], index) =>
          <button type="button" role="tab" key={name} id={`self-healing-tab-${name}`} aria-controls={`self-healing-panel-${name}`}
            aria-selected={activePlot === name} tabIndex={activePlot === name ? 0 : -1}
            ref={node => { tabButtons.current[index] = node }} onClick={() => selectPlot(index)}
            onKeyDown={event => {
              let next: number
              if (event.key === 'ArrowRight') next = (index + 1) % groups.length
              else if (event.key === 'ArrowLeft') next = (index + groups.length - 1) % groups.length
              else if (event.key === 'Home') next = 0
              else if (event.key === 'End') next = groups.length - 1
              else return
              event.preventDefault()
              selectPlot(next)
              tabButtons.current[next]?.focus()
            }}>{title}</button>)}</div>
        {groups.map(([name, title, description]) => <section className="fdi-plot-panel" key={name}
          role="tabpanel" id={`self-healing-panel-${name}`} aria-labelledby={`self-healing-tab-${name}`} hidden={activePlot !== name}>
        <header><div><h3>{title}</h3><p>{description}</p></div>{result && <div className="fdi-downloads">{['svg', 'png'].map(format => <a key={format} href={`/api/self-healing/tasks/${taskId}/files/${name}.${format}`} download>下载 {format.toUpperCase()}</a>)}</div>}</header>
        <DnnPlotViewer key={`${taskId}:${name}`} src={taskId && (result || task?.progress?.outputs?.[name]) && !imageErrors[name] ? `/api/self-healing/tasks/${taskId}/images/${name}.svg?v=${imageRevision}` : undefined}
          alt={`${title}曲线`} placeholder={imageErrors[name] ? '图片加载失败，请重试或重新运行。' : running ? '计算中，等待曲线更新…' : '运行仿真后显示结果图'}
          onError={() => setImageErrors(previous => ({ ...previous, [name]: true }))}/>
        {imageErrors[name] && <button type="button" onClick={() => setImageErrors(previous => ({ ...previous, [name]: false }))}>重新加载图片</button>}
        </section>)}
      </section>
      {!!task?.logs?.length && <details className="fdi-logs"><summary>运行日志</summary><pre>{task.logs.join('\n')}</pre></details>}
    </section>
    </div>
  </main>
}
