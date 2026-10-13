import { useEffect, useRef, useState } from 'react'
import { ApiError, api, post } from './api'
import { DnnPlotViewer } from './DnnPlotViewer'
import { ControlTaskStatus } from './ControlTaskStatus'
import { readSession, writeSession } from './taskSession'

type Result = { module: 'replay'; config: { sim_time: number }; elapsed_seconds: number; tracking_rmse: number; disturbance_rmse: number }
type Task = { id: string; status: string; error?: string; logs: string[]; cancellable?: boolean; cancel_requested?: boolean; progress?: { percent?: number; message?: string; image_revision?: number; outputs?: Record<string, string> }; result?: Result }
const groups = [
  ['tracking', '跟踪性能', '参考轨迹、系统输出与跟踪误差'],
  ['disturbance', '扰动观测', '失配扰动与估计误差；绿色阴影为恢复窗口，虚线为 1.8627 秒恢复时限'],
  ['state_control', '状态与控制', '系统状态与控制输入；阴影表示重放时段'],
  ['switching_replay', '切换与重放', '上方分别显示两路切换函数，下方对比实时状态与重放信号'],
] as const
// Preserve the running task when switching to another module and back.
type PlotName = typeof groups[number][0]
const session = readSession<{ duration: string; taskId: string | null; activePlot: PlotName }>('replay.session', { duration: '50', taskId: null, activePlot: 'tracking' })
if (!groups.some(([name]) => name === session.activePlot)) session.activePlot = 'tracking'

export function ReplayPage() {
  const [duration, setDuration] = useState(session.duration)
  const [taskId, setTaskId] = useState(session.taskId)
  const [task, setTask] = useState<Task | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [error, setError] = useState('')
  const [pollingError, setPollingError] = useState('')
  const [imageErrors, setImageErrors] = useState<Record<string, boolean>>({})
  const [activePlot, setActivePlot] = useState(session.activePlot)
  useEffect(() => { writeSession('replay.session', { duration, taskId, activePlot }) }, [duration, taskId, activePlot])
  const tabButtons = useRef<(HTMLButtonElement | null)[]>([])
  const running = submitting || (!!taskId && (!task || ['queued', 'running'].includes(task.status)))
  const result = task?.status === 'succeeded' ? task.result : undefined
  const imageRevision = task?.progress?.image_revision ?? 0
  useEffect(() => { setImageErrors({}) }, [taskId, imageRevision])
  const percent = task?.status === 'succeeded' ? 100 : Math.min(99, Math.max(0, task?.progress?.percent ?? 0))

  function selectPlot(index: number) {
    const name = groups[index][0]
    setActivePlot(name)
    session.activePlot = name
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
    const seconds = Number(duration)
    if (!duration.trim() || !Number.isFinite(seconds) || seconds < .1 || seconds > 50) {
      setError('仿真时长必须在 0.1～50 秒之间。')
      return
    }
    setError(''); setPollingError(''); setTask(null); setTaskId(null); setImageErrors({}); setSubmitting(true)
    session.taskId = null
    try {
      const created = await post<{ task_id: string }>('/replay/tasks', { sim_time: seconds })
      session.taskId = created.task_id
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
    ? `仿真完成 · ${result.config.sim_time} 秒场景 · 用时 ${result.elapsed_seconds.toFixed(2)} 秒`
    : task?.progress?.message ?? (running ? '等待执行' : '设置参数后开始运行')
  return <main className="fdi-page replay-page control-module" aria-label="重放攻击下的自适应控制仿真">
    <section className="panel fdi-controls">
      <h3>仿真设置</h3>
      <label htmlFor="replay-duration">仿真时长（秒）</label>
      <input id="replay-duration" type="number" min="0.1" max="50" step="0.1" value={duration} disabled={running}
        onChange={event => { setDuration(event.target.value); session.duration = event.target.value }} />
      <button type="button" className="primary" disabled={running} onClick={() => void start()}>{running ? '仿真运行中…' : '开始重放攻击仿真'}</button>
      <button type="button" disabled={!running || !task?.cancellable || cancelling || task.cancel_requested} onClick={() => void cancel()}>{running && (cancelling || task?.cancel_requested) ? '正在取消…' : '取消仿真'}</button>
      <dl><dt>初始状态</dt><dd>x₁ = 0，x₂ = 0</dd><dt>参考轨迹</dt><dd>yᵈ = sin(0.5t) + 0.5sin(t)</dd><dt>重放时段 / 回放延迟（秒）</dt><dd>5–9 / 4.8<br/>12–14 / 2.5<br/>18–21 / 3.5<br/>27–32 / 5.5<br/>36–39 / 3.7<br/>42–45 / 3.0</dd></dl>
      <p className="muted">橙色阴影表示重放攻击时段；扰动误差图的绿色阴影表示恢复窗口。</p>
      {result && <div className="fdi-metrics"><p>跟踪误差 RMSE<br/><strong>{result.tracking_rmse.toPrecision(4)}</strong></p><p>扰动估计 RMSE<br/><strong>{result.disturbance_rmse.toPrecision(4)}</strong></p>
        <div className="fdi-downloads">{[['signals.csv', '下载 CSV'], ['signals.npz', '下载完整数据'], ['result.json', '下载结果摘要']].map(([name, label]) =>
          <a key={name} href={`/api/replay/tasks/${taskId}/files/${name}`} download>{label}</a>)}</div></div>}
    </section>
    <section className="fdi-results" aria-label="重放攻击仿真结果">
      {(error || pollingError || task?.error) && <div className="error" role="alert">{error || pollingError || task?.error}</div>}
      <section className="panel fdi-output" aria-label="结果图示">
        <h3>结果图示</h3>
        <ControlTaskStatus state={running ? 'running' : task?.status === 'succeeded' ? 'succeeded' : task?.status === 'failed' ? 'failed' : task?.status === 'cancelled' ? 'cancelled' : 'idle'}
          message={status} percent={percent} label="重放攻击仿真进度"/>
        <div className="fdi-tabs" role="tablist" aria-label="选择输出图像">{groups.map(([name, title], index) =>
          <button type="button" role="tab" key={name} id={`replay-tab-${name}`} aria-controls={`replay-panel-${name}`}
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
          role="tabpanel" id={`replay-panel-${name}`} aria-labelledby={`replay-tab-${name}`} hidden={activePlot !== name}>
        <header><div><h3>{title}</h3><p>{description}</p></div>{result && <div className="fdi-downloads">{['svg', 'png'].map(format => <a key={format} href={`/api/replay/tasks/${taskId}/files/${name}.${format}`} download>下载 {format.toUpperCase()}</a>)}</div>}</header>
        <DnnPlotViewer key={`${taskId}:${name}`} src={taskId && (result || task?.progress?.outputs?.[name]) && !imageErrors[name] ? `/api/replay/tasks/${taskId}/images/${name}.svg?v=${imageRevision}` : undefined}
          alt={`${title}曲线`} placeholder={imageErrors[name] ? '图片加载失败，请重试或重新运行。' : running ? '计算中，等待曲线更新…' : '运行仿真后显示结果图'}
          onError={() => setImageErrors(previous => ({ ...previous, [name]: true }))}/>
        {imageErrors[name] && <button type="button" onClick={() => setImageErrors(previous => ({ ...previous, [name]: false }))}>重新加载图片</button>}
        </section>)}
      </section>
      {!!task?.logs?.length && <details className="fdi-logs"><summary>运行日志</summary><pre>{task.logs.join('\n')}</pre></details>}
    </section>
  </main>
}
