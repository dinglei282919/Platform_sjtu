import { scrollControlParameters } from './controlInteraction'
import { useCallback, useEffect, useState } from 'react'
import { api, post, ApiError } from './api'
import { readSession, useStoredState, writeSession } from './taskSession'
import { ControlTaskStatus } from './ControlTaskStatus'
import { DnnPlotViewer } from './DnnPlotViewer'
import { ReactorDiagram, compositionLabel, sampleIndexFor, validateReactorData, type ReactorData } from './ReactorDiagram'
import { ReactorLivePage } from './ReactorLivePage'

const defaults = { duration: 3, sample_dt: .002, prescribed_time: 1.5, ell: .15, xs: 1, xd: .5, observer_s: 0, observer_d: .5, r1: .3, r2: .2, attack_scale: 1, attack_frequency: .25, attack_phase: 0 }
type Config = typeof defaults
const fields: [keyof Config, string, number, number, number][] = [
  ['duration', '仿真时长 / s', .1, 3, .1], ['sample_dt', '输出间隔 / s', .0001, .1, .001],
  ['prescribed_time', '预设时间 T / s', .1, 3, .1], ['ell', '增益参数 ell', .01, 1, .01],
  ['xs', '组分初态 xs', -4, 4, .05], ['xd', '组分初态 xd', -2, 2, .05],
  ['observer_s', '组分估计初态 xs', -4, 4, .05], ['observer_d', '组分估计初态 xd', -2, 2, .05],
  ['r1', '循环参数 R1', 0, 1, .05], ['r2', '循环参数 R2', 0, .95, .05],
  ['attack_scale', 'FDI 强度 a（0 为无攻击）', 0, 1.8, .1], ['attack_frequency', 'FDI 角频率 ω / rad·s⁻¹', .01, 2, .05], ['attack_phase', 'FDI 相位 φ / rad', -Math.PI, Math.PI, .1],
]
type Task = { status: string; error?: string; logs?: string[]; cancel_requested?: boolean; cancellable?: boolean;
  progress?: {percent: number; message: string}; result?: {metrics: Record<string, number>; config: Config; imported?: boolean} }

export function ReactorPage() {
  const [view, setView] = useState<'live' | 'experiment'>('live')
  const [experimentBusy, setExperimentBusy] = useState(() => !!readSession<string | null>('reactor.task', null))
  const onExperimentBusyChange = useCallback((busy: boolean, checkingTask: boolean) => {
    setExperimentBusy(busy)
    if (busy && !checkingTask) setView('experiment')
  }, [])
  const activeView = experimentBusy ? 'experiment' : view
  return <section className="reactor-page reactor-workspace">
    <nav className="reactor-live-tabs" aria-label="反应器工作模式">
      <button aria-pressed={activeView === 'live'} disabled={experimentBusy} title={experimentBusy ? '离线任务运行或恢复中，请等待完成后切换' : undefined} onClick={() => { if (!experimentBusy) setView('live') }}>实时控制系统</button>
      <button aria-pressed={activeView === 'experiment'} onClick={() => setView('experiment')}>离线实验与导出</button>
    </nav>
    <div hidden={activeView !== 'live'}><ReactorLivePage active={activeView === 'live'}/></div>
    <div hidden={activeView !== 'experiment'}><ReactorExperimentPage active={activeView === 'experiment'} onBusyChange={onExperimentBusyChange}/></div>
  </section>
}

function ReactorExperimentPage({ active, onBusyChange }: { active: boolean; onBusyChange: (busy: boolean, checkingTask: boolean) => void }) {
  const [config, setConfig] = useStoredState<Config>('reactor.attack.config', defaults)
  const [taskId, setTaskId] = useStoredState<string | null>('reactor.task', null)
  const [task, setTask] = useState<Task | null>(null)
  const [data, setData] = useState<ReactorData | null>(null)
  const [error, setError] = useState('')
  const [pollError, setPollError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [selecting, setSelecting] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [animated, setAnimated] = useStoredState('reactor.animated', true)
  const [tab, setTab] = useStoredState('reactor.tab', 'diagram')
  const [index, setIndex] = useStoredState('reactor.index', 0)
  const [appliedImport, setAppliedImport] = useStoredState<string | null>('reactor.appliedImport', null)
  const [playing, setPlaying] = useState(false)
  const [reload, setReload] = useState(0)
  const running = selecting || submitting || (!!taskId && (!task || ['queued', 'running'].includes(task.status)))
  const sampleIndex = sampleIndexFor(data, index)
  useEffect(() => { onBusyChange(running, !!taskId && !task) }, [running, taskId, task, onBusyChange])
  useEffect(() => { if (!active) setPlaying(false) }, [active])
  useEffect(() => {
    if (!taskId) return
    let active = true
    let timer = 0
    async function poll() {
      try {
        const next = await api<Task>('/tasks/' + taskId)
        if (!active) return
        setTask(next); setPollError('')
        if (['queued', 'running'].includes(next.status)) timer = window.setTimeout(poll, 1000)
      } catch (e) {
        if (!active) return
        if (e instanceof ApiError && e.status === 404) {
          setTaskId(null); setTask(null); setData(null); setPlaying(false); setError('任务不存在，请重新运行。'); return
        }
        setPollError('状态查询暂时失败，正在重试：' + String(e))
        timer = window.setTimeout(poll, 2000)
      }
    }
    void poll()
    return () => { active = false; window.clearTimeout(timer) }
  }, [taskId, setTaskId])
  useEffect(() => {
    if (task?.status !== 'succeeded' || !taskId) return
    let active = true
    api<ReactorData>('/reactor/tasks/' + taskId + '/trajectory').then(value => {
      if (active) {
        validateReactorData(value)
        setData(value); setIndex(previous => sampleIndexFor(value, previous)); setError('')
        if (task.result?.imported && appliedImport !== taskId) {
          setConfig(Object.fromEntries(Object.keys(defaults).map(key => [key, task.result!.config[key as keyof Config]])) as Config)
          setAppliedImport(taskId)
        }
      }
    }).catch(e => { if (active) setError('数据加载失败，可重试：' + String(e)) })
    return () => { active = false }
  }, [task?.status, taskId, reload])
  useEffect(() => {
    if (!active || !playing || !data || running) return
    const start = performance.now(), offset = data.t[sampleIndexFor(data, index)]
    const timer = window.setInterval(() => {
      const t = offset + (performance.now() - start) / 1000
      const found = data.t.findIndex(value => value >= t)
      setIndex(found < 0 ? data.t.length - 1 : found)
      if (found < 0 || found === data.t.length - 1) setPlaying(false)
    }, 40)
    return () => window.clearInterval(timer)
    // index is the starting sample; timer itself advances the slider.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, playing, data, running])
  async function start() {
    if (Object.values(config).some(v => !Number.isFinite(v))) { setError('请输入有效数值。'); return }
    onBusyChange(true, false)
    setSubmitting(true); setError(''); setPollError(''); setData(null); setTask(null); setTaskId(null)
    writeSession('reactor.task', null)
    setPlaying(false); setCancelling(false); setIndex(0)
    try {
      const created = await post<{task_id: string}>('/reactor/tasks', config)
      // Persist even if navigation unmounted this page during the POST.
      writeSession('reactor.task', created.task_id)
      setTaskId(created.task_id)
    } catch (e) { setError(String(e)) } finally { setSubmitting(false) }
  }
  async function cancel() {
    setCancelling(true)
    try { setTask(await post<Task>('/tasks/' + taskId + '/cancel')) }
    catch (e) { setError(String(e)); setCancelling(false) }
  }
  async function loadSaved() {
    onBusyChange(true, false)
    setSelecting(true); setError(''); setPlaying(false)
    try {
      const { path } = await post<{path: string}>('/local-paths/select', {kind: 'directory'})
      if (!path) return
      setTaskId(null); writeSession('reactor.task', null); setTask(null); setData(null)
      setPlaying(false); setIndex(0); setPollError(''); setCancelling(false)
      const created = await post<{task_id: string}>('/reactor/import-result', {directory: path})
      writeSession('reactor.task', created.task_id); setTaskId(created.task_id)
    } catch (e) { setError(String(e)) }
    finally { setSelecting(false) }
  }
  const state = running ? 'running' : task?.status === 'succeeded' ? 'succeeded' : task?.status === 'failed' ? 'failed' : task?.status === 'cancelled' ? 'cancelled' : 'idle'
  const metrics = task?.result?.metrics
  return <main onWheelCapture={scrollControlParameters} className="reactor-page control-module" aria-label="化学反应器自适应控制仿真">
    <div className="reactor-layout">
      <section className="panel reactor-controls">
        <p className="muted reactor-control-objective">{compositionLabel("控制目标：期望组分 xs、xd 趋于 0。")}</p>
        <h3>仿真设置</h3><details><summary>模型与 FDI 注入说明</summary><p>xs=F(1−R2)x₁/(G₁G₂)，xd=Fx₂/G₂；G₁=0.4，G₂=0.5，F=0.5。Λ(t)=a[1+1.5sin(ωt+φ)]，κ=1+Λ；默认 a=1、ω=0.25、φ=0。控制器接收 κx₁ 和 κx̂₂，观测器保留原始输出反馈。目标为组分状态进入零点邻域，不表示浓度恢复至未定义的生产工作点。</p></details>
        {fields.map(([key, label, min, max, step]) => <label key={key}>{compositionLabel(label)}
          <input aria-label={label} type="number" value={Number.isNaN(config[key]) ? '' : config[key]} min={min} max={max} step={step} disabled={running}
            onChange={e => setConfig({...config, [key]: e.target.value === '' ? NaN : Number(e.target.value)})}/>
        </label>)}
        <button className="primary" disabled={running} onClick={() => void start()}>运行仿真</button>
        <button disabled={selecting || submitting || !running || !taskId || !task?.cancellable || cancelling || task?.cancel_requested} onClick={() => void cancel()}>{running && (cancelling || task?.cancel_requested) ? '正在取消…' : '取消计算'}</button>
        <button disabled={running} onClick={() => setConfig({...defaults})}>恢复默认参数</button>
        <button disabled={running} onClick={() => void loadSaved()}>载入已有仿真结果…</button>
        <p className="muted">输出间隔仅影响采样密度。显示反变换组分 xs/xd，保留正负值，不推定物理单位。R2 必须小于 1。</p>
        {metrics && <div><p>终态组分范数：{metrics.final_composition_norm.toPrecision(4)}</p><p>观测误差：{metrics.final_composition_observer_error.toPrecision(4)}</p>
          <p>采样 max |u|：{metrics.peak_abs_u_sampled.toPrecision(4)}</p><p>输出点数：{metrics.samples}</p><p>用时：{metrics.elapsed_seconds.toFixed(1)} s</p>
          <div className="fdi-downloads">{['trajectory.csv', 'trajectory.npz', 'summary.json', 'results.png'].map(name =>
            <a key={name} href={'/api/reactor/tasks/' + taskId + '/files/' + name} download>{name}</a>)}</div></div>}
      </section>
      <section className="panel reactor-view">
        <h3>结果图示</h3>
        {(error || pollError || task?.error) && <div className="error" role="alert">{error || pollError || task?.error}</div>}
        <ControlTaskStatus state={state} percent={task?.progress?.percent ?? 0} label="化学反应器仿真进度"
          message={selecting || (running && task?.cancellable === false) ? '正在载入已有结果 · 工艺动画已暂停' : running ? '后台计算中 · 工艺动画已暂停 · 可取消' : state === 'succeeded' ? '计算完成，可查看曲线、回放或下载结果' : state === 'failed' ? '计算失败，请检查错误信息后重试' : state === 'cancelled' ? '本次计算已取消' : '就绪 · 尚未运行'}/>
        <div className="fdi-tabs"><button aria-pressed={tab === 'diagram'} onClick={() => setTab('diagram')}>工艺框图</button>
          <button aria-pressed={tab === 'plots'} onClick={() => setTab('plots')}>仿真曲线</button></div>
        <div hidden={tab !== 'diagram'}>
          <ReactorDiagram data={data} index={sampleIndex} animated={active && animated && !running && tab === 'diagram'}/>
          <p className="muted">组分与参数使用本次实验数据，保留正负号；液面与冷却管路仅作工艺示意。</p>
          <div className="reactor-playback"><button disabled={running} onClick={() => setAnimated(!animated)}>{animated ? '暂停工艺动画' : '播放工艺动画'}</button>
            <button disabled={!data || running} onClick={() => { if (!playing && data && sampleIndex === data.t.length - 1) setIndex(0); setPlaying(!playing) }}>{playing ? '暂停数据回放' : '回放仿真数据'}</button>
            <input aria-label="仿真时间" type="range" min={0} max={data ? data.t.length - 1 : 0} value={sampleIndex} disabled={!data || running}
              onChange={e => { setPlaying(false); setIndex(Number(e.target.value)) }}/></div>
          <p className="reactor-readout">{compositionLabel(data ? '仿真时间：' + data.t[sampleIndex].toFixed(3) + ' s　切换模式：' + data.mode[sampleIndex] + '　组分 xs：' + data.xs[sampleIndex].toPrecision(5) + '　组分 xd：' + data.xd[sampleIndex].toPrecision(5) + '　控制输入 u：' + data.u[sampleIndex].toPrecision(5) : '运行后显示本次仿真数据')}</p>
        </div><div hidden={tab !== 'plots'}><DnnPlotViewer onError={() => setError("曲线图片加载失败，请切换页面后重试。")} src={state === 'succeeded' ? '/api/reactor/tasks/' + taskId + '/files/results.png' : undefined} alt="组分、组分估计误差、控制输入、FDI反馈、切换模式和权重曲线" placeholder="完成仿真后显示曲线"/></div>
        {state === 'succeeded' && !data && <button onClick={() => setReload(reload + 1)}>重新加载数据</button>}
        {!!task?.logs?.length && <details><summary>运行日志</summary><pre>{task.logs.join('\n')}</pre></details>}
      </section>
    </div>
  </main>
}
