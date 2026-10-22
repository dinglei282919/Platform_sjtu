import { useEffect, useState } from 'react'
import { api, post, ApiError } from './api'
import { useStoredState, writeSession } from './taskSession'
import { ControlTaskStatus } from './ControlTaskStatus'
import { DnnPlotViewer } from './DnnPlotViewer'

const defaults = { duration: 3, sample_dt: .002, prescribed_time: 1.5, ell: .15, x1: .5, x2: .5, observer1: 0, observer2: .5 }
type Config = typeof defaults
const fields: [keyof Config, string, number, number, number][] = [
  ['duration', '仿真时长 / s', .1, 3, .1], ['sample_dt', '输出间隔 / s', .0001, .1, .001],
  ['prescribed_time', '预设时间 T / s', .1, 3, .1], ['ell', '增益参数 ell', .01, 1, .01],
  ['x1', '初态 x1', -1, 1, .05], ['x2', '初态 x2', -1, 1, .05],
  ['observer1', '估计初态 xhat1', -1, 1, .05], ['observer2', '估计初态 xhat2', -1, 1, .05],
]
type Task = { status: string; error?: string; logs?: string[]; cancel_requested?: boolean;
  progress?: {percent: number; message: string}; result?: {metrics: Record<string, number>; config: Config} }
type Data = Record<string, number[]>

export function ReactorPage() {
  const [config, setConfig] = useStoredState<Config>('reactor.config', defaults)
  const [taskId, setTaskId] = useStoredState<string | null>('reactor.task', null)
  const [task, setTask] = useState<Task | null>(null)
  const [data, setData] = useState<Data | null>(null)
  const [error, setError] = useState('')
  const [pollError, setPollError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [animated, setAnimated] = useState(true)
  const [tab, setTab] = useState('diagram')
  const [index, setIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [reload, setReload] = useState(0)
  const running = submitting || (!!taskId && (!task || ['queued', 'running'].includes(task.status)))
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
          setTaskId(null); setTask(null); setError('任务不存在，请重新运行。'); return
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
    api<Data>('/reactor/tasks/' + taskId + '/trajectory').then(value => {
      if (active) { setData(value); setIndex(0); setError('') }
    }).catch(e => { if (active) setError('数据加载失败，可重试：' + String(e)) })
    return () => { active = false }
  }, [task?.status, taskId, reload])
  useEffect(() => {
    if (!playing || !data) return
    const start = performance.now(), offset = data.t[index]
    const timer = window.setInterval(() => {
      const t = offset + (performance.now() - start) / 1000
      const found = data.t.findIndex(value => value >= t)
      setIndex(found < 0 ? data.t.length - 1 : found)
      if (found < 0 || found === data.t.length - 1) setPlaying(false)
    }, 40)
    return () => window.clearInterval(timer)
    // index is the starting sample; timer itself advances the slider.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, data])
  async function start() {
    if (Object.values(config).some(v => !Number.isFinite(v))) { setError('请输入有效数值。'); return }
    setSubmitting(true); setError(''); setPollError(''); setData(null); setTask(null); setTaskId(null)
    writeSession('reactor.task', null)
    setPlaying(false); setCancelling(false)
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
  const state = running ? 'running' : task?.status === 'succeeded' ? 'succeeded' : task?.status === 'failed' ? 'failed' : task?.status === 'cancelled' ? 'cancelled' : 'idle'
  const metrics = task?.result?.metrics
  return <main className="reactor-page control-module" aria-label="化学反应器自适应控制仿真">
    <div className="reactor-layout">
      <section className="panel reactor-controls">
        <h3>仿真设置</h3>
        {fields.map(([key, label, min, max, step]) => <label key={key}>{label}
          <input aria-label={label} type="number" value={Number.isNaN(config[key]) ? '' : config[key]} min={min} max={max} step={step} disabled={running}
            onChange={e => setConfig({...config, [key]: e.target.value === '' ? NaN : Number(e.target.value)})}/>
        </label>)}
        <button className="primary" disabled={running} onClick={() => void start()}>运行仿真</button>
        <button disabled={!running || !taskId || cancelling || task?.cancel_requested} onClick={() => void cancel()}>{running && (cancelling || task?.cancel_requested) ? '正在取消…' : '取消计算'}</button>
        <button disabled={running} onClick={() => setConfig({...defaults})}>恢复默认参数</button>
        <p className="muted">输出间隔仅影响采样密度。状态与输入沿用模型符号，未推定物理单位。</p>
        {metrics && <div><p>终态 ||x||：{metrics.final_x_norm.toPrecision(4)}</p><p>观测误差：{metrics.final_observer_error.toPrecision(4)}</p>
          <p>采样 max |u|：{metrics.peak_abs_u_sampled.toPrecision(4)}</p><p>用时：{metrics.elapsed_seconds.toFixed(1)} s</p>
          <div className="fdi-downloads">{['trajectory.csv', 'trajectory.npz', 'summary.json', 'results.png'].map(name =>
            <a key={name} href={'/api/reactor/tasks/' + taskId + '/files/' + name} download>{name}</a>)}</div></div>}
      </section>
      <section className="panel reactor-view">
        <h3>结果图示</h3>
        {(error || pollError || task?.error) && <div className="error" role="alert">{error || pollError || task?.error}</div>}
        <ControlTaskStatus state={state} percent={task?.progress?.percent ?? 0} label="化学反应器仿真进度"
          message={running ? '后台计算中 · 工艺动画已暂停 · 可取消' : state === 'succeeded' ? '计算完成，可查看曲线、回放或下载结果' : state === 'failed' ? '计算失败，请检查错误信息后重试' : state === 'cancelled' ? '本次计算已取消' : '就绪 · 尚未运行'}/>
        <div className="fdi-tabs"><button aria-pressed={tab === 'diagram'} onClick={() => setTab('diagram')}>工艺框图</button>
          <button aria-pressed={tab === 'plots'} onClick={() => setTab('plots')}>仿真曲线</button></div>
        {tab === 'diagram' ? <div>
          <img className="reactor-diagram" width={1600} height={895} alt="化学反应器双路进料工艺框图" src={'/api/reactor/diagram/reactor_dual_feed.' + (animated && !running ? 'gif' : 'png')}/>
          <p className="muted">框图为固定工艺示意，液位、流速与双路进料不由数值状态驱动。</p>
          <div className="reactor-playback"><button disabled={running} onClick={() => setAnimated(!animated)}>{animated ? '暂停工艺动画' : '播放工艺动画'}</button>
            <button disabled={!data} onClick={() => { if (!playing && data && index === data.t.length - 1) setIndex(0); setPlaying(!playing) }}>{playing ? '暂停数据回放' : '回放仿真数据'}</button>
            <input aria-label="仿真时间" type="range" min={0} max={data ? data.t.length - 1 : 0} value={index} disabled={!data}
              onChange={e => { setPlaying(false); setIndex(Number(e.target.value)) }}/></div>
          <p className="reactor-readout">{data ? 't = ' + data.t[index].toFixed(3) + ' s　σ = ' + data.mode[index] + '　x1 = ' + data.x1[index].toPrecision(5) + '　x2 = ' + data.x2[index].toPrecision(5) + '　u = ' + data.u[index].toPrecision(5) : '运行后显示本次仿真数据'}</p>
        </div> : <DnnPlotViewer onError={() => setError("曲线图片加载失败，请切换页面后重试。")} src={state === 'succeeded' ? '/api/reactor/tasks/' + taskId + '/files/results.png' : undefined} alt="化学反应器状态、观测误差、控制输入、切换模式和权重曲线" placeholder="完成仿真后显示曲线"/>}
        {state === 'succeeded' && !data && <button onClick={() => setReload(reload + 1)}>重新加载数据</button>}
        {!!task?.logs?.length && <details><summary>运行日志</summary><pre>{task.logs.join('\n')}</pre></details>}
      </section>
    </div>
  </main>
}
