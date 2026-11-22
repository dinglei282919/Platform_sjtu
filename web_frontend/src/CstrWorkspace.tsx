import { Component, useEffect, useId, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import { CstrLiveTrends, TrendChart } from './CstrLiveTrends'
import { CstrProcessDiagram } from './CstrProcessDiagram'
import { pulseControls, type PulseReceipt } from './cstrDisturbance'
import { FormNumber, Page } from './PlatformUI'
import parameterHelp from '../../assets/cstr/parameter-help.json'
import { api, ApiError, post } from './api'
import { useTask } from './useTask'
import { useStoredState } from './taskSession'
import { dnnOwner, emptyHandle, healingOwner, useLivePresence } from './cstrOwner'
import { HealingPage } from './HealingPage'

export type RiskId =
  | 'risk-overview'
  | 'dnn-mpc'
  | 'fdi'
  | 'replay'
  | 'self-healing'
export const riskEntries: { id: RiskId; name: string }[] = [
  { id: 'risk-overview', name: '综合仿真展示' },
  { id: 'dnn-mpc', name: '深度神经网络优化控制' },
  { id: 'self-healing', name: '多变量过程自愈控制' },
  { id: 'fdi', name: '虚假数据注入下的自适应控制仿真' },
  { id: 'replay', name: '重放攻击下的自适应控制仿真' },
]
const stages = [
  '风险场景下控制系统智能模型训练算法',
  '风险场景下智能模型优化控制算法',
]
const status: Record<string, string> = {
  ready: '就绪',
  running: '运行中',
  paused: '已暂停',
  ended: '已结束',
  queued: '排队中',
  succeeded: '已完成',
  failed: '失败',
  cancelled: '已取消',
}
type Settings = {
  horizon: number
  flow_min: number
  flow_max: number
  move_weight: number
  speed: number
  reference: number
}
type Train = {
  epochs: number
  trajectories: number
  length: number
  seed: number
  lr: number
  device: string
}
type NumericParameter = keyof Settings | Exclude<keyof Train, 'device'> | 'duration'

function numberHint(key: NumericParameter, min: number, max: number, period: number, value: number) {
  const seconds = 60 * period
  const wall = key === 'speed' && value >= min && value <= max
    ? `，按此倍率约用 ${Number((seconds / value).toFixed(2))} 秒` : ''
  return parameterHelp[key].replace('{min}', String(min)).replace('{max}', String(max))
    .replace('{period}', String(seconds)).replace('{wall}', wall)
}

type DisturbanceOptions = {
  presets: { id: string; label: string; nominal: number; target: number; unit: string }[]
  default_cycles: number
  max_cycles: number
  recovery_cycles: number
}
type Config = {
  disturbances?: DisturbanceOptions
  period: number
  trend_intervals: number
  settings: Settings
  training: Train
  models: { id: string; version: string; label: string }[]
}
export type Point = {
  seq: number
  t: number
  T: number
  T_ref: number
  Qc: number
  Qc_cmd: number
  trial_id?: string
  disturbance_phase?: 'active' | 'recovery'
  T_hold?: number | null
  Qc_hold?: number
  solve_ms?: number
}
type LogEvent = {
  key?: string
  event: string
  t: number
  event_id: number
  new?: string
  reason?: string
  value?: number
}
type OnlineUpdate = {
  enabled: boolean
  device: string
  batch_intervals: number
  manual_ready: boolean
  status: string
  result: string
}
type Snapshot = {
  online_update?: OnlineUpdate
  disturbance?: { phase: string; clear_requested: boolean } | null
  disturbance_display?: { status: string; timing: string; metrics: string } | null
  id: string
  run: string
  status: string
  error: string | null
  seq: number
  model: string
  initial_model: string
  reference: number
  config: Settings
  update: string | null
  resync: boolean
  latest: Point
  samples: Point[]
  events: LogEvent[]
  display: Record<string, string>
}
type Task = {
  id: string
  status: string
  error?: string
  progress: { percent?: number; message?: string; history?: number[][] }
  result?: {
    model?: string
    ready?: boolean
    test?: { one_rmse: number[]; rollout_rmse: number[] }
    validation?: { reasons: string[] }
  }
}
const { ensure: ensureHandle, request: owned, save: saveHandle } = dnnOwner
export function useCstrPresence() {
  useLivePresence(dnnOwner)
  useLivePresence(healingOwner)
}
async function loadConfig(): Promise<Config> {
  const config = await api<Config | null>('/cstr/config')
  if (!config?.settings || !config.training ||
      !Array.isArray(config.models) || !config.models.every(model => typeof model?.id === 'string' && typeof model.label === 'string')) {
    throw new Error('CSTR 配置接口返回了不兼容的数据，请确认本机平台服务版本。')
  }
  return config
}

function StageSwitch({ stage, select }: { stage: number; select: (index: number) => void }) {
  function onKeyDown(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    let next: number
    switch (event.key) {
      case 'ArrowRight': next = (index + 1) % stages.length; break
      case 'ArrowLeft': next = (index + stages.length - 1) % stages.length; break
      case 'Home': next = 0; break
      case 'End': next = stages.length - 1; break
      default: return
    }
    event.preventDefault()
    select(next)
    document.getElementById(`cstr-stage-${next}`)?.focus()
  }
  return <div className="view-switches" role="tablist" aria-label="深度神经网络优化控制工作流程">
    {stages.map((name, index) => <button type="button" role="tab" key={name}
      id={`cstr-stage-${index}`} aria-controls="cstr-stage-panel"
      aria-selected={stage === index} tabIndex={stage === index ? 0 : -1}
      className={stage === index ? 'active' : ''} onClick={() => select(index)}
      onKeyDown={event => onKeyDown(event, index)}>{name}</button>)}
  </div>
}

function Device({
  value,
  change,
  label = '训练设备',
  disabled = false,
}: {
  value: string
  change: (v: string) => void
  label?: string
  disabled?: boolean
}) {
  const hintId = useId()
  return (
    <label>
      <span>{label}</span>
      <select aria-label={label} aria-describedby={hintId} disabled={disabled} value={value} onChange={(e) => change(e.target.value)}>
        {['auto', 'cpu', 'cuda'].map((v) => (
          <option key={v}>{v}</option>
        ))}
      </select>
      <small id={hintId} className="parameter-hint">{parameterHelp.device}</small>
    </label>
  )
}
class CstrPageBoundary extends Component<{ active: RiskId; children: ReactNode }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() { return { failed: true } }

  componentDidUpdate(previous: { active: RiskId }) {
    if (this.state.failed && previous.active !== this.props.active) this.setState({ failed: false })
  }

  render() {
    if (!this.state.failed) return this.props.children
    return <Page title="风险管控优化决策">
      <div role="alert" className="error">
        <strong>当前控制页面加载失败</strong>
        <p>可以重新打开当前页面，也可以通过顶部菜单切换模块。</p>
        <button type="button" className="secondary" onClick={() => this.setState({ failed: false })}>重新打开页面</button>
      </div>
    </Page>
  }
}

export function CstrWorkspace({ active }: { active: RiskId }) {
  return <CstrPageBoundary active={active}><CstrPage active={active} /></CstrPageBoundary>
}

function CstrPage({ active }: { active: RiskId }) {
  const [stage, setStage] = useState(1),
    [cfg, setCfg] = useState<Config | null>(null)
  const [settings, setSettings] = useState<Settings | null>(null),
    [train, setTrain] = useState<Train | null>(null)
  const [model, setModel] = useState('default')
  const csvInput = useRef<HTMLInputElement>(null)
  const [csvFile, setCsvFile] = useState<File | null>(null)
  const [preset, setPreset] = useState('')
  const [duration, setDuration] = useState<number | null>(null)
  const [pulseReceipt, setPulseReceipt] = useState<PulseReceipt | null>(null)
  const [trainId, setTrainId] = useStoredState<string | null>(
    'cstr-training',
    null,
  )
  const [snap, setSnap] = useState<Snapshot | null>(null),
    [samples, setSamples] = useState<Point[]>([])
  const [message, setMessage] = useState(''),
    [busy, setBusy] = useState(false)
  const [pendingPolicy, setPendingPolicy] = useState<Pick<OnlineUpdate, 'enabled' | 'device'> | null>(null)
  const lastSnapshot = useRef<Snapshot | null>(null)
  const visibleSamples = useRef<{ run: string; rows: Point[] }>({
    run: '',
    rows: [],
  })
  function showSamples(v: Snapshot, replace: boolean) {
    const prev = visibleSamples.current
    const last = prev.rows[prev.rows.length - 1]?.seq ?? -1
    if (!replace && prev.run === v.run && !v.samples.some(point => point.seq > last)) return
    const rows = (
      replace || prev.run !== v.run
        ? v.samples
        : [...prev.rows, ...v.samples.filter((p) => p.seq > last)]
    ).slice(-481)
    visibleSamples.current = { run: v.run, rows }
    setSamples(rows)
  }
  const live = active === 'dnn-mpc' && stage === 1
  const { task, pollingError: trainingError } = useTask<Task>(
    trainId,
    setTrainId,
  )
  const training = Boolean(
    trainId && (!task || ['queued', 'running'].includes(task.status)),
  )
  const isDnn = active === 'dnn-mpc'
  useEffect(() => { if (isDnn) setStage(1) }, [isDnn])
  const trainedModel = task?.status === 'succeeded' ? task.result?.model : undefined
  const [configError, setConfigError] = useState('')
  const [configAttempt, setConfigAttempt] = useState(0)

  useEffect(() => {
    if (!isDnn) return
    let current = true
    setConfigError('')
    loadConfig().then(config => {
      if (!current) return
      setCfg(config)
      setSettings(previous => previous ?? config.settings)
      setTrain(previous => previous ?? config.training)
    }).catch(error => {
      if (current) setConfigError(error instanceof Error ? error.message : String(error))
    })
    return () => { current = false }
  }, [isDnn, configAttempt, trainedModel])
  useEffect(() => {
    if (!isDnn || !cfg) return
    let active = true
    void ensureHandle().catch(e => { if (active) setMessage(String(e)) })
    return () => { active = false }
  }, [isDnn, Boolean(cfg)])

  useEffect(() => {
    if (!live || !cfg) return
    let alive = true,
      timer: number,
      cursor = -1,
      run = ''
    async function poll() {
      try {
        const h = await ensureHandle()
        let v = await owned<Snapshot>(h, '?after=' + cursor)
        if (!alive) return
        if (v.status === 'ready') {
          // Older saved sessions were created before automatic startup.
          await owned(h, '/commands', { kind: 'start', key: 'enter-' + v.run })
          v = await owned<Snapshot>(h, '?after=-1')
          if (!alive) return
        }
        if (run && run !== v.run && !v.resync) {
          v = await owned<Snapshot>(h, '?after=-1')
          if (!alive) return
          v.resync = true
        }
        if (cursor < 0 && lastSnapshot.current?.run !== v.run) {
          setSettings({ ...v.config, reference: v.reference })
          const original = cfg?.models.find(
            (m) => m.version === v.model,
          )
          if (original) setModel(original.id)
        }
        const replace = cursor < 0 || v.resync || v.run !== run
        cursor = v.seq
        run = v.run
        lastSnapshot.current = v
        setSnap(v)
        showSamples(v, replace)
        timer = window.setTimeout(poll, 350)
      } catch (e) {
        if (!alive) return
        if (e instanceof ApiError && e.status === 404) {
          saveHandle(emptyHandle)
          cursor = -1
          run = ''
        } else setMessage(String(e))
        timer = window.setTimeout(poll, 1000)
      }
    }
    void poll()
    return () => {
      alive = false
      clearTimeout(timer)
    }
  }, [live, Boolean(cfg)])
  async function action(fn: () => Promise<void>) {
    if (busy) return
    setBusy(true)
    setMessage('')
    try {
      await fn()
    } catch (e) {
      setMessage(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }
  const scenarios = cfg?.disturbances
  const selectedPreset = preset || scenarios?.presets[0]?.id
  const cycles = duration ?? scenarios?.default_cycles ?? 20
  const trialDisplay = snap?.disturbance_display
  const pulse = pulseControls(snap, pulseReceipt)
  const validCycles = !!scenarios && Number.isInteger(cycles) && cycles >= 1 && cycles <= scenarios.max_cycles
  async function command(kind: string) {
    if (!settings) return
    const h = await ensureHandle()
    const receipt = await owned<Omit<PulseReceipt, 'kind'>>(h, '/commands', {
      kind,
      key: crypto.randomUUID(),
      ...(kind === 'reset' ? { model_id: model, settings } : {}),
      ...(kind === 'disturbance' ? { preset: selectedPreset, duration_cycles: cycles } : {}),
      ...(kind === 'model' ? { model_id: model } : {}),
      ...(kind === 'reference' ? { value: settings.reference } : {}),
    })
    if (kind === 'disturbance' || kind === 'clear_disturbance') {
      setPulseReceipt({ ...receipt, kind })
      return
    }
    const v = await owned<Snapshot>(h, '?after=-1')
    lastSnapshot.current = v
    setSnap(v)
    showSamples(v, true)
  }
  const online = snap?.online_update
  async function setUpdatePolicy(enabled: boolean, device: string) {
    setPendingPolicy({ enabled, device })
    try {
      const h = await ensureHandle()
      const policy = await owned<OnlineUpdate>(h, '/updates/policy', { enabled, device })
      setSnap(current => current?.id === h.id ? { ...current, online_update: policy } : current)
    } finally {
      setPendingPolicy(null)
    }
  }
  const state = snap?.status ?? 'ready'
  const allowed: Record<string, string[]> = {
    ready: ['reset', 'model', 'reference', 'end'],
    running: ['model', 'pause', 'reference', 'reset', 'end'],
    paused: ['model', 'resume', 'reference', 'reset', 'end'],
    ended: ['reset'],
  }
  const cmdLabels: Record<string, string> = {
    model: '应用所选模型',
    pause: '暂停',
    resume: '继续',
    reset: '应用配置并复位',
    end: '结束',
    reference: '更新设定值',
  }
  const marked = snap?.events
    .filter((e) =>
      [
        'activated',
        'rollback',
        'reference_applied',
        'failed',
        'update_rejected',
      ].includes(e.event),
    )
    .slice(-1)[0]
  const eventLabels: Record<string, string> = {
    activated: '模型已切换',
    rollback: '模型已回退',
    reference_applied: '设定值已生效',
    failed: '求解暂停',
    update_rejected: '候选未通过',
  }
  const error = message || (stage === 0 ? trainingError : '')
  if (!isDnn) {
    if (active === 'self-healing') return <HealingPage />
    const overview = active === 'risk-overview'
    return <Page title={riskEntries.find(entry => entry.id === active)?.name ?? '风险管控优化决策'}>
      <section className="deferred">
        <h3>{overview ? '综合仿真展示待接入' : '仿真界面待接入'}</h3>
        <p>{overview
          ? '四种控制算法接入后，这里将展示统一 CSTR 的仿真结果。'
          : '该子模块的 Web 仿真界面将在后续阶段接入。'}</p>
        <p>从顶部“风险管控优化决策”菜单可进入深度神经网络优化控制或多变量过程自愈控制。</p>
      </section>
    </Page>
  }

  return (
    <Page title="深度神经网络优化控制" className="cstr-page">
      <StageSwitch stage={stage} select={setStage} />
      <section id="cstr-stage-panel" role="tabpanel" aria-labelledby={`cstr-stage-${stage}`} className="cstr-stage">
        {configError && <div role="alert" className="error">
          <strong>CSTR 配置加载失败</strong>　{configError}
          <div className="actions"><button type="button" className="small secondary"
            onClick={() => setConfigAttempt(previous => previous + 1)}>重新加载配置</button></div>
        </div>}
        {!cfg && !configError && <div role="status" className="module-status-bar">正在加载 CSTR 配置…</div>}
        {error && <div role="alert" className="error">{error}</div>}
        {cfg &&
          settings &&
          train &&
          (stage === 0 ? (
            <div className="two-column cstr-layout">
              <aside className="panel">
                <h3>模型训练配置</h3>
                <label>外部训练数据（CSV，可选）
                  <input ref={csvInput} type="file" accept=".csv,text/csv" onChange={event => setCsvFile(event.target.files?.[0] ?? null)} />
                </label>
                <p className="cstr-data-help">{csvFile ? csvFile.name : '默认使用机理生成的 CSTR 轨迹。'}　<a href="/api/cstr/training/example.csv" download>下载 CSV 示例</a>
                  {csvFile && <button type="button" className="small secondary" onClick={() => {
                    setCsvFile(null); if (csvInput.current) csvInput.current.value = ''
                  }}>改用生成轨迹</button>}</p>
                <div className="form-grid">
                  {(
                    [
                      ['trajectories', '轨迹数', 24, 600, 1],
                      ['length', '每条轨迹周期', 30, 120, 1],
                      ['epochs', '训练轮数', 1, 2000, 1],
                      ['lr', '学习率', 0.00001, 0.05, 0.0001],
                      ['seed', '随机种子', 0, 2147483647, 1],
                    ] as const
                  ).filter(([k]) => !csvFile || (k !== 'trajectories' && k !== 'length')).map(([k, l, lo, hi, st]) => (
                    <FormNumber
                      key={k}
                      label={l}
                      min={lo}
                      max={hi}
                      step={st}
                      value={train[k]}
                      hint={numberHint(k, lo, hi, cfg.period, train[k])}
                      onChange={(v) => setTrain({ ...train, [k]: v })}
                    />
                  ))}
                  <Device
                    value={train.device}
                    change={(v) => setTrain({ ...train, device: v })}
                  />
                </div>
                <div className="actions">
                  <button
                    disabled={busy || training}
                    onClick={() =>
                      void action(async () => {
                        const query = new URLSearchParams({ epochs: String(train.epochs), seed: String(train.seed),
                            lr: String(train.lr), device: train.device })
                          const r = csvFile
                            ? await api<{ task_id: string }>('/cstr/training/csv?' + query,
                                { method: 'POST', headers: { 'Content-Type': 'text/csv' }, body: csvFile })
                            : await post<{ task_id: string }>('/cstr/training', train)
                          setTrainId(r.task_id)
                      })
                    }
                  >
                    {csvFile ? '使用外部数据训练' : '生成轨迹并训练'}
                  </button>
                  <button
                    className="secondary"
                    disabled={busy || !training}
                    onClick={() =>
                      void action(async () => {
                        await post('/tasks/' + trainId + '/cancel')
                      })
                    }
                  >
                    取消训练
                  </button>
                </div>
              </aside>
              <article className="panel">
                <h3>模型训练进度</h3>
                <header className="module-status-bar cstr-status">
                  <strong><i className={`status-dot ${task?.status ?? 'ready'}`} />{status[task?.status ?? 'ready']}</strong>
                  <span>{task?.error ?? task?.progress.message ?? ''}</span>
                </header>
                <progress aria-label="模型训练进度" max={100} value={task?.progress.percent ?? 0} />
                {task?.progress.history?.length ? <TrendChart
                  title="预测模型训练误差"
                  series={['训练 MSE', '验证 MSE']}
                  axis="轮次"
                  rows={task.progress.history}
                /> : <div className="empty">开始训练后，这里将显示训练集和验证集的预测误差。</div>}
                {task?.result?.test && (
                  <div className="cstr-readout">
                    <strong>测试集 RMSE</strong>
                    <span>
                      单步 (C, T, Tc)　
                      {task.result.test.one_rmse
                        .map((n) => n.toPrecision(3))
                        .join(' / ')}
                    </span>
                    <span>
                      滚动 (C, T, Tc)　
                      {task.result.test.rollout_rmse
                        .map((n) => n.toPrecision(3))
                        .join(' / ')}
                    </span>
                    <span>模型　{task.result.model ?? '验证未通过'}</span>
                    {task.result.validation?.reasons.map((reason) => (
                      <span key={reason}>{reason}</span>
                    ))}
                    <a
                      href={'/api/cstr/training/' + trainId + '/report'}
                      download
                    >
                      导出训练报告
                    </a>
                  </div>
                )}
              </article>
            </div>
          ) : (
            <div className="two-column cstr-layout">
              <aside className="panel">
                {scenarios && <section className="cstr-disturbance-settings" aria-label="工况扰动">
                  <h3>工况扰动</h3>
                  <div className="form-grid">
                    <label className="cstr-model-field">扰动工况
                      <select aria-label="扰动工况" aria-describedby="cstr-preset-hint" value={selectedPreset} onChange={event => setPreset(event.target.value)}>
                        {scenarios.presets.map(p => <option key={p.id} value={p.id}>{p.label}（{p.nominal} → {p.target} {p.unit}）</option>)}
                      </select>
                      <small id="cstr-preset-hint" className="parameter-hint">{parameterHelp.preset}</small>
                    </label>
                    <FormNumber label="持续周期" value={cycles} min={1} max={scenarios.max_cycles} step={1}
                      hint={numberHint('duration', 1, scenarios.max_cycles, cfg.period, cycles)} onChange={setDuration} />
                    {snap && validCycles && <span className="muted">
                      {(cycles * cfg.period).toFixed(1)} 仿真分钟，按当前 {snap.config.speed}× 约 {(cycles * cfg.period * 60 / snap.config.speed).toFixed(1)} 秒
                    </span>}
                  </div>
                  <div className="actions">
                    <button type="button" disabled={busy || !pulse.canApply || !validCycles}
                      onClick={() => void action(() => command('disturbance'))}>施加扰动</button>
                    <button type="button" className="secondary" disabled={busy || !pulse.canClear}
                      onClick={() => void action(() => command('clear_disturbance'))}>提前撤除</button>
                  </div>
                  {pulse.status && <p className="cstr-data-help" role="status">{pulse.status}</p>}
                </section>}
                <h3>闭环参数</h3>
                <div className="form-grid">
                  <label className="cstr-model-field">
                    <span>预测模型</span>
                    <select
                      aria-label="预测模型" aria-describedby="cstr-model-hint"
                      value={model}
                      onChange={(e) => setModel(e.target.value)}
                    >
                      {cfg.models.map((v) => (
                        <option key={v.id} value={v.id}>
                          {v.label}
                        </option>
                      ))}
                    </select>
                    <small id="cstr-model-hint" className="parameter-hint">{parameterHelp.model}</small>
                  </label>
                  {(
                    [
                      ['horizon', '预测时域', 3, 40, 1],
                      ['flow_min', '最小流量 (L/min)', 95, 148.55, 1],
                      ['flow_max', '最大流量 (L/min)', 148.56, 200, 1],
                      ['move_weight', '流量变化权重', 0.005, 5, 0.01],
                      ['speed', '推进倍率', 0.25, 10, 0.25],
                      ['reference', '釜温设定 (K)', 428, 434, 0.25],
                    ] as const
                  ).map(([k, l, lo, hi, st]) => (
                    <FormNumber
                      key={k}
                      label={l}
                      min={lo}
                      max={hi}
                      step={st}
                      value={settings[k]}
                      hint={numberHint(k, lo, hi, cfg.period, settings[k])}
                      onChange={(v) => setSettings({ ...settings, [k]: v })}
                    />
                  ))}
                </div>
                <div className="actions">
                  {Object.entries(cmdLabels).map(([k, v]) => (
                    <button
                      key={k}
                      disabled={busy || !snap || !allowed[state]?.includes(k)}
                      onClick={() => void action(() => command(k))}
                    >
                      {v}
                    </button>
                  ))}
                </div>
                {online && <section className="cstr-online-updates" aria-label="在线模型更新">
                  <h4>在线模型更新</h4>
                  <label className="cstr-auto-toggle">
                    <input type="checkbox" checked={pendingPolicy?.enabled ?? online.enabled} disabled={busy || state === 'ended'}
                      onChange={e => void action(() => setUpdatePolicy(e.target.checked, online.device))} />
                    自动在线更新
                  </label>
                  <p className="cstr-data-help">每新增 {online.batch_intervals} 个运行区间，后台微调一次预测器。验证通过后在下一控制周期切换。</p>
                  <Device label="更新设备" value={pendingPolicy?.device ?? online.device} disabled={busy || state === 'ended'}
                    change={choice => void action(() => setUpdatePolicy(online.enabled, choice))} />
                  <p role="status">{online.status}</p>
                  {online.result && <p className="cstr-data-help">{online.result}</p>}
                  <div className="actions">
                    {!online.enabled && <button type="button" className="secondary"
                      disabled={busy || !online.manual_ready}
                      onClick={() => void action(async () => {
                        await owned(await ensureHandle(), '/updates', { device: online.device })
                      })}>手动更新一次</button>}
                    {snap?.update && <button type="button" className="secondary" disabled={busy}
                      onClick={() => void action(async () => {
                        await owned(await ensureHandle(), '/updates/' + snap.update + '/cancel', {})
                      })}>取消本轮更新</button>}
                  </div>
                </section>}
                {snap && (
                  <a
                    href={'/api/cstr/sessions/' + snap.id + '/export'}
                    download
                  >
                    导出实验记录
                  </a>
                )}
              </aside>
              <article className="panel">
                <h3>在线闭环仿真</h3>
                <header className="module-status-bar cstr-status">
                  <strong><i className={`status-dot ${state}`} />{snap ? status[state] : '正在连接'}</strong>
                  <span>仿真时间 {snap?.latest.t.toFixed(2) ?? '0.00'} min</span>
                  <span>温度偏差 {snap ? (snap.latest.T - snap.latest.T_ref).toFixed(2) : '—'} K</span>
                  <span>{cfg.models.find(entry => entry.id === 'default')?.version === snap?.model ? '默认 CSTR 模型' : '当前模型'}　{snap?.model ?? '—'}</span>
                  <span>
                    求解 {(snap?.latest.solve_ms ?? 0).toFixed(1)} ms
                  </span>
                </header>
                {snap && <CstrProcessDiagram key={'diagram-' + snap.run} display={snap.display}
                  appliedFlow={snap.latest.Qc} running={snap.status === 'running'} />}
                {trialDisplay && (trialDisplay.timing || trialDisplay.metrics) && <section className="module-status-bar cstr-trial-status" aria-label="扰动试验结果">
                  {trialDisplay.timing && <span>{trialDisplay.timing}</span>}
                  {trialDisplay.metrics && <span>{trialDisplay.metrics}</span>}
                </section>}
                {snap && <CstrLiveTrends key={snap.run} session={snap.id} run={snap.run} samples={samples}
                  period={cfg.period} intervals={cfg.trend_intervals ?? 120} />}
                {snap?.error && <p role="alert" className="error">{snap.error}</p>}
                {marked && (
                  <p>
                    {eventLabels[marked.event]} · {marked.t.toFixed(2)} min　
                    {marked.new ?? marked.value ?? marked.reason}
                  </p>
                )}
              </article>
            </div>
          ))}
      </section>
    </Page>
  )
}
