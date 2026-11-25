import { useEffect, useId, useRef, useState } from 'react'
import { CstrLiveTrends, TrendChart, type Point } from './CstrLiveTrends'
import { SimulationLayout } from './SimulationLayout'
import { CstrProcessDiagram } from './CstrProcessDiagram'
import { pulseControls, pulseDurationHint, speedLimits, type PulseReceipt } from './cstrDisturbance'
import { FormNumber, Page, SimulationBar, StageSwitch } from '../PlatformUI'
import parameterHelp from '../../../assets/cstr/parameter-help.json'
import { api, ApiError, post } from '../api'
import { useTask } from '../useTask'
import { useStoredState } from '../taskSession'
import { dnnOwner, emptyHandle } from './cstrOwner'

export const DNN_TITLE = '深度神经网络优化控制'

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
  control_display?: { active: boolean; timing: string; message: string }
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
async function loadConfig(): Promise<Config> {
  const config = await api<Config | null>('/cstr/config')
  if (!config?.settings || !config.training ||
      !Array.isArray(config.models) || !config.models.every(model => typeof model?.id === 'string' && typeof model.label === 'string')) {
    throw new Error('CSTR 配置接口返回了不兼容的数据，请确认本机平台服务版本。')
  }
  return config
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
export function DnnMpcPage({ visible }: { visible: boolean }) {
  const [stage, setStage] = useState(1),
    [cfg, setCfg] = useState<Config | null>(null)
  const [edits, setEdits] = useState<Partial<Settings>>({})
  const [train, setTrain] = useState<Train | null>(null)
  const [speedPreview, setSpeedPreview] = useState<number | null>(null)
  const [cyclesPreview, setCyclesPreview] = useState<number | null>(null)
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
  const commandRevision = useRef(0)
  const runningSettings = snap ? { ...snap.config, reference: snap.reference } : null
  const settings = cfg ? { ...(runningSettings ?? cfg.settings), ...edits } : null
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
  const live = visible && stage === 1
  useEffect(() => {
    setSpeedPreview(null)
    setCyclesPreview(null)
  }, [live, snap?.run])

  function receiveSnapshot(v: Snapshot, replace: boolean) {
    const previous = lastSnapshot.current
    if (previous?.run === v.run) {
      if (previous.seq > v.seq ||
          (previous.events[previous.events.length - 1]?.event_id ?? 0) > (v.events[v.events.length - 1]?.event_id ?? 0)) return
    } else {
      setEdits({})
      const original = cfg?.models.find(m => m.version === v.model)
      if (original) setModel(original.id)
    }
    lastSnapshot.current = v
    setSnap(v)
    showSamples(v, replace)
  }
  const { task, pollingError: trainingError } = useTask<Task>(
    trainId,
    setTrainId,
  )
  const training = Boolean(
    trainId && (!task || ['queued', 'running'].includes(task.status)),
  )
  useEffect(() => { if (visible) setStage(1) }, [visible])
  const trainedModel = task?.status === 'succeeded' ? task.result?.model : undefined
  const [configError, setConfigError] = useState('')
  const [configAttempt, setConfigAttempt] = useState(0)

  useEffect(() => {
    if (!visible) return
    let current = true
    setConfigError('')
    loadConfig().then(config => {
      if (!current) return
      setCfg(config)
      setTrain(previous => previous ?? config.training)
    }).catch(error => {
      if (current) setConfigError(error instanceof Error ? error.message : String(error))
    })
    return () => { current = false }
  }, [visible, configAttempt, trainedModel])
  useEffect(() => {
    if (!visible || !cfg) return
    let active = true
    void ensureHandle().catch(e => { if (active) setMessage(String(e)) })
    return () => { active = false }
  }, [visible, Boolean(cfg)])

  useEffect(() => {
    if (!live || !cfg) return
    let alive = true,
      timer: number,
      cursor = -1,
      run = ''
    async function poll() {
      try {
        const h = await ensureHandle()
        const ticket = commandRevision.current
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
        if (ticket !== commandRevision.current) {
          timer = window.setTimeout(poll, 0)
          return
        }
        const replace = cursor < 0 || v.resync || v.run !== run
        cursor = v.seq
        run = v.run
        receiveSnapshot(v, replace)
        timer = window.setTimeout(poll, 350)
      } catch (e) {
        if (!alive) return
        if (e instanceof ApiError && e.status === 404) {
          saveHandle(emptyHandle)
          lastSnapshot.current = null
          setSnap(null)
          setEdits({})
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
    commandRevision.current++
    if (kind === 'reference' || kind === 'reset') setEdits({})
    const v = await owned<Snapshot>(h, '?after=-1')
    commandRevision.current++
    receiveSnapshot(v, true)
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
  const lastReference = snap?.events.filter(e => e.event.startsWith('reference_')).slice(-1)[0]
  const queuedReference = state !== 'ended' && lastReference?.event === 'reference_queued' &&
    lastReference.value !== snap?.reference ? lastReference.value : undefined
  const allowed: Record<string, string[]> = {
    ready: ['reset', 'model', 'reference', 'end'],
    running: ['model', 'pause', 'reference', 'reset', 'end'],
    paused: ['model', 'resume', 'reference', 'reset', 'end'],
    ended: ['reset'],
  }
  const cmdLabels: Record<string, string> = {
    model: '应用所选模型',
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
  if (!visible) return null

  return (
    <Page title={DNN_TITLE} className="cstr-page">
      <StageSwitch id="cstr-stage" label="深度神经网络优化控制工作流程" tabs={stages} selected={stage} select={setStage}/>
      <section id="cstr-stage-panel" role="tabpanel" aria-labelledby={`cstr-stage-${stage}`} className="cstr-stage">
        {live && cfg && <SimulationBar state={state} time={snap?.latest.t} busy={busy || !snap}
          command={kind=>void action(()=>command(kind))}/>}
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
            <SimulationLayout controls={<>
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
                      decimals={k === 'lr' ? 5 : 0}
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
              </>}>
                <h3>模型训练进度</h3>
                <header className="module-status-bar cstr-status">
                  <strong><i className={`status-dot ${task?.status ?? 'ready'}`} />{status[task?.status ?? 'ready']}</strong>
                  <span>{task?.error ?? task?.progress.message ?? ''}</span>
                </header>
                <progress aria-label="模型训练进度" max={100} value={task?.progress.percent ?? 0} />
                {task?.progress.history?.length ? <TrendChart key={trainId}
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
              </SimulationLayout>
          ) : (
            <SimulationLayout controls={<>
                {scenarios && <section className="cstr-disturbance-settings" aria-label="工况扰动">
                  <h3>工况扰动</h3>
                  <div className="form-grid">
                    <label className="cstr-model-field">扰动工况
                      <select aria-label="扰动工况" aria-describedby="cstr-preset-hint" value={selectedPreset} onChange={event => setPreset(event.target.value)}>
                        {scenarios.presets.map(p => <option key={p.id} value={p.id}>{p.label}（{p.nominal} → {p.target} {p.unit}）</option>)}
                      </select>
                      <small id="cstr-preset-hint" className="parameter-hint">{parameterHelp.preset}</small>
                    </label>
                    <FormNumber label="持续周期" value={cycles} min={1} max={scenarios.max_cycles} step={1} decimals={0}
                      onPreview={setCyclesPreview}
                      aside={pulseDurationHint((cyclesPreview ?? cycles) * cfg.period, speedPreview ?? settings.speed, snap?.config.speed)}
                      hint={numberHint('duration', 1, scenarios.max_cycles, cfg.period, cycles)} onChange={setDuration} />
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
                      ['horizon', '预测时域', 3, 40, 1, 0],
                      ['flow_min', '最小流量 (L/min)', 95, 148.55, 1, 2],
                      ['flow_max', '最大流量 (L/min)', 148.56, 200, 1, 2],
                      ['move_weight', '流量变化权重', 0.005, 5, 0.01, 3],
                      ['speed', '推进倍率', ...speedLimits, 0.25, 2],
                      ['reference', '釜温设定 (K)', 428, 434, 0.25, 3],
                    ] as const
                  ).map(([k, l, lo, hi, st, decimals]) => (
                    <FormNumber
                      key={`${snap?.run}-${k}-${runningSettings?.[k]}`}
                      label={l}
                      min={lo}
                      max={hi}
                      step={st}
                      decimals={decimals}
                      disabled={!snap || busy}
                      runningValue={runningSettings?.[k]}
                      onPreview={k === 'speed' ? setSpeedPreview : undefined}
                      value={settings[k]}
                      hint={numberHint(k, lo, hi, cfg.period, settings[k]) + (k === 'reference' && queuedReference !== undefined
                        ? '\n' + parameterHelp.referenceQueued.replace('{value}', String(queuedReference)) : '')}
                      onChange={v => setEdits(previous => {
                        const next = { ...previous, [k]: v }
                        if (v === runningSettings?.[k]) delete next[k]
                        return next
                      })}
                    />
                  ))}
                </div>
                <p className="cstr-data-help">{parameterHelp.apply}</p>
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
              </>}>
                <h3>在线闭环仿真</h3>
                <header className="module-status-bar cstr-status">
                  <strong><i className={`status-dot ${state}`} />{snap ? status[state] : '正在连接'}</strong>
                  <span>仿真时间 {snap?.latest.t.toFixed(2) ?? '0.00'} min</span>
                  <span>温度偏差 {snap ? (snap.latest.T - snap.latest.T_ref).toFixed(2) : '—'} K</span>
                  <span>{cfg.models.find(entry => entry.id === 'default')?.version === snap?.model ? '默认 CSTR 模型' : '当前模型'}　{snap?.model ?? '—'}</span>
                  <span>
                    {snap?.control_display?.timing ?? (snap?.latest.solve_ms != null
                      ? `MPC 求解 ${snap.latest.solve_ms.toFixed(1)} ms` : '等待首个控制周期')}
                  </span>
                </header>
                {snap?.control_display?.message && (
                  <p role={snap.control_display.active ? 'alert' : 'status'}
                    className={snap.control_display.active ? 'error' : 'muted'}>
                    {snap.control_display.message}
                  </p>
                )}
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
              </SimulationLayout>
          ))}
      </section>
    </Page>
  )
}
