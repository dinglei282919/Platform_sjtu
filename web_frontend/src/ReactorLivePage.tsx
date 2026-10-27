import { useEffect, useMemo, useRef, useState } from 'react'
import * as echarts from 'echarts'
import { api, ApiError } from './api'
import { readSession, writeSession } from './taskSession'
import { scrollControlParameters } from './controlInteraction'
import { ReactorDiagram, compositionLabel, validateReactorData, type ReactorData } from './ReactorDiagram'

type AttackSettings = { scale: number; frequency: number; phase: number }
type Command = 'attack' | 'attack_enabled' | 'r1' | 'r2' | 'mode' | 'enabled' | 'pause' | 'resume' | 'reset'
type LiveSnapshot = {
  id: string; status: 'running' | 'paused' | 'error'; t: number
  controls: { r1: number; r2: number; mode: number; enabled: boolean; attack_enabled: boolean; attack_scale: number; attack_frequency: number; attack_phase: number }
  data: ReactorData; events: { time: number; kind: string; value?: unknown }[]; message?: string
}

const sessionKey = 'reactor.live.session'
const resumeKey = 'reactor.live.resumeOnOpen'
let cachedSnapshot: LiveSnapshot | null = null
let requestQueue: Promise<unknown> = Promise.resolve()

// Serializing creation and commands prevents late creation/resume responses from
// leaving a running orphan when React unmounts or switches to the experiment tab.
function enqueue<T>(action: () => Promise<T>): Promise<T> {
  const result = requestQueue.then(action, action)
  requestQueue = result.then(() => undefined, () => undefined)
  return result
}

function remember(snapshot: LiveSnapshot): LiveSnapshot {
  validateReactorData(snapshot.data)
  cachedSnapshot = snapshot
  writeSession(sessionKey, snapshot.id)
  return snapshot
}

async function sendCommand(id: string, kind: Command, value?: number | boolean | AttackSettings): Promise<LiveSnapshot> {
  return remember(await api<LiveSnapshot>(`/reactor/live/${id}/command`, {
    method: 'POST', keepalive: true, body: JSON.stringify({ kind, ...(value === undefined ? {} : { value }) }),
  }))
}

async function ensureSession(): Promise<LiveSnapshot> {
  const id = readSession<string | null>(sessionKey, null)
  if (id) {
    try { return remember(await api<LiveSnapshot>(`/reactor/live/${id}`)) }
    catch (error) {
      if (!(error instanceof ApiError) || error.status !== 404) throw error
      writeSession(sessionKey, null)
      cachedSnapshot = null
    }
  }
  return remember(await api<LiveSnapshot>('/reactor/live', { method: 'POST', body: '{}' }))
}

async function suspendSession(): Promise<void> {
  const id = readSession<string | null>(sessionKey, null)
  if (!id) return
  try { await sendCommand(id, 'pause') }
  catch { /* The server's heartbeat timeout also pauses disconnected sessions. */ }
}

function LiveCurves({ data, events }: { data: ReactorData | null; events: LiveSnapshot["events"] }) {
  const element = useRef<HTMLDivElement>(null)
  const chart = useRef<echarts.ECharts | null>(null)
  useEffect(() => {
    if (!element.current) return
    const instance = echarts.init(element.current)
    chart.current = instance
    const resize = () => instance.resize()
    const observer = new ResizeObserver(resize)
    observer.observe(element.current)
    return () => { observer.disconnect(); instance.dispose(); chart.current = null }
  }, [])
  const option = useMemo<echarts.EChartsOption>(() => {
    const t = data?.t ?? []
    const end = (t.length ? t[t.length-1] : 0)
    const panels = [
      [['xs', 'xs'], ['observer_s', 'xs 估计'], ['received_s', 'xs 接收值']],
      [['xd', 'xd'], ['observer_d', 'xd 估计'], ['received_d', 'xd 接收值']],
      [['u', '控制输入 u']], [['k', 'FDI 增益 κ']],
    ]
    return {
      animation: false, textStyle: { color: '#bdcfe3', fontFamily: 'Microsoft YaHei' },
      tooltip: { trigger: 'axis' },
      grid: panels.map((_, i) => ({ left: 58, right: 20, top: `${i * 25 + 6}%`, height: '14%' })),
      legend: panels.map((panel, i) => ({ top: `${i * 25}%`, data: panel.map(([, label]) => label), formatter: (name: string) => name.replace(/x([sd])/g, '{math|x}{sub|$1}'), textStyle: {color: '#dce8f5', rich: {math: {fontFamily: 'Times New Roman', fontStyle: 'italic',fontSize: 20},sub: {fontFamily: 'Times New Roman',fontSize: 11,padding: [10,0,0,0]}}} })),
      xAxis: panels.map((_, i) => ({ gridIndex: i, type: 'value', min: Math.max(0, end-60), max: Math.max(1,end), axisLabel: {color:'#bdcfe3'} })),
      yAxis: panels.map((_, i) => ({ gridIndex: i, type: 'value', scale: true, axisLabel: {color:'#bdcfe3'}, splitLine: {lineStyle:{color:'#263e58'}} })),
      series: panels.flatMap((panel, i) => panel.map(([key,label], j) => ({ name: label, type: 'line', xAxisIndex: i, yAxisIndex: i,
        showSymbol: false, lineStyle: {type: j ? 'dashed' : 'solid'},
        markLine: j === 0 ? {symbol: 'none', label: {show: i===0, color:'#fbbf24', formatter: 'FDI 变更'}, lineStyle: {color:'#fbbf24',type:'dotted'},
          data: events.filter(e => e.kind.startsWith('attack') && e.time >= (t[0] ?? 0)).slice(-8).map(e => ({xAxis:e.time}))} : undefined,
        data: t.map((time, n) => [time, data?.[key]?.[n] ?? null]) }))),
    } as echarts.EChartsOption
  }, [data, events])
  useEffect(() => { chart.current?.setOption(option, {notMerge: true}) }, [option])
  return <div className="reactor-live-chart" style={{minHeight: 620}} role="img" aria-label="组分 xs、xd 与估计、控制输入 u、FDI 增益曲线" ref={element}/>
}
const eventNames: Record<string, string> = {
  attack: 'FDI 参数', attack_enabled: 'FDI 开关', r1: 'R1 应用并复位', r2: 'R2 应用并复位', mode: '系统模式', enabled: '控制器',
  pause: '暂停', resume: '继续运行', reset: '复位', timeout: '连接超时暂停',
}
const formatValue = (value: unknown) => typeof value === 'boolean' ? value ? '投入' : '退出' : value == null ? '' : typeof value === 'object' ? JSON.stringify(value) : String(value)

export function ReactorLivePage({ active = true }: { active?: boolean }) {
  const [snapshot, setSnapshot] = useState<LiveSnapshot | null>(cachedSnapshot)
  const [connecting, setConnecting] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [retry, setRetry] = useState(0)
  const [r1, setR1] = useState('0.3')
  const [r2, setR2] = useState('0.2')
  const [mode, setMode] = useState('0')
  const [attackForm, setAttackForm] = useState({scale:'1',frequency:'0.25',phase:'0'})
  const lifetime = useRef(0)
  const epoch = useRef(0)
  const commandInFlight = useRef(false)

  useEffect(() => {
    if (!active) return
    const generation = ++lifetime.current
    let disposed = false, timer = 0
    let pollRequest: AbortController | null = null
    const current = () => !disposed && lifetime.current === generation
    const publish = (next: LiveSnapshot, initial = false) => {
      if (!current()) return
      setSnapshot(next); setError('')
      if (initial) {
        setR1(String(next.controls.r1)); setR2(String(next.controls.r2))
        setMode(String(next.controls.mode))
        setAttackForm({scale:String(next.controls.attack_scale),frequency:String(next.controls.attack_frequency),phase:String(next.controls.attack_phase)})
      }
    }
    const schedule = (delay: number) => {
      window.clearTimeout(timer)
      if (current() && !document.hidden) timer = window.setTimeout(() => void poll(), delay)
    }
    const open = async () => {
      setConnecting(true)
      try {
        const next = await enqueue(async () => {
          let value = await ensureSession()
          if (!current() || document.hidden || !readSession(resumeKey, true)) {
            if (value.status === 'running') value = await sendCommand(value.id, 'pause')
          } else if (value.status === 'paused') {
            value = await sendCommand(value.id, 'resume')
          }
          return value
        })
        publish(next, true)
        if (current()) { setConnecting(false); setBusy(false); schedule(200) }
      } catch (caught) {
        if (current()) { setConnecting(false); setError('连接实时系统失败：' + String(caught)) }
      }
    }
    const poll = async () => {
      if (!current() || document.hidden) return
      if (commandInFlight.current) { schedule(200); return }
      const id = readSession<string | null>(sessionKey, null)
      if (!id) { await open(); return }
      const startedEpoch = epoch.current
      pollRequest = new AbortController()
      let delay = 200
      try {
        const next = await api<LiveSnapshot>(`/reactor/live/${id}`, { signal: pollRequest.signal })
        if (current() && startedEpoch === epoch.current) publish(remember(next))
      } catch (caught) {
        if (!current() || pollRequest.signal.aborted) return
        if (caught instanceof ApiError && caught.status === 404) {
          writeSession(sessionKey, null); cachedSnapshot = null
          setNotice('原会话已结束，正在重新建立正常运行状态。')
          await open(); return
        }
        writeSession(resumeKey, false)
        setError('连接暂时中断，正在重试；连接超时后系统会自动暂停。')
        delay = 1000
      }
      schedule(delay)
    }
    const onHidden = () => {
      if (!document.hidden) { schedule(0); return }
      writeSession(resumeKey, false)
      epoch.current += 1
      window.clearTimeout(timer); pollRequest?.abort()
      setNotice('页面已转入后台，系统已请求暂停。返回后请点击“继续运行”。')
      void enqueue(suspendSession).then(() => { if (current() && cachedSnapshot) publish(cachedSnapshot) })
    }
    const onPageHide = () => { writeSession(resumeKey, false); void enqueue(suspendSession) }
    if (document.hidden) writeSession(resumeKey, false)
    document.addEventListener('visibilitychange', onHidden)
    window.addEventListener('pagehide', onPageHide)
    void open()
    return () => {
      disposed = true; lifetime.current += 1; epoch.current += 1
      window.clearTimeout(timer); pollRequest?.abort()
      document.removeEventListener('visibilitychange', onHidden)
      window.removeEventListener('pagehide', onPageHide)
      void enqueue(suspendSession)
    }
  }, [active, retry])

  async function command(kind: Command, value?: number | boolean | AttackSettings) {
    const id = snapshot?.id
    if (!id || busy || connecting || commandInFlight.current) return
    const generation = lifetime.current
    if (kind === 'pause' || kind === 'resume') writeSession(resumeKey, kind === 'resume')
    commandInFlight.current = true
    epoch.current += 1; setBusy(true); setError('')
    try {
      const next = await enqueue(() => sendCommand(id, kind, value))
      if (generation !== lifetime.current) return
      setSnapshot(next)
      setNotice(`${eventNames[kind]}指令已提交，实际状态以实时读数为准。`)
      if (['reset', 'r1', 'r2'].includes(kind)) setMode('0')
      if (kind === 'reset') { setR1(String(next.controls.r1)); setR2(String(next.controls.r2)) }
    } catch (caught) {
      if (kind === 'pause' || kind === 'resume') writeSession(resumeKey, false)
      if (generation === lifetime.current) setError(String(caught))
    } finally {
      commandInFlight.current = false
      if (generation === lifetime.current) setBusy(false)
    }
  }
  function applyNumber(kind: 'r1' | 'r2', text: string) {
    const value = Number(text)
    if (!text.trim() || !Number.isFinite(value) || value < 0 || value > (kind === 'r1' ? 1 : .95)) {
      setError('R1 范围 [0,1]，R2 范围 [0,0.95]；R2=1 使坐标变换退化。'); return
    }
    void command(kind, value)
  }

  function applyAttack(preset?: AttackSettings) {
    const params = preset ?? {scale: Number(attackForm.scale), frequency: Number(attackForm.frequency), phase: Number(attackForm.phase)}
    if ((!preset && Object.values(attackForm).some(v => !v.trim())) || !Object.values(params).every(Number.isFinite)
      || params.scale < 0 || params.scale > 1.8 || params.frequency < .01 || params.frequency > 2 || Math.abs(params.phase) > Math.PI) {
      setError('FDI 强度 a: 0～1.8；角频率 ω: 0.01～2；相位 φ: −π～π。'); return
    }
    if (preset) setAttackForm({scale:String(params.scale),frequency:String(params.frequency),phase:String(params.phase)})
    void command('attack', params)
  }

  const data = snapshot?.data ?? null
  const index = Math.max(0, (data?.t.length ?? 1) - 1)
  const running = snapshot?.status === 'running' && !connecting
  const unavailable = connecting || busy || !snapshot
  const value = (key: string) => data?.[key]?.[index]?.toPrecision(4) ?? '—'
  return <main className="reactor-page control-module reactor-live-page" onWheelCapture={scrollControlParameters} aria-label="化学反应器实时控制系统">
    <div className="reactor-live-layout">
      <section className="panel reactor-live-controls">
        <div className="reactor-live-controls-heading"><h3>在线调节</h3><p className="muted">实时 FDI 闭环组分调节；修改 R1/R2 后重新开始实验。</p></div>
        <div className="reactor-live-control-bar">
        {([['r1', '循环参数 R1', r1, setR1, 1], ['r2', '循环参数 R2', r2, setR2, .95]] as const).map(([key,label,value,setValue,max]) =>
          <div key={key} className="reactor-live-field"><label htmlFor={'reactor-live-'+key}>{label}</label>
          <div><input id={'reactor-live-'+key} type="number" min={0} max={max} step={.05} value={value} disabled={unavailable} onChange={e => setValue(e.target.value)}/>
          <button disabled={unavailable} onClick={() => applyNumber(key,value)}>应用并复位</button></div><small>当前 {snapshot?.controls[key].toFixed(3) ?? '—'}</small></div>)}
        <div className="reactor-live-field"><label htmlFor="reactor-live-mode">系统模式</label>
          <div><select id="reactor-live-mode" value={mode} disabled={unavailable} onChange={event => setMode(event.target.value)}><option value="0">MATLAB 自动切换</option><option value="1">固定模式 1</option><option value="2">固定模式 2</option></select>
            <button disabled={unavailable} onClick={() => void command('mode', Number(mode))}>应用模式</button></div></div>
        <div className="reactor-live-run-controls"><button className={snapshot?.controls.enabled ? '' : 'primary'} disabled={unavailable} onClick={() => void command('enabled', !snapshot?.controls.enabled)}>{snapshot?.controls.enabled ? '退出控制器' : '投入控制器'}</button>
        <div className="reactor-live-actions"><button disabled={unavailable || snapshot?.status === 'error'} onClick={() => void command(running ? 'pause' : 'resume')}>{running ? '暂停运行' : '继续运行'}</button>
          <button disabled={unavailable} onClick={() => void command('reset')}>按当前设置重跑</button></div>
        </div>
        </div>
        <div className="reactor-attack-controls">
          <h4>FDI 攻击调节</h4>
          <div className="reactor-live-control-bar">
            {([['scale','攻击强度 a',0,1.8,.1],['frequency','角频率 ω / rad·s⁻¹',.01,2,.05],['phase','相位 φ / rad',-Math.PI,Math.PI,.1]] as const).map(([key,label,min,max,step]) =>
              <div key={key} className="reactor-live-field"><label htmlFor={'reactor-attack-'+key}>{label}</label>
                <input id={'reactor-attack-'+key} type="number" min={min} max={max} step={step} value={attackForm[key]} disabled={unavailable} onChange={e => setAttackForm({...attackForm,[key]:e.target.value})}/></div>)}
            <div className="reactor-live-actions"><button disabled={unavailable} onClick={() => applyAttack()}>在线应用 FDI</button>
              <button disabled={unavailable} onClick={() => void command('attack_enabled', !snapshot?.controls.attack_enabled)}>{snapshot?.controls.attack_enabled ? '关闭 FDI' : '开启 FDI'}</button></div>
          </div>
          <div className="reactor-live-actions"><button disabled={unavailable} onClick={() => applyAttack({scale:1,frequency:.25,phase:0})}>参数复位</button><button disabled={unavailable} onClick={() => applyAttack({scale:1.8,frequency:.5,phase:0})}>增强参数</button></div>
          <p className="muted">当前 FDI：{snapshot?.controls.attack_enabled ? '开启' : '关闭'}；a={snapshot?.controls.attack_scale.toFixed(2) ?? '—'}，ω={snapshot?.controls.attack_frequency.toFixed(2) ?? '—'}，φ={snapshot?.controls.attack_phase.toFixed(2) ?? '—'}。在线应用保留状态；可按当前设置复位重跑，比较初始恢复过程。</p>
        </div>
      </section>
      <section className="panel reactor-live-system">
        <header className="reactor-live-header"><div><h3>反应器实时运行</h3><p>{connecting ? '正在连接正常运行状态…' : snapshot?.message || '观察 FDI 反馈下的组分稳定化响应。'}</p></div>
          <span className={`reactor-live-status ${snapshot?.status ?? 'paused'}`} role="status">{connecting ? '连接中' : running ? snapshot?.controls.enabled ? '闭环运行' : '控制器已退出' : snapshot?.status === 'error' ? '运行异常' : '已暂停'}</span></header>
        {error && <div className="error" role="alert">{error}<button onClick={() => { setBusy(false); setRetry(previous => previous + 1) }}>重试连接</button></div>}
        <p className="muted reactor-control-objective">{compositionLabel("控制目标：期望组分 xs、xd 趋于 0。")}</p>
        <div className="reactor-live-readouts"><div><span>运行时间</span><strong>{snapshot?.t.toFixed(2) ?? '—'} s</strong></div><div><span>{compositionLabel("组分 xs")}</span><strong>{value('xs')}</strong></div><div><span>{compositionLabel("组分 xd")}</span><strong>{value('xd')}</strong></div><div><span>控制输入 u</span><strong>{value('u')}</strong></div></div>
        <div className="reactor-live-monitor">
          <div className="reactor-live-diagram"><ReactorDiagram data={data} index={index} animated={active && running}/><p className="muted">控制输入 u 由右侧曲线展示；接收值曲线表示反馈篡改，竖线标记 FDI 调节时刻。</p></div>
          <section className="reactor-live-curves"><h4>实时控制曲线 <small>最近 60 s</small></h4><LiveCurves data={data} events={snapshot?.events ?? []}/></section>
        </div>
        {notice && <p className="reactor-live-notice" role="status">{notice}</p>}
        <details open className="reactor-live-events"><summary>模型与 FDI 注入说明</summary>
          <p>Λ(t)=a[1+1.5 sin(ωt+φ)]，κ(t)=1+Λ(t)。默认 a=1、ω=0.25、φ=0。控制器接收 κx₁、κx̂₂；观测器使用原始输出，u 直接进入对象。乘性攻击在反馈严格为零时不产生额外激励。</p>
          <p>xs=F(1−R2)x₁/(G₁G₂)，xd=Fx₂/G₂；G₁=0.4，G₂=0.5，F=0.5。显示有符号组分，不推定浓度单位或百分比。调节目标为零点邻域。最右侧罐为工艺示意，没有独立的液位、温度或第三组分状态方程。</p>
          <p>在线采用 T=0.6、ell=0.9 的有限增益以支持持续交互；离线保留 MATLAB 的 T=1.5、ell=0.15。固定模式为人工实验扩展；自动切换沿用原时间表。</p>
        </details>
        <details className="reactor-live-events"><summary>调节记录{snapshot?.events.length ? `（${snapshot.events.length}）` : ''}</summary>
          {snapshot?.events.length ? <ol>{snapshot.events.slice(-8).reverse().map((event, eventIndex) => <li key={`${event.time}-${event.kind}-${eventIndex}`}><time>{event.time.toFixed(2)} s</time> {eventNames[event.kind] ?? event.kind}{event.value === undefined ? '' : `：${formatValue(event.value)}`}</li>)}</ol> : <p className="muted">尚无参数调节。</p>}
        </details>
      </section>
    </div>
  </main>
}
