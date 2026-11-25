import { useEffect, useMemo, useRef, useState, type PointerEvent } from 'react'
import { getInstanceByDom, type EChartsOption, type LineSeriesOption } from 'echarts'
import { SampleChart } from './SampleChart'
import { api } from './api'
import type { Point } from './CstrWorkspace'

const timeIntervals = [15, 30, 60, 120, 240, 480]
const liveGrid = { top: 36, right: 16, bottom: 48, left: 72 }
const maxVerticalZoom = 10
type Metric = 'temperature' | 'flow'
type VerticalZoom = { level: number; min: number; max: number }
type TimeWindow = { span: number; pinned: number | null }
type PlotDrag = {
  pointer: number; x: number; y: number; width: number; height: number
  first: number; span: number; pinned: number | null; movedX: boolean; movedY: boolean
  metric: Metric; startY: VerticalZoom; previousY?: VerticalZoom
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

// Keep one real sample within a manually positioned window, including at startup.
function manualStart(first: number, span: number, latest: number) {
  return clamp(Math.round(first), 1 - span, Math.max(0, latest - 1))
}

function scaleTime(window: TimeWindow, span: number, anchor: number, latest: number): TimeWindow {
  if (window.pinned === null) return { span, pinned: null }
  const start = window.pinned + anchor * (window.span - span)
  return { span, pinned: manualStart(start, span, latest) }
}

function plotPosition(target: EventTarget | null, clientX: number, clientY: number) {
  if (!(target instanceof Element)) return
  const element = target.closest<HTMLDivElement>('.chart')
  if (!element) return
  const metric = element.closest<HTMLElement>('[data-metric]')?.dataset.metric
  if (metric !== 'temperature' && metric !== 'flow') return
  const rect = element.getBoundingClientRect()
  const width = rect.width - liveGrid.left - liveGrid.right
  const height = rect.height - liveGrid.top - liveGrid.bottom
  const x = (clientX - rect.left - liveGrid.left) / width
  const y = (rect.bottom - liveGrid.bottom - clientY) / height
  if (width > 0 && height > 0 && x >= 0 && x <= 1 && y >= 0 && y <= 1) {
    return { element, metric: metric as Metric, y, width, height }
  }
}

function verticalRange(element: HTMLDivElement, current?: VerticalZoom): VerticalZoom | undefined {
  if (current) return current
  const chart = getInstanceByDom(element)
  if (!chart) return
  const min = Number(chart.convertFromPixel({ yAxisIndex: 0 }, element.clientHeight - liveGrid.bottom))
  const max = Number(chart.convertFromPixel({ yAxisIndex: 0 }, liveGrid.top))
  if (Number.isFinite(min) && Number.isFinite(max) && max > min) return { min, max, level: 0 }
}

export function TrendChart({ title, series, axis, rows, domain, comparison, bands, metric, vertical }: {
  title: string
  series: readonly [string, string]
  axis: string
  rows: number[][]
  domain?: readonly [number, number]
  comparison?: number[][]
  bands?: number[][]
  metric?: Metric
  vertical?: VerticalZoom
}) {
  const [first, second] = series
  const samples = useMemo(() => rows.filter(row => Number.isFinite(row[1])), [rows])
  const option = useMemo<EChartsOption>(() => {
    const lines: LineSeriesOption[] = [first, second].map((name, index) => ({
      name, type: 'line', showSymbol: rows.length === 1,
      lineStyle: { type: index === 0 ? 'solid' : 'dashed' },
      data: rows.map(row => [row[0], row[index + 1]]),
    }))
    if (bands?.length) {
      lines[0].markArea = { silent: true, label: { show: false }, itemStyle: { color: 'rgba(237,185,103,.17)' },
        data: bands.map(([from, to]) => [{ xAxis: from }, { xAxis: to }]) }
    }
    if (comparison?.length) {
      lines.push({ name: '恒流对照', type: 'line', showSymbol: false, connectNulls: false,
        lineStyle: { type: 'dotted', width: 2 }, data: comparison })
    }
    return {
      animation: false,
      color: ['#69baff', '#f4bd73', '#c6a6ff'],
      legend: { top: 0, textStyle: { color: '#d7e7f8' } },
      grid: domain ? liveGrid : { top: 32, left: 8, right: 14, bottom: 35, containLabel: true },
      xAxis: { type: 'value', scale: true, min: domain?.[0], max: domain?.[1],
        interval: domain ? (domain[1] - domain[0]) / 6 : undefined, name: axis, nameLocation: 'middle', nameGap: 24,
        nameTextStyle: { color: '#b9cee2' }, axisLabel: { color: '#b9cee2', hideOverlap: true },
        axisLine: { lineStyle: { color: '#466385' } }, splitLine: { show: false } },
      yAxis: { type: 'value', scale: true, splitNumber: 4, min: vertical?.min, max: vertical?.max,
        axisLine: domain ? { onZero: false } : undefined,
        axisLabel: { color: '#b9cee2', hideOverlap: true, formatter: vertical
          ? (v: number) => Number(v.toFixed(clamp(1 - Math.floor(Math.log10((vertical.max - vertical.min) / 4)), 0, 7))).toString()
          : undefined },
        splitLine: { lineStyle: { color: '#30465d' } } },
      series: lines,
    }
  }, [first, second, axis, rows, domain, comparison, bands, vertical])

  return <section aria-label={title} data-metric={metric}>
    <h4>{title}{vertical && <span className="cstr-y-scale">
      {vertical.level ? '纵轴 ×' + 2 ** vertical.level : '纵轴已平移'}</span>}</h4>
    <SampleChart option={option} height={205} fill={!!metric} samples={samples} timeOf={row=>row[0]}
      epoch={axis==='轮次'} unit={metric==='temperature' ? 'K' : metric==='flow' ? 'L/min' : ''}
      readingsOf={row=>{
        const values: [string, number][] = [[first, row[1]], [second, row[2]]]
        const hold = comparison?.find(([t, v]) => Math.abs(t-row[0]) < 1e-9 && Number.isFinite(v))
        if (hold) values.push(['恒流对照', hold[1]])
        return values
      }}/>
  </section>
}


// Derive intervals from completed samples so the same marks survive history reads.
function disturbanceOverlay(rows: Point[], period: number) {
  const bands: number[][] = [], temperature: number[][] = [], flow: number[][] = []
  let previous: Point | undefined
  for (const row of rows) {
    const adjacent = !!previous && previous.seq + 1 === row.seq
    const sameTrial = adjacent && previous?.trial_id === row.trial_id
    if (row.disturbance_phase === 'active') {
      if (sameTrial && previous?.disturbance_phase === 'active') bands[bands.length - 1][1] = row.t
      else bands.push([Math.max(0, row.t - period), row.t])
    }
    if (row.trial_id && row.T_hold != null && row.Qc_hold != null) {
      if (!sameTrial || previous?.T_hold == null) {
        if (temperature.length) {
          temperature.push([row.t - period, NaN])
          flow.push([row.t - period, NaN])
        }
        if (adjacent && previous) {
          temperature.push([previous.t, previous.T])
          flow.push([previous.t, row.Qc_hold])
        }
      }
      temperature.push([row.t, row.T_hold])
      flow.push([row.t, row.Qc_hold])
    } else if (previous?.T_hold != null) {
      temperature.push([row.t, NaN])
      flow.push([row.t, NaN])
    }
    previous = row
  }
  return { bands, temperature, flow }
}

function signalRows(rows: Point[], measured: 'T' | 'Qc', target: 'T_ref' | 'Qc_cmd') {
  const data: number[][] = []
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i]
    if (i && row.seq > rows[i - 1].seq + 1) data.push([row.t, NaN, NaN])
    data.push([row.t, row[measured], row[target]])
  }
  return data
}

export function CstrLiveTrends({ session, run, samples, period, intervals }: {
  session: string
  run: string
  samples: Point[]
  period: number
  intervals: number
}) {
  const defaultSpan = timeIntervals.includes(intervals) ? intervals : 120
  const [{ span, pinned }, setWindow] = useState<TimeWindow>({ span: defaultSpan, pinned: null })
  const [vertical, setVertical] = useState<Partial<Record<Metric, VerticalZoom>>>({})
  const [archive, setArchive] = useState<{ first: number; last: number; rows: Point[] } | null>(null)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  const plots = useRef<HTMLDivElement>(null)
  const drag = useRef<PlotDrag | null>(null)
  const wheelMotion = useRef({ mode: '', delta: 0, at: 0 })
  const latest = samples[samples.length - 1]?.seq ?? 0
  const first = pinned === null ? Math.max(0, latest - span) : manualStart(pinned, span, latest)
  const earliestStart = 1 - span
  const latestStart = Math.max(0, latest - 1)
  const last = first + span
  const archiveFirst = Math.max(0, first)
  const availableLast = Math.min(last, latest)
  const archiveLast = Math.min(availableLast, (samples[0]?.seq ?? 0) - 1)
  const needsArchive = archiveFirst <= archiveLast
  const cached = !!archive && archive.first <= archiveFirst && archive.last >= archiveLast
  const domain = useMemo<readonly [number, number]>(() => [first * period, last * period], [first, last, period])

  useEffect(() => {
    setError('')
    if (!needsArchive || cached) return
    const controller = new AbortController()
    const timer = window.setTimeout(async () => {
      try {
        // Caching the complete available window keeps a pinned view stable as the live ring advances.
        const query = new URLSearchParams({ run, first: String(archiveFirst), count: String(availableLast - archiveFirst + 1) })
        const result = await api<{ run: string; samples: Point[] }>(
          '/cstr/sessions/' + session + '/history?' + query, { signal: controller.signal })
        if (!controller.signal.aborted && result.run === run) {
          setArchive({ first: archiveFirst, last: availableLast, rows: result.samples })
        }
      } catch (e) {
        if (!controller.signal.aborted) setError(e instanceof Error ? e.message : String(e))
      }
    }, 70)
    return () => { clearTimeout(timer); controller.abort() }
  }, [needsArchive, cached, archiveFirst, availableLast, run, session, attempt])
  const rows = useMemo(() => {
    const recent = samples.filter(p => p.seq >= first && p.seq <= last)
    if (!needsArchive || !archive) return recent
    const older = archive.rows.filter(p => p.seq >= archiveFirst && p.seq <= archiveLast)
    return older.concat(recent)
  }, [needsArchive, archive, archiveFirst, archiveLast, first, last, samples])
  const temperature = useMemo(() => signalRows(rows, 'T', 'T_ref'), [rows])
  const flow = useMemo(() => signalRows(rows, 'Qc', 'Qc_cmd'), [rows])
  const overlay = useMemo(() => disturbanceOverlay(rows, period), [rows, period])

  // Only modified wheel events are consumed; the native page scroll remains available over each plot.
  useEffect(() => {
    const node = plots.current
    if (!node) return
    function wheel(event: WheelEvent) {
      if (!event.shiftKey || event.ctrlKey || event.metaKey || event.altKey || drag.current) return
      const position = plotPosition(event.target, event.clientX, event.clientY)
      if (!position) return
      const { element, metric, y, height } = position
      const raw = event.deltaY || event.deltaX
      if (!raw) return
      const direction = Math.sign(raw)
      const newLevel = (vertical[metric]?.level ?? 0) - direction
      if (newLevel < 0 || newLevel > maxVerticalZoom) return
      const range = verticalRange(element, vertical[metric])
      if (!range) return
      event.preventDefault()
      const delta = raw * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? height : 1)
      const motion = wheelMotion.current
      if (motion.mode !== metric || event.timeStamp - motion.at > 200 || Math.sign(motion.delta) !== direction) motion.delta = 0
      motion.mode = metric
      motion.at = event.timeStamp
      motion.delta += delta
      if (Math.abs(motion.delta) < 32) return
      motion.delta = 0
      if (newLevel === 0) {
        setVertical(current => ({ ...current, [metric]: undefined }))
        return
      }
      const { min, max } = range
      const anchor = min + y * (max - min)
      const factor = direction < 0 ? .5 : 2
      setVertical(current => ({ ...current, [metric]: {
        level: newLevel, min: anchor + factor * (min - anchor), max: anchor + factor * (max - anchor),
      } }))
    }
    node.addEventListener('wheel', wheel, { passive: false, capture: true })
    return () => node.removeEventListener('wheel', wheel, true)
  }, [vertical])

  function pin(start: number | null) {
    setWindow(current => ({ ...current, pinned: start }))
  }
  function begin(event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0 || event.pointerType !== 'mouse') return
    const position = plotPosition(event.target, event.clientX, event.clientY)
    if (!position) return
    const { element, width, height, metric } = position
    const previousY = vertical[metric]
    const startY = verticalRange(element, previousY)
    if (!startY) return
    drag.current = {
      pointer: event.pointerId, x: event.clientX, y: event.clientY, width, height,
      first, span, pinned, movedX: false, movedY: false, metric, startY, previousY,
    }
    element.setPointerCapture(event.pointerId)
  }
  function move(event: PointerEvent<HTMLDivElement>) {
    const start = drag.current
    if (!start || event.pointerId !== start.pointer) return
    const dx = start.x - event.clientX
    const dy = event.clientY - start.y
    // Use the scales at pointer-down so subsequent live samples cannot move the drag anchor.
    if (Math.abs(dx) >= 3 || start.movedX) {
      const offset = Math.round(dx * start.span / start.width)
      const next = manualStart(start.first + offset, start.span, latest)
      if (next !== start.first || start.movedX) {
        start.movedX = true
        pin(next === start.first ? start.pinned : next)
      }
    }
    if (Math.abs(dy) >= 3 || start.movedY) {
      start.movedY = true
      const { min, max, level } = start.startY
      const offset = dy * (max - min) / start.height
      const bounds = dy === 0 ? start.previousY : { min: min + offset, max: max + offset, level }
      setVertical(current => ({ ...current, [start.metric]: bounds }))
    }
  }
  function end(event: PointerEvent<HTMLDivElement>) {
    if (drag.current?.pointer === event.pointerId) drag.current = null
  }
  function resetScale(follow = false) {
    drag.current = null
    setWindow(current => follow ? { span: defaultSpan, pinned: null } : scaleTime(current, defaultSpan, .5, latest))
    setVertical({})
  }
  return <section aria-label="在线趋势回看">
    <div className="cstr-history-tools">
      <span>{pinned === null ? '跟随实时' : '历史回看'}　{domain[0].toFixed(1)}–{domain[1].toFixed(1)} min</span>
      <progress aria-label="历史时间窗口位置" max={latestStart - earliestStart}
        value={samples.length ? first - earliestStart : 0} />
      <label className="cstr-window-size">时间窗口
        <select aria-label="时间窗口" value={span} onChange={e => {
          const next = Number(e.currentTarget.value)
          setWindow(current => scaleTime(current, next, .5, latest))
        }}>
          {timeIntervals.map(size => <option key={size} value={size}>{Number((size * period).toFixed(2))} min</option>)}
        </select>
      </label>
      <button type="button" className="small secondary" onClick={() => resetScale()}
        disabled={span === defaultSpan && !vertical.temperature && !vertical.flow}>重置比例尺</button>
      <button type="button" className="small secondary" disabled={pinned === null} onClick={() => pin(null)}>回到实时</button>
    </div>
    <p className="cstr-chart-help">在绘图区内按住左键拖动可平移曲线，双击恢复默认实时视图。滚轮滚动页面，Shift + 滚轮缩放当前纵轴。</p>
    {error && <div role="alert" className="error">{error} <button onClick={() => setAttempt(n => n + 1)}>重试</button></div>}
    <div className="cstr-plots cstr-history-plots" ref={plots} onPointerDown={begin} onPointerMove={move}
      onPointerUp={end} onPointerCancel={end} onLostPointerCapture={end}
      onDoubleClick={event => {
        if (plotPosition(event.target, event.clientX, event.clientY)) resetScale(true)
      }}>
      <TrendChart title="釜内温度 (K)" series={['实际釜温', '设定值']} axis="仿真时间 (min)" rows={temperature} domain={domain}
        metric="temperature" vertical={vertical.temperature} comparison={overlay.temperature} bands={overlay.bands} />
      <TrendChart title="冷却液流量 (L/min)" series={['实际施加', '控制指令']} axis="仿真时间 (min)" rows={flow} domain={domain}
        metric="flow" vertical={vertical.flow} comparison={overlay.flow} bands={overlay.bands} />
    </div>
    {!!(overlay.bands.length || overlay.temperature.length) && <p className="muted cstr-pulse-key">
      阴影为扰动施加时段，恒流对照保持施加前的冷却液流量。
    </p>}
  </section>
}
