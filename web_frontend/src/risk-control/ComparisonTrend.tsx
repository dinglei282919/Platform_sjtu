import type { EChartsOption, LineSeriesOption } from 'echarts'
import { SampleChart } from './SampleChart'
import {trendGrid, useTrendGestures, type Range} from './trendGestures'

export type SamplePosition = {run:string; at:number; first?:number}
// Each frame contains 120 sampled intervals. The inspector uses the last recorded sample in the view.
const windowSamples = 120
export const chartHelp = '在绘图区内按住左键拖动可平移曲线，双击回到实时并恢复自动纵轴。滚轮滚动页面，Shift + 滚轮缩放当前纵轴。'

export function comparisonDomain(at:number, period:number, first?:number):Range {
  const start = first ?? Math.max(0,at-windowSamples+1)
  return [start*period,(start+windowSamples)*period]
}

export function pannedPosition(run:string, latest:number, firstMin:number, period:number):SamplePosition {
  const first = Math.max(1-windowSamples, Math.min(Math.max(0,latest),Math.round(firstMin/period)))
  return {run, first, at:Math.min(Math.max(0,latest),first+windowSamples-1)}
}

export type ComparisonLine<Row> = {
  name: string
  color: string
  dashed?: boolean
  reference?: boolean
  read: (row: Row) => number | null
}

type Props<Row> = {
  title: string
  unit: string
  note?: string
  rows: readonly Row[]
  lines: ComparisonLine<Row>[]
  period: number
  intervals: ReadonlyArray<readonly [start: number, stop: number | null]>
  domain?: Range
  vertical?: Range
}

const textStyle = { color: '#a9c1d5', fontSize: 11, hideOverlap: true }

export function ComparisonTrend<Row extends { t: number }>({
  title, unit, note, rows, lines, period, intervals, domain, vertical,
}: Props<Row>) {
  if ((!rows.length && !domain) || !lines.length) return null
  const samples = domain ? rows.filter(row=>row.t>=domain[0]-1e-9 && row.t<=domain[1]+1e-9) : rows
  const first = samples[0]?.t ?? 0
  const last = samples[samples.length - 1]?.t ?? first
  let low = Infinity, high = -Infinity
  const series: LineSeriesOption[] = lines.map(line => ({
    name: line.name, type: 'line', symbol: 'circle', symbolSize: 4, showSymbol: samples.length === 1,
    itemStyle: { color: line.color },
    lineStyle: { color: line.color, type: line.dashed ? 'dashed' : 'solid', width: line.reference ? 1 : 2 },
    z: line.reference ? 1 : 2,
    data: samples.map(row => {
      const value = line.read(row)
      if (value !== null && Number.isFinite(value)) {
        low = Math.min(low, value)
        high = Math.max(high, value)
      }
      return [row.t, value]
    }),
  }))
  // Match the Qt Trend's minimum padding when the process is at steady state.
  const yRange = vertical ?? (low <= high && high - low < 0.001 ? [low - 0.001, high + 0.001] : undefined)
  const visibleBands = intervals
    .filter(([start, stop]) => samples.length && start <= last && (stop === null || stop > first))
    .map(([start, stop]): [{xAxis: number}, {xAxis: number}] => [
      { xAxis: Math.max(first, start) }, { xAxis: Math.min(domain?.[1]??Infinity, last + period, stop ?? last + period) },
    ])
  if (visibleBands.length) {
    series[0].markArea = {
      silent: true, itemStyle: { color: 'rgba(237,185,103,.14)' },
      data: visibleBands,
    }
  }
  const option: EChartsOption = {
    animation: false, color: lines.map(line => line.color), series,
    legend: { top: 1, type: 'scroll', textStyle: {color:'#c9dbe9', fontSize:11} },
    grid: trendGrid,
    xAxis: {
      type: 'value', min: domain?.[0] ?? first, max: domain?.[1] ?? Math.max(first + period, last),
      name: '仿真时间 (min)', nameLocation: 'middle', nameGap: 26,
      axisLabel: textStyle, nameTextStyle: textStyle, splitLine: { show: false },
    },
    yAxis: { type: 'value', scale: true, min:yRange?.[0], max:yRange?.[1], axisLabel: textStyle,
      splitLine: { lineStyle: { color: '#30465d' } } },
  }
  function readings(row: Row): [string, number][] {
    const values: [string, number][] = []
    for (const line of lines) {
      const value = line.read(row)
      if (value !== null) values.push([line.name, value])
    }
    return values
  }
  return <section className="cstr-trend-panel cstr-comparison-trend">
    <h3>{title} ({unit}){vertical && <span className="cstr-y-scale">
      纵轴 {Number(vertical[0].toPrecision(7))}–{Number(vertical[1].toPrecision(7))} {unit}</span>}</h3>
    {note && <p className="cstr-data-help">{note}</p>}
    <SampleChart option={option} height={235} samples={samples} timeOf={row => row.t}
      readingsOf={readings} unit={unit}/>
  </section>
}

export type TrendNavigation = {domain:Range; pan:(firstMin:number)=>void; restore:()=>void}

export function InteractiveComparisonTrend<Row extends {t:number}>(props:Props<Row> & TrendNavigation) {
  const {vertical,plotProps}=useTrendGestures(props.domain, props.pan, props.restore)
  return <div {...plotProps}><ComparisonTrend {...props} vertical={vertical}/></div>
}
