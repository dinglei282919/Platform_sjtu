/// <reference types="vite/client" />
import { useEffect, useState } from 'react'
import template from '../../assets/reactor/reactor_process.svg?raw'

export type ReactorData = Record<string, number[]>

const diagramFields = ['t', 'mode', 'xs', 'xd', 'u', 'r1', 'r2', 'k']

export function sampleIndexFor(data: ReactorData | null, index: number): number {
  return Math.max(0, Math.min(Number.isFinite(index) ? Math.floor(index) : 0, (data?.t?.length ?? 1) - 1))
}

export function validateReactorData(data: ReactorData): void {
  const length = data?.t?.length
  if (!length || diagramFields.some(key => !Array.isArray(data[key]) || data[key].length !== length
      || data[key].some(value => !Number.isFinite(value)))) {
    throw new Error('仿真轨迹缺少有效的工艺显示数据，请检查服务版本或重新运行。')
  }
}

const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, value))
const compact = (value: number) => Number(value.toPrecision(4)).toString()

const flowPaths = [
  [[20, 19], [583, 19], [583, 133]],
  [[20, 60], [244, 60], [244, 133]],
  [[237, 990], [237, 1096], [855, 1096], [855, 244], [1003, 244]],
  [[583, 990], [583, 1065], [787, 1065], [787, 198], [1003, 198]],
  [[1111, 221], [1207, 221], [1304, 268]],
  [[1485, 908], [1485, 1096]],
]

function flowArrows(phase: number): string {
  const pieces: string[] = []
  for (const points of flowPaths) {
    let distance = 24 + phase * 64
    for (let i = 1; i < points.length; i++) {
      const [x, y] = points[i - 1], [endX, endY] = points[i]
      const length = Math.hypot(endX - x, endY - y)
      const dx = (endX - x) / length, dy = (endY - y) / length
      while (distance < length) {
        const cx = x + dx * distance, cy = y + dy * distance
        pieces.push(`M${(cx + 5 * dx).toFixed(3)} ${(cy + 5 * dy).toFixed(3)} L${(cx - 4 * dx - 3 * dy).toFixed(3)} ${(cy - 4 * dy + 3 * dx).toFixed(3)} L${(cx - 4 * dx + 3 * dy).toFixed(3)} ${(cy - 4 * dy - 3 * dx).toFixed(3)} Z`)
        distance += 64
      }
      distance -= length
    }
  }
  return pieces.join(' ')
}

// Only numeric data and fixed labels/colors can enter the bundled SVG template.
// Physical-to-display mappings are calculated by the shared Python core.
export function reactorDiagramTokens(data: ReactorData | null, index: number, phase: number): Record<string, string> {
  const i = sampleIndexFor(data, index)
  const present = !!data && diagramFields.every(key => Number.isFinite(data[key]?.[i]))
  const value = (key: string, fallback: number) => present ? data![key][i] : fallback
  const level1 = clamp(value('level1', .5), 0, 1)
  const level2 = clamp(value('level2', .5), 0, 1)
  const mode = value('mode', 0)
  const signedU = clamp(value('u',0)/20,-1,1)
  const currentPhase = present && Number.isFinite(phase) ? ((phase % 1) + 1) % 1 : 0
  const bladeHalf = 90 + 120 * Math.abs(Math.cos(2 * Math.PI * currentPhase))
  return {
    XS: present ? compact(value('xs', 0)) : '—', XD: present ? compact(value('xd', 0)) : '—',
    KAPPA: present ? compact(value('k', 2)) : '—',
    TIME: present ? value('t', 0).toFixed(3) + ' s' : '等待仿真',
    MODE: present ? String(mode) : '—',
    FLOW1: value('r1', .3).toFixed(3), FLOW2: value('r2', .2).toFixed(3),
    LEVEL1: present ? Math.floor(100 * level1 + .5) + '%' : '—',
    LEVEL2: present ? Math.floor(100 * level2 + .5) + '%' : '—',
    FILL1_Y: (990 - 768 * level1).toFixed(3), FILL1_H: (768 * level1).toFixed(3),
    FILL2_Y: (990 - 768 * level2).toFixed(3), FILL2_H: (768 * level2).toFixed(3),
    SWITCH_Y: String(mode === 1 ? 244 : mode === 2 ? 198 : 221),
    ROUTE1: mode === 1 ? '#2563eb' : '#cbd5e1', ROUTE2: mode === 2 ? '#2563eb' : '#cbd5e1',
    U_BAR_X: (1420 + Math.min(0,signedU)*180).toFixed(3), U_BAR_WIDTH: (Math.abs(signedU)*180).toFixed(3), U_BAR_COLOR: signedU<0 ? '#d97706' : '#167c9d',
    U: present ? compact(value('u', 0)) : '—',
    LIQUID_COLOR: '#b7d9e5', FLOW_OFFSET: (-currentPhase * 48).toFixed(3), FLOW_ARROWS: flowArrows(currentPhase),
    BLADE_HALF: bladeHalf.toFixed(3), BLADE_LEFT: (1485 - bladeHalf).toFixed(3), BLADE_RIGHT: (1485 + bladeHalf).toFixed(3),
  }
}

export function ReactorDiagram({data, index, animated}: {data: ReactorData | null; index: number; animated: boolean}) {
  const [phase, setPhase] = useState(0)
  useEffect(() => { if (!data) setPhase(0) }, [data])
  useEffect(() => {
    if (!animated || !data) return
    const timer = window.setInterval(() => setPhase(previous => (previous + 1 / 120) % 1), 50)
    return () => window.clearInterval(timer)
  }, [animated, data])
  const tokens = reactorDiagramTokens(data, index, phase)
  const svg = template.replace(/__([A-Z0-9_]+)__/g, (_, key: string) => tokens[key] ?? '')
  const description = `化学反应器双路进料工艺框图。时间：${tokens.TIME}；切换模式：${tokens.MODE}；循环流量 R1：${tokens.FLOW1}，R2：${tokens.FLOW2}；组分 xs：${tokens.XS}，xd：${tokens.XD}；控制输入：${tokens.U}；FDI 增益：${tokens.KAPPA}。`
  return <div className="reactor-diagram" role="img" aria-label={description} dangerouslySetInnerHTML={{__html: svg}}/>
}

export function compositionLabel(text: string) {
  return text.split(/(xs|xd)/g).map((part, i) => part === 'xs' || part === 'xd'
    ? <span key={i} style={{fontFamily: 'Cambria, Times New Roman, serif', fontSize: '1.25em'}}><i>x</i><sub style={{fontSize: '.62em'}}>{part[1]}</sub></span>
    : part)
}
