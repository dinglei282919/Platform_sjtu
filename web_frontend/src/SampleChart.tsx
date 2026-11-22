import { useEffect, useRef, useState, type PointerEvent } from 'react'
import { getInstanceByDom, type EChartsOption } from 'echarts'
import { Chart } from './PlatformUI'

type Readings = [name: string, value: number][]
type Selection = { at: number; readings: Readings; left: boolean }

export function SampleChart<T>({ option, height, fill, samples, timeOf, readingsOf, unit = '', epoch = false, digits }: {
  option: EChartsOption; height: number; fill?: boolean
  samples: readonly T[]; timeOf: (row: T) => number; readingsOf: (row: T) => Readings
  unit?: string; epoch?: boolean; digits?: number
}) {
  const host = useRef<HTMLDivElement>(null)
  const guide = useRef<HTMLSpanElement>(null)
  const [selection, select] = useState<Selection | null>(null)
  const grid = Array.isArray(option.grid) ? option.grid[0] : option.grid

  function inspect(event: PointerEvent<HTMLDivElement>) {
    if (event.buttons || event.pointerType === 'touch') return select(null)
    const element = host.current?.querySelector<HTMLElement>('.chart')
    const chart = element && getInstanceByDom(element)
    if (!element || !chart || !samples.length) return select(null)
    const rect = element.getBoundingClientRect()
    const x = event.clientX - rect.left, y = event.clientY - rect.top
    if (!chart.containPixel({gridIndex: 0}, [x, y])) return select(null)
    const axis = {xAxisIndex: 0}
    const at = Number(chart.convertFromPixel(axis, x))
    const margin = Math.abs(Number(chart.convertFromPixel(axis, x + 6)) - at)
    if (!Number.isFinite(at) || at < timeOf(samples[0]) - margin || at > timeOf(samples[samples.length-1]) + margin) return select(null)
    let low = 0, high = samples.length - 1
    while (low < high) {
      const mid = Math.floor((low + high) / 2)
      if (timeOf(samples[mid]) < at) low = mid + 1
      else high = mid
    }
    const row = low > 0 && at - timeOf(samples[low-1]) <= timeOf(samples[low]) - at ? samples[low-1] : samples[low]
    const time = timeOf(row), left = x > rect.width / 2
    if (selection?.at === time && selection.left === left) return
    const readings = readingsOf(row).filter(([, value]) => Number.isFinite(value))
    select(readings.length ? {at: time, readings, left} : null)
  }

  // Keep the sampled values. Only its guide moves as the simulation window advances.
  useEffect(() => {
    const element = host.current?.querySelector<HTMLElement>('.chart')
    const chart = element && getInstanceByDom(element)
    const line = guide.current
    if (!element || !chart || !line || !selection) return
    const drawGuide = () => {
      const x = Number(chart.convertToPixel({xAxisIndex: 0}, selection.at))
      line.hidden = !Number.isFinite(x) || !chart.containPixel({gridIndex: 0}, [x, element.clientHeight / 2])
      if (!line.hidden) line.style.left = x + 'px'
    }
    drawGuide()
    chart.on('rendered', drawGuide)
    return () => { if (!chart.isDisposed()) chart.off('rendered', drawGuide) }
  }, [selection])

  return <div ref={host} className="cstr-sample-chart" onPointerMove={inspect}
    onPointerLeave={() => select(null)} onPointerDown={() => select(null)} onPointerCancel={() => select(null)}>
    <Chart {...{option, height, fill}}/>
    {selection && <>
      <span ref={guide} aria-hidden="true" className="cstr-sample-guide" style={{top:grid?.top, bottom:grid?.bottom}}/>
      <div role="tooltip" className={'cstr-sample-tooltip' + (selection.left ? ' on-left' : '')} style={{top:grid?.top}}>
        <strong>{epoch ? '轮次 ' + selection.at : '仿真时间 ' + selection.at.toFixed(1) + ' min'}</strong>
        <dl>{selection.readings.map(([name, value]) => <div key={name}><dt>{name}</dt>
          <dd>{digits === undefined ? Number(value.toPrecision(7)) : value.toFixed(digits)}{unit && ' ' + unit}</dd></div>)}</dl>
      </div>
    </>}
  </div>
}
