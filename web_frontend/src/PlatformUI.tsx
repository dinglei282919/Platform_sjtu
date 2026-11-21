import { useEffect, useId, useRef, type FocusEvent, type ReactNode } from 'react'
import * as echarts from 'echarts'

export function Page({ title, subtitle, className, children }: {
  title: string
  subtitle?: string
  className?: string
  children: ReactNode
}) {
  const classes = [title.startsWith('风险动态分析') && 'classification-page', className]
    .filter(Boolean).join(' ')
  return <main className={classes || undefined} aria-label={title}>
    {subtitle && <p className="page-subtitle">{subtitle}</p>}
    {children}
  </main>
}

export function FormNumber({ label, value, min, max, step, onChange, onBlur, hint }: {
  label: string
  hint?: string
  value: number
  min: number
  max: number
  step: number
  onChange: (value: number) => void
  onBlur?: (event: FocusEvent<HTMLInputElement>) => void
}) {
  const hintId = useId()
  return <label>{label}
    <input type="number" value={value} min={min} max={max} step={step}
      aria-label={hint ? label : undefined} aria-describedby={hint ? hintId : undefined}
      onChange={event => onChange(Number(event.target.value))} onBlur={onBlur} />
    {hint && <small id={hintId} className="parameter-hint">{hint}</small>}
  </label>
}

export function Chart({ option, height = 330, fill = false, className = '', onClick, onResize }: {
  option: echarts.EChartsOption
  height?: number
  fill?: boolean
  className?: string
  onClick?: (params: any) => void
  onResize?: (width: number, height: number) => void
}) {
  const element = useRef<HTMLDivElement>(null)
  const chart = useRef<echarts.ECharts | null>(null)

  useEffect(() => {
    if (!element.current) return
    const instance = echarts.init(element.current)
    chart.current = instance
    return () => {
      chart.current = null
      instance.dispose()
    }
  }, [])

  useEffect(() => { chart.current?.setOption(option, true) }, [option])

  useEffect(() => {
    const node = element.current
    const instance = chart.current
    if (!node || !instance) return
    const resize = () => {
      instance.resize()
      onResize?.(node.clientWidth, node.clientHeight)
    }
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(resize)
    observer?.observe(node)
    window.addEventListener('resize', resize)
    const frame = requestAnimationFrame(resize)
    return () => {
      cancelAnimationFrame(frame)
      window.removeEventListener('resize', resize)
      observer?.disconnect()
    }
  }, [onResize])

  useEffect(() => {
    const instance = chart.current
    if (!instance || !onClick) return
    instance.on('click', onClick)
    return () => { if (!instance.isDisposed()) instance.off('click', onClick) }
  }, [onClick])

  return <div className={`chart ${className}`.trim()} ref={element} style={fill ? undefined : { height }} />
}

