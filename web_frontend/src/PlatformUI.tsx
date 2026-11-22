import { useEffect, useId, useRef, useState, type FocusEvent, type ReactNode } from 'react'
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

export function SimulationBar({state, time, busy, command}: {
  state:string; time?:number; busy:boolean; command:(kind:'pause'|'resume')=>void
}) {
  const button=useRef<HTMLButtonElement>(null)
  useEffect(()=>{
    function shortcut(event:KeyboardEvent){
      if(event.key!=='Pause' || event.repeat || event.defaultPrevented || event.ctrlKey || event.altKey || event.metaKey ||
        !button.current || button.current.disabled)return
      event.preventDefault()
      button.current.click()
    }
    window.addEventListener('keydown',shortcut)
    return ()=>window.removeEventListener('keydown',shortcut)
  },[])
  const paused=state==='paused'
  const resume=paused || state==='ready'
  const title=paused?'已暂停':state==='running'?'运行中':state==='ended'?'已结束':'就绪'
  return <div className="cstr-simulation-bar" role="group" aria-label="当前仿真">
    <span>{title}　仿真时间 {time?.toFixed(1) ?? '—'} min</span>
    <button ref={button} type="button" className="secondary" aria-keyshortcuts="Pause" title="暂停或继续当前仿真（Pause 键）"
      disabled={busy || !(resume || state==='running')} onClick={()=>command(resume?'resume':'pause')}>
      {resume?'继续仿真':'暂停仿真'}</button>
  </div>
}

type NumberFieldProps = {
  label: string
  hint?: string
  describedBy?: string
  aside?: string
  value: number
  min: number
  max: number
  step: number | 'any'
  onChange: (value: number) => void
  onBlur?: (event: FocusEvent<HTMLInputElement>) => void
  decimals?: number
  disabled?: boolean
  onPreview?: (value: number | null) => void
  runningValue?: number
}

type NumericEditorProps = Pick<NumberFieldProps, 'value' | 'min' | 'max' | 'step' | 'onChange' | 'disabled' | 'onPreview'> &
  { decimals: number; 'aria-label'?: string; 'aria-describedby'?: string }

function NumericEditor({ value, min, max, step, decimals, disabled, onChange, onPreview, ...aria }: NumericEditorProps) {
  const [text, setText] = useState<string | null>(null)
  const rejected = useRef(false)
  function checkInsertion(event: { preventDefault(): void }, inserted: string) {
    // A native number input can silently discard letters from an otherwise valid prefix.
    if (/[^0-9.eE+-]/.test(inserted)) {
      event.preventDefault()
      rejected.current = true
      onPreview?.(null)
    }
  }
  function read(input: HTMLInputElement) {
    const n = input.valueAsNumber
    return !rejected.current && Number.isFinite(n) && n >= min && n <= max && Number(n.toFixed(decimals)) === n ? n : null
  }
  return <input type="number" {...aria} value={text ?? value} min={min} max={max} step={step} disabled={disabled}
    onBeforeInput={event => checkInsertion(event, event.data ?? '')}
    onPaste={event => checkInsertion(event, event.clipboardData.getData('text'))}
    onChange={event => {
      setText(event.currentTarget.value)
      onPreview?.(read(event.currentTarget))
    }}
    onKeyDown={event => {
      if (event.key === 'Delete' || event.key === 'Backspace' || ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'a')) {
        rejected.current = false
      }
      if (event.key === 'Enter' && !event.nativeEvent.isComposing) {
        event.preventDefault()
        event.currentTarget.blur()
      }
    }}
    onBlur={event => {
      const next = read(event.currentTarget)
      if (text !== null && next !== null && next !== value) onChange(next)
      event.currentTarget.value = String(next ?? value)
      rejected.current = false
      setText(null)
      onPreview?.(null)
    }} />
}

export function FormNumber({ label, value, min, max, step, onChange, onBlur, hint, describedBy, aside, decimals,
  disabled, onPreview, runningValue }: NumberFieldProps) {
  const hintId = useId()
  const [preview, setPreview] = useState<number | null>(null)
  const pending = runningValue !== undefined && runningValue !== (preview ?? value)
  const description = [describedBy, hint && hintId, aside && hintId + '-aside', pending && hintId + '-running'].filter(Boolean).join(' ')
  const aria = { 'aria-label': decimals !== undefined || description ? label : undefined, 'aria-describedby': description || undefined }
  return <label className={aside ? 'parameter-with-aside' : undefined}>{label}
    {decimals === undefined ? <input type="number" {...aria} value={value} min={min} max={max} step={step}
      onChange={event => onChange(Number(event.target.value))} onBlur={onBlur} /> :
      <NumericEditor {...{ value, min, max, step, decimals, disabled, onChange }} {...aria}
        onPreview={next => { setPreview(next); onPreview?.(next) }} />}
    {aside && <span id={hintId + '-aside'} className="parameter-aside">{aside}</span>}
    {hint && <small id={hintId} className="parameter-hint">{hint}</small>}
    {pending && <small id={hintId + '-running'} className="parameter-hint">当前运行值 {runningValue}，修改待应用</small>}
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

