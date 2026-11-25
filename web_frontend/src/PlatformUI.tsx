import { useEffect, useId, useRef, useState, type FocusEvent, type KeyboardEvent as ReactKeyboardEvent, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import * as echarts from 'echarts'
import { ErrorDialog } from './ErrorDialog'

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

export function StageSwitch({id, label, tabs, selected, select}: {
  id: string; label: string; tabs: readonly string[]; selected: number; select: (index: number) => void
}) {
  function onKeyDown(event: ReactKeyboardEvent<HTMLButtonElement>, index: number) {
    let next: number
    switch (event.key) {
      case 'ArrowRight': next = (index + 1) % tabs.length; break
      case 'ArrowLeft': next = (index + tabs.length - 1) % tabs.length; break
      case 'Home': next = 0; break
      case 'End': next = tabs.length - 1; break
      default: return
    }
    event.preventDefault()
    select(next)
    document.getElementById(id + '-' + next)?.focus()
  }
  return <div className="view-switches" role="tablist" aria-label={label}>
    {tabs.map((name, index) => <button type="button" role="tab" key={name} id={id + '-' + index}
      aria-controls={id + '-panel'} aria-selected={selected === index} tabIndex={selected === index ? 0 : -1}
      className={selected === index ? 'active' : ''} onClick={() => select(index)}
      onKeyDown={event => onKeyDown(event, index)}>{name}</button>)}
  </div>
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
      {state==='ready'?'开始仿真':resume?'继续仿真':'暂停仿真'}</button>
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
  const [warning, setWarning] = useState('')
  const field = useRef<HTMLInputElement>(null)
  const rejected = useRef(false)
  const pendingKey = useRef<string | null>(null)
  function checkInsertion(event: { preventDefault(): void }, inserted: string) {
    // Permit an unfinished numeric fragment, including a sign or exponent while editing.
    if (!/^[+-]?\d*\.?\d*(?:[eE][+-]?\d*)?$/.test(inserted)) {
      event.preventDefault()
      rejected.current = true
      onPreview?.(null)
    }
  }
  function read(input: HTMLInputElement) {
    const n = input.valueAsNumber
    return !rejected.current && Number.isFinite(n) && n >= min && n <= max && Number(n.toFixed(decimals)) === n ? n : null
  }
  return <><input ref={field} type="number" {...aria} value={text ?? value} min={min} max={max} step={step} disabled={disabled}
    onBeforeInput={event => checkInsertion(event, event.data ?? '')}
    onPaste={event => checkInsertion(event, event.clipboardData.getData('text'))}
    onInput={() => { pendingKey.current = null }}
    onChange={event => {
      setText(event.currentTarget.value)
      onPreview?.(read(event.currentTarget))
    }}
    onKeyDown={event => {
      if (event.key.length === 1 && !event.ctrlKey && !event.altKey && !event.metaKey && !event.nativeEvent.isComposing) {
        checkInsertion(event, event.key)
        pendingKey.current = event.defaultPrevented ? null : event.key
      }
      if (event.key === 'Delete' || event.key === 'Backspace' || ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'a')) {
        rejected.current = false
      }
      if (event.key === 'Enter' && !event.nativeEvent.isComposing) {
        event.preventDefault()
        event.currentTarget.blur()
      }
    }}
    onKeyUp={event => {
      // Chromium silently drops a second decimal point or exponent without an input event.
      if (pendingKey.current === event.key) {
        rejected.current = true
        onPreview?.(null)
      }
      pendingKey.current = null
    }}
    onBlur={event => {
      const input = event.currentTarget
      if (pendingKey.current !== null) rejected.current = true
      pendingKey.current = null
      const next = read(input)
      if (next === null && (text !== null || rejected.current || input.validity.badInput)) {
        const reason = rejected.current || input.validity.badInput ? '请输入有效数字' :
          input.value === '' ? '输入不能为空' : '输入值 '+input.value+' 无效'
        const format = decimals === 0 ? '整数' : '数值，最多 '+decimals+' 位小数'
        setWarning(aria['aria-label']+'\n'+reason+'。允许范围为 '+min+' 至 '+max+'，请填写'+format+'。\n已恢复为上一次确认的值 '+value+'。')
      }
      if (text !== null && next !== null && next !== value) onChange(next)
      input.value = String(next ?? value)
      rejected.current = false
      setText(null)
      onPreview?.(null)
    }} />
    {warning && createPortal(<ErrorDialog title="参数输入无效" message={warning} onClose={() => {
      setWarning('')
      requestAnimationFrame(()=>{field.current?.focus();field.current?.select()})
    }}/>, document.body)}
  </>
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

