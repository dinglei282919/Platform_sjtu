import { useCallback, useEffect, useRef, useState } from 'react'

type Point = { x: number; y: number }
type Size = { width: number; height: number }
type View = { mode: 'fit' | 'width' | 'manual'; zoom: number; offset: Point }
type Props = { src?: string; alt: string; placeholder: string; onError: () => void }
const initialView: View = { mode: 'fit', zoom: 1, offset: { x: 0, y: 0 } }
const clampZoom = (value: number) => Math.min(8, Math.max(0.5, value))

export function DnnPlotViewer({ src, alt, placeholder, onError }: Props) {
  const viewport = useRef<HTMLDivElement>(null)
  const drag = useRef<{ id: number; start: Point; offset: Point } | null>(null)
  const [dragging, setDragging] = useState(false)
  const [size, setSize] = useState<Size>({ width: 1, height: 1 })
  const [natural, setNatural] = useState<Size>({ width: 0, height: 0 })
  const [view, setView] = useState<View>(initialView)
  const ready = !!src && natural.width > 0
  const fitScale = Math.max(0.001, Math.min(Math.max(1, size.width - 16) / Math.max(1, natural.width),
    Math.max(1, size.height - 16) / Math.max(1, natural.height)))
  const widthZoom = clampZoom(Math.max(1, size.width - 16) / Math.max(1, natural.width) / fitScale)
  const zoom = view.mode === 'width' ? widthZoom : view.zoom

  useEffect(() => {
    const node = viewport.current
    if (!node) return
    const observer = new ResizeObserver(() => setSize({ width: node.clientWidth, height: node.clientHeight }))
    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  const boundOffset = useCallback((offset: Point, nextZoom: number) => {
    // Leave a visible edge so dragging cannot lose the plot completely.
    const limitX = Math.max(0, (natural.width * fitScale * nextZoom + size.width) / 2 - 40)
    const limitY = Math.max(0, (natural.height * fitScale * nextZoom + size.height) / 2 - 40)
    return { x: Math.max(-limitX, Math.min(limitX, offset.x)), y: Math.max(-limitY, Math.min(limitY, offset.y)) }
  }, [natural, fitScale, size])

  const zoomBy = useCallback((factor: number, anchor: Point = { x: 0, y: 0 }) => {
    if (!ready) return
    setView(previous => {
      const current = previous.mode === 'width' ? widthZoom : previous.zoom
      const next = clampZoom(current * factor)
      const ratio = next / current
      const offset = boundOffset({ x: anchor.x - (anchor.x - previous.offset.x) * ratio,
        y: anchor.y - (anchor.y - previous.offset.y) * ratio }, next)
      return { mode: 'manual', zoom: next, offset }
    })
  }, [ready, widthZoom, boundOffset])

  useEffect(() => {
    const node = viewport.current
    if (!node) return
    const wheel = (event: WheelEvent) => {
      if (!ready || document.activeElement !== node) return
      event.preventDefault()
      const rect = node.getBoundingClientRect()
      const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? size.height : 1)
      zoomBy(Math.exp(-Math.max(-600, Math.min(600, delta)) * 0.0015),
        { x: event.clientX - rect.left - size.width / 2, y: event.clientY - rect.top - size.height / 2 })
    }
    // A native non-passive listener lets the wheel zoom without scrolling the page.
    node.addEventListener('wheel', wheel, { passive: false })
    return () => node.removeEventListener('wheel', wheel)
  }, [ready, size, zoomBy])

  function fit(mode: 'fit' | 'width' = 'fit') {
    setView({ ...initialView, mode })
  }

  return <div className="dnn-plot-viewer">
    <div className="dnn-plot-toolbar" role="group" aria-label="图表查看工具">
      <button type="button" aria-label="图表缩小" onClick={() => zoomBy(1 / 1.2)} disabled={!ready || zoom <= 0.5}>缩小</button>
      <button type="button" aria-label="图表放大" onClick={() => zoomBy(1.2)} disabled={!ready || zoom >= 8}>放大</button>
      <button type="button" onClick={() => fit()} disabled={!ready}>适应窗口</button>
      <button type="button" onClick={() => fit('width')} disabled={!ready}>适应宽度</button>
      <output aria-label="图表缩放比例">{Math.round(zoom * 100)}%</output>
      <span>点击图表后滚轮缩放 · 按住左键拖动 · 双击适应窗口</span>
    </div>
    <div ref={viewport} className={`dnn-plot-viewport${dragging ? ' is-dragging' : ''}`} role="region" aria-label="可缩放图表" tabIndex={0}
      onDoubleClick={() => fit()}
      onPointerDown={event => {
        if (!ready || event.button !== 0) return
        event.preventDefault()
        event.currentTarget.focus()
        event.currentTarget.setPointerCapture(event.pointerId)
        drag.current = { id: event.pointerId, start: { x: event.clientX, y: event.clientY }, offset: view.offset }
        setDragging(true)
      }}
      onPointerMove={event => {
        const active = drag.current
        if (!active || active.id !== event.pointerId) return
        setView({ mode: 'manual', zoom, offset: boundOffset({ x: active.offset.x + event.clientX - active.start.x,
          y: active.offset.y + event.clientY - active.start.y }, zoom) })
      }}
      onPointerUp={event => {
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId)
        drag.current = null
        setDragging(false)
      }}
      onLostPointerCapture={() => { drag.current = null; setDragging(false) }}
      onPointerCancel={() => { drag.current = null; setDragging(false) }}
      onKeyDown={event => {
        if (!ready) return
        if (['+', '=', '-', '0', 'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) event.preventDefault()
        if (event.key === '+' || event.key === '=') zoomBy(1.2)
        else if (event.key === '-') zoomBy(1 / 1.2)
        else if (event.key === '0') fit()
        else if (event.key.startsWith('Arrow')) {
          const x = event.key === 'ArrowLeft' ? -40 : event.key === 'ArrowRight' ? 40 : 0
          const y = event.key === 'ArrowUp' ? -40 : event.key === 'ArrowDown' ? 40 : 0
          setView({ mode: 'manual', zoom, offset: boundOffset({ x: view.offset.x + x, y: view.offset.y + y }, zoom) })
        }
      }}>
      {src ? <img src={src} alt={alt} draggable={false}
        onLoad={event => {
          const img = event.currentTarget
          setNatural(previous => previous.width === img.naturalWidth && previous.height === img.naturalHeight
            ? previous : { width: img.naturalWidth, height: img.naturalHeight })
        }}
        onError={onError}
        style={{ width: natural.width * fitScale * zoom, height: natural.height * fitScale * zoom,
          transform: `translate(-50%, -50%) translate(${view.offset.x}px, ${view.offset.y}px)`, visibility: ready ? 'visible' : 'hidden' }}/>
        : <span className="dnn-plot-placeholder">{placeholder}</span>}
    </div>
  </div>
}
