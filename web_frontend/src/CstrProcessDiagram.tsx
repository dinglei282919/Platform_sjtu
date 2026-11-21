/// <reference types="vite/client" />
import { useLayoutEffect, useRef } from 'react'
import processSvg from '../../assets/cstr/process.svg?raw'

const diagram = { __html: processSvg }
const token = /@@([A-Z]+)@@/g
type Label = { node: SVGTextElement; format: string }
type Props = { display: Record<string, string>; appliedFlow: number; running: boolean }

export function CstrProcessDiagram({ display, appliedFlow, running }: Props) {
  const host = useRef<HTMLDivElement>(null)
  const labels = useRef<Label[]>([])
  const liquid = useRef<SVGRectElement | null>(null)

  // Keep one SVG tree for this run so updating a reading never restarts the moving stripes.
  useLayoutEffect(() => {
    if (labels.current.length) return
    const root = host.current!
    labels.current = Array.from(root.querySelectorAll<SVGTextElement>('text'))
      .map(node => ({ node, format: node.textContent ?? '' }))
      .filter(label => label.format.includes('@@'))
    liquid.current = root.querySelector<SVGRectElement>('[fill="@@COLOR@@"]')
    const layer = root.querySelector('#pipe-motion')!
    const pattern = layer.getAttribute('stroke-dasharray')!.split(/\s+/).map(Number)
    const pitch = pattern.reduce((sum, value) => sum + value, 0)
    root.style.setProperty('--cstr-dash-end', -pitch + 'px')
  }, [])

  useLayoutEffect(() => {
    for (const { node, format } of labels.current) {
      const text = format.replace(token, (_, key: string) => display[key] ?? '—')
      if (node.textContent !== text) node.textContent = text
    }
    if (display.COLOR) liquid.current?.setAttribute('fill', display.COLOR)
  }, [display])

  useLayoutEffect(() => {
    const layer = host.current!.querySelector('#pipe-motion')!
    const referenceFlow = Number(layer.getAttribute('data-reference-flow'))
    const coolant = layer.querySelector('[data-stream="coolant"]')!.getAnimations()[0]
    const rate = Number.isFinite(appliedFlow) ? Math.max(0, appliedFlow) / referenceFlow : 0
    coolant?.updatePlaybackRate(rate)
  }, [appliedFlow])

  return <div ref={host} className="cstr-diagram" role="img" aria-label="夹套式 CSTR 实时工艺图"
    aria-description={`釜内温度 ${display.TEMP} K，夹套出口温度 ${display.JACKET} K，实际冷却液流量 ${display.FLOW} L/min`}
    data-running={running} dangerouslySetInnerHTML={diagram} />
}
