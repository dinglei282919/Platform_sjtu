import {useEffect, useRef, useState, type MouseEvent, type PointerEvent} from 'react'
import {getInstanceByDom} from 'echarts'

export const trendGrid = {left:68, right:18, top:46, bottom:42}
export type Range = [number, number]

export function useTrendGestures(domain:Range, pan:(firstMin:number)=>void, restore:()=>void) {
  const host = useRef<HTMLDivElement>(null)
  const drag = useRef<{pointer:number; x:number; y:number; domain:Range; range?:Range; width:number; height:number} | null>(null)
  const [vertical,setVertical] = useState<Range>()

  function position(x:number, y:number) {
    const element = host.current?.querySelector<HTMLElement>('.chart')
    if (!element) return
    const rect = element.getBoundingClientRect()
    const width = rect.width-trendGrid.left-trendGrid.right, height = rect.height-trendGrid.top-trendGrid.bottom
    const fx = (x-rect.left-trendGrid.left)/width, fy = (rect.bottom-trendGrid.bottom-y)/height
    if (width>0 && height>0 && fx>=0 && fx<=1 && fy>=0 && fy<=1) return {element,width,height,fy}
  }
  function range(element:HTMLElement):Range|undefined {
    const chart = getInstanceByDom(element)
    if (!chart) return
    const low = Number(chart.convertFromPixel({yAxisIndex:0},element.clientHeight-trendGrid.bottom))
    const high = Number(chart.convertFromPixel({yAxisIndex:0},trendGrid.top))
    if (Number.isFinite(low) && Number.isFinite(high) && high>low) return [low,high]
  }
  useEffect(()=>{
    const node=host.current!
    function wheel(e:WheelEvent) {
      if (!e.shiftKey || e.ctrlKey || e.altKey || e.metaKey || drag.current) return
      const pos=position(e.clientX,e.clientY), delta=e.deltaY || e.deltaX
      const bounds=pos && range(pos.element)
      if (!pos || !bounds || !delta) return
      const size=(bounds[1]-bounds[0])*(delta<0 ? .5 : 2)
      const low=bounds[0]+pos.fy*(bounds[1]-bounds[0]-size)
      if (Number.isFinite(low+size) && size>1e-12) {
        e.preventDefault()
        setVertical([low,low+size])
      }
    }
    node.addEventListener('wheel',wheel,{passive:false})
    return ()=>node.removeEventListener('wheel',wheel)
  },[])
  function start(e:PointerEvent<HTMLDivElement>) {
    if (e.button!==0 || drag.current) return
    const pos=position(e.clientX,e.clientY)
    if (!pos) return
    drag.current={pointer:e.pointerId,x:e.clientX,y:e.clientY,domain,range:range(pos.element),width:pos.width,height:pos.height}
    e.currentTarget.setPointerCapture(e.pointerId)
  }
  function move(e:PointerEvent<HTMLDivElement>) {
    const d=drag.current
    if (!d || e.pointerId!==d.pointer) return
    const dx=e.clientX-d.x, dy=e.clientY-d.y
    if (Math.abs(dx)>=4) pan(d.domain[0]-dx*(d.domain[1]-d.domain[0])/d.width)
    if (Math.abs(dy)>=4 && d.range) {
      const offset=dy*(d.range[1]-d.range[0])/d.height
      setVertical([d.range[0]+offset,d.range[1]+offset])
    }
  }
  function end(e:PointerEvent<HTMLDivElement>) {
    if (drag.current?.pointer===e.pointerId) drag.current=null
  }
  return {vertical, plotProps:{ref:host, className:'cstr-interactive-plot', onPointerDown:start,
    onPointerMove:move, onPointerUp:end, onPointerCancel:end, onLostPointerCapture:end,
    onDoubleClick:(e:MouseEvent<HTMLDivElement>)=>{if(position(e.clientX,e.clientY)) restore()}}}
}
