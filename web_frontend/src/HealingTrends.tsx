import { useEffect, useMemo, useState } from 'react'
import type { EChartsOption, LineSeriesOption, ScatterSeriesOption } from 'echarts'
import {trendGrid as grid, useTrendGestures, type Range} from './trendGestures'
import { SampleChart } from './SampleChart'
import { api } from './api'
import { healingColors as colors, inputs, outputs, type Group, type Method, type Options, type Output, type Row, type Signal, type Snapshot } from './healingTypes'

const span = 600

export function HealingEffect({snap, method, select}: {snap: Snapshot; method: Method; select:(value:Method)=>void}) {
  const display = snap.view.effect
  const selected = display.methods[method]
  const option = useMemo<EChartsOption>(() => {
    const [left, right] = display.domain
    const column = method === 'scsc' ? 2 : 3
    const color = colors[column]
    const points = snap.effect.points
    const single = points.length === 1
    const style = {color: '#c9dbe9', fontSize: 12}
    return {
      animation: false, grid: {...grid, left: 76}, legend: {top: 1, textStyle: style},
      xAxis: {type: 'value', min: left, max: right, name: '仿真时间 (min)', nameLocation: 'middle', nameGap: 26,
        axisLabel: {color: style.color}, splitLine: {show: false}},
      yAxis: {type: 'value', min: selected.limits[0], max: selected.limits[1], interval: selected.tick,
        axisLabel: {color: style.color, formatter: (v: number) => v.toFixed(snap.effect.signal === 'C' ? 5 : 3)},
        splitLine: {lineStyle: {color: '#30465d'}}},
      series: [
        {name: '正常参考', type: 'line', showSymbol: false, itemStyle: {color: colors[4]},
          lineStyle: {type: 'dashed', width: 1}, data: [[left, 0], [right, 0]]},
        {name: '仅 PI', type: 'line', showSymbol: single, itemStyle: {color: colors[1]},
          data: points.map(p => [p[0], p[1]]),
          markLine: {silent: true, symbol: 'none', data: [
            ...(snap.fault_min === null ? [] : [{xAxis: snap.fault_min, lineStyle: {color: colors[1], type: 'dashed' as const},
              label: {formatter: '退化开始', position: 'insideEndTop' as const, rotate: 0, color: colors[1]}}]),
            ...(selected.first_min === null ? [] : [{xAxis: selected.first_min, lineStyle: {color, type: 'dashed' as const},
              label: {formatter: '退化后首次补偿', position: 'insideStartTop' as const, rotate: 0, color}}]),
          ]}},
        {name: 'PI + '+method.toUpperCase(), type: 'line', showSymbol: single, itemStyle: {color},
          lineStyle: {width: 2.5}, data: points.map(p => [p[0], p[column]])},
      ],
    }
  }, [display, method, selected, snap.effect, snap.fault_min])
  return <section className="healing-effect" aria-label="自愈效果对照">
    <div className="healing-effect-header"><h3>自愈效果对照</h3>
      <label>对照的自愈方法<select aria-label="对照的自愈方法" value={method}
        onChange={e=>select(e.target.value as Method)}>
        <option value="scsc">简单相关自愈（SCSC）</option><option value="ccsc">典型相关自愈（CCSC）</option>
      </select></label></div>
    <p className="cstr-data-help">{display.phase} {selected.step}</p>
    <h4>{display.title}</h4>
    <SampleChart key={snap.run+method} option={option} height={245} samples={snap.effect.points} timeOf={p=>p[0]}
      unit={display.unit} digits={snap.effect.signal==='C' ? 6 : 4}
      readingsOf={p=>[['正常参考',0],['仅 PI',p[1]],['PI + '+method.toUpperCase(),p[method==='scsc' ? 2 : 3]]]}/>
    <p className="cstr-data-help">{display.reference}</p>
    <p className="module-status-bar cstr-status">{selected.readout}</p>
  </section>
}

function Plot({signal,rows,cfg,snap,group,domain,pan,restore}: {
  signal: Signal; rows: Row[]; cfg: Options; snap: Snapshot; group: Group; domain: Range
  pan: (at:number)=>void; restore: ()=>void
}) {
  const {vertical, plotProps} = useTrendGestures(domain, pan, restore)
  const index=outputs.indexOf(signal as Output)
  const keys=useMemo(()=>index>=0 ? Object.keys(cfg.groups) as Group[] : [group], [index,cfg.groups,group])
  const center=snap.config.references[index]
  const bounds=useMemo(()=>index>=0 ? [center-snap.config.widths[index],center+snap.config.widths[index]] :
    signal==='Qc' ? cfg.flow_limits : [], [index,center,snap.config.widths,signal,cfg.flow_limits])
  const option=useMemo<EChartsOption>(()=>{
    const lines:(LineSeriesOption|ScatterSeriesOption)[]=[]
    for (const key of keys) lines.push({name:cfg.groups[key],type:'line',showSymbol:rows.length===1,
      lineStyle:{width:key===group?2.5:1.5},data:rows.map(r=>[r.t,r.groups[key][signal]])})
    bounds.forEach((v,i)=>lines.push({name:i?'上限':'下限',type:'line',showSymbol:false,lineStyle:{type:'dashed',width:1,color:colors[4]},
      data:[[domain[0],v],[domain[1],v]]}))
    if(index>=0) lines.push({name:'所选组触发',type:'scatter',symbolSize:8,itemStyle:{color:'#ff8f79'},
      data:rows.flatMap(r=>r.compensation.filter(e=>e.group===group && e.channels.includes(signal as Output)).map(e=>[e.at,center+e.error[index]]))})
    return {animation:false,color:colors,grid,legend:{top:1,textStyle:{color:'#c9dbe9',fontSize:11}},
      xAxis:{type:'value',min:domain[0],max:domain[1],name:'仿真时间 (min)',nameLocation:'middle',nameGap:26,axisLabel:{color:'#a9c1d5'},splitLine:{show:false}},
      yAxis:{type:'value',scale:true,min:vertical?.[0],max:vertical?.[1],axisLabel:{color:'#a9c1d5'},splitLine:{lineStyle:{color:'#30465d'}}},series:lines}
  },[cfg.groups,domain,group,rows,signal,vertical,index,keys,center,bounds])
  const [name,unit]=cfg.signals[signal]
  return <section className="cstr-trend-panel"><h3>{name} ({unit})</h3><div {...plotProps}>
    <SampleChart key={group} option={option} height={235} samples={rows} timeOf={r=>r.t} unit={unit}
      readingsOf={r=>{
        const values: [string,number][] = keys.map(key=>[cfg.groups[key],r.groups[key][signal]])
        bounds.forEach((v,i)=>values.push([i?'上限':'下限',v]))
        return values
      }}/>
  </div></section>
}

export function HealingTrends({cfg,snap,rows,group}:{cfg:Options;snap:Snapshot;rows:Row[];group:Group}) {
  const [set,setSet]=useState('outputs')
  const [pinned,pin]=useState<number|null>(null)
  const [reset,setReset]=useState(0)
  const [archive,save]=useState<{key:string;rows:Row[]}|null>(null)
  const [error,setError]=useState('')
  const [retry,setRetry]=useState(0)
  const start=pinned===null ? Math.max(0,snap.seq-span) : Math.min(pinned,Math.max(0,snap.seq-1))
  const end=start+span, key=snap.run+'/'+start
  const needsArchive=Boolean(rows.length && start<rows[0].seq)
  useEffect(()=>{
    if (!needsArchive) {setError('');return}
    const controller=new AbortController()
    setError('')
    api<{samples:Row[]}>('/cstr/healing/sessions/'+snap.id+'/history?run='+snap.run+'&first='+start+'&count='+(span+1),{signal:controller.signal})
      .then(data=>{if(!controller.signal.aborted)save({key,rows:data.samples})})
      .catch(e=>{if(!controller.signal.aborted)setError(String(e))})
    return ()=>controller.abort()
  },[key,needsArchive,snap.id,snap.run,start,retry])
  const points=(needsArchive ? archive?.key===key ? archive.rows : [] : rows).filter(r=>r.seq>=start && r.seq<=end)
  const domain:Range=[start*cfg.period,end*cfg.period]
  function restore(){pin(null);setReset(n=>n+1)}
  return <section className="healing-trends"><div className="cstr-history-tools"><label>趋势信号
    <select aria-label="趋势信号" value={set} onChange={e=>setSet(e.target.value)}><option value="outputs">输出与触发范围</option>
    <option value="inputs">实际入口与冷却流量</option></select></label><span>{pinned===null?'跟随实时':'历史回看'}　{domain[0].toFixed(1)}–{domain[1].toFixed(1)} min</span>
    <button className="small secondary" disabled={pinned===null} onClick={restore}>回到实时</button></div>
    {error && <div role="alert">{error} <button onClick={()=>setRetry(n=>n+1)}>重试</button></div>}
    {(set==='outputs'?outputs:inputs).map(signal=><Plot key={signal+reset} {...{signal,rows:points,cfg,snap,group,domain,restore}}
      pan={at=>pin(Math.max(0,Math.min(Math.max(0,snap.seq-1),Math.round(at/cfg.period))))}/>)}</section>
}
