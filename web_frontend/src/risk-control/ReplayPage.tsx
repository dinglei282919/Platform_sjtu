import { useEffect, useRef, useState } from 'react'
import { api, ApiError } from '../api'
import { emptyHandle, replayOwner } from './cstrOwner'
import { FormNumber, Page, SimulationBar } from '../PlatformUI'
import {chartHelp, comparisonDomain, InteractiveComparisonTrend, pannedPosition,
  type ComparisonLine, type SamplePosition, type TrendNavigation} from './ComparisonTrend'
import { SimulationLayout } from './SimulationLayout'
import { CstrProcessDiagram, ControlDiagram } from './CstrProcessDiagram'

type Config = {scenario:string; speed:number}
type Labels = Record<string,string>
type Line = {key:string; name:string; color:string; dashed:boolean; optional:boolean; arm?:string}
type ChartSpec = {title:string; unit:string; lines:Line[]}
type Options = {scenarios:Labels; groups:Labels; metrics:Labels; charts:ChartSpec[]; sample_min:number; defaults:Config
  guide:string; case_guides:Labels; sensors:Labels; observations:Record<string,ChartSpec>; heat_chart:ChartSpec}
type Sample = {t:number; [key:string]:number|null}
type Process = {labels:Labels; flow:number}
type Readout = {
  svg:string; link:string; active:boolean; preview:boolean; received:string; source:string; action:string
  local:string; observed:string; jacket:string; jacket_observed:string; base:string; comp:string; final:string; request:string; weight:string
  local_mode:string; local_prediction:string; local_residual:string; local_streak:string
}
type Frame = {
  live_run:string; live_seq:number; live_config:Config; run:string; last:number; status:string; error:string
  scenario:string; pending:string|null; phase:string; phase_label:string; segment:number
  point:{seq:number;t:number;start_seq:number|null}|null; samples:Sample[]; bands:[number,number][]
  display:Readout|null; process:Process|null; metrics:Record<string,Record<string,number|null>>; scored_through:number|null
  archives:{run:string; scenario:string; last:number}[]
  arm:string; available_arms:string[]
}
type ViewedFrame = Frame & {following:boolean}

const mainArms = ['raw','guard','candidate','continuous']

function ReplayTrend({spec, rows, bands, extra, availableArms, period, navigation}: {
  spec: ChartSpec; rows:Sample[]; bands:Frame['bands']; extra:boolean; availableArms:string[]; period:number; navigation:TrendNavigation
}) {
  const lines: ComparisonLine<Sample>[] = spec.lines
    .filter(line => (extra || !line.optional) && (!line.arm || availableArms.includes(line.arm)))
    .map(line => ({
      name: line.name, color: line.color, dashed: line.dashed,
      reference: line.key === 'setpoint' || line.key === 'heat', read: row => row[line.key] ?? null,
    }))
  return <InteractiveComparisonTrend title={spec.title} unit={spec.unit}
    rows={rows} lines={lines} period={period} intervals={bands} {...navigation}/>
}

export const REPLAY_TITLE = '重放攻击下的自适应控制'

export function ReplayPage() {
  const [options,setOptions]=useState<Options|null>(null), [frame,setFrame]=useState<ViewedFrame|null>(null)
  const [draft,setDraft]=useState<Config|null>(null), [position,setPosition]=useState<SamplePosition|null>(null)
  const [scaleReset,resetScales]=useState(0)
  const [arm,setArm]=useState('guard'), [extra,setExtra]=useState(false)
  const [sensor,setSensor]=useState('Tc')
  const [busy,setBusy]=useState(false), [error,setError]=useState(''), [revision,reload]=useState(0)
  const [configAttempt,retryConfig]=useState(0)
  const current=useRef(''), submitting=useRef(false)
  useEffect(()=>{
    const c=new AbortController()
    api<Options>('/cstr/replay/config',{signal:c.signal}).then(setOptions).catch(e=>{if(!c.signal.aborted)setError(String(e))})
    return ()=>c.abort()
  },[configAttempt])
  const historyRun=position?.run, historyAt=position?.at
  useEffect(()=>{
    if(!options)return
    let active=true, timer:number
    async function poll(){
      let delay=350
      try {
        if(submitting.current)return
        const h=await replayOwner.ensure()
        const q=new URLSearchParams({arm})
        if(historyRun && historyAt!==undefined){q.set('run',historyRun);q.set('at',String(historyAt))}
        let data:Frame
        try {data=await replayOwner.request<Frame>(h,'?'+q.toString())}
        catch(e){
          if(active && e instanceof ApiError && e.status===404){
            if(historyRun)setPosition(null)
            else if(replayOwner.stored()?.id===h.id)replayOwner.save(emptyHandle)
          }
          throw e
        }
        if(!active||submitting.current)return
        if(current.current!==data.live_run){current.current=data.live_run;setDraft(data.live_config)}
        if(arm!==data.arm)setArm(data.arm)
        setFrame({...data,following:!historyRun});setError(data.error)
      }catch(e){if(active)setError(String(e));delay=1000}
      finally{if(active)timer=window.setTimeout(poll,delay)}
    }
    timer=window.setTimeout(poll,historyRun?40:0)
    return ()=>{active=false;clearTimeout(timer)}
  },[options,arm,historyRun,historyAt,revision])

  async function command(kind:string){
    if(!frame||!draft||submitting.current)return
    submitting.current=true;setBusy(true);setError('')
    try{
      const h=await replayOwner.ensure()
      const settings = kind === 'begin' ? {scenario:draft.scenario} : kind === 'reset' ? draft : undefined
      await replayOwner.request(h,'/commands',{kind,run:frame.live_run,key:crypto.randomUUID(),settings})
      if(kind==='reset')setPosition(null)
    }catch(e){setError(String(e))}
    finally{submitting.current=false;setBusy(false);reload(v=>v+1)}
  }
  const v=frame?.display
  const process=frame?.process
  const sampleReady = frame?.arm === arm && (position
    ? !frame.following && frame.run === position.run && frame.point?.seq === position.at
    : frame.following)
  const archive = frame?.archives.find(entry => entry.run === position?.run)
  const lastSample = Math.max(0, archive?.last ?? frame?.live_seq ?? 0)
  const run=position?.run??frame?.live_run??'', at=position?.at??lastSample, period=options?.sample_min??.1
  const domain=comparisonDomain(at,period,position?.first)
  const sameSource=frame?.run===run && frame.arm===arm
  const rows=sameSource ? frame.samples.filter(row=>row.t<=at*period+1e-9) : []
  const bands=sameSource ? frame.bands : []
  const availableArms=frame?.available_arms??[]
  const navigation:TrendNavigation={domain,pan:left=>{
    const next=pannedPosition(run,lastSample,left,period)
    if(next.first!==Math.round(domain[0]/period)) setPosition(next)
  },restore:()=>{setPosition(null);resetScales(n=>n+1)}}
  return <Page title={REPLAY_TITLE} className="cstr-page">
    {error && <div role="alert" className="error">{error} {!options && <button onClick={()=>{setError('');retryConfig(k=>k+1)}}>重试</button>}</div>}
    {options&&frame&&draft?<div className="cstr-stage">
      <SimulationBar state={frame.status} time={Math.max(0,frame.live_seq)*options.sample_min} busy={busy} command={k=>void command(k)}/>
      <SimulationLayout controls={<>
        <h3>重放攻击 Replay Attack</h3>
        <p className="cstr-data-help">{options.guide}</p>
        <label>本次重放的热负荷工况<select value={draft.scenario} onChange={e=>setDraft({...draft,scenario:e.target.value})}>
          {Object.entries(options.scenarios).map(([k,n])=><option key={k} value={k}>{n}</option>)}</select></label>
        <p className="cstr-data-help">{options.case_guides[draft.scenario]}</p>
        <div className="actions"><button disabled={busy||!!position||!!frame.pending||!!frame.error} onClick={()=>void command('begin')}>
          {frame.segment?'再次注入测温重放':'注入测温重放'}</button>
          {frame.pending==='begin'&&<button className="secondary" disabled={busy||!!position} onClick={()=>void command('cancel')}>取消待注入</button>}
          {frame.phase==='active'&&!frame.pending&&<button className="secondary" disabled={busy||!!position} onClick={()=>void command('finish')}>提前结束本段</button>}</div>
        <p className="cstr-data-help">{position?'正在查看历史 · 回到实时后可注入':frame.phase_label}</p>
        <FormNumber label="推进倍率" value={draft.speed} runningValue={frame.live_config.speed} min={1} max={1200} step={1}
          decimals={0} onChange={speed=>setDraft({...draft,speed})}/>
        <div className="actions"><button className="secondary" disabled={busy} onClick={()=>void command('reset')}>应用配置并复位</button></div>
        <p className="cstr-data-help">每轮先生成 0–4.9 min 的正常测温记录，3.0 min 起额外放热。复位会新开一轮。</p>
      </>}>
        <h3>反应釜工艺与测温支路</h3>
        <div className="form-grid"><label>运行记录<select value={position?.run??frame.live_run} onChange={e=>{
          const entry=frame.archives.find(a=>a.run===e.target.value)
          setPosition(entry?{run:entry.run,at:entry.last}:null)
        }}><option value={frame.live_run}>本轮</option>{[...frame.archives].reverse().map(a=><option key={a.run} value={a.run} disabled={a.last<0}>
          {options.scenarios[a.scenario]} · {a.run.slice(0,8)}</option>)}</select></label>
        <label>查看分支<select value={arm} onChange={e=>setArm(e.target.value)}>{Object.entries(options.groups)
          .filter(([k])=>availableArms.includes(k)&&(extra||mainArms.includes(k))).map(([k,n])=><option key={k} value={k}>{n}</option>)}</select></label></div>
        <div className="cstr-sample-group" aria-busy={!sampleReady}>
          {process && <CstrProcessDiagram key={frame.run+frame.arm} display={process.labels} appliedFlow={process.flow}
            running={sampleReady && frame.following && frame.status==='running'}/>}
          {v && <><ControlDiagram svg={v.svg} label={'测温副本至热负荷观测器的链路 · '+v.link}/>
            <p className="cstr-data-help">{v.link}　接收 {v.received}　样本源 {v.source}　{v.action}</p>
            <p className="cstr-data-help">釜温 · 本拍 {v.local} / 观测器 {v.observed}　夹套出口 · 本拍 {v.jacket} / 观测器 {v.jacket_observed}</p>
            {(frame.arm==='candidate'||frame.arm==='continuous')&&<p className="cstr-data-help">
              本地重估 · {v.local_mode}{v.local_prediction!=='—'&&<>　釜温预测 {v.local_prediction}　实测减预测 {v.local_residual}
                {frame.arm==='candidate'&&<>　同向越限 {v.local_streak}</>}</>}
            </p>}
            <dl className="replay-flow">{[['补偿前（PI）',v.base],['附加流量',v.comp],['补偿后（下发）',v.final]].map(([k,a])=>
              <div key={k}><dt>{k}</dt><dd>{a}</dd></div>)}</dl></>}
        </div>
        <div className="actions cstr-history-tools">{sampleReady
          ? <span>{frame.following?'实时':'历史'} · #{frame.point?.seq??'—'} · {frame.point?.t.toFixed(1)??'—'} min　{frame.phase_label}　{options.scenarios[frame.scenario]}</span>
          : <span role="status">{error ? '所选采样暂不可用' : '正在读取所选采样…'}</span>}
          <button className="secondary small" disabled={!sampleReady||frame.point?.start_seq==null} onClick={()=>{
            if(frame.point?.start_seq!=null)setPosition({run:frame.run,at:frame.point.start_seq})
          }}>定位本段首拍</button>
          <button className="secondary small" disabled={!position} onClick={()=>setPosition(null)}>回到实时</button></div>
        <input type="range" aria-label="查看采样点" min={0} max={lastSample} value={position?.at??lastSample}
          onChange={e=>setPosition({run:position?.run??frame.live_run,at:Number(e.target.value)})}/>
        <label className="replay-references"><input type="checkbox" checked={extra} onChange={e=>{
          setExtra(e.target.checked);if(!e.target.checked&&!mainArms.includes(arm))setArm('guard')
        }}/>展开参照组</label>
        <div className="cstr-sample-group">
          <label>温度测点<select value={sensor} onChange={e=>setSensor(e.target.value)}>
            {Object.entries(options.sensors).map(([key,name])=><option key={key} value={key}>{name}</option>)}</select></label>
          <p className="cstr-data-help">阴影为观测支路收到旧测温的时段。切换分支可核对本地测温、观测副本及当拍的阀指令。</p>
          <p className="cstr-chart-help">{position?'历史回看':'跟随实时'}　{domain[0].toFixed(1)}–{domain[1].toFixed(1)} min　{chartHelp}</p>
          {[options.observations[sensor],...options.charts.slice(1)].map(c=><ReplayTrend key={run+arm+c.title+scaleReset}
            spec={c} {...{rows,bands,extra,availableArms,period,navigation}}/>)}
          <details className="cstr-details"><summary>评测指标与指令明细</summary>
            <ReplayTrend key={run+arm+scaleReset} spec={options.heat_chart} {...{rows,bands,extra,availableArms,period,navigation}}/>
            <div className="cstr-sample-group" aria-busy={!sampleReady}>
            {v&&<p className="cstr-data-help">限幅前请求 {v.request}　补偿权重 {v.weight}</p>}
            <p className="cstr-data-help">{options.groups[frame.arm]} · {frame.scored_through===null?'尚未注入重放':
              `第 ${frame.segment} 段评测至 ${(frame.scored_through*options.sample_min).toFixed(1)} min`}</p>
            <dl className="cstr-readings">{Object.entries(options.metrics).map(([k,n])=><div key={k}><dt>{n}</dt>
              <dd>{frame.metrics[frame.arm]?.[k]?.toFixed(3)??'—'}</dd></div>)}</dl></div>
          </details>
        </div>
      </SimulationLayout>
    </div>:!error&&<p className="module-status-bar" role="status">正在连接独立 PI 试验…</p>}
  </Page>
}
