import { useEffect, useRef, useState } from 'react'
import { api, ApiError } from './api'
import { emptyHandle, fdiOwners } from './cstrOwner'
import { FormNumber, Page, SimulationBar, StageSwitch } from './PlatformUI'
import { useStoredState } from './taskSession'
import { CstrProcessDiagram } from './CstrProcessDiagram'
import {chartHelp, comparisonDomain, InteractiveComparisonTrend, pannedPosition,
  type ComparisonLine, type SamplePosition, type TrendNavigation} from './ComparisonTrend'
import demoViews from '../../assets/cstr/fdi-demo.json'
import loopSvg from '../../assets/cstr/fdi-loops.svg?raw'

type Side = keyof typeof fdiOwners
type Labels = Record<string,string>
type Config = {side: Side; scenario: string; seed: number; speed: number}
type Options = {
  scenarios: Record<Side, Labels>; groups: Record<Side, Labels>; sample_min: number; recovery_samples: number; segment_samples: number
  previews: Record<Side,Record<string,Labels>>
  nominal_display:Record<Side,Labels>; nominal_flow:number; loop_labels:Record<string,Labels>
}
type Point = {valid: boolean; Qc: number|null; previous_Qc: number; [key:string]:unknown}
type Row = {seq: number; t: number; phase: string; groups: Record<string,Point>}
type Frame = {
  run:string; seq:number; config:Config; status:string; error:string; segment:number; scenario:string
  pending:string|null; pending_scenario:string|null; bands:[number,number][]; preview:boolean; samples:Row[]; latest:Row|null; resync?:boolean
  archives?:{run:string; seq:number; config:Config}[]
  timing:{budget_ms:number; p95_ms:number|null; effective_speed:number|null; overruns:number; intervals:number}
  view:{phase:string; timeline:string; metrics_title:string; diagrams:Record<string,Labels>; readings:Record<string,string[][]>; sources:Labels; table:string[][]; events:string[]}
}
type Signal = {name:string; group:string; key:string; color:string; dashed?:boolean}
type ChartSpec = {title:string; unit:string; note?:string; signals:Signal[]}
type Demo = {presets:Labels; preset_notes?:Labels; controller_note:string; protected_group:string; comparison:string; charts:ChartSpec[]}

function signalValue(row:Row, signal:Signal) {
  const group = row.groups[signal.group]
  const value = group?.[signal.key]
  return group?.valid && typeof value === 'number' ? value : null
}

function LoopDiagram({display,labels}:{display:Labels;labels:Labels}) {
  const values={...display,...labels}
  const svg=loopSvg.replace(/@@([A-Z]+)@@/g,(_,key:string)=>values[key]??'')
  return <details><summary>展开控制框图</summary>
    <img className="fdi-loop-diagram" alt="所选分支的闭环信号图" src={'data:image/svg+xml;charset=utf-8,'+encodeURIComponent(svg)}/>
  </details>
}

function Trend({spec, rows, bands, period, navigation}: {
  spec: ChartSpec; rows: Row[]; bands:Frame['bands']; period:number; navigation:TrendNavigation
}) {
  const lines: ComparisonLine<Row>[] = spec.signals.map(signal => ({
    name: signal.name, color: signal.color, dashed: signal.dashed,
    reference: signal.key === 'T_ref', read: row => signalValue(row, signal),
  }))
  return <InteractiveComparisonTrend title={spec.title} unit={spec.unit} note={spec.note}
    rows={rows} lines={lines} period={period} intervals={bands} {...navigation}/>
}

function Trial({side, visible, options}:{side:Side; visible:boolean; options:Options}) {
  const owner=fdiOwners[side], groups=options.groups[side], arms=Object.keys(groups)
  const demo:Demo=demoViews[side]
  const presets=Object.entries(demo.presets)
  const [snap,setSnap]=useState<Frame|null>(null), [rows,setRows]=useState<Row[]>([])
  const [draft,setDraft]=useState<Config|null>(null), [arm,setArm]=useState(demo.protected_group)
  const [busy,setBusy]=useState(false)
  const [position,setPosition]=useState<SamplePosition|null>(null), [scaleReset,resetScales]=useState(0)
  const [history,setHistory]=useState<Frame|null>(null)
  const [error,setError]=useState(''), [connection,setConnection]=useState('')
  const cursor=useRef({seq:-1,run:''}), generation=useRef(0), sending=useRef(false)

  function receive(data:Frame) {
    const old=cursor.current
    if(old.run===data.run && data.seq<old.seq)return
    const changed=old.run!==data.run, replace=changed||data.resync
    cursor.current={seq:data.seq,run:data.run}
    setRows(prev=>[...(replace ? [] : prev),...data.samples.filter(r=>replace || r.seq>old.seq)].slice(-240))
    if(changed){
      const cfg=data.config
      const next=data.pending_scenario??data.scenario
      setDraft({...cfg,scenario:next in demo.presets ? next : presets[0][0]})
      setPosition(null);setHistory(null)
    }
    setSnap(data);setConnection('')
  }
  async function read() {
    const ticket=generation.current
    const handle=await owner.ensure(), at=cursor.current
    try {
      const suffix='?after='+at.seq+'&run='+at.run
      let data=await owner.request<Frame>(handle,suffix)
      if(ticket!==generation.current)return
      if(data.status==='ready'){
        // A tab may still own a session created before automatic startup.
        await owner.request(handle,'/commands',{kind:'start',key:'enter-'+data.run,run:data.run})
        data=await owner.request<Frame>(handle,suffix)
      }
      if(ticket===generation.current)receive(data)
    } catch(e) {
      if(ticket===generation.current && e instanceof ApiError && e.status===404 && owner.stored()?.id===handle.id){
        owner.save(emptyHandle);cursor.current={seq:-1,run:''}
      }
      throw e
    }
  }
  useEffect(()=>{
    let active=true, timer:number
    async function poll(){
      // Both clocks start on entry. Only the visible trial needs ongoing screen updates.
      let delay=visible?320:null
      try {if(!sending.current)await read()}
      catch(e){
        if(!active)return
        setConnection(String(e))
        delay=1000
      }finally{if(active && delay!==null)timer=window.setTimeout(poll,delay)}
    }
    void poll()
    return ()=>{active=false;generation.current++;clearTimeout(timer)}
  },[owner,visible])

  const historyAt=position?.at, historyRun=position?.run
  useEffect(()=>{
    if(historyAt===undefined||!historyRun){setHistory(null);return}
    if(!visible)return
    let active=true
    const first=Math.max(0,historyAt-119)
    const timer=window.setTimeout(()=>{void owner.ensure().then(h=>owner.request<Frame>(h,'/history?run='+historyRun+'&first='+first+'&count='+(historyAt-first+1)))
      .then(data=>{if(active)setHistory(data)}).catch(e=>{if(active)setError(String(e))})
    },40)
    return ()=>{active=false;window.clearTimeout(timer)}
  },[owner,historyRun,historyAt,visible])

  async function command(kind:string) {
    if(!snap || !draft || sending.current)return
    sending.current=true;generation.current++;setBusy(true);setError('')
    try {
      const h=await owner.ensure()
      if(kind==='export'){
        const run=position?.run??snap.run
        const report=await owner.request(h,'/export?run='+run)
        const url=URL.createObjectURL(new Blob([JSON.stringify(report)],{type:'application/json'}))
        const a=document.createElement('a');a.href=url;a.download='cstr-fdi-'+side+'-'+run+'.json';a.click()
        window.setTimeout(()=>URL.revokeObjectURL(url),1000)
      }else{
        const settings=kind==='reset' ? draft : kind==='begin' ? {scenario:draft.scenario} : undefined
        await owner.request(h,'/commands',{kind, run:snap.run, key:crypto.randomUUID(), settings})
        await read()
      }
    }catch(e){setError(String(e))}finally{sending.current=false;setBusy(false)}
  }

  if(!visible)return null
  if(!snap||!draft)return <section className="panel" role="status">{connection||'正在连接试验会话…'}</section>
  const historical=!!position
  const frame=historical && history ? history : snap, point=frame.latest
  const ended=snap.status==='ended'
  const changed=draft.scenario!==snap.scenario
  const needsReset=!(snap.config.scenario in demo.presets)
  const presetNote=demo.preset_notes?.[draft.scenario]
  const visibleRun=position?.run??snap.run
  const entries=snap.archives??[]
  const upper=visibleRun===snap.run ? snap.seq : (entries.find(r=>r.run===visibleRun)?.seq??0)
  const at=position?.at??snap.seq, period=options.sample_min
  const matchingHistory=position && history?.run===visibleRun && history.latest?.seq===at ? history : null
  const sampleReady=!position || !!matchingHistory
  const domain=comparisonDomain(at,period,position?.first)
  const plotSource=matchingHistory ?? (visibleRun===snap.run ? snap : history?.run===visibleRun ? history : null)
  const plotRows=plotSource===snap ? rows : plotSource?.samples??[]
  const samples=plotRows.filter(row=>row.seq<=at)
  const navigation:TrendNavigation={domain,pan:left=>{
    const next=pannedPosition(visibleRun,upper,left,period)
    if(next.first!==Math.round(domain[0]/period)) setPosition(next)
  },restore:()=>{setPosition(null);resetScales(n=>n+1)}}
  const diag=point && frame.view.diagrams[arm]
  const selected=point?.groups[arm]
  const previewCase=!snap.preview && !changed ? 'healthy' : draft.scenario
  const preview=options.previews[side][previewCase]
  const display=diag && !historical && changed && point?.phase==='healthy' && !snap.pending ? {...diag,...preview} : diag
  const message=error||connection||snap.error
  const scenarioLabel=(key:string)=>demo.presets[key]??options.scenarios[side][key]
  return <>
    {message && <div role="alert" className="error">{message}</div>}
    <SimulationBar state={snap.status} time={Math.max(0,snap.seq)*options.sample_min} busy={busy}
      command={k=>void command(k)}/>
    <div className="two-column cstr-layout"><aside className="panel" key={snap.run}>
      <section className="cstr-disturbance-settings" aria-labelledby={'fdi-attack-heading-'+side}>
      <h3 id={'fdi-attack-heading-'+side}>虚假数据注入攻击FDI Attack</h3>
      <label>预置样本<select value={draft.scenario} disabled={presets.length===1}
        aria-describedby={presetNote?'fdi-preset-note-'+side:undefined} onChange={e=>setDraft({...draft,scenario:e.target.value})}>
        {presets.map(([k,v])=><option key={k} value={k}>{v}</option>)}</select></label>
      {presetNote && <p id={'fdi-preset-note-'+side} className="cstr-data-help">{presetNote}</p>}
      <div className="actions"><button disabled={busy||ended||!!snap.error||historical||!!snap.pending||needsReset}
        onClick={()=>void command('begin')}>{snap.segment?'再次注入 FDI':'注入 FDI 样本'}</button>
        <button className="secondary" disabled={busy||ended||historical||!(snap.pending==='begin'||(!snap.pending&&point?.phase==='active'))}
          onClick={()=>void command('finish')}>{snap.pending==='begin'?'取消待注入样本':'结束本次注入'}</button></div>
      <p className="cstr-data-help">每次持续 {options.segment_samples*options.sample_min} 仿真分钟。生效期间再次注入，先撤除一拍，再施加所选样本。</p>
      <p className="cstr-data-help">{demo.controller_note.split(/<sub>([^<]+)<\/sub>/).map((part, index) =>
        index % 2 === 1 ? <sub key={index}>{part}</sub> : part
      )}</p></section>
      <h3>仿真参数</h3>
      <div className="form-grid">
        <FormNumber label="噪声与初态种子" min={0} max={2147483647} step={1} decimals={0}
          hint="相同种子复用同一组初态和噪声。"
          value={draft.seed} runningValue={snap.config.seed} onChange={seed=>setDraft({...draft,seed})}/>
        <FormNumber label="推进倍率" min={1} max={1200} step="any" decimals={1}
          hint="可输入范围 1 至 1200 倍。"
          value={draft.speed} runningValue={snap.config.speed} onChange={speed=>setDraft({...draft,speed})}/>
      </div>
      <button className="secondary" disabled={busy} onClick={()=>void command('reset')}>应用配置并复位</button>
      <p className="cstr-data-help">{needsReset?'应用配置并复位可从验收工况切换到演示样本。':
        `每次注入单独评测，恢复观察最长 ${options.recovery_samples*options.sample_min} 仿真分钟。`}</p>
      <div className="actions"><button className="secondary" disabled={busy||ended} onClick={()=>void command('end')}>结束试验</button>
        <button className="secondary" disabled={busy} onClick={()=>void command('export')}>导出所选记录</button></div>
    </aside><section className="panel healing-results">
      <h3>闭环对照</h3>
      <header className="module-status-bar cstr-status" role="status">
        {sampleReady ? <><strong><i className={'status-dot '+frame.status}/>{frame.view.phase}</strong>
          <span>{scenarioLabel(frame.scenario)}</span><span>设定 {frame.config.speed}×</span></> : '正在读取所选采样…'}
      </header>
      <div className="form-grid"><label>运行记录<select value={visibleRun} onChange={e=>{
        const record=entries.find(r=>r.run===e.target.value)
        setPosition(record?{run:record.run,at:record.seq}:null)
      }}><option value={snap.run}>本轮</option>{[...entries].reverse().map(r=><option key={r.run} value={r.run} disabled={r.seq<0}>
        {scenarioLabel(r.config.scenario)} · {r.run.slice(0,8)}</option>)}</select></label></div>
      <div className="cstr-sample-group" aria-busy={!sampleReady}>
      {!point && <><CstrProcessDiagram display={{...options.nominal_display[side],...preview}}
        appliedFlow={options.nominal_flow} running={false}/><p className="cstr-data-help">名义工况预览 · 正在准备观测器</p></>}
      {point && selected && <>
        {display && <CstrProcessDiagram display={display} appliedFlow={selected.Qc??selected.previous_Qc}
          running={!historical && snap.status==='running' && selected.valid}/>}
        <p className="cstr-data-help">工艺图 · {groups[arm]}　反馈来源 · {frame.view.sources[arm]}</p>
        {display && <LoopDiagram display={display} labels={options.loop_labels[arm]}/>}
        {frame.view.timeline && <p className="cstr-data-help">{frame.view.timeline}</p>}
      </>}
      </div>
      <div className="actions cstr-history-tools"><span>{sampleReady
        ? `${historical?'历史':'实时'} · #${point?.seq??'—'} · ${point?.t.toFixed(1)??'—'} min` : '正在读取所选采样…'}</span>
        <button className="secondary small" disabled={!historical} onClick={()=>setPosition(null)}>回到实时</button></div>
      <input aria-label="查看采样点" type="range" min={0} max={Math.max(0,upper)} value={Math.max(0,at)}
        onChange={e=>setPosition({run:visibleRun,at:Number(e.target.value)})}/>
      <p className="cstr-data-help">{demo.comparison}</p>
      <p className="cstr-chart-help">{historical?'历史回看':'跟随实时'}　{domain[0].toFixed(1)}–{domain[1].toFixed(1)} min　{chartHelp}</p>
      {demo.charts.map(spec=><Trend key={visibleRun+spec.title+scaleReset} spec={spec} rows={samples}
        bands={plotSource?.bands??[]} period={period} navigation={navigation}/>)}
      <details className="fdi-details"><summary>试验详情</summary>
        <label>工艺图与采样详情分支<select value={arm} onChange={e=>setArm(e.target.value)}>
          {arms.map(k=><option key={k} value={k}>{groups[k]}</option>)}</select></label>
        {sampleReady && point && selected && <>
          <h3>所选采样点 · {groups[arm]}</h3><dl className="fdi-readings">
            {frame.view.readings[arm].map(([k,v])=><div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>
          <h3>{frame.view.metrics_title}</h3>
          <div className="healing-table"><table><thead><tr><th scope="col">指标</th>{arms.map(k=><th scope="col" key={k}>{groups[k]}</th>)}</tr></thead>
            <tbody>{frame.view.table.map(([k,...values])=><tr key={k}><th scope="row">{k}</th>{values.map((v,i)=><td key={arms[i]}>{v}</td>)}</tr>)}</tbody></table></div>
          <p className="cstr-data-help">恢复用时取真实釜温连续 10 拍处于设定值 ±0.10 K 内的首拍。请在相同的有效区间数下比较 IAE。</p>
        </>}
        {sampleReady && <><h3>采样事件</h3>{frame.view.events.length ? <ul className="healing-events">{frame.view.events.slice(-8).map((v,i)=><li key={i}>{v}</li>)}</ul> :
          <p className="cstr-data-help">暂无事件</p>}
        <p className="cstr-data-help">周期预算 {frame.timing.budget_ms.toFixed(1)} ms　计算 P95 {frame.timing.p95_ms?.toFixed(2)??'—'} ms　
          超时 {frame.timing.overruns} / {frame.timing.intervals}　实测推进 {frame.timing.effective_speed?.toFixed(1)??'—'}×</p></>}
      </details>
    </section></div></>
}

export const FDI_TITLE = '虚假数据注入下的自适应控制'

export function FdiPage() {
  const [selectedSide,setSide]=useStoredState<Side>('cstr-fdi-side','measurement')
  const side=selectedSide==='execution'?'execution':'measurement'
  const stage=side==='measurement'?0:1
  const [options,setOptions]=useState<Options|null>(null)
  const [error,setError]=useState(''), [attempt,retry]=useState(0)
  useEffect(()=>{
    const controller=new AbortController()
    api<Options>('/cstr/fdi/config',{signal:controller.signal}).then(setOptions)
      .catch(e=>{if(!controller.signal.aborted)setError(String(e))})
    return ()=>controller.abort()
  },[attempt])
  return <Page title={FDI_TITLE} className="cstr-page">
    <StageSwitch id="fdi-stage" label="独立试验" tabs={['测量端','执行端']} selected={stage}
      select={index=>setSide(index===0?'measurement':'execution')}/>
    <section id="fdi-stage-panel" role="tabpanel" aria-labelledby={'fdi-stage-'+stage} className="cstr-stage">
    {error && <div className="error" role="alert">{error} <button onClick={()=>{setError('');retry(n=>n+1)}}>重试</button></div>}
    {options ? (['measurement','execution'] as const).map(key=>
      <Trial key={key} side={key} visible={key===side} options={options}/>) : !error && <p role="status">正在读取试验配置…</p>}
    </section>
  </Page>
}
