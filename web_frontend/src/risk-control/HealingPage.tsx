import { Fragment, useEffect, useId, useRef, useState } from 'react'
import { api, ApiError } from '../api'
import { emptyHandle, healingOwner } from './cstrOwner'
import { FormNumber, Page, SimulationBar } from '../PlatformUI'
import { SimulationLayout } from './SimulationLayout'
import { CstrProcessDiagram } from './CstrProcessDiagram'
import { HealingEffect, HealingTrends } from './HealingTrends'
import { healingColors, outputs, type Config, type Group, type Method, type Options, type Row, type Snapshot } from './healingTypes'

function RecoveryTable({view}: {view: Snapshot['view']}) {
  const title = useId(), description = useId()
  const units = ['%', 'mol/L', 'K', 'min', 'min', '次']
  return <section className="healing-recovery">
    <h3 id={title}>四组恢复指标</h3>
    <div className="cstr-metrics-table" role="region" aria-labelledby={title} tabIndex={0}>
      <table aria-labelledby={title} aria-describedby={description}>
        <thead><tr><th scope="col">恢复指标</th>{view.metrics.map(([group], i) =>
          <th scope="col" key={group}><span className="healing-group-dot" style={{background:healingColors[i]}}/>{group}</th>
        )}</tr></thead>
        <tbody>{view.headings.slice(1).map((heading, i) => <tr key={heading}>
          <th scope="row">{heading} <small>{units[i]}</small></th>
          {view.metrics.map(row => {
            const raw = row[i+1], suffix = ' '+units[i]
            const value = raw.endsWith(suffix) ? raw.slice(0,-suffix.length) : raw
            return <td key={row[0]} className={value==='—' ? 'missing' : undefined}>{value}</td>
          })}
        </tr>)}</tbody>
      </table>
    </div>
    <p id={description} className="cstr-data-help">{view.recovery_help}</p>
  </section>
}

function TriggerBand({index, cfg, draft, running, change}: {index:number; cfg:Options; draft:Config; running:Config;
  change:(key:'references'|'widths', index:number, value:number)=>void}) {
  const [centerPreview,setCenterPreview]=useState<number|null>(null)
  const [widthPreview,setWidthPreview]=useState<number|null>(null)
  const key=outputs[index], unit=cfg.signals[key][1]
  const center=centerPreview ?? draft.references[index], width=widthPreview ?? draft.widths[index]
  const format=(n:number)=>String(Number(n.toFixed(12)))
  const {reference: [refMin, refMax], half_width: [widthMin, widthMax]} = cfg.trigger_limits[key]
  const lower='下限 '+format(center-width), upper='上限 '+format(center+width)+' '+unit
  const endpoints=key==='C' ? lower+'　'+upper : lower+' '+unit+'\n'+upper
  return <div className="form-grid healing-band">
    <FormNumber min={refMin} max={refMax} step="any" decimals={12} label={key+' 参考点 ('+unit+')'}
      value={draft.references[index]} runningValue={running.references[index]} hint={`可输入范围 ${refMin} 至 ${refMax} ${unit}`}
      onPreview={setCenterPreview} onChange={v=>change('references',index,v)}/>
    <FormNumber min={widthMin} max={widthMax} step="any" decimals={12} label={key+' 半宽 ('+unit+')'}
      value={draft.widths[index]} runningValue={running.widths[index]}
      onPreview={setWidthPreview} onChange={v=>change('widths',index,v)}
      hint={`可输入范围 ${widthMin} 至 ${widthMax} ${unit}\n${endpoints}`}/>
  </div>
}

export const HEALING_TITLE = '多变量过程自愈控制'

export function HealingPage() {
  const compensationHintId=useId()
  const [cfg,setCfg]=useState<Options|null>(null)
  const [draft,setDraft]=useState<Config|null>(null)
  const [snap,setSnap]=useState<Snapshot|null>(null)
  const [rows,setRows]=useState<Row[]>([])
  const [group,setGroup]=useState<Group>('scsc')
  const [method,setMethod]=useState<Method>('scsc')
  const [error,setError]=useState('')
  const [pollError,setPollError]=useState('')
  const [busy,setBusy]=useState(false)
  const [attempt,setAttempt]=useState(0)
  const cursor=useRef({run:'',seq:-1})
  const generation=useRef(0)

  function receive(data:Snapshot) {
    const previous=cursor.current
    const changed=data.run!==previous.run
    if(!changed && data.seq<previous.seq)return
    const replace=changed || data.resync
    cursor.current={run:data.run,seq:data.seq}
    setRows(old=>[...(replace?[]:old),...data.samples.filter(r=>replace || r.seq>previous.seq)].slice(-640))
    if(changed)setDraft(data.config)
    setSnap(data)
    setPollError('')
  }

  useEffect(()=>{
    let active=true
    let timer:number
    const controller=new AbortController()
    async function poll(){
      let handle
      try {
        handle=await healingOwner.ensure()
        if(!active)return
        const previous=cursor.current
        const ticket=generation.current
        const data=await healingOwner.request<Snapshot>(handle,'?after='+previous.seq+'&run='+previous.run)
        if(!active || generation.current!==ticket)return
        receive(data)
      } catch(e) {
        if(!active)return
        if(e instanceof ApiError && e.status===404 && handle && healingOwner.stored()?.id===handle.id) {
          healingOwner.save(emptyHandle);cursor.current={run:'',seq:-1}
        }
        setPollError(String(e))
      } finally {if(active)timer=window.setTimeout(poll,350)}
    }
    api<Options>('/cstr/healing/config',{signal:controller.signal}).then(data=>{
      if(!active)return
      if(data.fault_mode !== 'manual' || !data.parameters || !data.defaults?.references ||
        !outputs.every(key=>data.trigger_limits?.[key]?.reference && data.trigger_limits[key].half_width))
        throw new Error('自愈控制服务版本不匹配，请更新后台后重试。')
      setCfg(data);void poll()
    }).catch(e=>{if(active)setPollError(String(e))})
    return ()=>{active=false;controller.abort();clearTimeout(timer)}
  },[attempt])

  async function command(kind:string) {
    if(!snap||!draft||busy)return
    setBusy(true);setError('')
    try {
      const handle=await healingOwner.ensure()
      if(kind==='export'){
        const report=await healingOwner.request(handle,'/export')
        const url=URL.createObjectURL(new Blob([JSON.stringify(report)],{type:'application/json'}))
        const a=document.createElement('a');a.href=url;a.download='cstr-self-healing-'+snap.run+'.json';a.click()
        window.setTimeout(()=>URL.revokeObjectURL(url),1000)
      }else{
        await healingOwner.request(handle,'/commands',{kind,run:snap.run,key:crypto.randomUUID(),...(kind==='reset'?{settings:draft}:{})})
        generation.current++
        const previous=cursor.current
        const data=await healingOwner.request<Snapshot>(handle,'?after='+previous.seq+'&run='+previous.run)
        generation.current++
        receive(data)
      }
    }catch(e){setError(String(e))}
    finally{setBusy(false)}
  }
  function preset(scenario:string,mode:string){
    if(!cfg||!draft)return
    const p=cfg.presets[scenario][mode]
    setDraft({...draft,scenario,mode,scsc_gain:p.scsc_gain,ccsc_gain:p.ccsc_gain,scsc_interval:p.scsc_interval,ccsc_interval:p.ccsc_interval})
  }
  function selectGroup(key:Group){
    setGroup(key)
    if(key==='scsc'||key==='ccsc')setMethod(key)
  }
  function changeBand(key:'references'|'widths', index:number, value:number){
    setDraft(old=>old && {...old,[key]:old[key].map((v,j)=>j===index?value:v)})
  }
  const message=error||pollError||snap?.error
  return <Page title={HEALING_TITLE} className="cstr-page">
    {message && <div role="alert" className="error">{message} {pollError && <button onClick={()=>setAttempt(n=>n+1)}>重试</button>}</div>}
    {(!cfg||!snap||!draft) ? !message && <p className="module-status-bar" role="status">正在连接自愈仿真会话…</p> :
    <><SimulationBar state={snap.status} time={snap.latest.t} busy={busy} command={kind=>void command(kind)}/>
    <SimulationLayout controlsKey={snap.run} controls={<><h3>自愈控制配置</h3>
      <div className="form-grid"><label>退化场景<select aria-label="退化场景" value={draft.scenario} onChange={e=>preset(e.target.value,draft.mode)}>
        {Object.entries(cfg.scenarios).map(([k,v])=><option key={k} value={k}>{v}</option>)}</select></label>
      <label>补偿调度<select aria-label="补偿调度" value={draft.mode} onChange={e=>preset(draft.scenario,e.target.value)}>
        {Object.entries(cfg.modes).map(([k,v])=><option key={k} value={k}>{v}</option>)}</select></label></div>
      <div className="actions"><button type="button" disabled={busy || snap.fault_min !== null || snap.status === 'ended' ||
        draft.scenario !== snap.config.scenario} onClick={()=>void command('degrade')}>施加退化</button>
        <button type="button" className="secondary" disabled={busy} onClick={()=>void command('reset')}>应用配置并复位</button></div>
      <p className="cstr-data-help">{draft.scenario !== snap.config.scenario ? '场景已修改，请先应用配置并复位。' :
        snap.view.effect.instruction}</p>
      <h3>自愈触发范围</h3><p className="cstr-data-help">参考点用于越限判断和入口补偿。应用配置并复位后，曲线显示本轮生效的上下限。</p>
      <div className="healing-bands">{outputs.map((key,i)=>
        <TriggerBand key={key} index={i} cfg={cfg} draft={draft} running={snap.config} change={changeBand}/>)}</div>
      <h3>补偿参数</h3><div className="form-grid">
        {(['interval','gain'] as const).map(kind=>{
          const {hint,...limits}=cfg.parameters[kind]
          const id=compensationHintId+'-'+kind
          const title=kind==='interval'?'最小间隔 (min)':'增益'
          return <Fragment key={kind}>
            {(['scsc','ccsc'] as const).map(method=>{
              const key=`${method}_${kind}` as const
              return <FormNumber key={key} {...limits} label={method.toUpperCase()+' '+title} describedBy={id}
                step={kind==='gain'?'any':limits.step} disabled={kind==='interval' && draft.mode!=='event'}
                value={draft[key]} runningValue={snap.config[key]}
                onChange={v=>setDraft(previous=>previous && {...previous,[key]:v})}/>
            })}
            <small id={id} className="parameter-hint healing-shared-hint">{hint}</small>
          </Fragment>
        })}
        <FormNumber label="正常数据拟合种子" {...cfg.parameters.seed} value={draft.seed}
          runningValue={snap.config.seed} onChange={seed=>setDraft({...draft,seed})}/>
        <FormNumber label="推进倍率" {...cfg.parameters.speed} step="any" value={draft.speed}
          runningValue={snap.config.speed} onChange={speed=>setDraft({...draft,speed})}/>
      </div><p className="cstr-data-help">配置在复位后生效。手动施加退化后继续观测 {cfg.post_fault_min} 仿真分钟。</p>
      <div className="actions">{[['end','结束'],['export','导出实验记录']].map(([k,v])=>
        <button key={k} className="secondary" disabled={busy||(k==='end'&&snap.status==='ended')} onClick={()=>void command(k)}>{v}</button>)}</div>
    </>}>
      <header className="module-status-bar cstr-status" role="status"><strong>{snap.view.state}</strong><span>{snap.view.status}</span><span>{snap.view.speed}</span></header>
      <HealingEffect snap={snap} method={method} select={selectGroup}/>
      <label>工艺图与入口对照组<select aria-label="工艺图与入口对照组" value={group} onChange={e=>selectGroup(e.target.value as Group)}>
        {Object.entries(cfg.groups).map(([k,v])=><option value={k} key={k}>{v}</option>)}</select></label>
      <CstrProcessDiagram key={'diagram-'+snap.run} display={snap.view.diagrams[group]} appliedFlow={snap.latest.groups[group].Qc} running={snap.status==='running'}/>
      <p className="cstr-data-help">当前越限 {snap.latest.groups[group].outside.join('、')||'无'}　退化后首次触发 {snap.metrics.groups[group].first_update?.toFixed(1)??'—'} min　
        活性系数 {snap.latest.groups[group].a.toFixed(4)}　换热系数 {snap.latest.groups[group].b.toFixed(4)}</p>
      <RecoveryTable view={snap.view}/>
      <HealingTrends key={'trends-'+snap.run} {...{cfg,snap,rows,group}}/>
      <h3>补偿事件</h3>{snap.recent.length ? <ul className="cstr-events">{snap.recent.slice(-6).reverse().map(e=><li key={e.group+e.first_seq}>{e.label}</li>)}</ul>
        : <p className="cstr-data-help">暂无补偿事件</p>}
      <p className="cstr-data-help">{snap.view.timing.join('　')}</p>
    </SimulationLayout></>}</Page>
}
