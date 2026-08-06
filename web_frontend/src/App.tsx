import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import * as echarts from 'echarts'
import { api, post } from './api'

type ModuleId = 'governance' | 'score' | 'classification' | 'cdq' | 'sdg' | 'sil' | 'anomaly' | 'training' | 'dnn-mpc'
type TaskProgress = { percent?: number; message?: string; module?: string; timestamp?: string; revision?: number; outputs?: Record<string, unknown> }
type Task = { id: string; title: string; status: string; logs: string[]; result?: Record<string, unknown>; error?: string; progress?: TaskProgress }
type Recommendation = { node_id: string; node_name: string; frequency: number; severity: number; rrf: number; target_sil: number }
type SdgNode = { id: string; name: string; type: 'R' | 'P' | 'C'; probability: number }
type SdgEdge = { source: string; target: string; type: '+' | '-'; probability: number }

type NavigationItem = { id: ModuleId; name: string; deferred?: boolean }
type NavigationGroup = { id: string; icon: string; name: string; items: NavigationItem[] }
type SilForm = {
  m: number
  n: number
  lambda_fit: number
  ti: number
  mrt: number
  nsim: number
  years: number
  ccf_mode: 'total' | 'partial'
  total_beta: number
  partial_betas: Record<string, number>
  estimate_T: number
  estimate_k: number
  estimate_low: number
  estimate_high: number
}

const SIL_DEFAULT_FORM: SilForm = {
  m: 2,
  n: 4,
  lambda_fit: 111.11,
  ti: 8760,
  mrt: 8,
  nsim: 500,
  years: 10000,
  ccf_mode: 'total',
  total_beta: 0.1,
  partial_betas: { '2': 0.0333, '3': 0.0333, '4': 0.0333 },
  estimate_T: 876000,
  estimate_k: 5,
  estimate_low: 20,
  estimate_high: 80,
}

const navigationGroups: NavigationGroup[] = [
  { id: 'governance', icon: '📊', name: '异构数据治理', items: [] },
  { id: 'anomaly', icon: '🏭', name: '异常行为检测', items: [{ id: 'anomaly', name: '基于移动目标防御的异常检测' }] },
  {
    id: 'risk-analysis', icon: '📈', name: '风险动态分析', items: [
      { id: 'classification', name: '潜在安全威胁识别与自动分类' },
      { id: 'score', name: '多评估准则融合的风险学习分析' },
      { id: 'cdq', name: '风险场景动态匹配与适配方案生成算法' },
    ],
  },
  {
    id: 'risk-control', icon: '🎛', name: '风险管控优化决策', items: [
      { id: 'training', name: '控制模型训练评估' },
      { id: 'dnn-mpc', name: '优化控制仿真验证' },
    ],
  },
  { id: 'sis', icon: '🛡', name: 'SIS自主化检测', items: [{ id: 'sdg', name: 'SDG-HAZOP' }] },
  { id: 'sil', icon: '✅', name: '在线SIL验证', items: [{ id: 'sil', name: '基于GSPN-MC模型的动态化SIL验证方法' }] },
]

function Chart({ option, height = 330, fill = false, className = '', onClick, onResize }: { option: echarts.EChartsOption; height?: number; fill?: boolean; className?: string; onClick?: (params: any) => void; onResize?: (width: number, height: number) => void }) {
  const element = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!element.current) return
    const instance = echarts.getInstanceByDom(element.current) ?? echarts.init(element.current)
    instance.setOption(option, true)
    if (onClick) instance.on('click', onClick)
    const resize = () => {
      instance.resize()
      if (onResize && element.current) onResize(element.current.clientWidth, element.current.clientHeight)
    }
    window.addEventListener('resize', resize)
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(resize)
    observer?.observe(element.current)
    requestAnimationFrame(resize)
    return () => { if (onClick) instance.off('click', onClick); window.removeEventListener('resize', resize); observer?.disconnect() }
  }, [option, onClick, onResize])
  return <div className={`chart ${className}`} ref={element} style={fill ? undefined : { height }} />
}

function useTask(taskId: string | null) {
  const [task, setTask] = useState<Task | null>(null)
  useEffect(() => {
    setTask(null)
    if (!taskId) return
    let active = true
    const load = async () => {
      try {
        const next = await api<Task>(`/tasks/${taskId}`)
        if (active) setTask(next)
      } catch (error) {
        if (active) setTask({ id: taskId, title: '任务', status: 'failed', logs: [], error: error instanceof Error ? error.message : String(error) })
      }
    }
    void load()
    const handle = window.setInterval(() => void load(), 1000)
    return () => { active = false; window.clearInterval(handle) }
  }, [taskId])
  return task
}

function TaskPanel({ task }: { task: Task | null }) {
  if (!task) return null
  return <section className="task-panel">
    <div className="task-title"><span className={`status-dot ${task.status}`} />{task.title}：{task.status}</div>
    <pre>{task.logs.join('\n')}{task.error ? `\n错误：${task.error}` : ''}</pre>
  </section>
}

function AnomalyPage() {
  const [form, setForm] = useState({
    mcr_root: 'E:\\MATLAB2024',
    attack_min_pct: 5,
    attack_max_pct: 10,
    measurement_noise_pct: 2,
    process_disturbance_pct: 5,
  })
  const [taskId, setTaskId] = useState<string | null>(null)
  const [imageMode, setImageMode] = useState<'topology' | 'detection'>('topology')
  const [imageRevision, setImageRevision] = useState(0)
  const [imageError, setImageError] = useState(false)
  const [error, setError] = useState('')
  const task = useTask(taskId)
  const result = task?.result as Record<string, any> | undefined
  const running = task?.status === 'queued' || task?.status === 'running'
  const statusText = error || task?.status === 'failed' ? '执行失败' : task?.status === 'succeeded' ? '执行完成' : running ? '运行中' : '待执行'
  const statusEnglish = error || task?.status === 'failed' ? 'Failed' : task?.status === 'succeeded' ? 'Completed' : running ? 'Running' : 'Idle'
  const imageName = imageMode === 'detection' ? 'detection_probability.png' : 'topology.png'
  const imageTitle = imageMode === 'detection' ? '检测概率图 (detection_probability.png)' : '网络拓扑图 (topology.png)'
  const imageUrl = `/api/anomaly/images/${imageName}?v=${encodeURIComponent(String(result?.image_revision ?? imageRevision))}`
  const report = result ? JSON.stringify(result, null, 2) : task?.error ? `执行失败：${task.error}` : ''

  useEffect(() => {
    if (task?.status === 'succeeded') {
      setImageRevision(Number(task.result?.image_revision ?? Date.now()))
      setImageError(false)
    }
  }, [task?.status, task?.result])

  const update = (key: keyof typeof form, value: string | number) => setForm(previous => ({ ...previous, [key]: value }))
  const submit = async () => {
    try {
      setError('')
      const nextTask = await post<{ task_id: string }>('/anomaly/tasks', form)
      setTaskId(nextTask.task_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }
  const exportResult = () => {
    if (!result) return
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `anomaly_result_${new Date().toISOString().replace(/[-:]/g, '').slice(0, 15)}.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  return <Page title="异常行为检测 · MTD" className="anomaly-page">
    <div className="anomaly-layout">
      <section className="anomaly-box anomaly-settings-root">
        <h3>参数与导出设置</h3>
        <div className="anomaly-box-body">
          <section className="anomaly-box anomaly-param-group">
            <h3>参数设置</h3>
            <div className="anomaly-param-row"><label htmlFor="anomaly-mcr">MATLAB Runtime路径:</label><input id="anomaly-mcr" type="text" value={form.mcr_root} onChange={e => update('mcr_root', e.target.value)}/></div>
            <div className="anomaly-param-row"><label>随机攻击强度范围（%）:</label><div className="anomaly-range"><input aria-label="随机攻击强度最小值" type="number" value={form.attack_min_pct} min={5} max={50} step={1} onChange={e => update('attack_min_pct', Number(e.target.value))}/><span>~</span><input aria-label="随机攻击强度最大值" type="number" value={form.attack_max_pct} min={5} max={50} step={1} onChange={e => update('attack_max_pct', Number(e.target.value))}/></div></div>
            <div className="anomaly-param-row"><label htmlFor="anomaly-noise">测量噪声强度（%）:</label><input id="anomaly-noise" type="number" value={form.measurement_noise_pct} min={1} max={30} step={1} onChange={e => update('measurement_noise_pct', Number(e.target.value))}/></div>
            <div className="anomaly-param-row"><label htmlFor="anomaly-disturbance">过程扰动强度（%）:</label><input id="anomaly-disturbance" type="number" value={form.process_disturbance_pct} min={1} max={30} step={1} onChange={e => update('process_disturbance_pct', Number(e.target.value))}/></div>
            <div className="anomaly-centered-action"><button type="button" onClick={() => void submit()} disabled={running}>运行异常结果</button></div>
          </section>
          <section className="anomaly-box anomaly-export-group">
            <h3>导出设置</h3>
            <pre className="anomaly-result-preview">{report}</pre>
            <button type="button" className="anomaly-export-button" onClick={exportResult} disabled={!result || running}>导出结果JSON</button>
          </section>
        </div>
      </section>
      <section className="anomaly-box anomaly-result-root">
        <h3>结果图示</h3>
        <div className="anomaly-box-body">
          <div className="anomaly-image-switches"><button type="button" className={imageMode === 'topology' ? 'active' : ''} onClick={() => { setImageMode('topology'); setImageError(false) }} disabled={running}>网络拓扑图</button><button type="button" className={imageMode === 'detection' ? 'active' : ''} onClick={() => { setImageMode('detection'); setImageError(false) }} disabled={running}>检测概率图</button></div>
          <div className="anomaly-image-panel"><h4>{imageTitle}</h4><div className="anomaly-image-frame" role="img" aria-label={imageTitle} style={imageError ? undefined : { backgroundImage: `url("${imageUrl}")`, backgroundSize: 'contain', backgroundPosition: 'center', backgroundRepeat: 'no-repeat' }}>{imageError ? <div className="anomaly-image-placeholder">暂无{imageMode === 'detection' ? '检测概率图' : '拓扑图'}</div> : <img className="anomaly-result-image" src={imageUrl} alt="" aria-hidden="true" style={{ display: 'none' }} onError={() => setImageError(true)}/>}</div></div>
        </div>
      </section>
    </div>
    <div className="anomaly-status-bar"><span>状态：{statusText}</span><span>Status: {statusEnglish}</span></div>
    {(error || task?.error) && <ErrorBox text={error || task?.error || '异常检测失败'} />}
  </Page>
}

function ScorePage() {
  const [config, setConfig] = useState<Record<string, any> | null>(null)
  const [values, setValues] = useState<Record<string, number>>({})
  const [weights, setWeights] = useState<Record<string, number>>({})
  const [result, setResult] = useState<Record<string, any> | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { void api<Record<string, any>>('/score/config').then(data => { setConfig(data); setValues(data.defaults); setWeights(data.default_weights) }).catch(e => setError(e.message)) }, [])
  const evaluate = async () => { try { setError(''); setResult(await post('/score/evaluate', { values, weights })) } catch (e) { setError(e instanceof Error ? e.message : String(e)) } }
  const randomize = async () => { try { setValues(await post<Record<string, number>>('/score/random')) } catch (e) { setError(e instanceof Error ? e.message : String(e)) } }
  const radar = result ? {
    backgroundColor: 'transparent',
    tooltip: { formatter: (params: any) => `单项得分：${(params.value as number[]).map(value => Number(value).toFixed(2)).join(' / ')}` },
    radar: {
      indicator: result.radar.indicators.map((name: string) => ({ name, max: 100 })),
      axisName: { color: '#d7e7f8' },
      splitLine: { lineStyle: { color: '#48617a' } },
      splitArea: { areaStyle: { color: ['#17293b'] } },
    },
    series: [{ type: 'radar', symbol: 'circle', symbolSize: 8, label: { show: true, color: '#eef6ff', formatter: (params: any) => Array.isArray(params.value) ? params.value.map((value: number) => Number(value).toFixed(1)).join(' / ') : Number(params.value).toFixed(1) }, data: [{ value: result.radar.scores, name: '单项得分', areaStyle: { color: 'rgba(88,174,255,.35)' }, lineStyle: { color: '#63b9ff', width: 3 }, itemStyle: { color: '#63b9ff' } }] }],
  } as echarts.EChartsOption : null
  const statusText = error ? '状态：评分异常' : result ? '状态：评分完成' : '状态：待命'
  return <Page title="多评估准则融合的风险学习分析">
    <div className="two-column score-layout">
      <section className="panel score-input-panel"><h3>数据与评分规则设定</h3>{config && <div className="metric-table"><div className="table-head">指标名称</div><div className="table-head">当前数值</div><div className="table-head">权重配置</div>
        {Object.entries(config.metrics).map(([name, item]: [string, any]) => <div className="metric-row" key={name}><label>{name}</label><input type="number" min={item.minimum} max={item.maximum} step={item.step} value={values[name] ?? ''} onChange={e => setValues({ ...values, [name]: Number(e.target.value) })} /><input type="number" min="0" max="10" step="0.1" value={weights[name] ?? 0} onChange={e => setWeights({ ...weights, [name]: Number(e.target.value) })} /></div>)}
      </div>}<div className="actions score-actions"><button className="secondary" onClick={() => void randomize()}>随机生成数据</button><button onClick={() => void evaluate()}>执行加权评分</button></div></section>
      <section className="panel score-result-panel"><h3>评估报告与六维蛛网图</h3>{result ? <><div className="score-result-top"><div className="score-radar-wrap">{radar && <Chart option={radar} height={420} className="score-radar-chart" />}</div><div className="score-summary"><ScoreCard label="综合加权总分" value={Number(result.total_score).toFixed(2)} tone="blue" /><ScoreCard label="潜在危险分数" value={Number(result.danger_score).toFixed(2)} tone="red" /></div></div><div className="report-list score-report-list"><p><b>【各项指标得分明细】</b></p>{result.items.map((item: any) => <p key={item.metric}>{item.metric}（{item.value}）：{item.score.toFixed(2)} 分（权重：{(item.normalized_weight * 100).toFixed(2)}%）</p>)}<p><b>【综合评估】</b></p><p>加权总分：{Number(result.total_score).toFixed(2)} / 100.00</p><p>潜在危险分数：{Number(result.danger_score).toFixed(4)}</p></div></> : <Empty text="输入指标数值后执行评分。" />}</section>
    </div><div className="module-status-bar">{statusText}</div>{error && <ErrorBox text={error} />}
  </Page>
}

function SdgPage({ onRecommend }: { onRecommend: (recommendation: Recommendation) => void }) {
  const [nodes, setNodes] = useState<SdgNode[]>([])
  const [edges, setEdges] = useState<SdgEdge[]>([])
  const [config, setConfig] = useState<Record<string, any> | null>(null)
  const [result, setResult] = useState<Record<string, any> | null>(null)
  const [sisRequiredNodes, setSisRequiredNodes] = useState<string[]>([])
  const [graphRevision, setGraphRevision] = useState(0)
  const [graphSize, setGraphSize] = useState({ width: 0, height: 0 })
  const [logLines, setLogLines] = useState<string[]>([])
  const [error, setError] = useState('')
  const [nodeId, setNodeId] = useState('R1')
  const [nodeName, setNodeName] = useState('冷却水故障')
  const [nodeType, setNodeType] = useState<'R' | 'P' | 'C'>('R')
  const [nodeProbability, setNodeProbability] = useState('0.018')
  const [fuzzyTerm, setFuzzyTerm] = useState('中等')
  const [source, setSource] = useState('')
  const [target, setTarget] = useState('')
  const [edgeType, setEdgeType] = useState<'+' | '-'>('+')
  const [edgeProbability, setEdgeProbability] = useState('0.85')

  const applyExample = (data: Record<string, any>) => {
    setNodes(data.nodes ?? [])
    setEdges(data.edges ?? [])
    setResult(null)
    setSisRequiredNodes([])
    setLogLines(['✓ 已加载 TE 过程反应器超压示例模型。'])
  }

  useEffect(() => {
    void Promise.all([
      api<Record<string, any>>('/sdg/example'),
      api<Record<string, any>>('/sdg/config'),
    ]).then(([example, editorConfig]) => {
      setConfig(editorConfig)
      const nodeDefaults = editorConfig.node_defaults ?? {}
      const edgeDefaults = editorConfig.edge_defaults ?? {}
      setNodeId(nodeDefaults.id ?? 'R1')
      setNodeName(nodeDefaults.name ?? '冷却水故障')
      setNodeType(nodeDefaults.type ?? 'R')
      setNodeProbability(String(nodeDefaults.probability ?? 0.018))
      setFuzzyTerm(nodeDefaults.fuzzy_term ?? '中等')
      setEdgeType(edgeDefaults.type ?? '+')
      setEdgeProbability(String(edgeDefaults.probability ?? 0.85))
      applyExample(example)
    }).catch(e => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  useEffect(() => {
    if (!nodes.length) {
      setSource('')
      setTarget('')
      return
    }
    setSource(previous => nodes.some(node => node.id === previous) ? previous : nodes[0].id)
    const preferredTarget = nodes.find(node => node.id === 'P1')?.id ?? nodes[1]?.id ?? nodes[0].id
    setTarget(previous => nodes.some(node => node.id === previous) ? previous : preferredTarget)
  }, [nodes])

  const fuzzyTerms = config?.fuzzy_terms ?? []
  const nodeTypes = config?.node_types ?? [
    { value: 'R', label: '原因 (R)' },
    { value: 'P', label: '参数 (P)' },
    { value: 'C', label: '后果 (C)' },
  ]
  const edgeTypes = config?.edge_types ?? [
    { value: '+', label: '增量 (+)' },
    { value: '-', label: '减量 (-)' },
  ]

  const handleGraphResize = useCallback((width: number, height: number) => {
    setGraphSize(previous => previous.width === width && previous.height === height ? previous : { width, height })
  }, [])

  const addNode = () => {
    const id = nodeId.trim()
    const name = nodeName.trim()
    const probability = Number(nodeProbability)
    if (!id || !name) { setError('节点 ID 和名称不能为空'); return }
    if (nodes.some(node => node.id === id)) { setError(`节点 ID “${id}” 已存在`); return }
    if (!Number.isFinite(probability) || probability < 0) { setError('节点概率/频率必须是非负数'); return }
    const nextNode: SdgNode = { id, name, type: nodeType, probability: nodeType === 'R' ? probability : 0 }
    setNodes(previous => [...previous, nextNode])
    setResult(null)
    setSisRequiredNodes([])
    setError('')
    setLogLines(previous => [...previous, `节点添加: ${id} (${name}) [类型:${nodeType}, 概率:${nextNode.probability}]`])
    setNodeId('')
    setNodeName('')
    setNodeProbability('0.01')
  }

  const applyFuzzy = () => {
    const item = fuzzyTerms.find((entry: any) => entry.label === fuzzyTerm)
    if (!item) { setError('当前模糊术语配置不可用'); return }
    setNodeProbability(Number(item.probability).toFixed(6))
    setError('')
    setLogLines(previous => [...previous, `模糊术语 “${fuzzyTerm}” → 概率 ${Number(item.probability).toFixed(6)} 次/年`])
  }

  const addEdge = () => {
    const probability = Number(edgeProbability)
    if (!source || !target) { setError('请选择源节点和目标节点'); return }
    if (source === target) { setError('不能连接节点自身'); return }
    if (edges.some(edge => edge.source === source && edge.target === target)) { setError('该边已经存在'); return }
    if (!Number.isFinite(probability) || probability < 0 || probability > 1) { setError('条件概率必须在 0～1 之间'); return }
    setEdges(previous => [...previous, { source, target, type: edgeType, probability }])
    setResult(null)
    setSisRequiredNodes([])
    setError('')
    setLogLines(previous => [...previous, `边添加: ${source} → ${target} [${edgeType}, P=${probability}]`])
  }

  const clearAll = () => {
    setNodes([])
    setEdges([])
    setResult(null)
    setSisRequiredNodes([])
    setError('')
    setLogLines(['已清空所有数据。'])
  }

  const loadExample = async () => {
    try {
      setError('')
      applyExample(await api<Record<string, any>>('/sdg/example'))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const runAnalysis = async () => {
    try {
      setError('')
      setResult(null)
      setSisRequiredNodes([])
      setLogLines([])
      const next = await post<Record<string, any>>('/sdg/analyze', { nodes, edges })
      setResult(next)
      setSisRequiredNodes(next.sis_required_nodes ?? (next.sil_recommendations ?? []).map((item: Recommendation) => item.node_id))
      const logs: string[] = [
        '='.repeat(70),
        '  SDG-HAZOP 完整定量风险分析报告',
        '='.repeat(70),
        '\n--- 正向推理：风险路径与概率计算 ---',
      ]
      for (const item of next.forward_paths ?? []) {
        logs.push(`\n路径: ${(item.path ?? []).join(' → ')}`)
        logs.push(...(item.steps ?? []).map((step: string) => `  ${step}`))
        logs.push(`  >> 最终概率 = ${Number(item.probability).toFixed(8)} 次/年`)
      }
      logs.push('\n--- 后果节点总概率计算（并联/或门聚合）---')
      for (const item of next.consequences ?? []) {
        const pathItems = item.paths ?? []
        if (pathItems.length === 1) {
          logs.push(`\n后果 ${item.node_id}: 仅有1条路径 → 总概率 = ${Number(item.frequency).toFixed(8)}`)
        } else if (pathItems.length > 1) {
          logs.push(`\n后果 ${item.node_id}: 汇聚 ${pathItems.length} 条路径（或门）`)
          pathItems.forEach((path: any, index: number) => logs.push(`  路径${index + 1}概率: ${Number(path.probability).toFixed(8)}`))
          logs.push('  聚合计算:', ...(item.aggregation_steps ?? []).map((step: string) => `    ${step}`))
          logs.push(`  >> 总概率 = ${Number(item.frequency).toFixed(8)} 次/年`)
        }
        logs.push('', '  【风险矩阵定级】', `    可能性等级: P${item.risk.probability_level} (${item.risk.probability_description})`, `    严重性等级: S${item.risk.severity_level} (${item.risk.severity_description})`, `    ★ 综合风险: ${item.risk.level}`, `    建议措施: ${item.risk.action}`)
        if (Number(item.frequency) > 0) {
          const lopa = item.lopa
          logs.push('', '  【LOPA 保护层分析】', `    假设保护层: DCS(PFD=${lopa.pfd_dcs}) + 安全阀(PFD=${lopa.pfd_relief_valve})`, `    原始风险: ${Number(item.frequency).toFixed(8)} 次/年`, `    残余风险: ${Number(lopa.residual_frequency).toFixed(8)} 次/年`, `    可容忍标准: ${lopa.tolerance} 次/年`)
          if (Number(lopa.residual_frequency) <= Number(lopa.tolerance)) logs.push('    ✓ 残余风险可接受')
          else logs.push(`    ❌ 需要新增SIF，RRF=${Number(lopa.rrf).toFixed(2)}`, `    → 目标 SIL 等级: ${lopa.target_sil}`)
        }
      }
      logs.push('\n--- 反向推理：后果原因追溯 ---')
      for (const item of next.backward_paths ?? []) {
        logs.push(`\n后果 ${item.consequence} 的可能原因链:`)
        for (const path of item.paths ?? []) logs.push(`  ${(path ?? []).join(' → ')}`)
      }
      logs.push('\n' + '='.repeat(70), '  分析完成')
      if ((next.sis_required_nodes ?? []).length) logs.push('\n🔎 点击图中红色边框节点，可打开 SIL 验证工具。')
      setLogLines(logs)
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e)
      setError(message)
      setLogLines([`错误: ${message}`])
    }
  }

  const graph = useMemo(() => {
    // Force layout is deliberately avoided here.  With a small graph and a
    // resized panel it can converge to the same point before its animation
    // starts, making the labels unreadable.  Build a stable layered layout
    // from the directed edges instead; users can still pan, zoom and drag.
    const incoming = new Map(nodes.map(node => [node.id, 0]))
    const outgoing = new Map<string, string[]>()
    edges.forEach(edge => {
      incoming.set(edge.target, (incoming.get(edge.target) ?? 0) + 1)
      outgoing.set(edge.source, [...(outgoing.get(edge.source) ?? []), edge.target])
    })
    const roots = nodes.filter(node => (incoming.get(node.id) ?? 0) === 0).map(node => node.id)
    const levels = new Map<string, number>()
    const queue = [...(roots.length ? roots : nodes.slice(0, 1).map(node => node.id))]
    queue.forEach(id => levels.set(id, 0))
    for (let index = 0; index < queue.length; index += 1) {
      const id = queue[index]
      for (const next of outgoing.get(id) ?? []) {
        const level = Math.max(levels.get(next) ?? 0, (levels.get(id) ?? 0) + 1)
        levels.set(next, level)
        if (!queue.includes(next)) queue.push(next)
      }
    }
    nodes.forEach(node => { if (!levels.has(node.id)) levels.set(node.id, 0) })
    const maxLevel = Math.max(1, ...Array.from(levels.values()))
    const innerWidth = Math.max(1, graphSize.width - 128)
    const innerHeight = Math.max(1, graphSize.height - 104)
    // Keep the logical graph bounds in the same aspect ratio as the actual
    // canvas.  ECharts transforms the view coordinate system to the canvas;
    // matching these ratios prevents a circle from becoming an ellipse.
    const logicalHeight = 80 / Math.max(1, innerWidth / innerHeight)
    const columns = new Map<number, string[]>()
    nodes.forEach(node => {
      const level = levels.get(node.id) ?? 0
      columns.set(level, [...(columns.get(level) ?? []), node.id])
    })
    const positions = new Map<string, { x: number; y: number }>()
    columns.forEach((ids, level) => {
      ids.forEach((id, index) => {
        const y = ids.length === 1 ? logicalHeight / 2 : (logicalHeight * index) / (ids.length - 1)
        positions.set(id, { x: maxLevel === 0 ? 50 : 10 + (80 * level) / maxLevel, y })
      })
    })
    return ({
      backgroundColor: 'transparent',
      tooltip: { formatter: (params: any) => params.data?.name ?? params.name },
      series: [{
        type: 'graph',
        layout: 'none',
        roam: true,
        draggable: true,
        // Reserve enough room around the outermost node symbols and labels;
        // unlike force layout, the logical bounds of a `none` layout are
        // exactly the min/max node coordinates.
        left: 64,
        right: 64,
        top: 52,
        bottom: 52,
        zoom: 1,
        scaleLimit: { min: 0.55, max: 2.2 },
        edgeSymbol: ['none', 'arrow'],
        label: { show: true, color: '#e8f3ff', fontSize: 10, lineHeight: 12 },
        edgeLabel: { show: true, color: '#d8e7ff', fontSize: 10, formatter: (params: any) => `P=${Number(params.data?.probability ?? 0).toFixed(2)}` },
        data: nodes.map(node => ({
          id: node.id,
          name: `${node.id}\n${node.name}`,
          ...(positions.get(node.id) ?? { x: 50, y: 25 }),
          symbolSize: sisRequiredNodes.includes(node.id) ? 58 : 52,
          itemStyle: {
            color: node.type === 'R' ? '#ff7f7f' : node.type === 'C' ? '#7fbfff' : '#7fff7f',
            borderColor: sisRequiredNodes.includes(node.id) ? '#ff3f3f' : '#111c2b',
            borderWidth: sisRequiredNodes.includes(node.id) ? 3 : 1.5,
          },
        })),
      links: edges.map(edge => ({
        source: edge.source,
        target: edge.target,
        probability: edge.probability,
        label: { show: true, formatter: `P=${Number(edge.probability).toFixed(2)}`, color: '#d8e7ff', fontSize: 10 },
        lineStyle: { color: edge.type === '+' ? '#c2d8ec' : '#ff6262', width: 2 },
      })),
      }],
    } as echarts.EChartsOption)
  }, [nodes, edges, sisRequiredNodes, graphRevision, graphSize])

  const handleGraphClick = useCallback((params: any) => {
    const nodeIdValue = params?.data?.id
    const recommendation = (result?.sil_recommendations ?? []).find((item: Recommendation) => item.node_id === nodeIdValue)
    if (!recommendation) return
    setLogLines(previous => [...previous, `点击节点 ${nodeIdValue}，打开 SIL 验证。`])
    onRecommend(recommendation)
  }, [result, onRecommend])

  return <Page title="SIS自主化检测 · SDG-HAZOP" className="sdg-page">
    <div className="two-column wide-right sdg-layout">
      <section className="panel sdg-input-panel">
        <div className="sdg-input-scroll">
          <section className="sdg-group"><h3>添加节点</h3><div className="sdg-form-grid"><label>ID<input value={nodeId} onChange={e => setNodeId(e.target.value)} /></label><label>名称<input value={nodeName} onChange={e => setNodeName(e.target.value)} /></label><label>类型<select value={nodeType} onChange={e => setNodeType(e.target.value as 'R' | 'P' | 'C')}>{nodeTypes.map((item: any) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label><label>概率/频率<input type="number" min="0" step="0.000001" value={nodeProbability} onChange={e => setNodeProbability(e.target.value)} /></label></div><label>模糊术语<select value={fuzzyTerm} onChange={e => setFuzzyTerm(e.target.value)}>{fuzzyTerms.map((item: any) => <option key={item.label} value={item.label}>{item.label}</option>)}</select></label><div className="sdg-inline-actions"><button className="small" onClick={applyFuzzy}>应用模糊→概率</button><button className="small" onClick={addNode}>➕ 添加节点</button></div></section>
          <section className="sdg-group"><h3>添加因果关系边</h3><label>源节点<select value={source} onChange={e => setSource(e.target.value)}>{nodes.map(node => <option key={node.id} value={node.id}>{node.id} - {node.name}</option>)}</select></label><label>目标节点<select value={target} onChange={e => setTarget(e.target.value)}>{nodes.map(node => <option key={node.id} value={node.id}>{node.id} - {node.name}</option>)}</select></label><div className="sdg-form-grid"><label>影响类型<select value={edgeType} onChange={e => setEdgeType(e.target.value as '+' | '-')}>{edgeTypes.map((item: any) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label><label>条件概率<input type="number" min="0" max="1" step="0.01" value={edgeProbability} onChange={e => setEdgeProbability(e.target.value)} /></label></div><button className="small" onClick={addEdge}>🔗 添加边</button></section>
        </div>
        <div className="sdg-actions"><button className="secondary" onClick={() => void loadExample()}>📋 加载TE示例</button><button className="secondary" onClick={clearAll}>🗑️ 清空所有</button><button className="secondary" onClick={() => { setGraphRevision(previous => previous + 1); setLogLines(previous => [...previous, '已刷新视图。']) }}>🔄 刷新视图</button><button onClick={() => void runAnalysis()}>🚀 运行完整定量分析</button></div>
        <div className="sdg-node-status">节点: {nodes.length} | 边: {edges.length} | SIS需求: {sisRequiredNodes.length}</div>
      </section>
      <section className="panel sdg-output-panel">
        <section className="sdg-graph-panel"><h3>SDG 因果模型</h3><Chart option={graph} fill className="sdg-graph-chart" onClick={handleGraphClick} onResize={handleGraphResize} /></section>
        <section className="sdg-log-panel"><h3>SDG-HAZOP 分析日志</h3><pre>{logLines.length ? logLines.join('\n') : '运行完整定量分析后，详细报告将在这里显示。'}</pre></section>
      </section>
    </div>
    {error && <ErrorBox text={error} />}
  </Page>
}

function CdqPageReplica() {
  const [config, setConfig] = useState<Record<string, any> | null>(null)
  const [step, setStep] = useState(1)
  const [horizon, setHorizon] = useState(10)
  const [sampleIndex, setSampleIndex] = useState(0)
  const [cv, setCv] = useState<number[]>([])
  const [uNow, setUNow] = useState<number[]>([])
  const [uAfter, setUAfter] = useState<number[]>([])
  const [result, setResult] = useState<Record<string, any> | null>(null)
  const [error, setError] = useState('')
  const [running, setRunning] = useState(false)

  useEffect(() => {
    void api<Record<string, any>>('/cdq/config').then(data => {
      setConfig(data)
      setStep(Number(data.default_step ?? 1))
      setHorizon(Number(data.default_horizon ?? 10))
      setSampleIndex(Number(data.default_sample_index ?? 0))
      setCv(data.default_cv ?? [])
      setUNow((data.initial_u_now ?? []).map((value: number) => Number(Number(value).toFixed(2))))
      setUAfter((data.initial_u_after ?? []).map((value: number) => Number(Number(value).toFixed(2))))
    }).catch(e => setError(e.message))
  }, [])

  const run = async () => {
    try {
      setError('')
      setResult(null)
      setRunning(true)
      const next = await post<Record<string, any>>('/cdq/analyze', { step, horizon, sample_index: sampleIndex, cv, u_now: uNow, u_after: uAfter })
      setResult(next)
      if (next.inputs?.u_now) setUNow(next.inputs.u_now.map((value: number) => Number(Number(value).toFixed(2))))
      if (next.inputs?.u_after) setUAfter(next.inputs.u_after.map((value: number) => Number(Number(value).toFixed(2))))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setRunning(false)
    }
  }

  const uLabels = config?.u_labels ?? ['装焦量', '空气导入量', '排焦量', '循环空气流量', '放散阀门开度', '氮气补充量', '锅炉过热蒸汽流量']
  const cvLabels = config?.cv_labels ?? ['预存室料位', '气体成分H2', '气体成分CO', '气体成分CO2', '锅炉入口温度', '冷焦排出温度', '干熄炉入口温度']
  const series = result?.series
  const axis = () => ({ type: 'category', data: series?.steps ?? [], axisLabel: { color: '#b9cee2', fontSize: 10 }, axisLine: { lineStyle: { color: '#466385' } } })
  const yAxis = () => ({ type: 'value', axisLabel: { color: '#b9cee2', fontSize: 10 }, splitLine: { lineStyle: { color: '#30465d' } }, axisLine: { lineStyle: { color: '#466385' } } })
  const chartBase = { animation: false, backgroundColor: 'transparent', tooltip: { trigger: 'axis' }, grid: { left: 42, right: 12, top: 64, bottom: 38, containLabel: true } }
  const levelChart = series ? {
    ...chartBase,
    title: { text: '预存室料位预测', left: 'center', top: 8, textStyle: { color: '#d4e8ff', fontSize: 13 } },
    legend: { data: ['料位高度 (m)'], top: 32, textStyle: { color: '#d7e7f8', fontSize: 10 } },
    xAxis: axis(), yAxis: yAxis(),
    series: [{ name: '料位高度 (m)', type: 'line', data: series.level, symbol: 'circle', symbolSize: 6, lineStyle: { color: '#63b9ff', width: 2 }, itemStyle: { color: '#63b9ff' } }],
  } as echarts.EChartsOption : null
  const gasChart = series ? {
    ...chartBase,
    title: { text: '可燃气体演变趋势', left: 'center', top: 8, textStyle: { color: '#d4e8ff', fontSize: 13 } },
    legend: { data: ['H2 (%)', 'CO (%)', 'CO2 (%)'], top: 32, textStyle: { color: '#d7e7f8', fontSize: 10 } },
    xAxis: axis(), yAxis: yAxis(),
    series: [
      { name: 'H2 (%)', type: 'line', data: series.h2, symbol: 'x', symbolSize: 6, lineStyle: { color: '#ff9b6a', width: 2 }, itemStyle: { color: '#ff9b6a' } },
      { name: 'CO (%)', type: 'line', data: series.co, symbol: 'rect', symbolSize: 5, lineStyle: { color: '#6ad5ff', width: 2 }, itemStyle: { color: '#6ad5ff' } },
      { name: 'CO2 (%)', type: 'line', data: series.co2, symbol: 'diamond', symbolSize: 5, lineStyle: { color: '#b682ff', width: 2 }, itemStyle: { color: '#b682ff' } },
    ],
  } as echarts.EChartsOption : null
  const temperatureChart = series ? {
    ...chartBase,
    title: { text: '热力系统温度监控', left: 'center', top: 8, textStyle: { color: '#d4e8ff', fontSize: 13 } },
    legend: { data: ['锅炉温度 (°C)', '排焦温度 (°C)'], top: 32, textStyle: { color: '#d7e7f8', fontSize: 10 } },
    xAxis: axis(), yAxis: yAxis(),
    series: [
      { name: '锅炉温度 (°C)', type: 'line', data: series.boiler_temperature, symbol: 'triangle', symbolSize: 5, lineStyle: { color: '#ff6b6b', width: 2 }, itemStyle: { color: '#ff6b6b' } },
      { name: '排焦温度 (°C)', type: 'line', data: series.coke_temperature, symbol: 'arrow', symbolSize: 5, lineStyle: { color: '#ffe66a', width: 2 }, itemStyle: { color: '#ffe66a' } },
    ],
  } as echarts.EChartsOption : null

  const resultInputs = result?.inputs as Record<string, any> | undefined
  const source = result?.data_source as Record<string, any> | undefined
  const logLines = result ? [
    '正在执行状态感知与多步物理演化计算...',
    `数据源：${source?.path ?? config?.path ?? 'cdq_data.xlsx'} | 真实样本：${source?.samples ?? config?.samples ?? 0} 行 | 字段：${(source?.headers ?? config?.headers ?? []).join('、')}`,
    source?.available
      ? `已选取真实数据窗口：第 ${(Number(source.sample_index) || 0) + 1} 行开始，共 ${source.window_rows} 行。`
      : `未找到真实样本文件，使用默认演示数据。${source?.error ? ` ${source.error}` : ''}`,
    resultInputs?.u_now ? `首行动作样本：${JSON.stringify(resultInputs.u_now.map((value: number) => Number(value).toFixed(4)))}` : '',
    `有效建模步数：${resultInputs?.horizon ?? 0}`,
    '物理演化预测完成。正在启动算法匹配风险场景数据库...',
    '',
    '【算法匹配结果输出】',
    '='.repeat(50),
    ...result.risks.flatMap((risk: string, index: number) => [risk, result.schemes[index], '-'.repeat(40)]),
  ].filter(Boolean).join('\n') : ''
  const statusText = error
    ? `状态：异常：${error}`
    : running
      ? '状态：正在执行状态感知、物理演化与风险匹配...'
      : result
        ? '状态：算法运行完毕，已输出最佳适配干预方案。'
        : config?.available
          ? '状态：已载入 cdq_data.xlsx 真实样本，等待运行。'
          : config
            ? `状态：${config.error || '未找到真实样本文件，使用默认演示数据。'}`
            : '状态：正在载入数据源...'

  return <Page title="风险场景动态匹配与适配方案生成算法" className="cdq-page">
    <div className="two-column cdq-layout">
      <section className="panel cdq-input-panel">
        <h3>系统状态与动作空间设定</h3>
        <div className="cdq-input-scroll">
          <section className="cdq-section"><h4>控制动作指令集 (U)</h4><div className="cdq-u-table"><div className="cdq-table-head">指令名称</div><div className="cdq-table-head">当前动作 (U_now)</div><div className="cdq-table-head">预选动作 (U_after)</div>{uLabels.map((label: string, index: number) => <div className="cdq-table-row" key={label}><span>{label}</span><input type="number" value={uNow[index] ?? ''} min={-999999} max={9999999} step={0.01} onChange={e => { const next = [...uNow]; next[index] = Number(e.target.value); setUNow(next) }} /><input type="number" value={uAfter[index] ?? ''} min={-999999} max={9999999} step={0.01} onChange={e => { const next = [...uAfter]; next[index] = Number(e.target.value); setUAfter(next) }} /></div>)}</div></section>
          <section className="cdq-section"><h4>实时工况特征向量 (CV)</h4><div className="cdq-cv-table"><div className="cdq-table-head">特征名称</div><div className="cdq-table-head">实时感知数值</div>{cvLabels.map((label: string, index: number) => <div className="cdq-table-row" key={label}><span>{label}</span><input type="number" value={cv[index] ?? ''} min={-999999} max={9999999} step={0.001} onChange={e => { const next = [...cv]; next[index] = Number(e.target.value); setCv(next) }} /></div>)}</div></section>
          <section className="cdq-section"><h4>算法推演视界设定</h4><div className="cdq-settings-grid"><FormNumber label="时间步长 (Step)" value={step} onChange={setStep} min={0.1} max={100} step={0.1}/><FormNumber label="预测域 (Horizon)" value={horizon} onChange={setHorizon} min={1} max={500} step={1}/><FormNumber label="真实数据起始样本" value={sampleIndex} onChange={setSampleIndex} min={0} max={Math.max(0, (config?.samples ?? 2) - 2)} step={1}/></div></section>
          <p className="cdq-data-source">数据源：{config?.path ?? 'cdq_data.xlsx'} | 真实样本：{config?.samples ?? 0} 行 | 字段：{(config?.headers ?? uLabels).join('、')}</p>
        </div>
        <button className="cdq-run-button" disabled={running || !config} onClick={() => void run()}>{running ? '算法运行中...' : '启动 风险场景匹配与方案生成'}</button>
      </section>
      <section className="panel cdq-output-panel">
        <section className="cdq-subpanel cdq-chart-panel"><h3>系统状态多步预测演化轨迹</h3><div className="cdq-chart-area"><div className="cdq-chart-grid">{levelChart ? <div className="cdq-square-chart"><Chart option={levelChart} fill className="cdq-chart" /></div> : <div className="cdq-square-chart cdq-chart-placeholder" />}{gasChart ? <div className="cdq-square-chart"><Chart option={gasChart} fill className="cdq-chart" /></div> : <div className="cdq-square-chart cdq-chart-placeholder" />}{temperatureChart ? <div className="cdq-square-chart"><Chart option={temperatureChart} fill className="cdq-chart" /></div> : <div className="cdq-square-chart cdq-chart-placeholder" />}</div></div></section>
        <section className="cdq-subpanel cdq-log-panel"><h3>算法决策输出：风险匹配与自适应方案生成</h3><pre className="cdq-result-log">{logLines || '等待算法评估，生成干预方案...'}</pre></section>
      </section>
    </div>
    {error && <ErrorBox text={error} />}
    <div className="module-status-bar">{statusText}</div>
  </Page>
}

// Legacy compact CDQ page retained for reference; the application uses CdqPageReplica.
function CdqPageLegacy() {
  const [config, setConfig] = useState<Record<string, any> | null>(null)
  const [step, setStep] = useState(1)
  const [horizon, setHorizon] = useState(10)
  const [sampleIndex, setSampleIndex] = useState(0)
  const [cv, setCv] = useState<number[]>([])
  const [result, setResult] = useState<Record<string, any> | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { void api<Record<string, any>>('/cdq/config').then(data => { setConfig(data); setCv(data.default_cv) }).catch(e => setError(e.message)) }, [])
  const run = async () => { try { setError(''); setResult(await post('/cdq/analyze', { step, horizon, sample_index: sampleIndex, cv })) } catch (e) { setError(e instanceof Error ? e.message : String(e)) } }
  const series = result?.series
  const option = series ? { backgroundColor: 'transparent', tooltip: { trigger: 'axis' }, legend: { textStyle: { color: '#d7e7f8' } }, xAxis: { type: 'category', data: series.steps, axisLabel: { color: '#b9cee2' } }, yAxis: { type: 'value', axisLabel: { color: '#b9cee2' }, splitLine: { lineStyle: { color: '#30465d' } } }, series: [{ name: '料位', type: 'line', data: series.level }, { name: 'H2(%)', type: 'line', data: series.h2 }, { name: 'CO(%)', type: 'line', data: series.co }, { name: 'CO2(%)', type: 'line', data: series.co2 }, { name: '锅炉温度', type: 'line', data: series.boiler_temperature }, { name: '排焦温度', type: 'line', data: series.coke_temperature }] } as echarts.EChartsOption : null
  return <Page title="风险场景动态匹配与适配方案生成算法" subtitle="使用本机cdq_data.xlsx的连续真实样本执行多步物理演化与规则匹配。"><div className="two-column"><section className="panel"><h3>算法推演视界设定</h3>{config && <p className="muted">数据源：{config.path}，有效样本 {config.samples} 行。</p>}<FormNumber label="时间步长" value={step} onChange={setStep} min={0.1} max={100} step={0.1}/><FormNumber label="预测域" value={horizon} onChange={setHorizon} min={1} max={500} step={1}/><FormNumber label="起始样本" value={sampleIndex} onChange={setSampleIndex} min={0} max={Math.max(0, (config?.samples ?? 2) - 2)} step={1}/><h4>实时工况特征向量</h4><div className="compact-grid">{cv.map((value, index) => <input key={index} type="number" value={value} onChange={e => { const next = [...cv]; next[index] = Number(e.target.value); setCv(next) }} />)}</div><button onClick={() => void run()}>执行风险匹配与方案生成</button></section><section className="panel"><h3>预测趋势与适配方案</h3>{option ? <Chart option={option} height={370} /> : <Empty text="设置参数后执行推演。" />}{result?.risks.map((risk: string, index: number) => <article className="result-card" key={risk}><b>{risk}</b><pre>{result.schemes[index]}</pre></article>)}</section></div>{error && <ErrorBox text={error} />}</Page>
}

function formatSilNumber(value: unknown, digits = 4) {
  const number = Number(value)
  return Number.isFinite(number) ? number.toExponential(digits) : '-'
}

function estimateSilLambda(T: number, k: number) {
  const shape = k + 0.5
  const point = (shape - 1 / 3) / T
  const z = 1.959963984540054
  const gammaQuantile = (standardScore: number) => {
    const term = 1 - 1 / (9 * shape) + standardScore / (3 * Math.sqrt(shape))
    return Math.max(0, shape * Math.pow(term, 3)) / T
  }
  return {
    point: point * 1e9,
    low: gammaQuantile(-z) * 1e9,
    high: gammaQuantile(z) * 1e9,
  }
}

function buildSilReport(result: Record<string, any>) {
  const architecture = result.architecture ?? {}
  const ccf = result.ccf ?? {}
  const lines = [
    '='.repeat(60),
    `  表决架构: ${architecture.m}oo${architecture.n}`,
    `  失效率 λ = ${Number(result.lambda_fit).toFixed(2)} FIT`,
    `  测试间隔 TI = ${Number(result.ti).toFixed(0)} h`,
    `  平均修复时间 MRT = ${Number(result.mrt).toFixed(0)} h`,
    `  共因模式: ${ccf.mode ?? '-'}`,
  ]
  if (ccf.mode?.includes('全局')) {
    lines.push(`  共因因子 β = ${Number(ccf.total_beta).toFixed(3)}`)
  } else {
    const partial = Object.entries(ccf.partial_betas ?? {}).map(([order, value]) => `β${order}=${value}`).join(', ')
    lines.push(`  部分共因: ${partial || '-'}`)
  }
  lines.push(
    `  仿真次数 = ${result.simulations}`,
    `  仿真年数 = ${result.years}`,
    '='.repeat(60),
    '',
    '【仿真结果】',
    '',
    `  PFDavg = ${formatSilNumber(result.pfdavg)}  (标准差 ${formatSilNumber(result.std)})`,
    '',
    `  95% 置信区间: [${formatSilNumber(result.confidence_interval?.[0])}, ${formatSilNumber(result.confidence_interval?.[1])}]`,
    '',
    `  SIL 等级 = ${result.sil ?? '-'}`,
  )
  return lines.join('\n')
}

function SilPage({ recommendation }: { recommendation: Recommendation | null }) {
  const [form, setForm] = useState<SilForm>(SIL_DEFAULT_FORM)
  const [lambdaSource, setLambdaSource] = useState<'direct' | 'estimate'>('direct')
  const [lambdaEstimate, setLambdaEstimate] = useState('')
  const [taskId, setTaskId] = useState<string | null>(null)
  const [error, setError] = useState('')
  const task = useTask(taskId)
  useEffect(() => {
    void api<Record<string, any>>('/sil/defaults').then(data => setForm(previous => ({
      ...previous,
      ...data,
      partial_betas: { ...previous.partial_betas, ...(data.partial_betas ?? {}) },
    }))).catch(e => setError(e.message))
  }, [])

  const updateForm = <K extends keyof SilForm>(key: K, value: SilForm[K]) => {
    setForm(previous => ({ ...previous, [key]: value }))
  }
  const nForInputs = Number.isFinite(Number(form.n)) ? Math.max(1, Math.min(10, Math.trunc(Number(form.n)))) : 4
  const partialBetaOrders = Array.from({ length: Math.max(0, nForInputs - 1) }, (_, index) => index + 2)
  const defaultPartialBeta = 0.1 / Math.max(1, nForInputs - 1)
  const setVotingChannels = (value: number) => setForm(previous => {
    const nextN = Number.isFinite(value) ? Math.max(1, Math.min(10, Math.trunc(value))) : 4
    const partialBetas = { ...previous.partial_betas }
    for (let order = 2; order <= nextN; order += 1) {
      if (partialBetas[String(order)] === undefined) partialBetas[String(order)] = 0.1 / Math.max(1, nextN - 1)
    }
    return { ...previous, n: value, partial_betas: partialBetas }
  })
  const setPartialBeta = (order: number, value: number) => setForm(previous => ({
    ...previous,
    partial_betas: { ...previous.partial_betas, [String(order)]: value },
  }))

  const estimateLambda = (T = Number(form.estimate_T), k = Number(form.estimate_k)) => {
    if (!Number.isFinite(T) || !Number.isFinite(k) || T <= 0 || k < 0) {
      setError('运行时间和失效次数必须为有效数字，且运行时间>0、失效次数>=0。')
      return
    }
    const estimate = estimateSilLambda(T, k)
    updateForm('lambda_fit', Number(estimate.point.toFixed(6)))
    setLambdaEstimate(`λ = ${estimate.point.toFixed(2)} FIT  [95% CI: ${estimate.low.toFixed(2)}, ${estimate.high.toFixed(2)}]`)
    setError('')
  }

  const importSampleData = () => {
    const low = Number(form.estimate_low)
    const high = Number(form.estimate_high)
    if (!Number.isFinite(low) || !Number.isFinite(high) || low >= high) {
      setError('低限阈值必须小于高限阈值。')
      return
    }
    // 与 CS 示例数据入口保持相同的总运行时间和示例失效统计口径。
    const sampleT = 10 * 10 * 8760
    const sampleK = 5
    setForm(previous => ({ ...previous, estimate_T: sampleT, estimate_k: sampleK }))
    estimateLambda(sampleT, sampleK)
  }

  const toggleLambdaSource = () => setLambdaSource(previous => previous === 'direct' ? 'estimate' : 'direct')
  const submit = async () => {
    try {
      setError('')
      const partialBetas = Object.fromEntries(partialBetaOrders.map(order => [String(order), Number(form.partial_betas?.[String(order)] ?? defaultPartialBeta)]))
      const nextTask = await post<{ task_id: string }>('/sil/tasks', {
        m: Number(form.m),
        n: nForInputs,
        lambda_fit: Number(form.lambda_fit),
        ti: Number(form.ti),
        mrt: Number(form.mrt),
        nsim: Number(form.nsim),
        years: Number(form.years),
        ccf_mode: form.ccf_mode,
        total_beta: Number(form.total_beta),
        partial_betas: partialBetas,
      })
      setTaskId(nextTask.task_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const result = task?.result as Record<string, any> | undefined
  const progress = useMemo(() => {
    if (task?.status === 'succeeded') return 100
    const latest = [...(task?.logs ?? [])].reverse().find(line => line.includes('仿真进度'))
    const match = latest?.match(/(\d+)\s*\/\s*(\d+)/)
    if (!match) return 0
    return Math.min(100, Math.round(Number(match[1]) / Math.max(1, Number(match[2])) * 100))
  }, [task])
  const running = task?.status === 'queued' || task?.status === 'running'
  const statusText = task?.status === 'failed' || error ? '验证异常' : task?.status === 'succeeded' ? '仿真完成' : running ? '仿真运行中...' : '就绪'
  const report = result ? buildSilReport(result) : '示例参数已加载 (2oo4, λ=111.11 FIT, 全局β=0.1)'
  const histogramLabels = result?.histogram?.centers?.map((value: number) => value.toExponential(1)) ?? result?.histogram?.edges?.slice(0, -1).map((value: number) => value.toExponential(1)) ?? []
  const histogramValues = result?.histogram?.density ?? result?.histogram?.counts ?? []
  const histogramCenters = result?.histogram?.centers ?? result?.histogram?.edges?.slice(0, -1) ?? []
  const meanBinIndex = histogramCenters.length > 0 ? histogramCenters.reduce((closest: number, value: number, index: number, values: number[]) => Math.abs(value - Number(result?.pfdavg)) < Math.abs(values[closest] - Number(result?.pfdavg)) ? index : closest, 0) : 0
  const histogram = result?.histogram ? {
    backgroundColor: '#ffffff',
    grid: { left: 58, right: 20, top: 36, bottom: 56 },
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'category',
      data: histogramLabels,
      name: 'PFDavg',
      nameTextStyle: { color: '#111827' },
      axisLabel: { color: '#111827', rotate: 30, interval: Math.max(0, Math.floor(histogramLabels.length / 7)) },
      axisLine: { lineStyle: { color: '#111827' } },
    },
    yAxis: {
      type: 'value',
      name: '概率密度',
      nameTextStyle: { color: '#111827' },
      axisLabel: { color: '#111827' },
      axisLine: { lineStyle: { color: '#111827' } },
      splitLine: { lineStyle: { color: '#d6dbe2' } },
    },
    series: [{
      type: 'bar',
      data: histogramValues,
      itemStyle: { color: '#87ceeb', borderColor: '#111827', borderWidth: 0.5 },
      markLine: Number(result.pfdavg) > 0 ? {
        symbol: 'none',
        lineStyle: { color: 'red', type: 'dashed' },
        label: { color: 'red', formatter: `均值 = ${Number(result.pfdavg).toExponential(2)}` },
        data: [{ xAxis: meanBinIndex }],
      } : undefined,
    }],
  } as echarts.EChartsOption : null

  return <Page title="在线SIL验证 · GSPN-MC" className="sil-page">
    <div className="sil-layout">
      <section className="sil-input-panel">
        <div className="sil-input-scroll">
          {recommendation && <div className="notice">来自SIS检测：{recommendation.node_name}；频率 {recommendation.frequency.toExponential(3)} 次/年；RRF {recommendation.rrf.toFixed(2)}；目标SIL {recommendation.target_sil}。该信息仅作设计参考，不会自动篡改可靠性参数。</div>}
          <section className="sil-group"><h3>1. 表决架构 (MooN)</h3>
            <div className="sil-form-row"><label htmlFor="sil-m">M (表决阈值):</label><input id="sil-m" type="number" value={form.m} min={1} max={10} step={1} onChange={e => updateForm('m', Number(e.target.value))}/></div>
            <div className="sil-form-row"><label htmlFor="sil-n">N (通道总数):</label><input id="sil-n" type="number" value={form.n} min={1} max={10} step={1} onChange={e => setVotingChannels(Number(e.target.value))}/></div>
          </section>

          <section className="sil-group"><h3>2. 失效率配置</h3>
            {lambdaSource === 'direct' ? <div className="sil-form-row"><label htmlFor="sil-lambda">λ (FIT):</label><input id="sil-lambda" type="number" value={form.lambda_fit} min={0.01} max={100000} step={0.01} onChange={e => updateForm('lambda_fit', Number(e.target.value))}/></div> : <div className="sil-estimator">
              <div className="sil-form-row"><label htmlFor="sil-estimate-t">总运行时间 (h):</label><input id="sil-estimate-t" type="number" value={form.estimate_T} min={0.01} step={1} onChange={e => updateForm('estimate_T', Number(e.target.value))}/></div>
              <div className="sil-form-row"><label htmlFor="sil-estimate-k">失效次数 (等效):</label><input id="sil-estimate-k" type="number" value={form.estimate_k} min={0} step={0.0001} onChange={e => updateForm('estimate_k', Number(e.target.value))}/></div>
              <div className="sil-form-row"><label htmlFor="sil-estimate-low">低限阈值:</label><input id="sil-estimate-low" type="number" value={form.estimate_low} step={1} onChange={e => updateForm('estimate_low', Number(e.target.value))}/></div>
              <div className="sil-form-row"><label htmlFor="sil-estimate-high">高限阈值:</label><input id="sil-estimate-high" type="number" value={form.estimate_high} step={1} onChange={e => updateForm('estimate_high', Number(e.target.value))}/></div>
              <div className="sil-estimator-actions"><button type="button" onClick={importSampleData}>📥 导入示例数据</button><button type="button" onClick={() => estimateLambda()}>📊 估计 λ (手动T/k)</button></div>
              {lambdaEstimate && <p className="sil-lambda-estimate">{lambdaEstimate}</p>}
            </div>}
            <button type="button" className="sil-toggle-button" onClick={toggleLambdaSource}>{lambdaSource === 'direct' ? '切换到 运行数据估计' : '切换到 直接输入'}</button>
          </section>

          <section className="sil-group"><h3>3. 共因失效模式</h3>
            <div className="sil-form-row"><label htmlFor="sil-ccf-mode">共因模式:</label><select id="sil-ccf-mode" value={form.ccf_mode} onChange={e => updateForm('ccf_mode', e.target.value as SilForm['ccf_mode'])}><option value="total">全局共因 (Total β)</option><option value="partial">部分共因 (Partial β)</option></select></div>
            {form.ccf_mode === 'total' ? <div className="sil-form-row"><label htmlFor="sil-total-beta">β (共因因子):</label><input id="sil-total-beta" type="number" value={form.total_beta} min={0} max={0.99} step={0.01} onChange={e => updateForm('total_beta', Number(e.target.value))}/></div> : <div className="sil-beta-list">{partialBetaOrders.map(order => <div className="sil-form-row" key={order}><label htmlFor={`sil-beta-${order}`}>β{order} (影响{order}个通道):</label><input id={`sil-beta-${order}`} type="number" value={form.partial_betas?.[String(order)] ?? defaultPartialBeta} min={0} max={0.99} step={0.0001} onChange={e => setPartialBeta(order, Number(e.target.value))}/></div>)}</div>}
          </section>

          <section className="sil-group"><h3>4. 仿真控制参数</h3>
            <div className="sil-form-row"><label htmlFor="sil-ti">测试间隔 TI (h):</label><input id="sil-ti" type="number" value={form.ti} min={1} max={100000} step={1} onChange={e => updateForm('ti', Number(e.target.value))}/></div>
            <div className="sil-form-row"><label htmlFor="sil-mrt">平均修复 MRT (h):</label><input id="sil-mrt" type="number" value={form.mrt} min={0} max={10000} step={1} onChange={e => updateForm('mrt', Number(e.target.value))}/></div>
            <div className="sil-form-row"><label htmlFor="sil-nsim">仿真次数:</label><input id="sil-nsim" type="number" value={form.nsim} min={1} max={2000} step={1} onChange={e => updateForm('nsim', Number(e.target.value))}/></div>
            <div className="sil-form-row"><label htmlFor="sil-years">仿真年数:</label><input id="sil-years" type="number" value={form.years} min={1001} max={100000} step={1} onChange={e => updateForm('years', Number(e.target.value))}/></div>
          </section>
        </div>
        <div className="sil-run-row"><button type="button" onClick={() => void submit()} disabled={running}>▶ 开始验证</button><progress value={progress} max={100} aria-label="SIL 仿真进度"/><span>{statusText}</span></div>
      </section>
      <section className="sil-output-panel">
        <pre className="sil-result-text" aria-label="SIL 验证文本结果">{report}</pre>
        <div className="sil-chart-panel">{histogram ? <Chart option={histogram} fill className="sil-chart"/> : <div className="sil-chart-placeholder" aria-label="等待仿真结果"/>}</div>
      </section>
    </div>
    {(error || task?.error) && <ErrorBox text={error || task?.error || '验证失败'} />}
  </Page>
}

function ClassificationPage() {
  const [datasets, setDatasets] = useState<Array<{ id: string; name: string }>>([
    { id: 'original', name: '数据集1' },
    { id: 'easy', name: '数据集2' },
    { id: 'hard', name: '数据集3' },
  ])
  const [form, setForm] = useState({ dataset: 'original', epochs: 50, batch_size: 32, learning_rate: 0.001 })
  const [taskId, setTaskId] = useState<string | null>(null)
  const [error, setError] = useState('')
  const task = useTask(taskId)
  useEffect(() => { void api<Array<{ id: string; name: string }>>('/classification/datasets').then(setDatasets).catch(e => setError(e.message)) }, [])
  const submit = async () => {
    try {
      setError('')
      const nextTask = await post<{ task_id: string }>('/classification/tasks', form)
      setTaskId(nextTask.task_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }
  const result = task?.result as Record<string, any> | undefined
  const trainingActive = taskId !== null && (!task || task.status === 'queued' || task.status === 'running')
  const matrix = result?.confusion_matrix as number[][] | undefined
  const classNames = result?.class_names as string[] | undefined
  const confusionData = matrix && classNames
    ? matrix.flatMap((row, rowIndex) => row.map((value, columnIndex) => [columnIndex, rowIndex, value]))
    : []
  const maxCount = Math.max(1, ...confusionData.map(item => Number(item[2])))
  const recalls = matrix
    ? matrix.map((row, index) => row[index] / Math.max(1, row.reduce((sum, value) => sum + value, 0)))
    : []
  const resultChart = matrix && classNames ? {
    animation: false,
    backgroundColor: 'transparent',
    title: [
      { text: '混淆矩阵', left: '15%', textStyle: { color: '#d4e8ff', fontSize: 13 } },
      { text: '识别召回率', left: '67%', textStyle: { color: '#d4e8ff', fontSize: 13 } },
    ],
    grid: [
      { left: 48, right: '53%', top: 42, bottom: 54, containLabel: true },
      { left: '59%', right: 22, top: 42, bottom: 54, containLabel: true },
    ],
    xAxis: [
      { type: 'category', data: classNames, gridIndex: 0, axisLabel: { color: '#d4e8ff', rotate: 20, fontSize: 11 }, axisLine: { lineStyle: { color: '#466385' } } },
      { type: 'category', data: classNames, gridIndex: 1, axisLabel: { color: '#d4e8ff', rotate: 20, fontSize: 11 }, axisLine: { lineStyle: { color: '#466385' } } },
    ],
    yAxis: [
      { type: 'category', data: classNames, gridIndex: 0, inverse: true, axisLabel: { color: '#d4e8ff', fontSize: 11 }, axisLine: { lineStyle: { color: '#466385' } } },
      { type: 'value', min: 0, max: 1, gridIndex: 1, axisLabel: { color: '#d4e8ff', formatter: (value: number) => `${Math.round(value * 100)}%` }, splitLine: { lineStyle: { color: '#203445' } } },
    ],
    visualMap: { min: 0, max: maxCount, calculable: false, orient: 'vertical', left: '47%', top: 72, itemWidth: 9, itemHeight: 110, textStyle: { color: '#a9c6e2', fontSize: 10 }, inRange: { color: ['#132b43', '#286aa3', '#63b9ff'] } },
    series: [
      { type: 'heatmap', xAxisIndex: 0, yAxisIndex: 0, data: confusionData, label: { show: true, color: '#f4f8ff', fontWeight: 600 } },
      { type: 'bar', xAxisIndex: 1, yAxisIndex: 1, data: recalls, barMaxWidth: 28, itemStyle: { color: '#63b9ff', borderColor: '#2e4a63', borderWidth: 1 } },
    ],
  } as echarts.EChartsOption : null
  const statusText = !task ? '状态：待命' : task.status === 'succeeded' && result
    ? `状态：训练完成 (最高精度: ${(Number(result.best_accuracy) * 100).toFixed(2)}%)`
    : task.status === 'failed' ? '状态：训练异常中断' : '状态：正在训练模型...'
  return <Page title="风险动态分析 - 潜在安全威胁识别与自动分类">
    <div className="two-column classification-layout">
      <section className="panel classification-input-panel">
        <h3>输入与训练日志</h3>
        <div className="classification-controls">
          <label>数据集<select value={form.dataset} disabled={trainingActive} onChange={e => setForm({ ...form, dataset: e.target.value })}>{datasets.map(dataset => <option key={dataset.id} value={dataset.id}>{dataset.name}</option>)}</select></label>
          <FormNumber label="Epochs" value={form.epochs} onChange={v => setForm({ ...form, epochs: v })} min={1} max={500} step={1}/>
          <FormNumber label="Batch Size" value={form.batch_size} onChange={v => setForm({ ...form, batch_size: v })} min={8} max={256} step={8}/>
          <FormNumber label="Learning Rate" value={form.learning_rate} onChange={v => setForm({ ...form, learning_rate: v })} min={0.0001} max={0.1} step={0.001}/>
          <button className="classification-run-button" disabled={trainingActive} onClick={() => void submit()}>{trainingActive ? '训练任务执行中…' : '开始训练'}</button>
        </div>
        <TaskPanel task={task}/>
        {!task && <div className="classification-log-placeholder">训练日志将在这里显示。</div>}
      </section>
      <section className="panel classification-result-panel">
        <h3>分类结果</h3>
        {result && matrix && classNames ? <>
          <ClassificationMatrix matrix={matrix} names={classNames} />
          {resultChart && <Chart option={resultChart} fill className="classification-result-chart" />}
        </> : <div className="classification-result-placeholder" aria-label="等待训练结果" />}
      </section>
    </div>
    <div className="module-status-bar">{statusText}</div>
    {error && <ErrorBox text={error} />}
  </Page>
}

function TrainingPage() {
  const [form, setForm] = useState({
    package_dir: '', package_name: 'dnnmpcpkg', mcr_root: 'E:\\MATLAB2024', output_dir: '', model_path: '',
    sample_count: 1000, epochs: 50, hidden_layers: '64,64', dataset_path: '',
  })
  const [taskId, setTaskId] = useState<string | null>(null)
  const [imageMode, setImageMode] = useState<'training' | 'prediction'>('training')
  const [imageRevision, setImageRevision] = useState(0)
  const [imageError, setImageError] = useState(false)
  const [error, setError] = useState('')
  const task = useTask(taskId)
  const taskResult = task?.result as Record<string, any> | undefined
  const result = taskResult?.result as Record<string, any> | undefined
  const running = task?.status === 'queued' || task?.status === 'running'
  const taskProgress = task?.progress
  const progress = task?.status === 'succeeded' ? 100 : running ? Math.max(0, Math.min(100, Number(taskProgress?.percent ?? 0))) : 0
  const progressMessage = task?.status === 'succeeded'
    ? '\u6267\u884c\u5b8c\u6210'
    : task?.status === 'failed'
      ? '\u6267\u884c\u5931\u8d25'
      : taskProgress?.message || (running ? '\u542f\u52a8\u4e2d' : '\u7b49\u5f85\u542f\u52a8')
  const progressLabel = `${progress}%  ${progressMessage}`
  const imageName = imageMode === 'prediction' ? 'prediction_error.png' : 'training_performance.png'
  const imageTitle = imageMode === 'prediction' ? '预测误差 (prediction_error.png)' : '训练曲线 (training_performance.png)'
  const imageRevisionKey = taskProgress?.revision ?? taskProgress?.timestamp ?? taskResult?.image_revision ?? imageRevision
  const imageUrl = `${taskId ? `/api/training/tasks/${taskId}/images` : '/api/training/images'}/${imageName}?v=${encodeURIComponent(String(imageRevisionKey))}`

  useEffect(() => {
    void api<Record<string, any>>('/training/defaults').then(defaults => setForm(previous => ({ ...previous, ...defaults }))).catch(e => setError(e.message))
  }, [])
  useEffect(() => {
    if (task?.status === 'succeeded') {
      setImageRevision(Number(taskResult?.image_revision ?? Date.now()))
      setImageError(false)
    }
  }, [task?.status, taskResult?.image_revision])
  useEffect(() => {
    setImageError(false)
  }, [imageUrl])

  const update = (key: string, value: string | number) => setForm(previous => ({ ...previous, [key]: value }))
  const submit = async () => {
    try {
      setError('')
      setImageMode('training')
      setImageError(false)
      const nextTask = await post<{ task_id: string }>('/training/tasks', form)
      setTaskId(nextTask.task_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const statusText = error || task?.status === 'failed'
    ? '状态：执行失败'
    : task?.status === 'succeeded'
      ? `状态：训练完成${result?.best_accuracy == null ? '' : `（最佳测试准确率：${(Number(result.best_accuracy) * 100).toFixed(2)}%）`}`
      : running ? '状态：正在训练模型...' : '状态：待执行'
  const progressStatusText = running && taskProgress?.message
    ? `状态：${taskProgress.module ? `${taskProgress.module} - ` : ''}${taskProgress.message}`
    : statusText
  const jsonPath = taskResult?.json_path ? String(taskResult.json_path) : '运行后显示 JSON 结果文件路径。'

  return <Page title="控制模型训练评估" className="training-page">
    <div className="training-layout">
      <section className="training-panel training-config-panel">
        <h3>DNNTrain</h3>
        <div className="training-config-scroll">
          <section className="training-subpanel"><h4>路径设置</h4>
            <TrainingPathField label="Python包目录:" value={form.package_dir} onChange={value => update('package_dir', value)} placeholder="可留空；使用当前环境已安装的包" disabled={running}/>
            <TrainingPathField label="Python包名:" value={form.package_name} onChange={value => update('package_name', value)} disabled={running} browse={false}/>
            <TrainingPathField label="MCR_ROOT:" value={form.mcr_root} onChange={value => update('mcr_root', value)} disabled={running}/>
            <TrainingPathField label="输出目录:" value={form.output_dir} onChange={value => update('output_dir', value)} disabled={running}/>
            <TrainingPathField label="模型文件:" value={form.model_path} onChange={value => update('model_path', value)} disabled={running}/>
          </section>
          <section className="training-subpanel training-model-subpanel"><h4>DNNTrain</h4>
            <label className="training-number-field">训练样本数:<input type="number" min={100} max={100000} step={100} value={form.sample_count} onChange={e => update('sample_count', Number(e.target.value))} disabled={running}/></label>
            <label className="training-number-field">训练轮数:<input type="number" min={1} max={5000} step={1} value={form.epochs} onChange={e => update('epochs', Number(e.target.value))} disabled={running}/></label>
            <label>隐藏层规模:<input type="text" value={form.hidden_layers} onChange={e => update('hidden_layers', e.target.value)} placeholder="例如 64,64" disabled={running}/></label>
            <TrainingPathField label="外部数据集:" value={form.dataset_path} onChange={value => update('dataset_path', value)} placeholder="留空则自动生成数据集；选择 .mat 则使用外部 X_data/Y_data" disabled={running} allowClear/>
            <div className="training-run-action"><button type="button" onClick={() => void submit()} disabled={running}>{running ? '训练执行中...' : '运行训练模块'}</button></div>
          </section>
          <section className="training-subpanel"><h4>结果文件</h4><div className="training-result-path">{jsonPath}</div></section>
        </div>
      </section>
      <section className="training-panel training-result-panel">
        <h3>结果图</h3>
        <div className="training-status-panel"><span>{progressStatusText}</span><div className="training-progress-wrap"><progress value={progress} max={100} aria-label="训练进度"/><span>{progressLabel}</span></div></div>
        <div className="training-image-switches"><button type="button" className={imageMode === 'training' ? 'active' : ''} onClick={() => { setImageMode('training'); setImageError(false) }} disabled={running}>训练曲线</button><button type="button" className={imageMode === 'prediction' ? 'active' : ''} onClick={() => { setImageMode('prediction'); setImageError(false) }} disabled={running}>预测误差</button></div>
        <h4 className="training-image-title">{imageTitle}</h4>
        <div className="training-image-frame" role="img" aria-label={imageTitle} style={imageError ? undefined : { backgroundImage: `url("${imageUrl}")`, backgroundSize: 'contain', backgroundPosition: 'center', backgroundRepeat: 'no-repeat' }}>{imageError ? <div className="training-image-placeholder">暂无图像</div> : <img src={imageUrl} alt="" aria-hidden="true" style={{ display: 'none' }} onError={() => setImageError(true)}/>}</div>
      </section>
    </div>
    <div className="module-status-bar training-module-status">{progressStatusText}</div>
    {(error || task?.error) && <ErrorBox text={error || task?.error || '训练执行失败'} />}
  </Page>
}

function MpcPage() {
  const [form, setForm] = useState({
    package_dir: '', package_name: 'dnnmpcpkg', mcr_root: 'E:\\MATLAB2024', output_dir: '', model_path: '',
    sim_time: 1.0, prediction_horizon: 5,
  })
  const [taskId, setTaskId] = useState<string | null>(null)
  const [imageMode, setImageMode] = useState<'trajectory' | 'control' | 'tracking' | 'cost'>('trajectory')
  const [imageRevision, setImageRevision] = useState(0)
  const [imageError, setImageError] = useState(false)
  const [error, setError] = useState('')
  const task = useTask(taskId)
  const taskResult = task?.result as Record<string, any> | undefined
  const result = taskResult?.result as Record<string, any> | undefined
  const running = task?.status === 'queued' || task?.status === 'running'
  const taskProgress = task?.progress
  const progress = task?.status === 'succeeded' ? 100 : running ? Math.max(0, Math.min(100, Number(taskProgress?.percent ?? 0))) : 0
  const progressMessage = task?.status === 'succeeded'
    ? '执行完成'
    : task?.status === 'failed'
      ? '执行失败'
      : taskProgress?.message || (running ? '启动中' : '等待启动')
  const progressLabel = `${progress}%  ${progressMessage}`
  const imageMap = {
    trajectory: ['process_control_trajectory.png', '状态轨迹 (process_control_trajectory.png)'],
    control: ['control_input.png', '控制输入 (control_input.png)'],
    tracking: ['tracking_error.png', '跟踪误差 (tracking_error.png)'],
    cost: ['cost_curve.png', '代价曲线 (cost_curve.png)'],
  } as const
  const [imageName, imageTitle] = imageMap[imageMode]
  const imageRevisionKey = taskProgress?.revision ?? taskProgress?.timestamp ?? taskResult?.image_revision ?? imageRevision
  const imageUrl = `${taskId ? `/api/mpc/tasks/${taskId}/images` : '/api/mpc/images'}/${imageName}?v=${encodeURIComponent(String(imageRevisionKey))}`

  useEffect(() => {
    void api<Record<string, any>>('/mpc/defaults').then(defaults => setForm(previous => ({ ...previous, ...defaults }))).catch(e => setError(e.message))
  }, [])
  useEffect(() => {
    if (task?.status === 'succeeded') {
      setImageRevision(Number(taskResult?.image_revision ?? Date.now()))
      setImageError(false)
    }
  }, [task?.status, taskResult?.image_revision])
  useEffect(() => {
    setImageError(false)
  }, [imageUrl])

  const update = (key: string, value: string | number) => setForm(previous => ({ ...previous, [key]: value }))
  const submit = async () => {
    try {
      setError('')
      setImageMode('trajectory')
      setImageError(false)
      const nextTask = await post<{ task_id: string }>('/mpc/tasks', form)
      setTaskId(nextTask.task_id)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const statusText = error || task?.status === 'failed'
    ? '状态：执行失败'
    : task?.status === 'succeeded'
      ? `状态：MPC 仿真完成${result?.tracking_mse == null ? '' : `（跟踪 MSE：${Number(result.tracking_mse).toFixed(4)}）`}`
      : running ? '状态：正在运行 MPC 仿真...' : '状态：待执行'
  const progressStatusText = running && taskProgress?.message
    ? `状态：${taskProgress.module ? `${taskProgress.module} - ` : ''}${taskProgress.message}`
    : statusText
  const jsonPath = taskResult?.json_path ? String(taskResult.json_path) : '运行后显示 JSON 结果文件路径。'

  return <Page title="优化控制仿真验证" className="training-page mpc-page">
    <div className="training-layout">
      <section className="training-panel training-config-panel">
        <h3>MPC simulation</h3>
        <div className="training-config-scroll">
          <section className="training-subpanel"><h4>路径设置</h4>
            <TrainingPathField label="Python包目录:" value={form.package_dir} onChange={value => update('package_dir', value)} placeholder="可留空；使用当前环境已安装的包" disabled={running}/>
            <TrainingPathField label="Python包名:" value={form.package_name} onChange={value => update('package_name', value)} disabled={running} browse={false}/>
            <TrainingPathField label="MCR_ROOT:" value={form.mcr_root} onChange={value => update('mcr_root', value)} disabled={running}/>
            <TrainingPathField label="输出目录:" value={form.output_dir} onChange={value => update('output_dir', value)} disabled={running}/>
            <TrainingPathField label="模型文件:" value={form.model_path} onChange={value => update('model_path', value)} disabled={running}/>
          </section>
          <section className="training-subpanel training-model-subpanel"><h4>MPC simulation</h4>
            <label className="training-number-field">仿真时长(s):<input type="number" min={0.2} max={20} step={0.2} value={form.sim_time} onChange={e => update('sim_time', Number(e.target.value))} disabled={running}/></label>
            <label className="training-number-field">预测步长:<input type="number" min={1} max={60} step={1} value={form.prediction_horizon} onChange={e => update('prediction_horizon', Number(e.target.value))} disabled={running}/></label>
            <div className="training-run-action"><button type="button" onClick={() => void submit()} disabled={running}>{running ? '仿真执行中...' : '运行 MPC 仿真'}</button></div>
          </section>
          <section className="training-subpanel"><h4>结果文件</h4><div className="training-result-path">{jsonPath}</div></section>
        </div>
      </section>
      <section className="training-panel training-result-panel">
        <h3>结果图</h3>
        <div className="training-status-panel"><span>{progressStatusText}</span><div className="training-progress-wrap"><progress value={progress} max={100} aria-label="MPC 仿真进度"/><span>{progressLabel}</span></div></div>
        <div className="training-image-switches"><button type="button" className={imageMode === 'trajectory' ? 'active' : ''} onClick={() => setImageMode('trajectory')}>状态轨迹</button><button type="button" className={imageMode === 'control' ? 'active' : ''} onClick={() => setImageMode('control')}>控制输入</button><button type="button" className={imageMode === 'tracking' ? 'active' : ''} onClick={() => setImageMode('tracking')}>跟踪误差</button><button type="button" className={imageMode === 'cost' ? 'active' : ''} onClick={() => setImageMode('cost')}>代价曲线</button><button type="button" className="mpc-refresh-button" onClick={() => { setImageRevision(Date.now()); setImageError(false) }}>刷新图像</button></div>
        <h4 className="training-image-title">{imageTitle}</h4>
        <div className="training-image-frame" role="img" aria-label={imageTitle} style={imageError ? undefined : { backgroundImage: `url("${imageUrl}")`, backgroundSize: 'contain', backgroundPosition: 'center', backgroundRepeat: 'no-repeat' }}>{imageError ? <div className="training-image-placeholder">暂无图像</div> : <img src={imageUrl} alt="" aria-hidden="true" style={{ display: 'none' }} onError={() => setImageError(true)}/>}</div>
      </section>
    </div>
    <div className="module-status-bar training-module-status">{progressStatusText}</div>
    {(error || task?.error) && <ErrorBox text={error || task?.error || 'MPC 仿真执行失败'} />}
  </Page>
}

function TrainingPathField({ label, value, onChange, placeholder, disabled, browse = true, allowClear = false }: { label: string; value: string; onChange: (value: string) => void; placeholder?: string; disabled?: boolean; browse?: boolean; allowClear?: boolean }) {
  const fileInput = useRef<HTMLInputElement>(null)
  return <div className="training-path-field"><span>{label}</span><div className="training-path-control"><input type="text" value={value} placeholder={placeholder} onChange={e => onChange(e.target.value)} disabled={disabled}/>{browse && <button type="button" onClick={() => fileInput.current?.click()} disabled={disabled}>选择</button>}{allowClear && <button type="button" onClick={() => onChange('')} disabled={disabled}>清空</button>}<input ref={fileInput} type="file" className="training-hidden-file-input" onChange={e => { const selected = e.target.files?.[0]; if (selected) onChange(selected.name) }} /></div></div>
}

function ClassificationMatrix({ matrix, names }: { matrix: number[][]; names: string[] }) {
  const rowTotals = matrix.map(row => row.reduce((sum, value) => sum + value, 0))
  const columnTotals = names.map((_, column) => matrix.reduce((sum, row) => sum + row[column], 0))
  const total = rowTotals.reduce((sum, value) => sum + value, 0)
  return <div className="matrix classification-matrix"><table><thead><tr><th className="matrix-corner"></th>{names.map(name => <th key={name}>预测-{name}</th>)}<th>合计</th></tr></thead><tbody>{matrix.map((row, index) => <tr key={names[index]}><th>真实-{names[index]}</th>{row.map((value, column) => <td key={column}>{value}</td>)}<td>{rowTotals[index]}</td></tr>)}<tr><th>合计</th>{columnTotals.map((value, index) => <td key={index}>{value}</td>)}<td>{total}</td></tr></tbody></table></div>
}
function ScoreCard({ label, value, tone }: { label: string; value: string | number; tone: string }) { return <div className={`score-card ${tone}`}><small>{label}</small><strong>{value}</strong></div> }
function FormNumber({ label, value, onChange, min, max, step }: { label: string; value: number; onChange: (value: number) => void; min: number; max: number; step: number }) { return <label>{label}<input type="number" value={value} min={min} max={max} step={step} onChange={e => onChange(Number(e.target.value))}/></label> }
function Empty({ text }: { text: string }) { return <div className="empty">{text}</div> }
function ErrorBox({ text }: { text: string }) { return <div className="error">{text}</div> }
function Page({ title, subtitle, className, children }: { title: string; subtitle?: string; className?: string; children: ReactNode }) { const classificationPage = title.startsWith('风险动态分析'); const classes = [classificationPage ? 'classification-page' : '', className ?? ''].filter(Boolean).join(' '); return <main className={classes || undefined} aria-label={title}>{subtitle && <p className="page-subtitle">{subtitle}</p>}{children}</main> }

function App() {
  const [active, setActive] = useState<ModuleId>('sdg')
  const [recommendation, setRecommendation] = useState<Recommendation | null>(null)
  const [openMenu, setOpenMenu] = useState<string | null>(null)
  const activeGroup = navigationGroups.find(group => group.id === 'governance' ? active === 'governance' : group.items.some(item => item.id === active))
  const activeItem = activeGroup?.items.find(item => item.id === active)
  const contentTitle = active === 'governance'
    ? '异构数据治理'
    : `${activeGroup?.name ?? ''} - ${activeItem?.name ?? ''}`
  const selectModule = (moduleId: ModuleId) => {
    setActive(moduleId)
    setOpenMenu(null)
  }
  useEffect(() => {
    const locked = active === 'anomaly' || active === 'training' || active === 'dnn-mpc'
    document.documentElement.classList.toggle('anomaly-page-locked', locked)
    document.body.classList.toggle('anomaly-page-locked', locked)
    return () => {
      document.documentElement.classList.remove('anomaly-page-locked')
      document.body.classList.remove('anomaly-page-locked')
    }
  }, [active])
  const content = active === 'score' ? <ScorePage/> : active === 'anomaly' ? <AnomalyPage/> : active === 'training' ? <TrainingPage/> : active === 'dnn-mpc' ? <MpcPage/> : active === 'sdg' ? <SdgPage onRecommend={value => { setRecommendation(value); selectModule('sil') }}/> : active === 'sil' ? <SilPage recommendation={recommendation}/> : active === 'cdq' ? <CdqPageReplica/> : active === 'classification' ? <ClassificationPage/> : active === 'governance' ? <Page title="异构数据治理" subtitle="该一级导航已保留，当前暂无可用子模块。"><Empty text="暂无可用子模块，后续可在此扩展异构数据接入、清洗和治理能力。"/></Page> : <Page title={activeItem?.name ?? ''} subtitle="该模块仍可在桌面端使用。"><div className="deferred"><h3>桌面端可用，Web版后续迁移</h3><p>此模块依赖 MATLAB Runtime。首期本机 Web 版不会调用或重新编译 Runtime 包，以避免影响当前桌面端。</p></div></Page>
  return <div className={`desktop-shell ${active === 'anomaly' ? 'anomaly-shell' : ''} ${active === 'training' ? 'training-shell' : ''} ${active === 'dnn-mpc' ? 'mpc-shell' : ''}`}><header className="platform-header"><div className="platform-brand"><span className="logo-block">◈</span><h1>流程行业动态风险管控工具集平台</h1></div><div className="header-actions" aria-label="平台工具"><button type="button" title="用户">👤</button><button type="button" title="设置">⚙</button><button type="button" title="退出">⏻</button></div></header><nav className="function-bar" aria-label="功能导航"><div className="function-bar-scroll">{navigationGroups.map(group => { const selected = activeGroup?.id === group.id; const isOpen = openMenu === group.id; const hasMenu = group.items.length > 0; return <div className="nav-menu" key={group.id}><button type="button" className={`top-nav-button ${selected ? 'active' : ''}`} aria-expanded={hasMenu ? isOpen : undefined} onClick={() => { if (!hasMenu) { selectModule('governance'); return } setOpenMenu(isOpen ? null : group.id) }}><span>{group.icon}</span>{group.name}{hasMenu && <b>⌄</b>}</button>{hasMenu && isOpen && <div className="nav-dropdown">{group.items.map(item => <button type="button" key={item.id} className={active === item.id ? 'drop-active' : ''} onClick={() => selectModule(item.id)}>{item.name}{item.deferred && <em>桌面端</em>}</button>)}</div>}</div> })}</div></nav><section className="content-shell"><header className={`content-title-bar ${active === 'sil' ? 'sil-content-title-bar' : active === 'anomaly' ? 'anomaly-content-title-bar' : active === 'training' || active === 'dnn-mpc' ? 'training-content-title-bar' : ''}`}><h2>{contentTitle}</h2></header><div className={`content-body ${active === 'classification' ? 'classification-content-body' : active === 'cdq' ? 'cdq-content-body' : active === 'sdg' ? 'sdg-content-body' : active === 'sil' ? 'sil-content-body' : active === 'anomaly' ? 'anomaly-content-body' : active === 'training' || active === 'dnn-mpc' ? 'training-content-body' : ''}`}>{content}</div></section></div>
}

export default App
