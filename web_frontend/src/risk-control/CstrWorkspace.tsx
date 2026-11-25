import { Component, type ReactNode } from 'react'
import { Page } from '../PlatformUI'
import { dnnOwner, fdiOwners, healingOwner, replayOwner, useLivePresence } from './cstrOwner'
import { DNN_TITLE, DnnMpcPage } from './DnnMpcPage'
import { HEALING_TITLE, HealingPage } from './HealingPage'
import { FDI_TITLE, FdiPage } from './FdiPage'
import { REPLAY_TITLE, ReplayPage } from './ReplayPage'
import './risk-control.css'

export type RiskId =
  | 'risk-overview'
  | 'dnn-mpc'
  | 'fdi'
  | 'replay'
  | 'self-healing'
export const riskEntries: { id: RiskId; name: string }[] = [
  { id: 'risk-overview', name: '综合仿真展示' },
  { id: 'dnn-mpc', name: DNN_TITLE },
  { id: 'self-healing', name: HEALING_TITLE },
  { id: 'fdi', name: FDI_TITLE },
  { id: 'replay', name: REPLAY_TITLE },
]

export function useCstrPresence() {
  useLivePresence(dnnOwner)
  useLivePresence(healingOwner)
  useLivePresence(fdiOwners.measurement)
  useLivePresence(fdiOwners.execution)
  useLivePresence(replayOwner)
}

class CstrPageBoundary extends Component<{ active: RiskId; children: ReactNode }, { failed: boolean }> {
  state = { failed: false }

  static getDerivedStateFromError() { return { failed: true } }

  componentDidUpdate(previous: { active: RiskId }) {
    if (this.state.failed && previous.active !== this.props.active) this.setState({ failed: false })
  }

  render() {
    if (!this.state.failed) return this.props.children
    return <Page title="风险管控优化决策">
      <div role="alert" className="error">
        <strong>当前控制页面加载失败</strong>
        <p>可以重新打开当前页面，也可以通过顶部菜单切换模块。</p>
        <button type="button" className="secondary" onClick={() => this.setState({ failed: false })}>重新打开页面</button>
      </div>
    </Page>
  }
}

export function CstrWorkspace({ active }: { active: RiskId }) {
  return <CstrPageBoundary active={active}>
    {/* Keep drafts and the training task when switching between control modules. */}
    <DnnMpcPage visible={active === 'dnn-mpc'} />
    {active === 'self-healing' && <HealingPage />}
    {active === 'fdi' && <FdiPage />}
    {active === 'replay' && <ReplayPage />}
    {active === 'risk-overview' && <Page title="综合仿真展示">
      <section className="deferred">
        <h3>综合仿真展示待接入</h3>
        <p>从顶部“风险管控优化决策”菜单可分别运行 DNN-MPC、自愈控制、FDI 和重放控制试验。</p>
      </section>
    </Page>}
  </CstrPageBoundary>
}
