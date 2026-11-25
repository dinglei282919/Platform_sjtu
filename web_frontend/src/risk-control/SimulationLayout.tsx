import type { ReactNode } from 'react'

type Props = {
  controls: ReactNode
  controlsKey?: string
  children: ReactNode
}

export function SimulationLayout({ controls, controlsKey, children }: Props) {
  return <div className="two-column cstr-layout">
    <aside className="panel" key={controlsKey}>{controls}</aside>
    <section className="panel cstr-results">{children}</section>
  </div>
}
