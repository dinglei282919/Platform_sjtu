
type State = 'idle' | 'running' | 'succeeded' | 'failed' | 'cancelled'
const labels: Record<State, string> = { idle: '待执行', running: '运行中', succeeded: '已完成', failed: '失败', cancelled: '已取消' }

/** Shared task presentation for the training, MPC and FDI pages. */
export function ControlTaskStatus({ state, message, percent, label }: {
  state: State; message: string; percent: number; label: string
}) {
  const value = state === 'succeeded' ? 100 : state === 'idle' ? 0
    : Math.round(Math.min(99, Math.max(0, Number.isFinite(percent) ? percent : 0)))
  return <div className="control-task-status" data-state={state}>
    <div className="control-task-summary">
      <span className="control-task-badge">{labels[state]}</span>
      <span className="control-task-message" role="status" aria-live="polite">{message}</span>
      <span className="control-task-percent" aria-hidden="true">{value}%</span>
    </div>
    <progress value={value} max={100} aria-label={label}/>
  </div>
}
