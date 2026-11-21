export type PulseReceipt = {
  kind: 'disturbance' | 'clear_disturbance'
  key: string
  run: string
  seq: number
}

type PulseSnapshot = {
  run: string
  seq: number
  status: string
  events: { key?: string }[]
  disturbance?: { phase: string; clear_requested: boolean } | null
  disturbance_display?: { status: string } | null
}

export function pulseControls(snapshot: PulseSnapshot | null, receipt: PulseReceipt | null) {
  if (!snapshot || snapshot.status === 'ended') {
    return { canApply: false, canClear: false, status: snapshot?.disturbance_display?.status ?? '' }
  }

  // The command acknowledgement can precede the sample poll, even within the same interval.
  if (receipt && receipt.run === snapshot.run && snapshot.seq <= receipt.seq &&
      !snapshot.events.some(event => event.key === receipt.key)) {
    const applied = receipt.kind === 'disturbance'
    return { canApply: false, canClear: applied, status: applied ? '施加指令已接收，可提前撤除' : '撤除指令已接收' }
  }

  const trial = snapshot.disturbance
  const canClear = trial?.phase === 'queued' || trial?.phase === 'active'
  return {
    canApply: !canClear && trial?.phase !== 'recovery',
    canClear: canClear && !trial?.clear_requested,
    status: snapshot.disturbance_display?.status ?? '',
  }
}
