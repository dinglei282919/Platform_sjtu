import type { WheelEvent } from 'react'

// Preserve native panel scrolling instead of stepping a focused parameter.
export function scrollControlParameters(event: WheelEvent<HTMLElement>) {
  const target = event.target
  if (target instanceof HTMLInputElement && ['number', 'range'].includes(target.type)
      || target instanceof HTMLSelectElement) target.blur()
}
