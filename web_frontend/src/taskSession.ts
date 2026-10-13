import { useEffect, useState } from 'react'

// Tab-scoped: keep a task across navigation/reload without mixing browser tabs.
const prefix = 'platform-sjtu.tasks.v1.'
const memory = new Map<string, unknown>()

function matches(value: unknown, fallback: unknown): boolean {
  if (fallback === null) return value === null || (typeof value === 'string' && value.length > 0 && value.length < 200)
  if (typeof fallback === 'number') return typeof value === 'number' && Number.isFinite(value)
  if (typeof fallback !== 'object') return typeof value === typeof fallback
  if (!value || Array.isArray(value) || typeof value !== 'object') return false
  return Object.entries(fallback as Record<string, unknown>).every(([key, item]) => matches((value as Record<string, unknown>)[key], item))
}

export function readSession<T>(key: string, fallback: T): T {
  let value = memory.get(key)
  try {
    const raw = window.sessionStorage.getItem(prefix + key)
    if (raw !== null) value = JSON.parse(raw)
  } catch { /* Storage may be disabled; retain the in-page session. */ }
  return matches(value, fallback) ? value as T : fallback
}

export function writeSession(key: string, value: unknown): void {
  memory.set(key, value)
  try { window.sessionStorage.setItem(prefix + key, JSON.stringify(value)) } catch { /* In-page navigation still works. */ }
}

export function useStoredState<T>(key: string, fallback: T) {
  const [value, setValue] = useState<T>(() => readSession(key, fallback))
  useEffect(() => { writeSession(key, value) }, [key, value])
  return [value, setValue] as const
}
