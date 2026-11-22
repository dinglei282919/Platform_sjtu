import { useEffect } from 'react'
import { api, post } from './api'
import { readSession, writeSession } from './taskSession'

type Handle = { id: string; owner: string }
export const emptyHandle: Handle = { id: '', owner: '' }

function liveOwner(path: string, storage: string, event: string) {
  let creating: Promise<Handle> | null = null
  function stored() {
    const h = readSession(storage, emptyHandle)
    return h.id && h.owner ? h : null
  }
  function save(h: Handle) {
    writeSession(storage, {id: h.id, owner: h.owner})
    window.dispatchEvent(new Event(event))
  }
  async function ensure() {
    const h = stored()
    if (h) return h
    if (!creating) creating = post<Handle>(path, { auto_start: true })
      .then(next => { save(next); return next })
      .finally(() => { creating = null })
    return creating
  }
  function request<T>(h: Handle, suffix: string, body?: unknown, keepalive = false) {
    return api<T>(path+'/'+h.id+suffix, {
      headers: {'X-Control-Owner': h.owner, 'Content-Type': 'application/json'}, keepalive,
      method: body === undefined ? 'GET' : 'POST', body: body === undefined ? undefined : JSON.stringify(body),
    })
  }
  return {stored, save, ensure, request, path, event}
}

export const dnnOwner = liveOwner('/cstr/sessions', 'cstr-control-owner', 'cstr-owner-changed')
export const healingOwner = liveOwner('/cstr/healing/sessions', 'cstr-healing-owner', 'cstr-healing-owner-changed')

// Both connections belong to the platform shell and survive navigation away from the module.
export function useLivePresence(owner: typeof dnnOwner) {
  useEffect(() => {
    let controller: AbortController | undefined
    let timer: number
    function connect() {
      controller?.abort()
      clearTimeout(timer)
      const h = owner.stored()
      if (!h) return
      const current = new AbortController()
      controller = current
      async function listen(handle: Handle) {
        try {
          const response = await fetch('/api'+owner.path+'/'+handle.id+'/presence', {
            headers: {'X-Control-Owner': handle.owner}, signal: current.signal,
          })
          if (response.status===404 && owner.stored()?.id===handle.id && !current.signal.aborted) {
            owner.save(emptyHandle)
            return
          }
          if (!response.ok || !response.body) throw new Error('连接已断开')
          const reader = response.body.getReader()
          try { while (!(await reader.read()).done) { /* Open stream holds the lease. */ } }
          finally { reader.releaseLock() }
        } catch { /* Retry transient connections with the same owner. */ }
        if (!current.signal.aborted) timer = window.setTimeout(connect, 1500)
      }
      void listen(h)
    }
    connect()
    window.addEventListener(owner.event, connect)
    return () => { window.removeEventListener(owner.event, connect); controller?.abort(); clearTimeout(timer) }
  }, [owner])
}
