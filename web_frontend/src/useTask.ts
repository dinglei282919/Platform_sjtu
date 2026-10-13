import { useEffect, useState } from 'react'
import { ApiError, api } from './api'

type TaskState = { id: string; status: string }

/** A query failure is not a task failure. Only a confirmed 404 clears the ID. */
export function useTask<T extends TaskState>(taskId: string | null, onMissing: (id: null) => void) {
  const [task, setTask] = useState<T | null>(null)
  const [pollingError, setPollingError] = useState('')
  useEffect(() => {
    setTask(null)
    if (!taskId) return
    setPollingError('')
    let active = true
    let timer: ReturnType<typeof setTimeout> | undefined
    let retryDelay = 1000
    const controller = new AbortController()
    const load = async () => {
      try {
        const next = await api<T>(`/tasks/${taskId}`, { signal: controller.signal })
        if (!active) return
        setTask(next)
        setPollingError('')
        retryDelay = 1000
        if (['succeeded', 'failed', 'cancelled'].includes(next.status)) return
      } catch (error) {
        if (!active) return
        if (error instanceof ApiError && error.status === 404) {
          setTask(null)
          onMissing(null)
          setPollingError('任务不存在或记录已被移除，请重新运行。')
          return
        }
        setPollingError(`状态查询暂时失败，${retryDelay / 1000} 秒后自动重试：${error instanceof Error ? error.message : String(error)}`)
        timer = setTimeout(load, retryDelay)
        retryDelay = Math.min(retryDelay * 2, 5000)
        return
      }
      timer = setTimeout(load, 1000)
    }
    void load()
    return () => { active = false; controller.abort(); clearTimeout(timer) }
  }, [taskId, onMissing])
  return { task, pollingError }
}
