export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(body.detail ?? `请求失败（${response.status}）`)
  return body as T
}

export const post = <T>(path: string, payload?: unknown) => api<T>(path, {
  method: 'POST', body: JSON.stringify(payload ?? {}),
})
