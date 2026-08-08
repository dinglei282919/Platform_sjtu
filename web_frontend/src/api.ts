type ApiValidationIssue = {
  type?: string
  loc?: Array<string | number>
  msg?: string
  ctx?: Record<string, unknown>
}

const fieldLabels: Record<string, string> = {
  mcr_root: 'MATLAB Runtime 路径',
  attack_min_pct: '随机攻击强度最小值',
  attack_max_pct: '随机攻击强度最大值',
  measurement_noise_pct: '测量噪声强度',
  process_disturbance_pct: '过程扰动强度',
}

function formatValidationIssue(issue: ApiValidationIssue): string {
  const location = issue.loc ?? []
  const fieldName = String(location[location.length - 1] ?? '')
  const label = fieldName === 'body' ? '' : fieldLabels[fieldName] ?? fieldName
  const unit = fieldName.endsWith('_pct') ? '%' : ''
  if (issue.type === 'greater_than_equal') return `${label}不能小于 ${String(issue.ctx?.ge)}${unit}。`
  if (issue.type === 'less_than_equal') return `${label}不能大于 ${String(issue.ctx?.le)}${unit}。`
  if (issue.type === 'finite_number') return `${label}必须是有限数值。`
  if (issue.type === 'float_parsing' || issue.type === 'float_type') return `${label}必须是有效数值。`
  if (issue.type === 'string_type') return `${label}必须是有效文本。`
  const message = issue.msg?.replace(/^Value error,\s*/i, '').trim()
  if (issue.type === 'value_error' && message) return message
  if (message) return label && !message.includes(label) ? `${label}：${message}` : message
  return label ? `${label}参数无效。` : '请求参数无效。'
}

function formatApiError(detail: unknown, status: number): string {
  if (typeof detail === 'string' && detail.trim()) return detail
  if (Array.isArray(detail)) {
    const messages = detail.map(item => formatValidationIssue(item as ApiValidationIssue))
    if (messages.length) return messages.join('\n')
  }
  return `请求失败（${status}）`
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  })
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(formatApiError(body.detail, response.status))
  return body as T
}

export const post = <T>(path: string, payload?: unknown) => api<T>(path, {
  method: 'POST', body: JSON.stringify(payload ?? {}),
})
