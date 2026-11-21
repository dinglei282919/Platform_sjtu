const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const ts = require('../web_frontend/node_modules/typescript')

const file = path.join(__dirname, '../web_frontend/src/cstrDisturbance.ts')
const compiled = ts.transpileModule(fs.readFileSync(file, 'utf8'), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
})
const moduleExports = {}
vm.runInNewContext(compiled.outputText, { exports: moduleExports })
const { pulseControls } = moduleExports

const before = { run: 'run-a', seq: 22, status: 'running', events: [], disturbance: null }
const apply = { run: before.run, seq: 22, key: 'pulse', kind: 'disturbance' }
const clear = { run: before.run, seq: 23, key: 'withdraw', kind: 'clear_disturbance' }
const active = { ...before, seq: 23, events: [{ key: apply.key }], disturbance: { phase: 'active', cycles: 5, clear_requested: false } }

test('a five-cycle pulse is withdrawable on acknowledgement while the sample poll is delayed', () => {
  assert.equal(pulseControls(before, null).canClear, false)
  assert.equal(pulseControls(before, apply).canClear, true)
  assert.equal(pulseControls(before, apply).canApply, false)
  assert.equal(pulseControls(active, apply).canClear, true)
  assert.equal(pulseControls(active, clear).canClear, false)
})

test('the session phase takes precedence once a receipt appears in the journal, including during pause', () => {
  const waiting = { ...before, status: 'paused', events: [{ key: apply.key }], disturbance: { phase: 'queued', clear_requested: false } }
  assert.equal(pulseControls(waiting, apply).canClear, true)
  const cancelled = { ...waiting, events: [{ key: apply.key }, { key: clear.key }], disturbance: { phase: 'cancelled', clear_requested: false } }
  assert.equal(pulseControls(cancelled, clear).canApply, true)
  const withdrawing = { ...active, events: [{ key: clear.key }], disturbance: { ...active.disturbance, clear_requested: true } }
  assert.equal(pulseControls(withdrawing, clear).canClear, false)
})

test('a stale acknowledgement cannot reopen an expired pulse or carry over to a reset or ended run', () => {
  const recovery = { ...before, seq: 27, events: [], disturbance: { phase: 'recovery', clear_requested: false } }
  assert.equal(pulseControls(recovery, apply).canClear, false)
  assert.equal(pulseControls(recovery, apply).canApply, false)
  assert.equal(pulseControls({ ...before, run: 'run-b', seq: 0 }, apply).canClear, false)
  const ended = pulseControls({ ...active, status: 'ended', disturbance_display: { status: '运行已结束' } }, apply)
  assert.equal(ended.canClear, false)
  assert.equal(ended.status, '运行已结束')
  assert.equal(pulseControls(null, apply).canClear, false)
})
