// Run: node --test tests/test_task_polling.cjs (uses the frontend's installed TypeScript).
// Exercise the real hook, API and storage modules with deterministic timers/HTTP.
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const ts = require('../web_frontend/node_modules/typescript')
const flush = () => new Promise(resolve => setImmediate(resolve))
const source = name => ts.transpileModule(fs.readFileSync(path.join(__dirname, '../web_frontend/src', name + '.ts'), 'utf8'), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText
function load(name, globals) {
  const context = vm.createContext({ Error, AbortController, ...globals, exports: {} })
  vm.runInContext(source(name), context)
  return context.exports
}
function fixture(responses) {
  const states = [], timers = new Map(), signals = []
  let cleanup, requests = 0, sequence = 0, taskId = 'saved-task', cleared = 0
  const api = load('api', {
    fetch: async (_url, init) => {
      signals.push(init.signal)
      const response = responses[Math.min(requests++, responses.length - 1)]
      const resolved = typeof response === 'function' ? await response() : response
      if (resolved instanceof Error) throw resolved
      return { ok: typeof resolved !== 'number', status: typeof resolved === 'number' ? resolved : 200,
        json: async () => typeof resolved === 'number' ? { detail: `HTTP ${resolved}` } : resolved }
    },
  })
  const react = {
    useState: value => { const index = states.length; states.push(value); return [value, next => { states[index] = next }] },
    useEffect: effect => { cleanup = effect() },
  }
  const hook = load('useTask', {
    require: name => name === 'react' ? react : api,
    setTimeout: (fn, delay) => { timers.set(++sequence, { fn, delay }); return sequence },
    clearTimeout: id => timers.delete(id),
  })
  hook.useTask(taskId, id => { taskId = id; cleared++ })
  return {
    snapshot: () => ({ task: states[0], error: states[1], taskId, cleared, requests }),
    timers, signals, stop: () => cleanup(),
    tick: async () => { const [id, timer] = timers.entries().next().value; timers.delete(id); timer.fn(); await flush() },
  }
}
const running = { id: 'saved-task', status: 'running', progress: { percent: 40 } }

test('temporary HTTP/network errors retain last progress and task ID, then recover', async () => {
  const page = fixture([running, 503, new Error('offline'), running])
  await flush()
  await page.tick()
  assert.equal(page.snapshot().task, running)
  assert.equal(page.snapshot().taskId, 'saved-task')
  assert.match(page.snapshot().error, /自动重试/)
  assert.equal([...page.timers.values()][0].delay, 1000)
  await page.tick()
  assert.equal([...page.timers.values()][0].delay, 2000)
  assert.equal(page.snapshot().cleared, 0)
  await page.tick()
  assert.equal(page.snapshot().error, '')
  assert.equal([...page.timers.values()][0].delay, 1000)
  page.stop()
})

test('only a confirmed 404 clears the saved ID and stops polling', async () => {
  const page = fixture([running, 500, 404])
  await flush(); await page.tick()
  assert.equal(page.snapshot().cleared, 0)
  await page.tick()
  assert.equal(page.snapshot().taskId, null)
  assert.equal(page.snapshot().task, null)
  assert.equal(page.snapshot().cleared, 1)
  assert.match(page.snapshot().error, /任务不存在/)
  assert.equal(page.timers.size, 0)
  page.stop()
})

test('repeated temporary failure backs off with a five-second cap', async () => {
  const page = fixture([503])
  await flush()
  for (const delay of [1000, 2000, 4000, 5000, 5000]) {
    assert.equal([...page.timers.values()][0].delay, delay)
    assert.equal(page.timers.size, 1)
    await page.tick()
  }
  assert.equal(page.snapshot().taskId, 'saved-task')
  page.stop()
})

for (const status of ['succeeded', 'failed', 'cancelled']) {
  test(`terminal ${status} is retained without further polling`, async () => {
    const page = fixture([{ ...running, status }])
    await flush()
    assert.equal(page.snapshot().task.status, status)
    assert.equal(page.snapshot().cleared, 0)
    assert.equal(page.timers.size, 0)
    page.stop()
  })
}

test('an outstanding request cannot overlap; unmount aborts and ignores its late 404', async () => {
  let release
  const page = fixture([() => new Promise(resolve => { release = resolve })])
  await flush()
  assert.equal(page.snapshot().requests, 1)
  assert.equal(page.timers.size, 0)
  page.stop()
  assert.equal(page.signals[0].aborted, true)
  release(404)
  await flush()
  assert.equal(page.snapshot().cleared, 0)
  assert.equal(page.snapshot().error, '')
  assert.equal(page.timers.size, 0)
})

function storageFixture(storage = new Map()) {
  return load('taskSession', { require: () => ({}), window: { sessionStorage: {
    getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value),
  } } })
}
const fallback = { taskId: null, form: { duration: 10, path: '' } }
test('task and parameters survive module reload; clearing ID persists', () => {
  const storage = new Map(), first = storageFixture(storage)
  const saved = { taskId: 'saved-task', form: { duration: 5, path: 'chosen.npz' } }
  first.writeSession('simulation', saved)
  const reloaded = storageFixture(storage)
  assert.equal(JSON.stringify(reloaded.readSession('simulation', fallback)), JSON.stringify(saved))
  reloaded.writeSession('simulation', { ...saved, taskId: null })
  assert.equal(storageFixture(storage).readSession('simulation', fallback).taskId, null)
  assert.equal(storageFixture().readSession('simulation', fallback), fallback, 'Other tabs remain independent')
})
test('invalid persisted sessions fall back safely', () => {
  for (const raw of ['{', '[]', 'null', '{"taskId":42,"form":{}}', '{"taskId":null,"form":{"duration":"bad","path":""}}']) {
    const module = storageFixture(new Map([['platform-sjtu.tasks.v1.simulation', raw]]))
    assert.equal(module.readSession('simulation', fallback), fallback)
  }
})
test('disabled browser storage still preserves in-page navigation', () => {
  const module = load('taskSession', { require: () => ({}), window: { get sessionStorage() { throw new Error('disabled') } } })
  const saved = { taskId: 'saved-task', form: { duration: 5, path: 'chosen.npz' } }
  module.writeSession('simulation', saved)
  assert.equal(module.readSession('simulation', fallback), saved)
})
