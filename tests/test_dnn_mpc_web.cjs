// Run: node --test tests/test_dnn_mpc_web.cjs (after npm install in web_frontend).
// Execute the actual polling effect with controlled HTTP responses and timers.
// Browser checks separately cover rendering, resubmission and navigation.
const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const ts = require('../web_frontend/node_modules/typescript')

const frontend = path.join(__dirname, '../web_frontend/src')
const source = ts.createSourceFile('DnnMpcPage.tsx', fs.readFileSync(path.join(frontend, 'DnnMpcPage.tsx'), 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
let pollingEffect, runningExpression
function visit(node) {
  if (ts.isCallExpression(node) && node.expression.getText(source) === 'useEffect' && node.getText(source).includes('/tasks/${taskId}')) pollingEffect = node.getText(source)
  if (ts.isVariableDeclaration(node) && node.name.getText(source) === 'running') runningExpression = node.initializer.getText(source)
  ts.forEachChild(node, visit)
}
visit(source)
assert.ok(pollingEffect && runningExpression, 'DNN-MPC polling effect must be found')
const transpile = code => ts.transpileModule(code, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
const flush = () => new Promise(resolve => setImmediate(resolve))

function fixture(mode, responses, status = 'running') {
  const timers = new Map()
  let sequence = 0, requests = 0, cleanup
  const context = vm.createContext({
    Error, String, Math, exports: {},
    fetch: async () => {
      const response = responses[Math.min(requests++, responses.length - 1)]
      if (response instanceof Error) throw response
      return { ok: response === 200, status: response, json: async () => response === 200
        ? { id: 'task-before-restart', status, logs: [], result: { model_path: 'C:/trained.npz' } }
        : { detail: `HTTP ${response}` } }
    },
    useEffect: effect => { cleanup = effect() },
    setTimeout: (fn, delay) => { timers.set(++sequence, { fn, delay }); return sequence },
    clearTimeout: id => timers.delete(id),
  })
  vm.runInContext(transpile(fs.readFileSync(path.join(frontend, 'api.ts'), 'utf8')), context)
  vm.runInContext(transpile(`
    const taskId = 'task-before-restart';
    let currentTaskId = taskId, task = { id: taskId, status: 'running' }, submitting = false;
    const mode = ${JSON.stringify(mode)}, training = mode === 'training';
    const sessions = { mpc: { taskId: 'finished-mpc', form: { model_path: 'C:/trained.npz' }, manualModel: false }, [mode]: { taskId, form: { model_path: 'C:/chosen.npz' }, manualModel: true } };
    let error = '';
    const setTask = value => task = value, setTaskId = value => currentTaskId = value, setError = value => error = value;
    ${pollingEffect};
    function snapshot() { const taskId = currentTaskId; return { taskId, task, running: (${runningExpression}), error, saved: sessions[mode], mpc: sessions.mpc }; }
  `), context)
  return {
    snapshot: () => JSON.parse(JSON.stringify(vm.runInContext('snapshot()', context))),
    timers,
    requests: () => requests,
    stop: () => cleanup(),
    retry: async () => {
      const [id, timer] = timers.entries().next().value
      timers.delete(id)
      timer.fn()
      await flush()
    },
  }
}

for (const mode of ['training', 'mpc']) {
  test(`${mode}: 404 clears task and saved session without losing chosen model`, async () => {
    const page = fixture(mode, [404])
    await flush()
    const state = page.snapshot()
    assert.equal(state.taskId, null)
    assert.equal(state.task, null)
    assert.equal(state.saved.taskId, null)
    assert.equal(state.saved.form.model_path, 'C:/chosen.npz')
    assert.equal(state.saved.manualModel, true)
    assert.equal(state.running, false)
    assert.match(state.error, /任务不存在/)
    assert.equal(page.timers.size, 0)
    page.stop()
  })

  test(`${mode}: transient HTTP and network failures retain task and recover`, async () => {
    const page = fixture(mode, [503, new Error('network disconnected'), 200, 404])
    await flush()
    for (let attempt = 0; attempt < 2; attempt++) {
      const state = page.snapshot()
      assert.equal(state.running, true)
      assert.equal(state.taskId, 'task-before-restart')
      assert.equal(state.saved.taskId, state.taskId)
      assert.match(state.error, /自动重试/)
      await page.retry()
    }
    assert.equal(page.snapshot().error, '')
    assert.equal(page.snapshot().task.status, 'running')
    await page.retry()
    assert.equal(page.snapshot().running, false)
    assert.equal(page.timers.size, 0)
    assert.equal(page.requests(), 4)
    page.stop()
  })

  test(`${mode}: late 404 after unmount cannot erase a saved task`, async () => {
    const page = fixture(mode, [404])
    page.stop()
    await flush()
    assert.equal(page.snapshot().saved.taskId, 'task-before-restart')
    assert.equal(page.snapshot().error, '')
    assert.equal(page.timers.size, 0)
  })
}


test('revisiting completed training retains the completed MPC task', async () => {
  const page = fixture('training', [200], 'succeeded')
  await flush()
  assert.equal(page.snapshot().task.status, 'succeeded')
  assert.equal(page.snapshot().mpc.taskId, 'finished-mpc')
  assert.equal(page.timers.size, 0)
  page.stop()
})
