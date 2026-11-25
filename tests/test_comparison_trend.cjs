const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const ts = require('../web_frontend/node_modules/typescript')
const jsx = require('../web_frontend/node_modules/react/jsx-runtime')

function load(file, imports) {
  const source = ts.transpileModule(fs.readFileSync(path.join(__dirname,'../web_frontend/src/risk-control', file), 'utf8'), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
  })
  const result = {}
  vm.runInNewContext(source.outputText, { exports: result, require: name => imports[name] })
  return result
}
const sampleChart = Symbol('SampleChart')
const component = load('ComparisonTrend.tsx', { 'react/jsx-runtime': jsx, './SampleChart': { SampleChart: sampleChart },
  './trendGestures':load('trendGestures.ts', {}) })

function chart(rows, intervals = [], navigation = {}) {
  const props = {
    title: '釜温', unit: 'K', rows, period: 0.1, intervals, ...navigation,
    lines: [
      { name: '估计值', color: '#88aaff', read: row => row.estimate },
      { name: '设定值', color: '#cccccc', reference: true, dashed: true, read: () => 431 },
    ],
  }
  const section = component.ComparisonTrend(props)
  return section?.props.children.find(child => child?.type === sampleChart)?.props
}

test('plot and hover read the same sample, preserving missing predictions and zero', () => {
  const rows = [{ t: 5, estimate: null }, { t: 5.1, estimate: 0 }]
  const { option, readingsOf } = chart(rows, [[4, 6]])
  const [estimated, reference] = option.series
  assert.equal(estimated.data[0][1], null)
  assert.equal(readingsOf(rows[0]).length, 1)
  assert.equal(readingsOf(rows[1])[0][1], estimated.data[1][1])
  assert.equal(estimated.data[1][1], 0)
  assert.equal(reference.lineStyle.type, 'dashed')
  const [start, stop] = estimated.markArea.data[0]
  assert.equal(start.xAxis, 5)
  assert.equal(stop.xAxis, rows[1].t + 0.1)
})

test('one sample has a full interval and shading follows the selected window', () => {
  const rows = [{ t: 5, estimate: 430 }]
  assert.equal(chart([]), undefined)
  const single = chart(rows, [[5, null]]).option
  assert.equal(single.xAxis.max, 5.1)
  assert.equal(single.series[0].showSymbol, true)
  assert.equal(single.series[0].markArea.data[0][1].xAxis, 5.1)
  for (const intervals of [[], [[3, 4]], [[4, 5]], [[6, 7]]]) {
    assert.equal(chart(rows, intervals).option.series[0].markArea, undefined)
  }
})

test('nearly constant startup temperatures remain visible on an adjustable axis', () => {
  const rows = [{t:0,estimate:431}, {t:0.1,estimate:431 + 1e-12}]
  const {option,readingsOf} = chart(rows)
  assert.ok(option.yAxis.min < 431)
  assert.ok(option.yAxis.max > rows[1].estimate)
  assert.equal(readingsOf(rows[1])[0][1],rows[1].estimate)
  assert.equal(chart([{t:0,estimate:425}]).option.yAxis.min,undefined)
  assert.equal(chart([],[],{domain:[0,12]}).option.yAxis.max,undefined)
})

test('repeated exposures keep the intervening nominal intervals unshaded', () => {
  const rows = [{ t: 5, estimate: 430 }, { t: 7, estimate: 431 }]
  const { option } = chart(rows, [[2, 3], [4, 5.5], [5.6, 6.2], [6.9, null], [8, 9]])
  const bands = Array.from(option.series[0].markArea.data, ([a, b]) => [a.xAxis, b.xAxis])
  assert.deepEqual(bands, [[5, 5.5], [5.6, 6.2], [6.9, 7.1]])
})

test('pan keeps the cursor on a recorded sample and leaves uncomputed time blank', () => {
  const {comparisonDomain, pannedPosition} = component
  assert.deepEqual(Array.from(comparisonDomain(239,.1)),[12,24])
  const left = pannedPosition('run',240,-100,.1)
  assert.equal(left.first,-119)
  assert.equal(left.at,0)
  const right = pannedPosition('run',4,100,.1)
  assert.equal(right.first,4)
  assert.equal(right.at,4)
  const rows = [{t:.3,estimate:2},{t:.4,estimate:3}]
  const frame = chart(rows, [[.3,1]], {domain:comparisonDomain(right.at,.1,right.first),vertical:[1,4]})
  assert.equal(frame.samples.length,1)
  assert.equal(frame.samples[0].t,.4)
  assert.equal(frame.option.yAxis.min,1)
  assert.equal(frame.option.yAxis.max,4)
  assert.deepEqual(Array.from(frame.option.series[0].markArea.data[0], p=>p.xAxis),[.4,.5])
  assert.ok(chart([], [], {domain:[0,12]}))
})
