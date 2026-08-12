import { chromium } from 'playwright-core';
import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const out = path.join(root, 'docs', 'assets', 'bs-manual');
fs.mkdirSync(out, { recursive: true });
const edge = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';

const browser = await chromium.launch({
  headless: true,
  executablePath: edge,
  args: ['--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage'],
});
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
page.setDefaultTimeout(15000);
await page.goto('http://127.0.0.1:8000', { waitUntil: 'networkidle' });
await page.screenshot({ path: path.join(out, '01-home.png'), fullPage: true });

async function clickModule(groupName, itemName) {
  const group = page.locator('.top-nav-button').filter({ hasText: groupName }).first();
  await group.click();
  await page.waitForTimeout(150);
  const item = page.locator('.nav-dropdown button').filter({ hasText: itemName }).first();
  await item.click();
  await page.waitForTimeout(350);
}

async function shot(name) {
  await page.screenshot({ path: path.join(out, name), fullPage: true });
}

await clickModule('异常行为检测', '基于移动目标防御的异常检测');
await shot('02-anomaly-input.png');
const anomalyTab = page.getByRole('button', { name: '异常检测概率图' });
if (await anomalyTab.count()) { await anomalyTab.click(); await page.waitForTimeout(250); }
await shot('03-anomaly-result.png');

await clickModule('风险动态分析', '多评估准则融合的风险学习分析');
await shot('04-score.png');
await clickModule('风险动态分析', '潜在安全威胁识别与自动分类');
await shot('05-classification.png');
await clickModule('风险动态分析', '风险场景动态匹配与适配方案生成算法');
await shot('06-cdq.png');
await clickModule('风险管控优化决策', '控制模型训练评估');
await shot('07-training.png');
await clickModule('风险管控优化决策', '优化控制仿真验证');
await shot('08-mpc.png');
await clickModule('SIS自主化检测', 'SDG-HAZOP');
await shot('09-sdg.png');
await clickModule('在线SIL验证', '基于GSPN-MC模型的动态化SIL验证方法');
await shot('10-sil.png');

// A real boundary/error state: clear the required Runtime path on the anomaly page.
await clickModule('异常行为检测', '基于移动目标防御的异常检测');
const inputs = page.locator('input');
if (await inputs.count()) {
  await inputs.first().fill('');
  await inputs.first().blur();
  await page.waitForTimeout(250);
}
await shot('11-validation.png');

await browser.close();
console.log('screenshots written to', out);
