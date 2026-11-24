// G7: keep the visible house open in headless Chrome and render head-camera views on request.
//   node tools/house_render_server.mjs [BASE_URL]
// stdin: one JSON view per line ({"x","y","z","heading","pitch","width","height","hfov"}, see house_render.mjs);
// stdout: one JSON line per view {"jpeg": base64} (or {"error": ...}). "ready" is printed once the house loaded.
import readline from 'node:readline'

const base = process.argv[2] ?? 'http://127.0.0.1:4010'
const playwright = process.env.PLAYWRIGHT_MODULE ?? new URL('../.runtime/node-tools/node_modules/playwright/index.mjs', import.meta.url).href  // playwright 1.64.1 (/tmp is wiped on reboot)
const { chromium } = await import(playwright)
const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] })
const page = await browser.newPage({ viewport: { width: 800, height: 700 } })
await page.goto(`${base}/world-model/house?capture=1`, { waitUntil: 'domcontentloaded', timeout: 300000 })
await page.waitForFunction(() => window.__houseLab, null, { timeout: 300000 })
await page.evaluate(() => window.__houseLab.ready)
process.stdout.write('ready\n')
const lines = readline.createInterface({ input: process.stdin })
for await (const line of lines) {
  if (!line.trim()) continue
  try {
    const data = await page.evaluate((v) => window.__houseLab.render(v), JSON.parse(line))
    process.stdout.write(JSON.stringify({ jpeg: data.split(',')[1] }) + '\n')
  } catch (error) {
    process.stdout.write(JSON.stringify({ error: String(error) }) + '\n')
  }
}
await browser.close()
