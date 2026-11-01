// G7: render the head-camera views of the visible house (/world-model/house) in headless Chrome.
//   node tools/house_render.mjs VIEWS.json OUT_DIR [BASE_URL]
// VIEWS.json: [{ "name": "...", "x": .., "y": .., "z": .. (eye height, world), "heading": rad, "pitch": deg,
//               "width": 640, "height": 480, "hfov": 90 }, ...]; writes OUT_DIR/<name>.jpg.
// CPU rendering (SwiftShader): about 2-3 s per 640x480 view.
import fs from 'node:fs/promises'
import path from 'node:path'

const [viewsPath, outDir, base = 'http://127.0.0.1:4010'] = process.argv.slice(2)
if (!viewsPath || !outDir) { console.error('usage: node tools/house_render.mjs VIEWS.json OUT_DIR [BASE_URL]'); process.exit(2) }
const playwright = process.env.PLAYWRIGHT_MODULE ?? '/tmp/gakucho-motion-browser-check/node_modules/playwright/index.mjs'
const { chromium } = await import(playwright)
const views = JSON.parse(await fs.readFile(viewsPath, 'utf8'))
await fs.mkdir(outDir, { recursive: true })
const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] })
const page = await browser.newPage({ viewport: { width: 800, height: 700 } })
const errors = []
page.on('pageerror', (e) => errors.push(String(e)))
await page.goto(`${base}/world-model/house?capture=1`, { waitUntil: 'domcontentloaded', timeout: 300000 })
await page.waitForFunction(() => window.__houseLab, null, { timeout: 300000 })
await page.evaluate(() => window.__houseLab.ready)
const started = Date.now()
for (const view of views) {
  const data = await page.evaluate((v) => window.__houseLab.render(v), view)
  await fs.writeFile(path.join(outDir, `${view.name}.jpg`), Buffer.from(data.split(',')[1], 'base64'))
}
console.log(JSON.stringify({ rendered: views.length, seconds: (Date.now() - started) / 1000, errors }))
await browser.close()
