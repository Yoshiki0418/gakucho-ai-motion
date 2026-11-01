'use client'

/**
 * 物理シミュレーションでの学習の様子(/world-model/training)。
 *
 * ホストの書き出し係(world_model/tracker/monitor.py)が 30 秒ごとに更新する
 * `/training-monitor/status.json` を読み、進み具合・途中評価の推移・最新の重みでの再生を表示する。
 */

import Link from 'next/link'
import React, { useEffect, useMemo, useState } from 'react'

type EvalPoint = { step: number; failed: number; positionErrorCm: number; reward: number }
type RecordOption = { url: string; title: string; key: string; success: boolean; meanErrorCm: number; failedAtSeconds: number | null }
type FinalEvaluation = {
  mocapTest?: [number, number]
  mocapMeanErr?: number
  t2m?: [number, number]
  nav?: [number, number]
  navContacts?: number
  navFalls?: number
  seated?: [number, number]
  seatedMeanErr?: number
}
type Run = {
  name: string
  label: string
  state: 'running' | 'finished' | 'stopped'
  reason: string | null
  startedAt: string
  updatedAt: string | null
  step: number
  totalSteps: number
  resumedFrom: number | null
  elapsedSeconds: number
  stepsPerSecond: number | null
  etaSeconds: number | null
  includeSeated: boolean
  evals: EvalPoint[]
  finalEvaluation: FinalEvaluation | null
  records: RecordOption[]
}
type Status = { version: number; updatedAt: string; runs: Run[] }

// 暗い面(#132840)で検証済みの 1 系列色と、状態の色(必ず文字と記号を添える)
const SERIES = '#3987e5'
const STATE = {
  running: { color: '#0ca30c', label: '学習中', icon: '●' },
  finished: { color: '#8aa0b8', label: '完了', icon: '✓' },
  stopped: { color: '#fab219', label: '停止', icon: '!' },
} as const
const ink = { primary: '#e1edf9', secondary: '#b2c3d4', muted: '#8aa0b8', grid: '#2a3f58', surface: '#132840' }

const oku = (step: number) => `${(step / 1e8).toFixed(2)}億`
const duration = (seconds: number) => {
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `約${minutes}分`
  return `約${Math.floor(minutes / 60)}時間${minutes % 60 ? `${minutes % 60}分` : ''}`
}
const clock = (iso: string | null) => (iso ? iso.slice(11, 16) : '—')

function LineChart({ title, note, points, unit, digits }: {
  title: string
  note: string
  points: { step: number; value: number }[]
  unit: string
  digits: number
}) {
  const [hover, setHover] = useState<number | null>(null)
  const width = 460
  const height = 150
  const pad = { left: 44, right: 16, top: 12, bottom: 26 }
  const xs = points.map((p) => p.step)
  const ys = points.map((p) => p.value)
  const xMin = Math.min(...xs)
  // 評価が 1 回だけのときも横軸に幅を持たせる
  const xMax = Math.max(...xs, xMin + 1e7)
  const yMax = Math.max(...ys, 0) * 1.15 || 1
  const x = (v: number) => pad.left + ((v - xMin) / (xMax - xMin)) * (width - pad.left - pad.right)
  const y = (v: number) => pad.top + (1 - v / yMax) * (height - pad.top - pad.bottom)
  const ticks = [0, yMax / 2, yMax].map((v) => Number(v.toFixed(digits)))
  const path = points.map((p, i) => `${i ? 'L' : 'M'}${x(p.step).toFixed(1)},${y(p.value).toFixed(1)}`).join(' ')
  const last = points.at(-1)
  const active = hover !== null ? points[hover] : null
  if (points.length === 0) return null
  return (
    <figure style={{ margin: 0, minWidth: 0 }}>
      <figcaption style={{ fontSize: 13, color: ink.primary, marginBottom: 2 }}>{title}</figcaption>
      <div style={{ fontSize: 11, color: ink.muted, marginBottom: 6 }}>{note}</div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{ width: '100%', height: 'auto', display: 'block', touchAction: 'none' }}
        role="img"
        aria-label={`${title}の推移`}
        onPointerMove={(event) => {
          const box = event.currentTarget.getBoundingClientRect()
          const px = ((event.clientX - box.left) / box.width) * width
          let best = 0
          points.forEach((p, i) => { if (Math.abs(x(p.step) - px) < Math.abs(x(points[best].step) - px)) best = i })
          setHover(best)
        }}
        onPointerLeave={() => setHover(null)}
      >
        {ticks.map((t) => (
          <g key={t}>
            <line x1={pad.left} x2={width - pad.right} y1={y(t)} y2={y(t)} stroke={ink.grid} strokeWidth={1} />
            <text x={pad.left - 6} y={y(t) + 4} textAnchor="end" fontSize={10} fill={ink.muted}>{t}</text>
          </g>
        ))}
        <text x={pad.left} y={height - 6} fontSize={10} fill={ink.muted}>{oku(xMin)}</text>
        <text x={width - pad.right} y={height - 6} fontSize={10} fill={ink.muted} textAnchor="end">{oku(xMax)} ステップ</text>
        <path d={path} fill="none" stroke={SERIES} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
        {last && (
          <>
            <circle cx={x(last.step)} cy={y(last.value)} r={5} fill={SERIES} stroke={ink.surface} strokeWidth={2} />
            <text x={x(last.step) + (x(last.step) < width / 2 ? 8 : -8)} y={y(last.value) - 9}
              textAnchor={x(last.step) < width / 2 ? 'start' : 'end'} fontSize={11} fill={ink.primary}>
              {last.value.toFixed(digits)}{unit}
            </text>
          </>
        )}
        {active && (
          <>
            <line x1={x(active.step)} x2={x(active.step)} y1={pad.top} y2={height - pad.bottom} stroke={ink.muted} strokeWidth={1} />
            <circle cx={x(active.step)} cy={y(active.value)} r={5} fill={SERIES} stroke={ink.surface} strokeWidth={2} />
          </>
        )}
      </svg>
      <div style={{ minHeight: 16, fontSize: 11, color: ink.secondary }}>
        {active ? `${oku(active.step)}ステップ: ${active.value.toFixed(digits)}${unit}` : ''}
      </div>
    </figure>
  )
}

function RunCard({ run, now }: { run: Run; now: number }) {
  const [showTable, setShowTable] = useState(false)
  const state = STATE[run.state]
  const progress = Math.min(run.step / Math.max(run.totalSteps, 1), 1)
  const finishAt = run.etaSeconds ? new Date(now + run.etaSeconds * 1000) : null
  const failed = run.evals.map((e) => ({ step: e.step, value: e.failed * 100 }))
  const error = run.evals.map((e) => ({ step: e.step, value: e.positionErrorCm }))
  const final = run.finalEvaluation
  return (
    <section style={{ background: ink.surface, border: '1px solid #2a3f58', borderRadius: 12, padding: 18, marginBottom: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap', alignItems: 'baseline' }}>
        <div style={{ minWidth: 0 }}>
          <h2 style={{ fontSize: 17, margin: 0, color: ink.primary }}>{run.name}</h2>
          <p style={{ margin: '4px 0 0', fontSize: 13, color: ink.secondary, lineHeight: 1.6 }}>{run.label}</p>
        </div>
        <span style={{ fontSize: 13, color: ink.primary, whiteSpace: 'nowrap' }}>
          <span style={{ color: state.color, marginRight: 6 }}>{state.icon}</span>{state.label}
        </span>
      </div>
      {run.reason && <p style={{ margin: '8px 0 0', fontSize: 12, color: ink.secondary }}>理由:{run.reason}</p>}

      <div style={{ marginTop: 14 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: ink.secondary, marginBottom: 6, gap: 8, flexWrap: 'wrap' }}>
          <span>{oku(run.step)} / {oku(run.totalSteps)} ステップ({Math.round(progress * 100)}%)</span>
          <span>
            {run.stepsPerSecond ? `${(run.stepsPerSecond / 1e4).toFixed(1)}万ステップ/秒` : ''}
            {run.state === 'running' && run.etaSeconds ? `・残り${duration(run.etaSeconds)}(${finishAt?.toTimeString().slice(0, 5)} ごろ完了)` : ''}
          </span>
        </div>
        <div role="progressbar" aria-valuenow={Math.round(progress * 100)} aria-valuemin={0} aria-valuemax={100}
          style={{ height: 8, background: '#22364f', borderRadius: 4, overflow: 'hidden' }}>
          <div style={{ width: `${progress * 100}%`, height: '100%', background: SERIES, borderRadius: 4 }} />
        </div>
        <div style={{ fontSize: 11, color: ink.muted, marginTop: 6 }}>
          開始 {run.startedAt.replace('T', ' ').slice(0, 16)}・学習の記録の最終更新 {clock(run.updatedAt)}
          {run.resumedFrom ? `・${oku(run.resumedFrom)}ステップから再開` : ''}
        </div>
      </div>

      {run.evals.length > 0 && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 18, marginTop: 16 }}>
            <LineChart title="追従の失敗率" note="評価中に体の位置ずれが 0.5m を超えた(崩れ・転倒)割合。低いほど良い"
              points={failed} unit="%" digits={1} />
            <LineChart title="体の位置ずれ" note="手本の動作に対する、体の各部の平均のずれ。低いほど良い"
              points={error} unit="cm" digits={1} />
          </div>
          <button onClick={() => setShowTable((v) => !v)}
            style={{ background: 'none', border: 'none', color: ink.secondary, fontSize: 12, padding: 0, cursor: 'pointer', marginTop: 4 }}>
            {showTable ? '▾ 数値を閉じる' : '▸ 数値で見る'}
          </button>
          {showTable && (
            <table style={{ width: '100%', fontSize: 12, color: ink.secondary, borderCollapse: 'collapse', marginTop: 6 }}>
              <thead><tr style={{ color: ink.muted, textAlign: 'right' }}>
                <th style={{ textAlign: 'left', fontWeight: 400 }}>ステップ</th><th style={{ fontWeight: 400 }}>失敗率</th><th style={{ fontWeight: 400 }}>位置ずれ</th>
              </tr></thead>
              <tbody>{run.evals.map((e) => (
                <tr key={e.step} style={{ textAlign: 'right', borderTop: `1px solid ${ink.grid}` }}>
                  <td style={{ textAlign: 'left', padding: '3px 0' }}>{oku(e.step)}</td><td>{(e.failed * 100).toFixed(1)}%</td><td>{e.positionErrorCm.toFixed(1)}cm</td>
                </tr>))}</tbody>
            </table>
          )}
        </>
      )}

      {run.records.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <h3 style={{ fontSize: 14, margin: '0 0 8px', color: ink.primary }}>最新の重みでの動き(学習に使っていない動作で毎回同じものを再生)</h3>
          <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'grid', gap: 6 }}>
            {run.records.map((record) => (
              <li key={record.url} style={{ display: 'flex', justifyContent: 'space-between', gap: 10, flexWrap: 'wrap', fontSize: 13, color: ink.secondary }}>
                <span>
                  <span style={{ color: record.success ? '#0ca30c' : '#fab219', marginRight: 6 }}>{record.success ? '✓' : '!'}</span>
                  {record.title.replace(/^【[^】]*】/, '')}・平均ずれ {record.meanErrorCm}cm
                  {record.failedAtSeconds ? `・${record.failedAtSeconds.toFixed(1)}秒で失敗` : ''}
                  <span style={{ color: ink.muted }}>({record.title.match(/\d+\.\d+億/)?.[0]}ステップ時点)</span>
                </span>
                <Link href={`/world-model?stage=1&trace=${encodeURIComponent(record.url)}`} style={{ color: '#9ec7f5', fontSize: 12 }}>3Dで再生 →</Link>
              </li>
            ))}
          </ul>
        </div>
      )}

      {final && (
        <div style={{ marginTop: 16, fontSize: 13, color: ink.secondary }}>
          <h3 style={{ fontSize: 14, margin: '0 0 6px', color: ink.primary }}>最終評価(CPU の物理で独立に再生)</h3>
          {final.mocapTest && <div>保留モーキャプ:{final.mocapTest[0]} / {final.mocapTest[1]} 本が最後まで追従(平均ずれ {((final.mocapMeanErr ?? 0) * 100).toFixed(1)}cm)</div>}
          {final.t2m && <div>T2M の生成動作:{final.t2m[0]} / {final.t2m[1]} 本</div>}
          {final.nav && <div>家具を避けて目的地へ:{final.nav[0]} / {final.nav[1]} 配置(接触 {final.navContacts ?? 0}・転倒 {final.navFalls ?? 0})</div>}
          {final.seated && <div>椅子を使う保留動作(座る・立つ・座ったまま):{final.seated[0]} / {final.seated[1]} 本(平均ずれ {((final.seatedMeanErr ?? 0) * 100).toFixed(1)}cm)</div>}
        </div>
      )}
    </section>
  )
}

export function TrainingMonitor() {
  const [status, setStatus] = useState<Status | null>(null)
  const [error, setError] = useState('')
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    let alive = true
    const load = () => {
      fetch('/training-monitor/status.json', { cache: 'no-store' })
        .then(async (response) => {
          if (!response.ok) throw new Error('学習の記録をまだ読めません(書き出し係が止まっている可能性があります)')
          const data = (await response.json()) as Status
          if (alive) { setStatus(data); setError(''); setNow(Date.now()) }
        })
        .catch((e) => { if (alive) setError(e instanceof Error ? e.message : String(e)) })
    }
    load()
    const timer = window.setInterval(load, 15000)
    return () => { alive = false; window.clearInterval(timer) }
  }, [])

  const runs = useMemo(() => status?.runs ?? [], [status])
  return (
    <main className="training-monitor" style={{ height: '100dvh', overflow: 'auto', background: '#0b172b', color: ink.primary, padding: 'clamp(16px, 4vw, 40px)', boxSizing: 'border-box' }}>
      {/* GlobalStyles が全要素を color: black にしているため、この画面では親の色を継ぐ */}
      <style>{`.training-monitor :is(div, span, h1, h2, h3, p, ul, li, table, tr, td, th, figure, figcaption, section, header, strong) { color: inherit; }`}</style>
      <div style={{ maxWidth: 1080, margin: '0 auto' }}>
        <header style={{ display: 'flex', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap', marginBottom: 20 }}>
          <div>
            <p style={{ color: '#91a5bc', margin: '0 0 6px', fontSize: 12, letterSpacing: 1 }}>学長AI · 物理と学習の検証</p>
            <h1 style={{ fontSize: 'clamp(22px, 3vw, 28px)', margin: 0 }}>学習の様子</h1>
            <p style={{ color: '#a6b6c8', fontSize: 13, margin: '10px 0 0', lineHeight: 1.7 }}>
              物理シミュレーション上で、学長の身体がモーキャプの動きを追従できるように学習しています。15 秒ごとに自動で更新します。
            </p>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start', flexWrap: 'wrap' }}>
            <Link href="/world-model/live" style={{ color: '#fff', background: '#1f5fae', border: '1px solid #3987e5', borderRadius: 8, padding: '9px 13px', textDecoration: 'none', fontSize: 13 }}>ライブで見る(9 体を実時間で)</Link>
            <Link href="/world-model" style={{ color: '#c5dcf4', border: '1px solid #34475f', borderRadius: 8, padding: '9px 13px', textDecoration: 'none', fontSize: 13 }}>物理の検証記録へ</Link>
            <Link href="/" style={{ color: '#c5dcf4', border: '1px solid #34475f', borderRadius: 8, padding: '9px 13px', textDecoration: 'none', fontSize: 13 }}>対話画面へ</Link>
          </div>
        </header>
        {error && <p role="status" style={{ color: '#fab219', fontSize: 13 }}>! {error}</p>}
        {status && <p style={{ fontSize: 12, color: ink.muted, margin: '0 0 12px' }}>書き出しの最終更新 {status.updatedAt.replace('T', ' ')}</p>}
        {runs.map((run) => <RunCard key={run.name} run={run} now={now} />)}
      </div>
    </main>
  )
}
