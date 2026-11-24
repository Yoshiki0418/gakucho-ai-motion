'use client'

import { useEffect, useRef, useState } from 'react'
import { bvhDurationSeconds, type ScheduledMotion } from './useTextChat'

export function useUnrealBodyMotion(
  enabled: boolean,
  turn: number,
  motions: ScheduledMotion[],
  getAudioTime: () => number | null,
) {
  const [error, setError] = useState<string | null>(null)
  const client = useRef('')
  const sequence = useRef(0)
  const sent = useRef(false)
  const consumed = useRef(new Set<string>())
  const lastMotion = useRef<ScheduledMotion | null>(null)

  useEffect(() => {
    setError(null)
    if (!enabled) return
    const stop = () => {
      consumed.current.clear()
      lastMotion.current = null
      if (!sent.current) return
      sent.current = false
      // 中断時に play のアップロード中でも、後の sequence の stop を優先する。
      void fetch('/api/ue-motion/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ client_id: client.current, sequence: ++sequence.current }),
        keepalive: true,
      }).catch(() => {})
    }
    window.addEventListener('pagehide', stop)
    return () => {
      window.removeEventListener('pagehide', stop)
      stop()
    }
  }, [enabled, turn])

  useEffect(() => {
    if (!enabled) return
    const pending = motions.filter(motion => !consumed.current.has(motion.id))
      .sort((a, b) => a.startAt - b.startAt)
    let timer: number | undefined

    async function play(motion: ScheduledMotion, offset: number, remaining?: number) {
      client.current ||= crypto.randomUUID()
      const command = ++sequence.current
      sent.current = true
      try {
        const response = await fetch('/api/ue-motion/play', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            client_id: client.current,
            sequence: command,
            segment_id: motion.segmentId,
            clip_id: `${turn}-${motion.layered?.group ?? motion.id}`,
            revision: motion.layered?.revision ?? 0,
            bvh: motion.bvh,
            offset_seconds: offset,
            remaining_seconds: remaining,
          }),
        })
        if (!response.ok || (await response.json()).ok !== true) throw new Error(`HTTP ${response.status}`)
        if (sequence.current === command) setError(null)
      } catch (cause) {
        if (sequence.current === command) {
          console.warn('[UE body motion]', cause)
          setError('Unreal Editor に身体動作を送れませんでした。UE の身体モーション受信が起動しているか確認してください。')
        }
      }
    }

    const tick = () => {
      if (!pending.length) return
      const now = getAudioTime()
      if (now === null) {
        timer = window.setTimeout(tick, 100)
        return
      }

      // 画面切替・バックグラウンド復帰で複数の時刻を過ぎていたら、今の発話を選ぶ。
      let selected: ScheduledMotion | undefined
      while (pending.length && pending[0].startAt <= now) {
        const motion = pending.shift()!
        consumed.current.add(motion.id)
        const previous = selected ?? lastMotion.current
        if (previous && (motion.startAt < previous.startAt ||
          (motion.startAt === previous.startAt && motion.segmentId < previous.segmentId) ||
          (motion.layered && previous.layered?.group === motion.layered.group &&
            previous.layered.revision >= motion.layered.revision))) continue
        selected = motion
      }
      if (selected) {
        lastMotion.current = selected
        const elapsed = now - selected.startAt
        const remaining = selected.segmentDuration > 0 ? selected.segmentDuration - elapsed : undefined
        if (elapsed < bvhDurationSeconds(selected.bvh) && (remaining === undefined || remaining > 0)) {
          void play(selected, elapsed, remaining)
        }
      }
      if (pending.length) {
        timer = window.setTimeout(tick, Math.max(1, (pending[0].startAt - now) * 1000))
      }
    }
    tick()
    return () => window.clearTimeout(timer)
  }, [enabled, turn, motions, getAudioTime])

  return error
}
