'use client'

import type { ChatProvider } from '../ChatModelSettings'
import { useState, useRef, useEffect, useCallback } from 'react'

interface Message {
  id: string
  text: string
  role: 'user' | 'assistant'
}

// ===== SSE ペイロード型 =====
/**
 * 発話タイムライン上にスケジュール済みのモーション。
 *
 * 従来はモーションが届いた瞬間に再生していたため、開始時刻が
 * 「T2M の生成が終わった時刻」で決まってしまい発話とズレていた。
 * これを「どの発話が、AudioContext のどの時刻から鳴るか」に紐付けて渡す。
 */
export type ScheduledMotion = {
  /** 同じモーションを二重に適用しないための識別子 */
  id: string
  segmentId: number
  bvh: string
  prompt: string
  /** AudioContext.currentTime 基準の再生開始時刻 */
  startAt: number
  /** 合わせる発話の長さ(秒)。尺合わせに使う。 */
  segmentDuration: number
  /** 層構成方式の動作。発話ごとの 1 本で、同じ group の新しい版が届けば差し替える。 */
  layered?: LayeredMotionInfo
}

/** BVH の最終フレームの時刻。発話中に届いた動作が再生可能か判定する。 */
export function bvhDurationSeconds(text: string): number {
  const frames = Number(/Frames:\s*(\d+)/.exec(text)?.[1] ?? 0)
  const frameTime = Number(/Frame Time:\s*([\d.]+)/.exec(text)?.[1] ?? 0)
  return Math.max(frames - 1, 0) * frameTime
}

/** 動作生成モード(会話画面で切り替える)。 */
export type MotionMode = 'current' | 'layered' | 'layered-semantic'

/** 層構成方式で、どの層をどう重ねたか(比較表示用)。 */
export type LayeredMotionInfo = {
  mode: string
  group: string
  revision: number
  playNow?: boolean
  base?: 'idle' | 'rhythm'
  layer?: string
  duration?: number
  rhythm?: 'pending' | 'ok' | 'off'
  rhythmTiming?: Record<string, number>
  overlays?: {
    tag: string
    id: string
    label: string
    source: string
    start: number
    stroke: number
    length: number
    amplitude: number
    explicit: boolean
  }[]
  skipped?: { tag: string; reason: string; with?: string }[]
}

/** 直近に届いた動作(画面での比較表示用)。 */
export type MotionActivity = {
  receivedAt: number
  prompt: string
  latencyMs?: number
  layered?: LayeredMotionInfo
  error?: string
}

/** バックエンドが実際に使ったモード(座位中などは現行方式に戻る)。 */
export type MotionModeStatus = {
  requested: string
  effective: string
  reason?: string | null
}

export type StartChatOptions = {
  llmProvider?: ChatProvider
  llmModel?: string
  ttsProvider?: string
  motionMode?: MotionMode
  /** アバターの現在の姿勢。座位中は層構成方式を使わない。 */
  posture?: string
}

/** 収録時だけ使う音声キャプチャ。`window.__syncCapture` が有れば収集する。 */
type SyncAudioCapture = {
  chunks: {
    perf: number
    startAt: number
    duration: number
    ctxNow: number
    base64: string
  }[]
}

type SSEText = { type: 'text_chunk'; content: string }
type SSEAudio = {
  type: 'audio_chunk'
  /** この発話の通し番号。モーションはこの番号を指して再生位置を決める。 */
  segment_id?: number
  duration_ms?: number
  audio: string
  sentence?: string
  mime?: string
}
type SSEFrame = {
  type: 'frame_chunk'
  image: string
  frame_index?: number
  fps?: number
}
type SSEDone = { type: 'done'; message?: string }
type SSEMetadata = {
  type: 'metadata_chunk'
  dataType: string
  content: string
}
type SSEMotion = {
  type: 'motion_chunk'
  /** 合わせるべき発話の通し番号 */
  segment_id?: number
  /** その発話の開始からの追加オフセット(ミリ秒) */
  start_offset_ms?: number
  /**
   * 文中のどのあたりで動くか(0.0=文頭, 1.0=文末)。
   * 行動タグが文の途中に置かれた場合に、その位置で動かすための比率。
   * 秒数はここでは決まらない(発話長を知っているのはフロントエンド側)。
   */
  start_offset_ratio?: number
  bvh: string
  prompt?: string
  latency_ms?: number
  layered?: LayeredMotionInfo
}
type SSEMotionMode = {
  type: 'motion_mode'
  requested: string
  effective: string
  reason?: string | null
}
type SSEMotionError = {
  type: 'motion_error'
  prompt?: string
  message?: string
}
// ★ any 付きの index signature は削除
type SSEAny =
  | SSEText
  | SSEAudio
  | SSEFrame
  | SSEDone
  | SSEMetadata
  | SSEMotion
  | SSEMotionError
  | SSEMotionMode
  | { type: 'error'; message: string }

// UE expires a lip-sync client after seven seconds without a renewal.
function beginLipSync() {
  const client = crypto.randomUUID()
  let active = true
  let renewing = false
  async function send(action: 'start' | 'stop') {
    try {
      const response = await fetch(`/lip-sync/${action}?client_id=${client}`, {
        cache: 'no-store', keepalive: action === 'stop',
        signal: AbortSignal.timeout(4000),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
    } catch (error) {
      console.warn(`[LipSync] ${action}`, error)
    }
  }
  async function renew() {
    if (!active || renewing) return
    renewing = true
    await send('start')
    renewing = false
    // A late start from this playback must not leave the mouth moving.
    if (!active) await send('stop')
  }
  void renew()
  const timer = window.setInterval(renew, 2000)
  return () => {
    active = false
    window.clearInterval(timer)
    void send('stop')
  }
}

export function useTextChat(endpoint: string) {
  const [chatError, setChatError] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [speakingMessageId, setSpeakingMessageId] = useState<string | null>(null)
  const [avatarFrameSrc, setAvatarFrameSrc] = useState<string | null>(null) // ★ 最新フレーム
  const [latestUrl, setLatestUrl] = useState<string | null>(null)
  const [bvhText, setBvhText] = useState<string | null>(null) // ★ 最新BVHモーション
  const [bvhPrompt, setBvhPrompt] = useState<string | null>(null)
  // 発話タイムライン上にスケジュール済みのモーション列。BvhViewer が音声時計で再生する。
  const [scheduledMotions, setScheduledMotions] = useState<ScheduledMotion[]>([])
  const [motionTurn, setMotionTurn] = useState(0)
  // segment_id → その発話の再生開始時刻(AudioContext 基準)と長さ
  const segmentTimingRef = useRef<Map<number, { startAt: number; duration: number }>>(
    new Map(),
  )
  // 対応する発話がまだ届いていないモーション。発話が来たら解決してスケジュールする。
  const pendingMotionsRef = useRef<
    {
      id: string
      bvh: string
      prompt: string
      segmentId: number
      offsetMs: number
      offsetRatio: number
      layered?: LayeredMotionInfo
    }[]
  >([])
  const [motionModeStatus, setMotionModeStatus] = useState<MotionModeStatus | null>(null)
  const [motionActivity, setMotionActivity] = useState<MotionActivity[]>([])
  const motionSeqRef = useRef(0)
  // デコード待ちの音声の数。0 になるまで「発話が来なかった」と判断してはいけない。
  const decodingCountRef = useRef(0)
  const flushWhenIdleRef = useRef(false)

  const eventSourceRef = useRef<EventSource | null>(null)
  const currentAssistantIdRef = useRef<string | null>(null)
  const lastAudioChunkTimeRef = useRef<number>(0)

  // ========= Web Audio 再生制御 =========
  const audioCtxRef = useRef<AudioContext | null>(null)
  const playbackTimeRef = useRef<number>(0)
  const lipSyncStopRef = useRef<(() => void) | null>(null)
  const stopLipSync = useCallback(() => {
    lipSyncStopRef.current?.()
    lipSyncStopRef.current = null
  }, [])

  const resetChat = async () => {
    stopAllAudio()
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
    setChatError(null)
    setMessages([])
    setSpeakingMessageId(null)
    setAvatarFrameSrc(null)
    setLatestUrl(null)
    setBvhText(null)
    setBvhPrompt(null)
    setScheduledMotions([])
    setMotionActivity([])
    setMotionModeStatus(null)
    segmentTimingRef.current.clear()
    pendingMotionsRef.current = []
    currentAssistantIdRef.current = null

    // ★ バックエンドの履歴もリセットする
    try {
      const resetUrl = '/api/text-chat/reset'
      console.log('[useTextChat] Resetting history on backend:', resetUrl)
      await fetch(resetUrl, { method: 'POST' })
    } catch (err) {
      console.error('[useTextChat] Failed to reset backend history:', err)
    }
  }

  const ensureAudioReady = () => {
    if (!audioCtxRef.current) {
      const AudioContextClass =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext?: typeof AudioContext })
          .webkitAudioContext

      if (!AudioContextClass) {
        console.error('AudioContext not supported')
        return
      }

      audioCtxRef.current = new AudioContextClass()
    }

    if (audioCtxRef.current.state === 'suspended') {
      audioCtxRef.current.resume().catch(() => { })
    }
  }

  const b64ToArrayBuffer = (b64: string): ArrayBuffer => {
    b64 = b64.replace(/[\r\n\s]/g, '')
    while (b64.length % 4 !== 0) b64 += '='
    const binary = atob(b64)
    const buf = new ArrayBuffer(binary.length)
    const view = new Uint8Array(buf)
    for (let i = 0; i < binary.length; i++) view[i] = binary.charCodeAt(i)
    return buf
  }

  /**
   * 発話の再生時刻が確定したら、その発話を待っていたモーションをスケジュールする。
   *
   * モーションは T2M の生成が終わり次第届くので、対応する発話より先に来ることが多い。
   * その場合は pending に積んでおき、発話が届いた時点でここで解決する。
   */
  const resolvePendingMotions = () => {
    const timing = segmentTimingRef.current
    const still: typeof pendingMotionsRef.current = []
    const ready: ScheduledMotion[] = []

    for (const pending of pendingMotionsRef.current) {
      const segment = timing.get(pending.segmentId)
      if (!segment) {
        still.push(pending)
        continue
      }
      // 文中の位置(比率)を発話長に掛けて秒に直す。ミリ秒指定があればそれを足す。
      const offsetSeconds =
        pending.offsetRatio * segment.duration + pending.offsetMs / 1000
      ready.push({
        id: pending.id,
        segmentId: pending.segmentId,
        bvh: pending.bvh,
        prompt: pending.prompt,
        startAt: segment.startAt + offsetSeconds,
        // 文頭から動く場合は残り全部、途中から動く場合は残りぶんに尺を合わせる
        segmentDuration: Math.max(segment.duration - offsetSeconds, 0.2),
        layered: pending.layered,
      })
    }

    pendingMotionsRef.current = still
    if (ready.length === 0) return

    for (const motion of ready) {
      console.log(
        `[SYNC] motion_scheduled prompt="${motion.prompt}" ` +
        `startAt=${motion.startAt.toFixed(3)} ` +
        `segmentDuration=${motion.segmentDuration.toFixed(3)} ` +
        `ctxNow=${audioCtxRef.current?.currentTime.toFixed(3) ?? 'n/a'}`
      )
    }
    setScheduledMotions((prev) => [...prev, ...ready])
  }

  /**
   * 対応する発話が最後まで来なかったモーションを、その場で再生させる。
   * TTS が失敗した場合などに、モーションが永久に出ないのを防ぐ。
   *
   * `done` は音声のデコード完了を待たずに届くため、デコード中に呼ばれたら
   * 保留し、すべて片付いてから改めて判断する。
   */
  const flushPendingMotions = () => {
    if (decodingCountRef.current > 0) {
      flushWhenIdleRef.current = true
      return
    }
    resolvePendingMotions()
    if (pendingMotionsRef.current.length === 0) return
    const now = audioCtxRef.current?.currentTime ?? 0
    const flushed = pendingMotionsRef.current.map((pending) => ({
      id: pending.id,
      segmentId: pending.segmentId,
      bvh: pending.bvh,
      prompt: pending.prompt,
      startAt: now,
      segmentDuration: 0,
      layered: pending.layered,
    }))
    pendingMotionsRef.current = []
    console.warn(
      `[SYNC] motion_flushed count=${flushed.length} ` +
      '(対応する発話が来なかったため即時再生)'
    )
    setScheduledMotions((prev) => [...prev, ...flushed])
  }

  const enqueueAudioBuffer = (
    buf: AudioBuffer,
    base64?: string,
    segmentId?: number,
  ) => {
    const ctx = audioCtxRef.current!
    const src = ctx.createBufferSource()
    src.buffer = buf
    src.connect(ctx.destination)

    const now = ctx.currentTime
    if (playbackTimeRef.current < now) playbackTimeRef.current = now + 0.02

    if (currentAssistantIdRef.current) {
      setSpeakingMessageId(currentAssistantIdRef.current)
    }

    // 同期計測用。音声は AudioContext のタイムライン上に正確にスケジュールされる。
    // モーションがこの時刻に対してどれだけ遅れて適用されたかを後から算出できるようにする。
    console.log(
      `[SYNC] audio_scheduled perf=${performance.now().toFixed(1)} ` +
      `startAt=${playbackTimeRef.current.toFixed(3)} ` +
      `duration=${buf.duration.toFixed(3)} ctxNow=${now.toFixed(3)}`
    )

    // 収録スクリプトから明示的に有効化されたときだけ、鳴らした音声とその
    // スケジュール時刻を控える。画面収録には音声トラックが入らないため、
    // これを後から動画へ合成して「発話と行動のズレ」を耳でも確認できるようにする。
    // 通常の利用ではフラグが未定義なので何も起きない。
    const capture = (window as unknown as { __syncCapture?: SyncAudioCapture })
      .__syncCapture
    if (capture && base64) {
      capture.chunks.push({
        perf: performance.now(),
        startAt: playbackTimeRef.current,
        duration: buf.duration,
        ctxNow: now,
        base64,
      })
    }

    // この発話がいつからいつまで鳴るかを控える。モーションはこれを基準に並ぶ。
    if (segmentId !== undefined) {
      segmentTimingRef.current.set(segmentId, {
        startAt: playbackTimeRef.current,
        duration: buf.duration,
      })
    }

    src.start(playbackTimeRef.current)
    playbackTimeRef.current += buf.duration
    const endsAt = playbackTimeRef.current
    lipSyncStopRef.current ??= beginLipSync()

    // 先に届いていたモーションがこの発話を待っていれば、ここで時刻が確定する
    if (segmentId !== undefined) resolvePendingMotions()

    src.onended = () => {
      if (audioCtxRef.current === ctx && playbackTimeRef.current === endsAt) stopLipSync()
      const checkDelay = 600
      setTimeout(() => {
        const elapsed = Date.now() - lastAudioChunkTimeRef.current
        const remaining = playbackTimeRef.current - ctx.currentTime
        if (audioCtxRef.current === ctx && elapsed > checkDelay && remaining <= 0.5) {
          setSpeakingMessageId(null)
        }
      }, checkDelay)
    }
  }

  const handleAudioChunk = async (base64: string, segmentId?: number) => {
    ensureAudioReady()
    const ctx = audioCtxRef.current
    if (!ctx) return
    lastAudioChunkTimeRef.current = Date.now()
    decodingCountRef.current += 1

    try {
      const arr = b64ToArrayBuffer(base64)
      const audioBuffer = await ctx.decodeAudioData(arr.slice(0))
      if (audioCtxRef.current !== ctx) return
      enqueueAudioBuffer(audioBuffer, base64, segmentId)
      if (currentAssistantIdRef.current) {
        setSpeakingMessageId(currentAssistantIdRef.current)
      }
    } catch (e) {
      if (audioCtxRef.current !== ctx) return
      console.error('decodeAudioData 失敗:', e)
      await new Promise((r) => setTimeout(r, 150))
      if (audioCtxRef.current !== ctx) return
      try {
        const arr = b64ToArrayBuffer(base64)
        const audioBuffer = await ctx.decodeAudioData(arr.slice(0))
        if (audioCtxRef.current !== ctx) return
        enqueueAudioBuffer(audioBuffer, base64, segmentId)
      } catch (e2) {
        console.error('再試行でも失敗:', e2)
      }
    } finally {
      if (audioCtxRef.current === ctx) {
        decodingCountRef.current -= 1
        if (decodingCountRef.current === 0 && flushWhenIdleRef.current) {
          flushWhenIdleRef.current = false
          flushPendingMotions()
        }
      }
    }
  }

  const stopAllAudio = () => {
    stopLipSync()
    // 音声を止めたらタイムラインも無効になるので、スケジュール済みの
    // モーションと発話時刻の対応表も破棄する。
    segmentTimingRef.current.clear()
    pendingMotionsRef.current = []
    decodingCountRef.current = 0
    flushWhenIdleRef.current = false
    setScheduledMotions([])
    setMotionTurn((turn) => turn + 1)

    const ctx = audioCtxRef.current
    if (!ctx) return
    ctx.close().catch(() => { })
    audioCtxRef.current = null
    playbackTimeRef.current = 0
    currentAssistantIdRef.current = null
    setSpeakingMessageId(null)
  }

  const interruptChat = () => {
    stopAllAudio()
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
  }

  // ==========================================
  const startChat = (
    userText: string,
    mode: string = 'general',
    options: StartChatOptions = {},
  ) => {
    // 新しい会話のたびにオーディオはリセット
    stopAllAudio()
    ensureAudioReady()
    setChatError(null)
    setLatestUrl(null)
    setMotionModeStatus(null)

    // ユーザーメッセージ追加
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), text: userText, role: 'user' },
    ])

    // 既存 SSE を閉じる
    eventSourceRef.current?.close()

    // Nginx / Next.js と同じオリジンを経由してAPIへ接続する。
    // ビルド時の固定IPを使わないことで、どちらのLANからでもアクセスできる。
    const params = new URLSearchParams({ text: userText, mode })
    if (options.motionMode) params.set('motion_mode', options.motionMode)
    if (options.posture) params.set('posture', options.posture)
    if (options.llmProvider) params.set('llm_provider', options.llmProvider)
    if (options.llmModel) params.set('llm_model', options.llmModel)
    if (options.ttsProvider) params.set('tts_provider', options.ttsProvider)
    const fullUrl = `${endpoint}?${params}`

    console.log('[useTextChat] SSE connect to:', fullUrl)

    const es = new EventSource(fullUrl)
    eventSourceRef.current = es

    es.onmessage = (event) => {
      try {
        const data: SSEAny = JSON.parse(event.data)

        console.log('🔹 [SSE EVENT RECEIVED]', data)

        // 🟦 テキストチャンク
        if (data.type === 'text_chunk') {
          const chunk = data.content
          if (!chunk) return

          setMessages((prev) => {
            const last = prev.at(-1)
            if (!last || last.role !== 'assistant') {
              // 🎯 新しい assistant メッセージ開始
              const id = crypto.randomUUID()
              currentAssistantIdRef.current = id
              // 音声再生時にセットするためここでは呼ばない
              return [...prev, { id, text: chunk, role: 'assistant' }]
            } else {
              // 直近の assistant メッセージに追記
              const id = currentAssistantIdRef.current ?? last.id
              currentAssistantIdRef.current = id
              // 音声再生時にセットするためここでは呼ばない

              let updated = last.text + chunk
              if (/[。！？!?]$/.test(updated)) updated += '\n\n'

              return [...prev.slice(0, -1), { ...last, text: updated, id }]
            }
          })
          return
        }

        // 🟦 音声チャンク
        if (data.type === 'audio_chunk') {
          if (data.audio) handleAudioChunk(data.audio, data.segment_id)
          return
        }

        // 🟦 フレームチャンク（Ditto）
        if (data.type === 'frame_chunk') {
          const imgBase64: string | undefined = data.image
          if (imgBase64) {
            setAvatarFrameSrc(`data:image/jpeg;base64,${imgBase64}`)
          }
          return
        }

        // 🟦 モーションチャンク（BVH）
        if (data.type === 'motion_chunk') {
          if (data.bvh && data.bvh.length > 0) {
            console.log(
              `[useTextChat] motion_chunk received: prompt="${data.prompt}", ` +
              `bvh_len=${data.bvh.length}, latency_ms=${data.latency_ms ?? 'n/a'}`
            )
            // 同期計測用。音声は AudioContext のタイムライン上にスケジュール済みなので、
            // この ctxNow と [SYNC] audio_scheduled の startAt を突き合わせれば
            // 「どの発話に対してどれだけ遅れてモーションが出たか」が算出できる。
            const ctx = audioCtxRef.current
            // モーションは最初の音声より先に届くことがあり、その時点では AudioContext が
            // まだ存在しない。ctxNow だけでは測れないので壁時計(perf)も併記する。
            console.log(
              `[SYNC] motion_received perf=${performance.now().toFixed(1)} ` +
              `ctxNow=${ctx ? ctx.currentTime.toFixed(3) : 'n/a'} ` +
              `segment=${data.segment_id ?? 'n/a'} ` +
              `ratio=${(data.start_offset_ratio ?? 0).toFixed(2)} ` +
              `prompt="${data.prompt ?? ''}" latency_ms=${data.latency_ms ?? 'n/a'}`
            )

            motionSeqRef.current += 1
            // 状態更新の関数は後で実行されるので、識別子はここで確定させておく
            const motionId = `m${motionSeqRef.current}`
            const layered = data.layered
            setMotionActivity((prev) => [
              ...prev.slice(-11),
              {
                receivedAt: performance.now(),
                prompt: data.prompt ?? '',
                latencyMs: data.latency_ms,
                layered,
              },
            ])
            setBvhPrompt(data.prompt ?? null)
            if (layered?.playNow) {
              // 待機・傾聴の層(受け取りのうなずき)。発話を待たずにすぐ再生する。
              ensureAudioReady()
              const now = audioCtxRef.current?.currentTime ?? 0
              setScheduledMotions((prev) => [
                ...prev,
                {
                  id: motionId,
                  segmentId: data.segment_id ?? 0,
                  bvh: data.bvh,
                  prompt: data.prompt ?? '',
                  startAt: now,
                  segmentDuration: 0,
                  layered,
                },
              ])
              return
            }
            if (layered) {
              // 同じ発話の古い版がまだ発話待ちなら、新しい版で置き換える
              pendingMotionsRef.current = pendingMotionsRef.current.filter(
                (p) => !(p.layered?.group === layered.group && p.layered.revision < layered.revision),
              )
            }
            // 届いた瞬間に再生するのではなく、指定された発話の再生開始に合わせる。
            pendingMotionsRef.current.push({
              id: motionId,
              bvh: data.bvh,
              prompt: data.prompt ?? '',
              segmentId: data.segment_id ?? 1,
              offsetMs: data.start_offset_ms ?? 0,
              offsetRatio: data.start_offset_ratio ?? 0,
              layered,
            })
            // 対応する発話がすでに届いていれば即座に確定する
            resolvePendingMotions()
          }
          return
        }

        if (data.type === 'motion_mode') {
          setMotionModeStatus({
            requested: data.requested,
            effective: data.effective,
            reason: data.reason,
          })
          return
        }

        if (data.type === 'error') {
          setChatError(data.message || '応答の生成に失敗しました。対話モデルの設定を確認してください。')
          es.close()
          eventSourceRef.current = null
          return
        }

        if (data.type === 'motion_error') {
          setMotionActivity((prev) => [
            ...prev.slice(-11),
            {
              receivedAt: performance.now(),
              prompt: data.prompt ?? '',
              error: data.message ?? 'motion error',
            },
          ])
          console.warn(
            `[useTextChat] asynchronous motion generation failed: ` +
            `prompt="${data.prompt ?? ''}", message="${data.message ?? ''}"`
          )
          return
        }

        // 🟦 メタデータチャンク
        if (data.type === 'metadata_chunk') {
          if (data.dataType === 'url') {
            let url = data.content
            if (url.startsWith('www.')) {
              url = `https://${url}`
            }
            setLatestUrl(url)
          }
          return
        }

        // 🟦 完了イベント
        if (data.type === 'done') {
          // 応答が終わってもスケジュールできていないモーションは、
          // 対応する発話が来なかったということなので即時再生へ回す。
          flushPendingMotions()
          setMessages((prev) => {
            const last = prev.at(-1)
            if (last && last.role === 'assistant') {
              return [
                ...prev.slice(0, -1),
                { ...last, text: last.text.replace(/\n+$/g, '') },
              ]
            }
            return prev
          })
          es.close()
          eventSourceRef.current = null
          return
        }

        // その他の type（'start' など）は一旦無視でもOK
      } catch (e) {
        console.error('Parse error:', e)
      }
    }

    es.onerror = (err) => {
      setChatError('会話への接続が中断されました。対話モデル・音声合成の設定とバックエンドの起動状態を確認してください。')
      console.error('❌ SSE Error:', err)
      es.close()
      eventSourceRef.current = null
      setSpeakingMessageId(null)
      currentAssistantIdRef.current = null
    }
  }

  // ==========================================

  useEffect(() => {
    window.addEventListener('pagehide', stopLipSync)
    return () => window.removeEventListener('pagehide', stopLipSync)
  }, [stopLipSync])

  useEffect(() => {
    const onUserInteract = () => {
      ensureAudioReady()
      window.removeEventListener('click', onUserInteract)
      window.removeEventListener('keydown', onUserInteract)
      window.removeEventListener('touchstart', onUserInteract)
    }
    window.addEventListener('click', onUserInteract)
    window.addEventListener('keydown', onUserInteract)
    window.addEventListener('touchstart', onUserInteract)

    return () => {
      eventSourceRef.current?.close()
      stopAllAudio()
      window.removeEventListener('click', onUserInteract)
      window.removeEventListener('keydown', onUserInteract)
      window.removeEventListener('touchstart', onUserInteract)
    }
  }, [])

  return {
    chatError,
    messages,
    startChat,
    speakingMessageId,
    resetChat,
    motionTurn,
    scheduledMotions,
    /**
     * 発話タイムラインの現在時刻(秒)。まだ音声が始まっていなければ null。
     * BvhViewer はこれをマスタークロックにして、音声とモーションの時計を一本化する。
     */
    getAudioTime: useCallback(
      () => audioCtxRef.current?.state === 'running' ? audioCtxRef.current.currentTime : null,
      [],
    ),
    avatarFrameSrc,
    latestUrl,
    bvhText,
    bvhPrompt,
    interruptChat,
    motionModeStatus,
    motionActivity,
  }
}
