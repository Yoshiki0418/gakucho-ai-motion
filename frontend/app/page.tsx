'use client'

import React, { useState, useEffect, useCallback, useRef } from 'react'
import ChatPanel from '@/components/organisms/ChatPanel'
import {
  useTextChat,
  type MotionActivity,
  type MotionMode,
} from '@/features/text-chat/hooks/useTextChat'
import { AvatarPanel } from '@/components/organisms/AvatarPanel'
import { BvhViewer, type PostureKind } from '@/components/organisms/BvhViewer'
import { QRCodeDisplay } from '@/components/organisms/QRCodeDisplay'

type AvatarViewMode = 'bvh' | 'ue'

// 会話画面の動作生成モード。現行方式と新方式(層構成)を同じ画面で比べられるようにする。
const MOTION_MODES: { id: MotionMode; label: string; hint: string }[] = [
  { id: 'current', label: '現行', hint: '行動タグごとに T2M-GPT で生成' },
  { id: 'layered', label: '新方式', hint: '意味の層(収録クリップ)＋リズムの層(SentiAvatar・研究評価用)' },
  { id: 'layered-semantic', label: '新方式・リズムなし', hint: '意味の層(収録クリップ)のみ' },
]
const MOTION_MODE_STORAGE_KEY = 'gakucho.motionMode'

const describeActivity = (activity: MotionActivity): string => {
  if (activity.error) return `⚠ ${activity.error}`
  const layered = activity.layered
  if (!layered) {
    const seconds = activity.latencyMs !== undefined ? ` ${(activity.latencyMs / 1000).toFixed(1)}秒` : ''
    return `T2M生成${seconds}：${activity.prompt.slice(0, 28)}`
  }
  if (layered.layer === 'listening') return '受け取りのうなずき（待機・傾聴の層）'
  const segment = layered.group.replace('seg-', '発話')
  const gestures = (layered.overlays ?? []).map((overlay) =>
    `${overlay.tag}${overlay.source === 't2m' ? '(T2M事前生成)' : ''}${overlay.start < 0 ? '(前の発話から継続)' : ''}`)
  const base = layered.base === 'rhythm'
    ? `リズム層 SentiAvatar ${layered.rhythmTiming?.total?.toFixed(1) ?? '?'}秒`
    : layered.rhythm === 'pending' ? '待機の層（リズム生成中）' : '待機の層'
  const skipped = (layered.skipped ?? []).length > 0 ? `・省略${layered.skipped!.length}` : ''
  return `${segment}${layered.revision > 0 ? '（差し替え）' : ''}：${gestures.join('＋') || '意味の動作なし'}／${base}${skipped}`
}

export default function ChatDemoPage() {
  const [inputValue, setInputValue] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [closedUrls, setClosedUrls] = useState<Set<string>>(new Set())
  const [appMode, setAppMode] = useState<'general' | 'ceremony'>('general')
  const [avatarViewMode, setAvatarViewMode] = useState<AvatarViewMode>('bvh')
  const [motionMode, setMotionMode] = useState<MotionMode>('current')
  const [motionPanelOpen, setMotionPanelOpen] = useState(true)
  const postureRef = useRef<PostureKind>('standing')
  const handlePostureChange = useCallback((posture: PostureKind) => {
    postureRef.current = posture
  }, [])

  // URL(?motion_mode=)を優先し、無ければ前回選んだモードを使う
  useEffect(() => {
    const isMode = (value: string | null): value is MotionMode =>
      MOTION_MODES.some((mode) => mode.id === value)
    const fromUrl = new URLSearchParams(window.location.search).get('motion_mode')
    let stored: string | null = null
    try {
      stored = window.localStorage.getItem(MOTION_MODE_STORAGE_KEY)
    } catch {
      stored = null
    }
    if (isMode(fromUrl)) setMotionMode(fromUrl)
    else if (isMode(stored)) setMotionMode(stored)
  }, [])

  const selectMotionMode = (mode: MotionMode) => {
    setMotionMode(mode)
    try {
      window.localStorage.setItem(MOTION_MODE_STORAGE_KEY, mode)
    } catch {
      // 保存できなくても、このタブでは選んだモードで動く
    }
  }

  const [chatWidth, setChatWidth] = useState<number | string>('35%')
  const [isResizing, setIsResizing] = useState(false)

  useEffect(() => {
    setChatWidth(window.innerWidth * 0.35)
  }, [])

  useEffect(() => {
    if (!isResizing) return

    const handleMouseMove = (e: MouseEvent) => {
      // Limit minimum to 320px and maximum to 80% inner width
      const newWidth = Math.max(320, Math.min(e.clientX, window.innerWidth * 0.8))
      setChatWidth(newWidth)
    }

    const handleMouseUp = () => {
      setIsResizing(false)
    }

    document.addEventListener('mousemove', handleMouseMove)
    document.addEventListener('mouseup', handleMouseUp)
    return () => {
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isResizing])

  const {
    messages,
    startChat,
    speakingMessageId,
    resetChat,
    avatarFrameSrc,
    latestUrl,
    bvhText,
    bvhPrompt,
    interruptChat,
    scheduledMotions,
    getAudioTime,
    motionModeStatus,
    motionActivity,
  } =
    useTextChat('/api/text-chat/char-stream-orchestrator')

  const handleMicClick = () => {
    console.log('🎙️ Mic clicked')
  }

  const handleSend = async (value: string) => {
    const message = value.trim()
    if (!message) return

    setIsSending(true)
    try {
      await startChat(message, appMode, { motionMode, posture: postureRef.current })
    } finally {
      setIsSending(false)
    }

    setInputValue('')
  }

  const handleTemplateClick = (text: string) => {
    setInputValue(text)
    handleSend(text)
  }



  // ユーザーが閉じたURLでなければ表示する (式典モード時は非表示)
  const displayUrl = appMode === 'general' && latestUrl && !closedUrls.has(latestUrl) ? latestUrl : null

  const handleCloseQRCode = () => {
    if (latestUrl) {
      setClosedUrls((prev) => new Set(prev).add(latestUrl))
    }
  }

  return (
    <main
      style={{
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'flex-start',
        height: '100vh',
        backgroundColor: '#000',
        overflow: 'hidden',
        cursor: isResizing ? 'col-resize' : 'auto',
        userSelect: isResizing ? 'none' : 'auto',
      }}
    >
      <style>{`
        @media (max-width: 760px) {
          .gakucho-layout { flex-direction: column-reverse !important; }
          .gakucho-chat-column { width: 100% !important; flex: 1 !important; min-height: 0; }
          .gakucho-viewer-column { flex: none !important; height: 62vh; min-height: 480px; max-height: 560px; }
          .gakucho-resize-handle { display: none !important; }
          .gakucho-motion-mode { top: 56px !important; right: auto !important; left: 16px; }
        }
      `}</style>
      <section
        className="gakucho-layout"
        style={{
          display: 'flex',
          justifyContent: 'flex-start',
          alignItems: 'stretch',
          height: '100vh',
          width: '100%',
        }}
      >
        <div className="gakucho-chat-column" style={{ width: chatWidth, flexShrink: 0, minWidth: 0 }}>
        <ChatPanel
          messages={messages}
          inputValue={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onMicClick={handleMicClick}
          onTemplateClick={handleTemplateClick}
          onSend={handleSend}
          isSending={isSending}
          onResetChat={resetChat}
          speakingMessageId={speakingMessageId}
          onInterrupt={interruptChat}
          appMode={appMode}
          onToggleMode={() => setAppMode((prev) => (prev === 'general' ? 'ceremony' : 'general'))}
          width="100%"
        />
        </div>

        {/* リサイズ用ハンドル */}
        <div
          className="gakucho-resize-handle"
          onMouseDown={(e) => {
            e.preventDefault()
            setIsResizing(true)
          }}
          style={{
            width: '12px',
            cursor: 'col-resize',
            backgroundColor: isResizing ? 'rgba(59, 130, 246, 0.4)' : 'transparent',
            zIndex: 50,
            transition: 'background-color 0.2s',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
          }}
          onMouseEnter={(e) => {
            if (!isResizing) e.currentTarget.style.backgroundColor = 'rgba(59, 130, 246, 0.2)'
          }}
          onMouseLeave={(e) => {
            if (!isResizing) e.currentTarget.style.backgroundColor = 'transparent'
          }}
        >
          <div style={{ width: '2px', height: '30px', backgroundColor: 'rgba(255,255,255,0.2)', borderRadius: '1px' }} />
        </div>

        <div className="gakucho-viewer-column" style={{ flex: 1, minWidth: 0, minHeight: 0, position: 'relative', overflow: 'hidden' }}>
          {avatarViewMode === 'bvh' ? (
            <BvhViewer
              bvhText={bvhText}
              motionPrompt={bvhPrompt}
              scheduledMotions={scheduledMotions}
              getAudioTime={getAudioTime}
              onPostureChange={handlePostureChange}
            />
          ) : (
            <AvatarPanel frameSrc={avatarFrameSrc} />
          )}

          {/* アバター表示モード切替トグル */}
          <div
            style={{
              position: 'absolute',
              top: 12,
              left: 16,
              display: 'flex',
              flexWrap: 'wrap',
              maxWidth: 'calc(100% - 32px)',
              gap: 4,
              padding: 4,
              borderRadius: 999,
              background: 'rgba(15,23,42,0.85)',
              backdropFilter: 'blur(8px)',
              border: '1px solid rgba(148,163,184,0.3)',
              zIndex: 10,
            }}
          >
            <button
              onClick={() => setAvatarViewMode('bvh')}
              style={{
                padding: '6px 14px',
                borderRadius: 999,
                border: 'none',
                fontSize: 12,
                cursor: 'pointer',
                background:
                  avatarViewMode === 'bvh' ? 'rgba(59,130,246,0.85)' : 'transparent',
                color: avatarViewMode === 'bvh' ? '#fff' : '#94a3b8',
                transition: 'background 0.2s, color 0.2s',
              }}
            >
              🦴 動作生成
            </button>
            <button
              onClick={() => setAvatarViewMode('ue')}
              style={{
                padding: '6px 14px',
                borderRadius: 999,
                border: 'none',
                fontSize: 12,
                cursor: 'pointer',
                background:
                  avatarViewMode === 'ue' ? 'rgba(59,130,246,0.85)' : 'transparent',
                color: avatarViewMode === 'ue' ? '#fff' : '#94a3b8',
                transition: 'background 0.2s, color 0.2s',
              }}
            >
              🎥 UE ストリーミング
            </button>
            <a
              href="/world-model"
              style={{ padding: '6px 10px', borderRadius: 999, fontSize: 12, color: '#cbd5e1', textDecoration: 'none', whiteSpace: 'nowrap' }}
            >
              物理・学習
            </a>
          </div>

          {/* 動作生成モードの切り替えと、直近の動作の内訳(比較用) */}
          {avatarViewMode === 'bvh' && (
            <div
              className="gakucho-motion-mode"
              style={{
                position: 'absolute',
                top: 12,
                right: 16,
                maxWidth: 'min(400px, calc(100% - 32px))',
                padding: '8px 10px',
                borderRadius: 12,
                background: 'rgba(15,23,42,0.85)',
                backdropFilter: 'blur(8px)',
                border: '1px solid rgba(148,163,184,0.3)',
                color: '#cbd5e1',
                fontSize: 11,
                lineHeight: 1.5,
                zIndex: 10,
              }}
            >
              <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 4 }}>
                <button
                  onClick={() => setMotionPanelOpen((open) => !open)}
                  title={motionPanelOpen ? '折りたたむ' : '開く'}
                  style={{ marginRight: 4, padding: 0, border: 'none', background: 'none', color: '#94a3b8', fontSize: 11, cursor: 'pointer' }}
                >
                  {motionPanelOpen ? '▾' : '▸'} 動作生成
                </button>
                {(motionPanelOpen ? MOTION_MODES : MOTION_MODES.filter((mode) => mode.id === motionMode)).map((mode) => (
                  <button
                    key={mode.id}
                    title={mode.hint}
                    onClick={() => selectMotionMode(mode.id)}
                    style={{
                      padding: '4px 10px',
                      borderRadius: 999,
                      border: 'none',
                      fontSize: 11,
                      cursor: 'pointer',
                      background: motionMode === mode.id ? 'rgba(16,185,129,0.85)' : 'rgba(51,65,85,0.6)',
                      color: motionMode === mode.id ? '#fff' : '#cbd5e1',
                    }}
                  >
                    {mode.label}
                  </button>
                ))}
              </div>
              {motionPanelOpen && (
                <div style={{ marginTop: 4, color: '#94a3b8' }}>
                  {MOTION_MODES.find((mode) => mode.id === motionMode)?.hint}
                </div>
              )}
              {motionModeStatus?.reason && (
                <div style={{ marginTop: 2, color: '#fbbf24' }}>※ {motionModeStatus.reason}</div>
              )}
              {motionPanelOpen && motionActivity.length > 0 && (
                <ul style={{ margin: '4px 0 0', paddingLeft: 14, maxHeight: 96, overflowY: 'auto' }}>
                  {motionActivity.slice(-4).map((activity) => (
                    <li key={activity.receivedAt}>{describeActivity(activity)}</li>
                  ))}
                </ul>
              )}
            </div>
          )}

          <QRCodeDisplay url={displayUrl} onClose={handleCloseQRCode} />
        </div>
      </section>
    </main>
  )
}
