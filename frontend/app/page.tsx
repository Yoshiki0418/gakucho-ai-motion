'use client'

import React, { useState, useEffect, useCallback, useRef } from 'react'
import ChatPanel from '@/components/organisms/ChatPanel'
import {
  useTextChat,
  type MotionActivity,
  type MotionMode,
} from '@/features/text-chat/hooks/useTextChat'
import { useChatModelSettings } from '@/features/text-chat/ChatModelSettings'
import { useUnrealBodyMotion } from '@/features/text-chat/hooks/useUnrealBodyMotion'
import { ConversationSettings, MOTION_MODES } from '@/features/settings/ConversationSettings'
import { AvatarPanel } from '@/components/organisms/AvatarPanel'
import { BvhViewer, type PostureKind } from '@/components/organisms/BvhViewer'
import { QRCodeDisplay } from '@/components/organisms/QRCodeDisplay'

type AvatarViewMode = 'bvh' | 'ue'

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
  const modelSettings = useChatModelSettings()
  const [inputValue, setInputValue] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [closedUrls, setClosedUrls] = useState<Set<string>>(new Set())
  const [appMode, setAppMode] = useState<'general' | 'ceremony'>('general')
  const [avatarViewMode, setAvatarViewMode] = useState<AvatarViewMode>('bvh')
  const [motionMode, setMotionMode] = useState<MotionMode>('current')
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [viewerSettingsHost, setViewerSettingsHost] = useState<HTMLDivElement | null>(null)
  const [inputMode, setInputMode] = useState<'text' | 'voice'>('text')
  const [voiceMode, setVoiceMode] = useState<'ptt' | 'vad'>('ptt')
  const closeSettings = useCallback(() => setSettingsOpen(false), [])
  const postureRef = useRef<PostureKind>('standing')
  const handlePostureChange = useCallback((posture: PostureKind) => {
    postureRef.current = posture
  }, [])

  // URL(?motion_mode=)を優先し、無ければ前回選んだモードを使う
  useEffect(() => {
    const isMode = (value: string | null): value is MotionMode =>
      MOTION_MODES.some((mode) => mode.id === value)
    const query = new URLSearchParams(window.location.search)
    if (query.get('settings') === '1') setSettingsOpen(true)
    const fromUrl = query.get('motion_mode')
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
    motionTurn,
    getAudioTime,
    motionModeStatus,
    motionActivity,
    chatError,
  } =
    useTextChat('/api/text-chat/char-stream-orchestrator')
  const unrealMotionError = useUnrealBodyMotion(avatarViewMode === 'ue', motionTurn, scheduledMotions, getAudioTime)

  const handleMicClick = () => {
    console.log('🎙️ Mic clicked')
  }

  const handleSend = async (value: string) => {
    const message = value.trim()
    if (!message || !modelSettings.server) return

    setIsSending(true)
    try {
      await startChat(message, appMode, { motionMode, posture: postureRef.current, ...modelSettings.request })
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
        .gakucho-motion-status :is(summary, ul, li) { color: inherit; }
        .gakucho-motion-status p { color: #fbbf24; }
        .gakucho-model-retry { color: inherit; cursor: pointer; text-decoration: underline; }
        @media (max-width: 760px) {
          .gakucho-layout { flex-direction: column-reverse !important; }
          .gakucho-chat-column { width: 100% !important; flex: 1 !important; min-height: 0; }
          .gakucho-viewer-column { flex: none !important; height: min(48dvh, 500px); }
          .gakucho-resize-handle { display: none !important; }
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
            disabled={!modelSettings.server}
            inputMode={inputMode}
            voiceMode={voiceMode}
            settingsOpen={settingsOpen}
            onOpenSettings={() => setSettingsOpen(true)}
            notice={<>
              {modelSettings.failed && (
                <div role="alert" style={{ padding: '6px 16px', color: '#fbbf24' }}>
                  対話モデル設定を取得できません。 <button type="button" className="gakucho-model-retry" onClick={modelSettings.retry}>再試行</button>
                </div>
              )}
              {modelSettings.ttsNotice && <div role="status" style={{ padding: '6px 16px', color: '#fbbf24' }}>{modelSettings.ttsNotice}</div>}
              {chatError && <div role="alert" style={{ padding: '6px 16px', color: '#fbbf24' }}>{chatError}</div>}
              {avatarViewMode === 'ue' && unrealMotionError && <div role="alert" style={{ padding: '6px 16px', color: '#fbbf24' }}>{unrealMotionError}</div>}
              {(motionModeStatus || motionActivity.length > 0) && (
                <details className="gakucho-motion-status" style={{ padding: '6px 16px', color: '#cbd5e1', fontSize: 12 }}>
                  <summary style={{ cursor: 'pointer' }}>
                    動作の実行状況{motionModeStatus && ` · ${MOTION_MODES.find(mode => mode.id === motionModeStatus.effective)?.label ?? motionModeStatus.effective}`}
                  </summary>
                  {motionModeStatus?.reason && <p>{motionModeStatus.reason}</p>}
                  {motionActivity.length > 0 && (
                    <ul style={{ maxHeight: 96, overflowY: 'auto', paddingLeft: 18 }}>
                      {motionActivity.slice(-4).map(activity => <li key={activity.receivedAt}>{describeActivity(activity)}</li>)}
                    </ul>
                  )}
                </details>
              )}
            </>}
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
              settingsHost={viewerSettingsHost}
            />
          ) : (
            <AvatarPanel frameSrc={avatarFrameSrc} />
          )}

          {avatarViewMode === 'bvh' && (
            <a
              href="/world-model"
              style={{ position: 'absolute', top: 12, right: 16, zIndex: 3,
                padding: '6px 12px', borderRadius: 8,
                background: 'rgba(15,23,42,0.85)', color: '#cbd5e1', fontSize: 12, textDecoration: 'none' }}
            >
              物理・学習
            </a>
          )}
          <QRCodeDisplay url={displayUrl} onClose={handleCloseQRCode} />
        </div>
      </section>
      <ConversationSettings
        open={settingsOpen}
        onClose={closeSettings}
        modelSettings={modelSettings}
        appMode={appMode}
        onAppModeChange={setAppMode}
        inputMode={inputMode}
        onInputModeChange={setInputMode}
        voiceMode={voiceMode}
        onVoiceModeChange={setVoiceMode}
        avatarViewMode={avatarViewMode}
        onAvatarViewModeChange={setAvatarViewMode}
        motionMode={motionMode}
        onMotionModeChange={selectMotionMode}
        setViewerSettingsHost={setViewerSettingsHost}
      />
    </main>
  )
}
