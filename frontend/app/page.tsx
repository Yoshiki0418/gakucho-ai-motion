'use client'

import React, { useState, useEffect } from 'react'
import ChatPanel from '@/components/organisms/ChatPanel'
import { useTextChat } from '@/features/text-chat/hooks/useTextChat'
import { AvatarPanel } from '@/components/organisms/AvatarPanel'
import { BvhViewer } from '@/components/organisms/BvhViewer'
import { QRCodeDisplay } from '@/components/organisms/QRCodeDisplay'

type AvatarViewMode = 'bvh' | 'ue'

export default function ChatDemoPage() {
  const [inputValue, setInputValue] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [closedUrls, setClosedUrls] = useState<Set<string>>(new Set())
  const [appMode, setAppMode] = useState<'general' | 'ceremony'>('general')
  const [avatarViewMode, setAvatarViewMode] = useState<AvatarViewMode>('bvh')

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

  const { messages, startChat, speakingMessageId, resetChat, avatarFrameSrc, latestUrl, bvhText, interruptChat } =
    useTextChat('/api/text-chat/char-stream-orchestrator')

  const handleMicClick = () => {
    console.log('🎙️ Mic clicked')
  }

  const handleSend = async (value: string) => {
    const message = value.trim()
    if (!message) return

    setIsSending(true)
    try {
      await startChat(message, appMode)
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
      <section
        style={{
          display: 'flex',
          justifyContent: 'flex-start',
          alignItems: 'stretch',
          height: '100vh',
          width: '100%',
        }}
      >
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
          width={chatWidth}
        />

        {/* リサイズ用ハンドル */}
        <div
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

        <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
          {avatarViewMode === 'bvh' ? (
            <BvhViewer bvhText={bvhText} />
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
          </div>

          <QRCodeDisplay url={displayUrl} onClose={handleCloseQRCode} />
        </div>
      </section>
    </main>
  )
}