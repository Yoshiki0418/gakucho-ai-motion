'use client'

import { useEffect, useRef, useState } from 'react'
import styles from './AvatarPanel.module.css'

// The UE editor and the portrait camera share signalling. Wait specifically
// for the camera, including when the editor starts before the Level Sequence.
const PORTRAIT_URL = '/ue-stream/uiless.html?' + new URLSearchParams({
  StreamerId: 'GakuchoPortrait',
  WaitForStreamer: 'true',
  AutoConnect: 'true',
  AutoPlayVideo: 'true',
  StartVideoMuted: 'true',
  MatchViewportRes: 'false',
  MouseInput: 'false',
  KeyboardInput: 'false',
  TouchInput: 'false',
}).toString()

export function AvatarPanel({ frameSrc, streamUrl = PORTRAIT_URL }: {
  frameSrc: string | null
  streamUrl?: string
}) {
  const player = useRef<HTMLIFrameElement>(null)
  const portrait = useRef<HTMLVideoElement>(null)
  const [attempt, setAttempt] = useState(0)
  const [status, setStatus] = useState<'connecting' | 'playing' | 'unavailable'>('connecting')

  useEffect(() => {
    const output = portrait.current
    if (!output) return
    let lastFrameAt = performance.now()
    let lastTime = -1
    setStatus('connecting')

    const timer = window.setInterval(() => {
      // Use only the decoded media. Player controls, settings panels and
      // reconnect overlays in the signalling iframe never enter the portrait.
      const source = player.current?.contentDocument?.querySelector('video')
      const now = performance.now()
      const hasPortrait = source?.srcObject && source.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
        Math.abs(source.videoWidth / source.videoHeight - 1.5) < 0.01
      if (hasPortrait && source.currentTime !== lastTime) {
        lastTime = source.currentTime
        lastFrameAt = now
      }
      if (!hasPortrait || now - lastFrameAt > 12000) {
        output.srcObject = null
        setStatus(now - lastFrameAt > 12000 ? 'unavailable' : 'connecting')
        return
      }
      if (output.srcObject !== source.srcObject) {
        setStatus('connecting')
        output.srcObject = source.srcObject
        void output.play().catch(() => setStatus('unavailable'))
      } else if (!output.paused && output.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        setStatus('playing')
      }
    }, 250)
    return () => {
      window.clearInterval(timer)
      output.srcObject = null
    }
  }, [attempt, streamUrl])

  return (
    <div className={styles.panel}>
      <iframe key={attempt} ref={player} src={streamUrl} title="学長カメラの配信接続"
        aria-hidden="true" tabIndex={-1} allow="autoplay" className={styles.connection} />
      <video ref={portrait} autoPlay playsInline muted disablePictureInPicture
        aria-label={frameSrc ? '発話中の学長の上半身' : '学長の上半身'}
        className={styles.portrait} hidden={status !== 'playing'} />
      {status !== 'playing' && <div className={styles.message} role="status">
        {status === 'connecting' ? '学長のカメラに接続中…' : <>
          学長のカメラに接続できません。
          <button type="button" onClick={() => setAttempt(n => n + 1)}>再接続</button>
        </>}
      </div>}
    </div>
  )
}
