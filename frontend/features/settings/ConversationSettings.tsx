'use client'

import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react'
import { Activity, Box, Cpu, Settings2, SlidersHorizontal, Volume2, X } from 'lucide-react'
import { ChatModelSettings, SpeechModelSettings, type useChatModelSettings } from '@/features/text-chat/ChatModelSettings'
import type { MotionMode } from '@/features/text-chat/hooks/useTextChat'
import styles from './ConversationSettings.module.css'

type AppMode = 'general' | 'ceremony'
type InputMode = 'text' | 'voice'
type VoiceMode = 'ptt' | 'vad'
type AvatarViewMode = 'bvh' | 'ue'

type Choice<T extends string> = { id: T; label: string; hint?: string }

export const MOTION_MODES: { id: MotionMode; label: string; hint: string }[] = [
  { id: 'current', label: 'T2M生成', hint: '従来方式。行動タグごとにT2M-GPTが動作を生成します。' },
  { id: 'layered', label: 'クリップ＋音声', hint: '新方式。タグに対応する身振りを、SentiAvatar（研究用）が発話音声から作るリズムに重ねます。' },
  { id: 'layered-semantic', label: 'クリップ＋待機', hint: '新方式。タグに対応する身振りを待機姿勢に重ねます。音声リズムを使わずに動作を確認できます。' },
]

const sections = [
  { id: 'general', label: '一般', icon: SlidersHorizontal },
  { id: 'model', label: '対話モデル', icon: Cpu },
  { id: 'speech', label: '音声合成', icon: Volume2 },
  { id: 'motion', label: '動作生成', icon: Activity },
  { id: 'avatar', label: 'アバター・空間', icon: Box },
] as const

type Section = typeof sections[number]['id']

function Choices<T extends string>({ legend, options, value, onChange }: {
  legend: string; options: readonly Choice<T>[]; value: T; onChange: (value: T) => void
}) {
  const group = useId()
  return (
    <fieldset className={styles.choices}>
      <legend>{legend}</legend>
      <div className={options.some(option => option.hint) ? styles.list : styles.pair}>
        {options.map(option => (
          <label key={option.id} className={styles.choice}>
            <input type="radio" name={group} checked={value === option.id} onChange={() => onChange(option.id)}
              aria-label={option.label} aria-describedby={option.hint ? `${group}-${option.id}` : undefined} />
            <span>{option.label}{option.hint && <small id={`${group}-${option.id}`}>{option.hint}</small>}</span>
          </label>
        ))}
      </div>
    </fieldset>
  )
}

interface SettingsProps {
  open: boolean
  onClose: () => void
  modelSettings: ReturnType<typeof useChatModelSettings>
  appMode: AppMode
  onAppModeChange: (mode: AppMode) => void
  inputMode: InputMode
  onInputModeChange: (mode: InputMode) => void
  voiceMode: VoiceMode
  onVoiceModeChange: (mode: VoiceMode) => void
  avatarViewMode: AvatarViewMode
  onAvatarViewModeChange: (mode: AvatarViewMode) => void
  motionMode: MotionMode
  onMotionModeChange: (mode: MotionMode) => void
  setViewerSettingsHost: (element: HTMLDivElement | null) => void
}

export function ConversationSettings({ open, onClose, modelSettings, appMode, onAppModeChange, inputMode,
  onInputModeChange, voiceMode, onVoiceModeChange, avatarViewMode, onAvatarViewModeChange, motionMode,
  onMotionModeChange, setViewerSettingsHost }: SettingsProps) {
  const dialog = useRef<HTMLDialogElement>(null)
  const tabs = useRef<HTMLDivElement>(null)
  const fromBackdrop = useRef(false)
  const [section, setSection] = useState<Section>('general')
  const id = useId()

  useEffect(() => {
    const element = dialog.current
    if (!element || !open) return
    element.showModal()
    return () => element.close()
  }, [open])

  function navigate(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    let next: number
    switch (event.key) {
      case 'ArrowRight': case 'ArrowDown': next = (index + 1) % sections.length; break
      case 'ArrowLeft': case 'ArrowUp': next = (index + sections.length - 1) % sections.length; break
      case 'Home': next = 0; break
      case 'End': next = sections.length - 1; break
      default: return
    }
    event.preventDefault()
    setSection(sections[next].id)
    tabs.current?.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next]?.focus()
  }

  return (
    <dialog ref={dialog} className={styles.dialog} aria-labelledby={`${id}-title`}
      onCancel={event => { event.preventDefault(); onClose() }}
      onPointerDown={event => { fromBackdrop.current = event.target === event.currentTarget }}
      onClick={event => { if (fromBackdrop.current && event.target === event.currentTarget) onClose() }}>
      <div className={styles.layout}>
        <header className={styles.header}>
          <h2 id={`${id}-title`}><Settings2 size={20} aria-hidden="true" />設定</h2>
          <button type="button" aria-label="設定を閉じる" className={styles.close} onClick={onClose}><X size={19} /></button>
        </header>
        <div className={styles.tabs} ref={tabs} role="tablist" aria-label="設定項目">
          {sections.map(({ id: key, label, icon: Icon }, index) => (
            <button key={key} type="button" role="tab" id={`${id}-tab-${key}`} aria-selected={key === section}
              aria-controls={`${id}-panel-${key}`} tabIndex={key === section ? 0 : -1}
              onKeyDown={event => navigate(event, index)} onClick={() => setSection(key)}>
              <Icon size={17} aria-hidden="true" />{label}
            </button>
          ))}
        </div>
        {sections.map(({ id: key, label }) => (
          <section key={key} role="tabpanel" id={`${id}-panel-${key}`} aria-labelledby={`${id}-tab-${key}`}
            className={styles.panel} hidden={section !== key}>
            <h3>{label}</h3>
            {key === 'general' && <>
              <Choices<AppMode> legend="会話モード" options={[
                { id: 'general', label: '一般モード', hint: '必要に応じて、つなぎの発話や関連リンクのQRコードを使います。' },
                { id: 'ceremony', label: '式典モード', hint: '式典向けに応答し、つなぎの発話とQRコードを省きます。' },
              ]} value={appMode} onChange={onAppModeChange} />
              <Choices<InputMode> legend="入力方法" options={[
                { id: 'text', label: 'テキスト入力' }, { id: 'voice', label: '音声入力' },
              ]} value={inputMode} onChange={onInputModeChange} />
              {inputMode === 'voice' && <Choices<VoiceMode> legend="音声入力方式" options={[
                { id: 'ptt', label: 'Push-to-Talk', hint: 'マイクボタンを押して録音し、もう一度押して送信します。' },
                { id: 'vad', label: '自動（VAD）', hint: '録音を開始すると、声の区切りを検出して送信します。' },
              ]} value={voiceMode} onChange={onVoiceModeChange} />}
            </>}
            {key === 'model' && <ChatModelSettings settings={modelSettings} />}
            {key === 'speech' && <SpeechModelSettings settings={modelSettings} />}
            {key === 'motion' && <>
              <p className={styles.note}>変更は次の発話から適用されます。</p>
              <Choices legend="生成方式" options={MOTION_MODES} value={motionMode} onChange={onMotionModeChange} />
            </>}
            {key === 'avatar' && <>
              <Choices<AvatarViewMode> legend="表示方式" options={[
                { id: 'bvh', label: '動作生成（BVH）' }, { id: 'ue', label: 'UE ストリーミング' },
              ]} value={avatarViewMode} onChange={onAvatarViewModeChange} />
              {avatarViewMode === 'bvh' && <div className={styles.viewer} ref={setViewerSettingsHost} />}
              {avatarViewMode === 'ue' && <a href="/world-model">物理・学習</a>}
            </>}
          </section>
        ))}
      </div>
    </dialog>
  )
}
