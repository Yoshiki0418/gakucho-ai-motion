'use client'

import { useEffect, useState } from 'react'
import styles from '@/features/settings/ConversationSettings.module.css'

export type ChatProvider = 'openai' | 'local'
type Preference = { provider: 'auto' | ChatProvider; local: string; openai: string; ttsProvider: string }
type SpeechCatalog = {
  default_provider: string
  providers: { id: string; label: string; available: boolean }[]
}
type ServerSettings = {
  default_provider: ChatProvider
  openai_available: boolean
  openai_model: string
  local_model: string
  tts?: SpeechCatalog
}

const STORAGE_KEY = 'gakucho.chatModel'
const DEFAULT: Preference = { provider: 'auto', local: '', openai: '', ttsProvider: 'auto' }
const LABELS = { openai: 'OpenAI API', local: 'ローカルLLM' }

function readPreference(): Preference {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? 'null')
    if (value && ['auto', 'openai', 'local'].includes(value.provider)) {
      return {
        provider: value.provider,
        local: typeof value.local === 'string' ? value.local : '',
        openai: typeof value.openai === 'string' ? value.openai : '',
        ttsProvider: typeof value.ttsProvider === 'string' ? value.ttsProvider : 'auto',
      }
    }
  } catch {
    // ブラウザが保存を制限しているときは初期設定を使う。
  }
  return DEFAULT
}

export function useChatModelSettings() {
  const [server, setServer] = useState<ServerSettings | null>(null)
  const [preference, setPreference] = useState(DEFAULT)
  const [failed, setFailed] = useState(false)
  const [loading, setLoading] = useState(true)
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    setPreference(readPreference())
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    async function load() {
      try {
        const response = await fetch('/api/text-chat/model-settings', {
          signal: controller.signal, cache: 'no-store',
        })
        if (!response.ok) throw new Error('model-settings')
        const settings: ServerSettings = await response.json()
        if (!['openai', 'local'].includes(settings.default_provider) ||
            typeof settings.openai_available !== 'boolean' ||
            typeof settings.openai_model !== 'string' || typeof settings.local_model !== 'string') {
          throw new Error('model-settings')
        }
        const tts = settings.tts
        const hasSpeechCatalog = tts && typeof tts.default_provider === 'string' &&
          Array.isArray(tts.providers) && tts.providers.every(item => item &&
            typeof item.id === 'string' && typeof item.label === 'string' && typeof item.available === 'boolean')
        if (!controller.signal.aborted) setServer({ ...settings, tts: hasSpeechCatalog ? tts : undefined })
      } catch {
        if (!controller.signal.aborted) setFailed(true)
      } finally {
        if (!controller.signal.aborted) setLoading(false)
      }
    }
    void load()
    return () => controller.abort()
  }, [attempt])

  function save(next: Preference) {
    setPreference(next)
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
    } catch {
      // タブ内では引き続き設定を利用できる。
    }
  }

  const provider = preference.provider === 'openai' && !server?.openai_available
    ? 'auto' : preference.provider
  const model = provider === 'auto' ? '' : preference[provider]
  const catalog = server?.tts
  const speechSelection = catalog?.providers.find(item => item.id === preference.ttsProvider && item.available)
  const defaultSpeech = catalog?.providers.find(item => item.id === catalog.default_provider)
  let ttsNotice = ''
  if (server && !speechSelection) {
    if (defaultSpeech && !defaultSpeech.available) {
      ttsNotice = 'サーバー既定の音声合成が利用できません。「設定」の「音声合成」からエンジンを選んでください。'
    } else if (preference.ttsProvider !== 'auto') {
      ttsNotice = '以前選んだ音声合成エンジンが利用できないため、サーバー既定の音声を使用します。'
    }
  }

  return {
    server, failed, loading, provider, model,
    ttsProvider: speechSelection?.id ?? 'auto',
    ttsNotice,
    selectTtsProvider: (value: string) => save({ ...preference, ttsProvider: value }),
    selectProvider: (value: Preference['provider']) => save({ ...preference, provider: value }),
    setModel: (value: string) => {
      if (provider !== 'auto') save({ ...preference, [provider]: value })
    },
    retry: () => {
      setFailed(false)
      setLoading(true)
      setAttempt((last) => last + 1)
    },
    request: {
      ...(provider === 'auto' ? {} : { llmProvider: provider, llmModel: model.trim() || undefined }),
      ttsProvider: speechSelection?.id,
    },
  }
}

type ModelSettings = ReturnType<typeof useChatModelSettings>

function SettingsStatus({ failed, retry }: Pick<ModelSettings, 'failed' | 'retry'>) {
  return (
    <div role="status" className={styles.status}>
      {failed ? <>サーバーの設定を取得できません。 <button type="button" onClick={retry}>再試行</button></>
        : 'サーバーの設定を読み込み中…'}
    </div>
  )
}

export function SpeechModelSettings({ settings }: { settings: ModelSettings }) {
  const { server, failed, loading, retry, ttsProvider, ttsNotice, selectTtsProvider } = settings
  if (!server) return <SettingsStatus failed={failed} retry={retry} />

  const catalog = server.tts
  if (!catalog) {
    return <p className={styles.note}>このサーバーでは既定の音声合成を使用します。エンジンの切り替えには対応するバックエンドが必要です。</p>
  }

  const defaultLabel = catalog.providers.find(item => item.id === catalog.default_provider)?.label ?? catalog.default_provider
  return (
    <div className={styles.modelFields}>
      <label>
        <span className={styles.fieldLabel}>音声合成エンジン</span>
        <select value={ttsProvider} onChange={event => selectTtsProvider(event.target.value)}>
          <option value="auto">サーバー既定（{defaultLabel}）</option>
          {catalog.providers.map(item => (
            <option key={item.id} value={item.id} disabled={!item.available}>
              {item.label}{item.available ? '' : '（利用できません）'}
            </option>
          ))}
        </select>
      </label>
      {ttsNotice && <p role="status" className={styles.status}>{ttsNotice}</p>}
      {catalog.providers.some(item => !item.available) && (
        <div className={styles.status}>
          <button type="button" disabled={loading} onClick={retry}>
            {loading ? '確認中…' : '利用状況を再確認'}
          </button>
        </div>
      )}
      <p className={styles.note}>変更は次のメッセージへの返答から適用されます。</p>
    </div>
  )
}

export function ChatModelSettings({ settings }: { settings: ModelSettings }) {
  const { server, failed, provider, model, setModel, selectProvider, retry } = settings
  if (!server) return <SettingsStatus failed={failed} retry={retry} />
  const effective = provider === 'auto' ? server.default_provider : provider
  const defaultModel = effective === 'openai' ? server.openai_model : server.local_model
  return (
    <div className={styles.modelFields}>
      <label>
        <span className={styles.fieldLabel}>接続先</span>
        <select value={provider} onChange={(event) => selectProvider(event.target.value as Preference['provider'])}>
          <option value="auto">自動（{LABELS[server.default_provider]}）</option>
          <option value="openai" disabled={!server.openai_available}>
            OpenAI API{server.openai_available ? '' : '（キー未登録）'}
          </option>
          <option value="local">ローカルLLM</option>
        </select>
      </label>
      {provider === 'auto' ? (
        <p className={styles.note}>使用するモデル　{LABELS[effective]} / {defaultModel}</p>
      ) : (
        <label>
          <span className={styles.fieldLabel}>モデルID（空欄でサーバー既定）</span>
          <input value={model} onChange={(event) => setModel(event.target.value)}
            placeholder={defaultModel} maxLength={256} spellCheck={false} autoComplete="off" />
        </label>
      )}
    </div>
  )
}
