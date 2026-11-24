'use client'

import { useEffect, useState } from 'react'
import styled from 'styled-components'

export type ChatProvider = 'openai' | 'local'
type Preference = { provider: 'auto' | ChatProvider; local: string; openai: string }
type ServerSettings = {
  default_provider: ChatProvider
  openai_available: boolean
  openai_model: string
  local_model: string
}

const STORAGE_KEY = 'gakucho.chatModel'
const DEFAULT: Preference = { provider: 'auto', local: '', openai: '' }
const LABELS = { openai: 'OpenAI API', local: 'ローカルLLM' }

function readPreference(): Preference {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? 'null')
    if (value && ['auto', 'openai', 'local'].includes(value.provider)) {
      return {
        provider: value.provider,
        local: typeof value.local === 'string' ? value.local : '',
        openai: typeof value.openai === 'string' ? value.openai : '',
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
  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    setPreference(readPreference())
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
        if (!controller.signal.aborted) setServer(settings)
      } catch {
        if (!controller.signal.aborted) setFailed(true)
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
  return {
    server, failed, provider, model,
    selectProvider: (value: Preference['provider']) => save({ ...preference, provider: value }),
    setModel: (value: string) => {
      if (provider !== 'auto') save({ ...preference, [provider]: value })
    },
    retry: () => {
      setFailed(false)
      setAttempt((last) => last + 1)
    },
    request: provider === 'auto' ? {} : { llmProvider: provider, llmModel: model.trim() || undefined },
  }
}

const Panel = styled.details`
  color: #cbd5e1;
  background: #0f172a;
  border-bottom: 1px solid #334155;
  font-size: 0.85rem;
  padding: 8px 16px;
  summary { cursor: pointer; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  strong { color: #f1f5f9; margin-right: 10px; }
  summary::before { content: '▸'; margin-right: 6px; }
  &[open] summary::before { content: '▾'; }
  summary { list-style: none; }
  summary::-webkit-details-marker { display: none; }
  label { display: grid; gap: 4px; margin: 8px 0 4px; }
  input, select, button {
    color: #f1f5f9; background: #1e293b; border: 1px solid #64748b;
    font: inherit; border-radius: 6px; padding: 4px 8px; min-width: 0;
  }
`

export function ChatModelSettings({ settings }: { settings: ReturnType<typeof useChatModelSettings> }) {
  const { server, failed, provider, model, setModel, selectProvider, retry } = settings
  if (!server) {
    return (
      <div role="status" style={{ padding: '8px 16px', color: '#cbd5e1', fontSize: '0.85rem' }}>
        {failed ? <>対話モデル設定を取得できません。 <button onClick={retry}>再試行</button></>
          : '対話モデル設定を読み込み中…'}
      </div>
    )
  }
  const effective = provider === 'auto' ? server.default_provider : provider
  const defaultModel = effective === 'openai' ? server.openai_model : server.local_model
  return (
    <Panel>
      <summary>
        <strong>対話モデル</strong>{provider === 'auto' && '自動 · '}{LABELS[effective]} / {model.trim() || defaultModel}
      </summary>
      <label>
        接続先
        <select value={provider} onChange={(event) => selectProvider(event.target.value as Preference['provider'])}>
          <option value="auto">自動（{LABELS[server.default_provider]}）</option>
          <option value="openai" disabled={!server.openai_available}>
            OpenAI API{server.openai_available ? '' : '（キー未登録）'}
          </option>
          <option value="local">ローカルLLM</option>
        </select>
      </label>
      {provider !== 'auto' && (
        <label>
          モデルID（空欄でサーバー既定）
          <input value={model} onChange={(event) => setModel(event.target.value)}
            placeholder={defaultModel} maxLength={256} spellCheck={false} autoComplete="off" />
        </label>
      )}
    </Panel>
  )
}
