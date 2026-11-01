import { readdir } from 'fs/promises'
import path from 'path'
import { NextResponse } from 'next/server'

/**
 * public/avatars 配下の .glb / .gltf を列挙して返す。
 *
 * Unreal Engine から書き出したアバターをこのフォルダへ置くだけで、
 * ビューアのアバターセレクタに自動で並ぶ。
 *
 * パスを /api/ 配下にしていないのは、本番の nginx が /api/ をバックエンド(FastAPI)へ
 * 振り分けるため。/avatar-list は location / でフロントエンドへ届く。
 */
export const dynamic = 'force-dynamic'

const AVATAR_DIR = path.join(process.cwd(), 'public', 'avatars')
const MODEL_EXTENSIONS = new Set(['.glb', '.gltf'])

export async function GET() {
  try {
    const entries = await readdir(AVATAR_DIR, { withFileTypes: true })
    const avatars = entries
      .filter(
        (entry) =>
          entry.isFile() && MODEL_EXTENSIONS.has(path.extname(entry.name).toLowerCase()),
      )
      .map((entry) => ({
        label: entry.name,
        url: `/avatars/${entry.name}`,
      }))
      .sort((a, b) => a.label.localeCompare(b.label))

    return NextResponse.json({ avatars })
  } catch (error) {
    console.error('[api/avatars] failed to list avatars:', error)
    return NextResponse.json({ avatars: [] }, { status: 200 })
  }
}
