import type { NextConfig } from "next";

/**
 * 開発時に `/api` `/ws` をバックエンドへ中継するための向き先。
 * 本番は nginx が同じ振り分けを行うため、この変数は設定しない(= 中継しない)。
 *
 *   DEV_API_PROXY_TARGET=http://127.0.0.1:8087 npm run dev
 */
const DEV_API_PROXY_TARGET = process.env.DEV_API_PROXY_TARGET;

/**
 * 学習の様子(/world-model/training)が読む、学習の書き出しを配る静的配信コンテナ。
 * 本番では同じ Docker ネットワーク上の `training-monitor` を指す(world_model/tracker/monitor.py)。
 */
const TRAINING_MONITOR_URL = process.env.TRAINING_MONITOR_URL ?? 'http://training-monitor:80';
/** 学習中の重みで身体を実時間で動かして配る(/world-model/live)。world_model/tracker/live_server.py */
const TRAINING_LIVE_URL = process.env.TRAINING_LIVE_URL ?? 'http://training-live:8120';

const nextConfig: NextConfig = {
  output: "standalone",
  async rewrites() {
    const rules = [
      {
        source: '/lip-sync/:path*',
        destination: 'http://127.0.0.1:8080/:path*',
      },
      {
        source: '/training-monitor/:path*',
        destination: `${TRAINING_MONITOR_URL}/:path*`,
      },
      {
        source: '/training-live/:path*',
        destination: `${TRAINING_LIVE_URL}/:path*`,
      },
    ]

    if (DEV_API_PROXY_TARGET) {
      // バックエンドのルータは /api/text-chat のように /api を含むパスで公開されている
      rules.push({
        source: '/api/:path*',
        destination: `${DEV_API_PROXY_TARGET}/api/:path*`,
      })
      rules.push({
        source: '/ws/:path*',
        destination: `${DEV_API_PROXY_TARGET}/ws/:path*`,
      })
    }

    return rules
  },
};

export default nextConfig;
