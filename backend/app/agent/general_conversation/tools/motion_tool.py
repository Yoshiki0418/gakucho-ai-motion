"""
モーション生成ツール - Text-to-Motion サーバーと連携
エージェントが会話の文脈に基づいて、適切な身体動作（BVHモーション）を自律的に生成します。
"""

import json
import os
import urllib.request
import urllib.error

import contextvars
from agents import function_tool

# T2M サーバーの URL（環境変数で上書き可能。ホストマシンのT2Mサーバーを参照）
T2M_SERVER_URL = os.getenv("T2M_SERVER_URL", "http://192.168.29.19:8099")

# フロントエンドにBVHデータを直接送るためのサイドチャネル（コンテキスト変数）
motion_data_queue_var = contextvars.ContextVar('motion_data_queue_var', default=None)


@function_tool
def generate_motion(motion_description: str) -> str:
    """
    テキストで指定された動作に基づいて、3Dモーション（BVHデータ）を生成します。
    身振り手振り・ジェスチャー・動作表現が会話に適していると判断した場合に呼び出してください。

    motion_descriptionは必ず英語で記述してください。
    例: "a person waves their right hand", "a person bows politely",
        "a person nods their head", "a person gestures with both hands while explaining"

    Args:
        motion_description: 生成したいモーションの英語テキスト説明
    """
    try:
        # 1. T2M サーバーにモーション生成をリクエスト
        generate_url = f"{T2M_SERVER_URL}/api/realtime/generate"
        payload = json.dumps({
            "text": motion_description,
            "model": "t2m_gpt",
        }).encode("utf-8")

        req = urllib.request.Request(
            generate_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        bvh_path = result.get("bvh_path", "")
        latency = result.get("latency_seconds", 0)

        if not bvh_path:
            return json.dumps({
                "status": "error",
                "message": "モーション生成に失敗しました（BVHパスが空です）",
            })

        # 2. 生成された BVH ファイルの内容を取得
        bvh_url = f"{T2M_SERVER_URL}/bvh/{bvh_path}"
        bvh_req = urllib.request.Request(bvh_url, method="GET")

        with urllib.request.urlopen(bvh_req, timeout=30) as bvh_resp:
            bvh_text = bvh_resp.read().decode("utf-8")

        motion_data = {
            "status": "success",
            "prompt": motion_description,
            "latency_seconds": latency,
            "bvh": bvh_text,
        }
        
        # キューがあれば追加（サイドチャネルでフロントへ送信）
        print("motion data is ", motion_data)
        queue = motion_data_queue_var.get()
        if queue is not None:
            queue.append(motion_data)

        # エージェント（LLM）には結果のステータスだけを返す（トークン節約）
        return json.dumps({
            "status": "success",
            "message": "Motion generated and sent to frontend successfully. Do not mention this action in your response.",
        })

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            detail = json.loads(error_body).get("detail", "")
            # モデルがロードされていない場合は自動でロードして再試行
            if isinstance(detail, str) and "No model is currently loaded" in detail:
                try:
                    load_url = f"{T2M_SERVER_URL}/api/realtime/load"
                    load_req = urllib.request.Request(
                        load_url,
                        data=json.dumps({"model": "t2m_gpt"}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(load_req, timeout=60):
                        pass
                    
                    # 再試行
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        result = json.loads(resp.read().decode("utf-8"))
                        
                    bvh_path = result.get("bvh_path", "")
                    latency = result.get("latency_seconds", 0)
                    
                    if not bvh_path:
                        return json.dumps({"status": "error", "message": "モーション生成に失敗しました（再試行後もBVHパスが空です）"})
                        
                    bvh_url = f"{T2M_SERVER_URL}/bvh/{bvh_path}"
                    bvh_req = urllib.request.Request(bvh_url, method="GET")
                    with urllib.request.urlopen(bvh_req, timeout=30) as bvh_resp:
                        bvh_text = bvh_resp.read().decode("utf-8")
                        
                    motion_data = {
                        "status": "success",
                        "prompt": motion_description,
                        "latency_seconds": latency,
                        "bvh": bvh_text,
                    }
                    
                    queue = motion_data_queue_var.get()
                    if queue is not None:
                        queue.append(motion_data)
                        
                    return json.dumps({
                        "status": "success",
                        "message": "Motion generated and sent to frontend successfully.",
                    })
                except Exception as retry_e:
                    return json.dumps({"status": "error", "message": f"モデルの自動ロード・再試行に失敗しました: {retry_e}"})

            return json.dumps({
                "status": "error",
                "message": f"T2Mサーバーでエラーが発生しました (HTTP {e.code}): {detail or error_body}",
            })
        except Exception:
            return json.dumps({
                "status": "error",
                "message": f"T2Mサーバーでエラーが発生しました (HTTP {e.code}): {error_body}",
            })
    except urllib.error.URLError as e:
        return json.dumps({
            "status": "error",
            "message": f"T2Mサーバーに接続できません: {e}",
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"モーション生成中にエラーが発生しました: {e}",
        })

