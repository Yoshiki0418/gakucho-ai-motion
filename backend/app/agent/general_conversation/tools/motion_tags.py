"""応答テキストに埋め込む行動タグ(インラインタグ)の解析。

なぜタグにするのか
------------------
従来は LLM が `generate_motion` ツールを呼ぶ方式だったが、これには 2 つの限界があった。

1. **1 応答に 1 モーションになりやすい** — ツール呼び出しは重いので、
   モデルは応答あたり 1 回しか呼ばない傾向がある。
2. **どの文に合わせるか指定できない** — ツールは「動作」しか渡せないので、
   応答のどの位置で動くべきかという情報が失われる。

そこで応答本文に `[act:wave]` のようなタグを直接書かせる。

    こんにちは[act:wave]。本日はご来場ありがとうございます[act:bow]。

タグは
  - 発話・表示テキストからは取り除かれる(TTS にも画面にも出ない)
  - 「何文目の、どのあたりか」を保持したままモーション生成へ渡される

ため、1 応答に複数のモーションを、文中の狙った位置に置ける。

タグ名は `MOT_xxx` という生の ID ではなく短い英単語にしている。
モデルに 29 個の ID を覚えさせるより誤りが少なく、ログも読みやすい。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .motion_catalog import _RULES, normalize_motion_prompt

# 応答本文に書けるタグ。値は Custom-T2M-Lab のカタログ ID。
ACTION_TAGS: dict[str, str] = {
    "wave": "MOT_701",  # 手を振って挨拶
    "bye": "MOT_740",  # さようならと手を振る
    "raise_hand": "MOT_783",  # 手を挙げて示す
    "bow": "MOT_711",  # お辞儀
    "nod": "MOT_702",  # うなずく
    "shake_head": "MOT_703",  # 首を横に振る
    "thumbs_up": "MOT_714",  # いいね
    "thumbs_down": "MOT_715",  # ダメ
    "wait": "MOT_774",  # ちょっと待って
    "stop": "MOT_707",  # ストップ
    "calm": "MOT_765",  # 落ち着いて
    "ok": "MOT_706",  # OK サイン
    "cross": "MOT_734",  # バツ・却下
    "offer": "MOT_720",  # どうぞ
    "come": "MOT_719",  # こちらへ
    "point": "MOT_705",  # 正面を指差す
    "point_right": "MOT_721",  # 右方向を指差す
    "thanks": "MOT_738",  # 感謝
    "please": "MOT_739",  # お願いします
    "shrug": "MOT_708",  # わからない
    "think_tap": "MOT_733",  # 考えて
    "thinking": "MOT_748",  # 考え中
    "clap": "MOT_709",  # 拍手
    "cheer": "MOT_735",  # ガッツポーズ
    "phone": "MOT_729",  # 電話
    "heart": "MOT_768",  # ハート
    "shout": "MOT_766",  # 叫ぶ
    "peace": "MOT_716",  # ピースサイン
    "salute": "MOT_710",  # 敬礼
}

# カタログ ID → 生成プロンプト
_PROMPT_BY_ID: dict[str, str] = {rule.catalog_id: rule.prompt for rule in _RULES}

# `[act:wave]` / `[act: wave]` / 全角ブラケットも拾う
TAG_PATTERN = re.compile(
    r"[\[［]\s*act\s*[:：]\s*([A-Za-z0-9_ぁ-んァ-ヶ一-龠]+)\s*[\]］]"
)


@dataclass(frozen=True)
class ActionTag:
    """本文から取り出した 1 個の行動タグ。"""

    name: str
    """タグ名(そのまま。未知タグの場合は自由記述として扱う)"""

    prompt: str
    """モーション生成へ渡す日本語プロンプト"""

    catalog_id: str | None
    """対応するカタログ ID。自由記述なら None"""

    position: float
    """タグが文中のどのあたりにあったか(0.0=先頭, 1.0=末尾)"""


def resolve_tag(name: str) -> tuple[str, str | None] | None:
    """タグ名を (生成プロンプト, カタログID) に解決する。

    1. 既知のタグ名 → カタログ ID 経由でプロンプトを引く
    2. 未知なら日本語の自由記述とみなし、カタログのキーワード照合にかける
    3. それでも当たらなければ、記述そのものをプロンプトとして使う
    """
    key = name.strip()
    catalog_id = ACTION_TAGS.get(key.lower())
    if catalog_id:
        prompt = _PROMPT_BY_ID.get(catalog_id)
        if prompt:
            return prompt, catalog_id

    if not key:
        return None

    match = normalize_motion_prompt(key)
    return match.prompt, match.catalog_id


def extract_action_tags(text: str) -> tuple[str, list[ActionTag]]:
    """本文からタグを取り除き、取り出したタグを返す。

    `position` はタグを除去した後の本文における文字位置の比率。
    フロントエンドはこの比率に発話長を掛けて再生開始位置を決める。
    """
    tags: list[ActionTag] = []
    clean_parts: list[str] = []
    cursor = 0
    raw_positions: list[tuple[int, str]] = []

    for match in TAG_PATTERN.finditer(text):
        clean_parts.append(text[cursor : match.start()])
        # 除去後の本文における位置
        raw_positions.append((sum(len(p) for p in clean_parts), match.group(1)))
        cursor = match.end()
    clean_parts.append(text[cursor:])

    clean = "".join(clean_parts)
    length = max(len(clean), 1)

    for offset, name in raw_positions:
        resolved = resolve_tag(name)
        if resolved is None:
            continue
        prompt, catalog_id = resolved
        tags.append(
            ActionTag(
                name=name,
                prompt=prompt,
                catalog_id=catalog_id,
                position=min(max(offset / length, 0.0), 1.0),
            )
        )

    return clean, tags


# 開きかけのタグ。ストリーミングでは `[act:` までしか届いていない瞬間がある。
# `[` だけで途切れる瞬間もあるので、ブラケット単独も未完了として扱う。
# 本文が本当に `[` で終わっていた場合も次のチャンクまで待つだけで済む。
OPEN_TAG_PATTERN = re.compile(r"[\[［]\s*(a(c(t(\s*[:：][^\]］]*)?)?)?)?$")


def has_incomplete_tag(text: str) -> bool:
    """末尾が「閉じていない行動タグ」で終わっているか。

    ストリーミングではタグが 2 つのチャンクにまたがることがある。
    その状態で文末の句点を見つけて処理してしまうと、
    `…です。[act:` が本文として画面と TTS に流れ、タグも検出できない。
    そうならないよう、タグが閉じるまで文の確定を待つために使う。
    """
    return bool(OPEN_TAG_PATTERN.search(text))


def strip_action_tags(text: str) -> str:
    """本文からタグだけを取り除く(タグ情報が要らない場面向け)。"""
    return TAG_PATTERN.sub("", text)


def tag_reference_for_prompt() -> str:
    """システムプロンプトへ埋め込む用のタグ一覧を組み立てる。"""
    lines = []
    for name, catalog_id in ACTION_TAGS.items():
        prompt = _PROMPT_BY_ID.get(catalog_id, "")
        lines.append(f"  [act:{name}] … {prompt}")
    return "\n".join(lines)


def action_tag_instructions(indent: str = "                ") -> str:
    """全エージェント共通で使う、行動タグの指示ブロック。

    エージェントごとに書き分けると必ず食い違うので、ここ 1 箇所で持つ。
    handoff 先のドメインエージェントにも同じものを入れないと、
    そのエージェントが応答したターンだけ動作が出なくなる。
    """
    body = f"""# =========================================================
# 行動タグ（最優先。generate_motion ツールより先にこちらを使う）
# =========================================================
返答本文に [act:タグ名] を直接書き込むと、その位置で身体動作が再生されます。
タグは画面にも音声にも出ません。動作は「タグを書いた文が読み上げられる瞬間」に始まります。

★ 必ず守ること ★
- ユーザーが「左手を振って」「お辞儀して」など身体動作を直接指定した場合は、
  指示した動作を最優先し、身体部位・左右・速度・順序を保った日本語で generate_motion を一度だけ呼ぶ。
  システムが先に同じ要求を受理していた場合はツールが受理済みと返すので、呼び直さない。
  そのターンに行動タグや別の動作を追加せず、返答の意図から別の動作に置き換えない。
- 動作を付けたい文に、その動作が起きてほしい位置へタグを書く。
  例: 「こんにちは[act:wave]。本日はお越しいただきありがとうございます[act:bow]。」
- 1回の返答に複数のタグを書いてよい。長い説明では2〜4文に1回を目安に添えること。
  黙って立っているだけの時間が長くならないようにする。
- タグは「その動作が指し示す語の直前」に置く。**文頭にも文末にも固めない**こと。
  ○ 本学の特徴は[act:point]実践的な教育にあります。   ← 指し示す対象の直前
  ○ まずは[act:calm]深呼吸から始めましょう。          ← 「深呼吸」を導く位置
  ○ とても良い[act:nod]着眼点ですね。                ← 同意する語の直前
  × [act:point]本学の特徴は実践的な教育にあります。   ← まだ何も言っていないのに動く
  × 本学の特徴は実践的な教育にあります[act:point]。   ← 言い終わってから動く
- **長い文には複数入れる**。目安は30文字ごとに1つ。
  ○ 基礎教育部では[act:point]自己管理能力を養い、さらに[act:offer]専門分野への
    橋渡しも行っています。
- 使えるタグは下の一覧のみ。一覧に無いタグを作らないこと。

■ 使えるタグ:
{tag_reference_for_prompt()}

■ generate_motion ツールを使う場面（タグで表せないときだけ）:
- 座る / 立ち上がる などの姿勢変化
- 一覧に無い動作を日本語で細かく指定したいとき
姿勢変化以外は原則タグを使い、同じ動作をタグとツールの両方で出さないこと。
"""
    return "\n".join(indent + line for line in body.split("\n"))


# 簡潔版で提示するタグ(会話でよく使うものに絞る)。
# 長い一覧を文脈の末尾に置くと読み飛ばされるため、RAG のように
# 検索結果が前段に入るプロンプトではこちらを冒頭付近で使う。
BRIEF_TAGS = (
    "nod",
    "wave",
    "bow",
    "thanks",
    "point",
    "offer",
    "thinking",
    "shrug",
    "ok",
    "cheer",
    "please",
    "stop",
)


# 簡潔版で使う短い日本語ラベル(生成プロンプトを途中で切ると意味が壊れるため)
SHORT_LABELS: dict[str, str] = {
    "nod": "うなずく",
    "wave": "手を振る",
    "bow": "お辞儀",
    "thanks": "感謝",
    "point": "指し示す",
    "offer": "どうぞと促す",
    "thinking": "考える",
    "shrug": "わからない",
    "ok": "OKサイン",
    "cheer": "励ます",
    "please": "お願いする",
    "stop": "制止する",
}


def action_tag_instructions_brief() -> str:
    """短い指示。文脈が長いプロンプトの冒頭に置く用。"""
    listing = "、".join(
        f"[act:{name}]={SHORT_LABELS.get(name, name)}" for name in BRIEF_TAGS
    )
    return (
        "【身体動作(必須)】\n"
        "回答本文に [act:タグ名] を埋め込むこと。タグは画面にも音声にも出ず、"
        "その位置で学長が動く。2〜4文に1回は必ず入れ、無言で立ち尽くさないようにする。\n"
        "タグは動作が指し示す語の直前に置く。文頭にも文末にも固めない。\n"
        f"使えるタグ: {listing}\n"
        "○ 本学の特徴は[act:point]実践的な教育にあります。\n"
        "× [act:point]本学の特徴は…(まだ何も言っていない) / …にあります[act:point]。(言い終わってから)"
    )


# ---------------------------------------------------------------- 自動補完

# 日本語 TTS の実測がおよそ 6 文字/秒。長さから発話秒数を見積もるのに使う。
CHARS_PER_SECOND = 6.0

# この秒数につき 1 つの動作を目安に置く。
# 人が身振りを交えて話すときの間隔はおよそ 2〜4 秒。
# 5.0 だと 4 秒前後の文が 1 つも拾えず無動作のまま残ったため 3.5 にしている。
SECONDS_PER_GESTURE = 3.5

# 自動補完で使うタグ。内容を選ばない中立的なものだけに絞り、
# 同じ動作の反復に見えないよう順番に回す。
AUTO_FILL_ROTATION = ("nod", "point", "offer", "thinking")

# 既存タグのすぐ近くには入れない(重なって不自然になるため)。
MIN_TAG_DISTANCE = 0.18


def plan_auto_gestures(
    text: str,
    existing: list[ActionTag],
    rotation_offset: int = 0,
) -> list[ActionTag]:
    """長い発話に対して、タグが足りない分の動作を機械的に補う。

    なぜ必要か:
        タグ方式は「LLM がタグを書いてくれること」に依存している。
        実測では 11 秒や 13 秒の文が丸ごと無動作になり、
        発話時間の 85% で体が止まっていた。
        モデルの遵守に頼らず、長さから決まるぶんだけはここで確保する。

    やること:
        想定発話秒数から必要な動作数を出し、不足ぶんを等間隔に置く。
        既存タグの近くは避け、中立的な動作を順番に回して反復感を減らす。
    """
    length = len(text.strip())
    if length == 0:
        return []

    seconds = length / CHARS_PER_SECOND
    wanted = int(seconds // SECONDS_PER_GESTURE)
    missing = wanted - len(existing)
    if missing <= 0:
        return []

    taken = [tag.position for tag in existing]
    added: list[ActionTag] = []

    # 文頭・文末を避けた等間隔の候補位置
    for i in range(missing):
        position = (i + 1) / (missing + 1)
        position = 0.15 + position * 0.7  # 0.15〜0.85 の範囲に収める
        if any(abs(position - t) < MIN_TAG_DISTANCE for t in taken):
            continue

        name = AUTO_FILL_ROTATION[
            (rotation_offset + len(added)) % len(AUTO_FILL_ROTATION)
        ]
        resolved = resolve_tag(name)
        if resolved is None:
            continue
        prompt, catalog_id = resolved
        added.append(
            ActionTag(
                name=f"auto:{name}",
                prompt=prompt,
                catalog_id=catalog_id,
                position=position,
            )
        )
        taken.append(position)

    return added
