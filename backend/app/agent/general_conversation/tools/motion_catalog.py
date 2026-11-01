"""Motion prompt normalization for Custom-T2M-Lab.

Custom-T2M-Lab is now trained with a communication-motion catalog.  The T2M
model can generalize, but generation is more stable when we send the catalog's
original Japanese label instead of a short ad-hoc phrase from the LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class MotionCatalogMatch:
    original_prompt: str
    prompt: str
    catalog_id: Optional[str] = None
    reason: str = "as_is"


@dataclass(frozen=True)
class MotionCatalogRule:
    catalog_id: str
    prompt: str
    any_keywords: tuple[str, ...]
    all_keywords: tuple[str, ...] = ()
    reason: str = "catalog_keyword"


def _normalize(text: str) -> str:
    return re.sub(r"[\s　。、，,.!！?？「」『』（）()・\-ー]+", "", text.lower())


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _contains_all(text: str, keywords: Iterable[str]) -> bool:
    return all(keyword in text for keyword in keywords)


def _is_chair_seated(text: str) -> bool:
    return _contains_any(text, ("椅子", "イス", "いす", "腰掛"))


def _is_floor_seated(text: str) -> bool:
    return _contains_any(text, ("床", "地面", "あぐら", "胡座", "正座"))


def _with_seated_context(prompt: str, normalized: str) -> str:
    """Prefer catalog-like seated prompts when the body state is explicit."""
    if _is_chair_seated(normalized):
        if _contains_any(normalized, ("手を振", "挨拶", "こんにちは", "バイバイ")):
            return "椅子に座っている状態から、手を振る。"
        if _contains_any(normalized, ("うなず", "頷", "同意", "了解")):
            return "椅子に座っている状態から、うなずく。"
        if _contains_any(normalized, ("首を横", "首を振", "否定", "違う", "拒否")):
            return "椅子に座っている状態から、首を横に振る。"
        if _contains_any(normalized, ("拍手",)):
            return "椅子に座っている状態から、両手でリズミカルに拍手する。"
        if _contains_any(normalized, ("電話",)):
            return "椅子に座っている状態から、電話をかける。"
        if _contains_any(normalized, ("胸に手", "胸を", "感謝", "ありがとう")):
            return "椅子に座っている状態から、胸に手をあてる。"
        if _contains_any(normalized, ("右手を上", "右手をあげ", "右手を挙", "右腕を上", "挙手")):
            return "椅子に座っている状態から、右手を上げる。"

    if _is_floor_seated(normalized):
        if _contains_any(normalized, ("手を振", "挨拶", "こんにちは", "バイバイ")):
            return "床に座った状態で、右手を振る。"
        if _contains_any(normalized, ("うなず", "頷", "同意", "了解")):
            return "床に座った状態で、うなずく。"
        if _contains_any(normalized, ("首を横", "首を振", "否定", "違う", "拒否")):
            return "床に座った状態で、首を横に振る。"
        if _contains_any(normalized, ("拍手",)):
            return "床に座った状態で、両手でリズミカルに拍手する。"
        if _contains_any(normalized, ("右手を上", "右手をあげ", "右手を挙", "右腕を上", "挙手")):
            return "床に座った状態で、右手を上げる。"

    return prompt


_RULES: tuple[MotionCatalogRule, ...] = (
    MotionCatalogRule(
        "MOT_740",
        "右腕を前方に大きく振りながら手のひらを前方に向けて「さようなら」と手を振る",
        ("さようなら", "バイバイ", "またね", "別れ"),
    ),
    MotionCatalogRule(
        "MOT_701",
        "右手を肩の高さに上げ、手のひらを前方に向けて軽く左右に振って挨拶する",
        ("こんにちは", "挨拶", "手を振", "右手を振", "左手を振", "両手を振"),
    ),
    MotionCatalogRule(
        "MOT_783",
        # Match the verified training label exactly.  Standing prefixes and
        # punctuation materially change this model's generated motion.
        "右手を上げる",
        ("右手を上", "右手をあげ", "右手を挙", "右腕を上", "右腕を高く", "挙手"),
    ),
    MotionCatalogRule(
        "MOT_711",
        "立っている状態から、お辞儀する。",
        ("お辞儀", "礼をする", "会釈", "頭を下げ"),
    ),
    MotionCatalogRule(
        "MOT_702",
        "頭部をゆっくり上下に動かして同意のうなずきをする",
        ("うなず", "頷", "同意", "了解", "わかった"),
    ),
    MotionCatalogRule(
        "MOT_703",
        "頭部をゆっくり左右に振って否定・拒否の意思を示す",
        ("首を横", "首を振", "否定", "拒否", "違う"),
    ),
    MotionCatalogRule(
        "MOT_714",
        "右手の親指を上に立てて「いいね・賛成」のサムズアップを示す",
        ("サムズアップ", "いいね", "賛成", "グッド"),
    ),
    MotionCatalogRule(
        "MOT_774",
        "右手の人差し指を立てて「ちょっと待って」と示すジェスチャーをする",
        ("ちょっと待", "少し待", "少々待", "一旦待"),
    ),
    MotionCatalogRule(
        "MOT_715",
        "右手の親指を下に向けて「ダメ・反対」のサムズダウンを示す",
        ("サムズダウン", "反対", "ダメ"),
    ),
    MotionCatalogRule(
        "MOT_707",
        "右手のひらを前方に向けて腕を前方に伸ばし「ストップ」のサインを示す",
        ("ストップ", "止ま", "待って", "制止"),
    ),
    MotionCatalogRule(
        "MOT_765",
        "右手のひらを下向きに左右に小刻みに振って「落ち着いて・ゆっくり」を示す",
        ("落ち着", "慌てない", "冷静"),
    ),
    MotionCatalogRule(
        "MOT_706",
        "右手の親指と人差し指で輪を作り「OK」のサインを前方に向けて示す",
        ("ok", "オッケー", "大丈夫", "了解"),
    ),
    MotionCatalogRule(
        "MOT_734",
        "両腕を胸の前でX字に交差させて「ノー・バツ・却下」を示す",
        ("バツ", "却下", "禁止", "だめ", "ダメ"),
    ),
    MotionCatalogRule(
        "MOT_720",
        "右手のひらを上に向けて前方に差し出して「どうぞ・お先に」と示す",
        ("どうぞ", "お先に", "案内", "すすめ"),
    ),
    MotionCatalogRule(
        "MOT_719",
        "右手のひらを下に向けて前後に動かして「こちらに来て」と招く",
        ("こちらに来", "こっちに来", "招く", "来て"),
    ),
    MotionCatalogRule(
        "MOT_705",
        "右腕を前方に伸ばし、人差し指で正面を指差す",
        ("指差", "前を指", "そちら"),
    ),
    MotionCatalogRule(
        "MOT_721",
        "右腕を肩の高さに横に伸ばし、右方向を人差し指で指差して方向を示す",
        ("右方向", "右を指", "方向を示"),
    ),
    MotionCatalogRule(
        "MOT_738",
        "右手のひらを自分の胸に当てて「ありがとう・感謝しています」を示す",
        ("ありがとう", "感謝", "ありがたい"),
    ),
    MotionCatalogRule(
        "MOT_739",
        "両手を合わせて「お願いします」という懇願のジェスチャーをする",
        ("お願い", "頼む", "頼み"),
    ),
    MotionCatalogRule(
        "MOT_708",
        "両肩を同時に上方に持ち上げ、少し間をおいて元に戻すことで「わからない」を表す",
        ("わからない", "知らない", "困った", "疑問"),
    ),
    MotionCatalogRule(
        "MOT_733",
        "右手の人差し指でこめかみをトントンと叩いて「考えて・頭を使って」を示す",
        ("考えて", "頭を使", "こめかみ"),
    ),
    MotionCatalogRule(
        "MOT_748",
        "右手の人差し指を唇に当てて「考え中・迷っている」を示す",
        ("考え中", "迷って", "悩んで"),
    ),
    MotionCatalogRule(
        "MOT_709",
        "両手のひらを合わせてリズミカルに拍手する",
        ("拍手", "祝福", "おめでとう"),
    ),
    MotionCatalogRule(
        "MOT_735",
        "右腕を肘から折り曲げて上に伸ばし、握りこぶしを作って「頑張れ・ガッツポーズ」を示す",
        ("ガッツ", "頑張", "応援", "よし"),
    ),
    MotionCatalogRule(
        "MOT_729",
        "右手の親指と小指を立てて耳と口の近くに持ってきて電話で話すジェスチャーをする",
        ("電話", "通話"),
    ),
    MotionCatalogRule(
        "MOT_768",
        "両手の親指と人差し指でハート型を作って前方に向けて提示する",
        ("ハート", "愛", "好き"),
    ),
    MotionCatalogRule(
        "MOT_766",
        "右手を丸めて口の前に持ってきて叫ぶポーズを取り「大声で呼ぶ・叫ぶ」を示す",
        ("叫ぶ", "大声", "呼ぶ"),
    ),
    MotionCatalogRule(
        "MOT_716",
        "右手の人差し指と中指をV字に開いてピースサインを作る",
        ("ピース", "vサイン"),
    ),
    MotionCatalogRule(
        "MOT_710",
        "右手を右こめかみ付近に当てて敬礼のポーズを取る",
        ("敬礼",),
    ),
)


# Explicit physical details must survive catalog normalization.  A keyword
# match must not turn a left-hand wave into a right-hand wave, or a slow bow
# into a calming hand gesture.  Exact catalog labels are still accepted.
_PHYSICAL_DETAILS = (
    "右", "左", "両", "片", "だけ", "のみ", "てから", "その後", "ずに", "せず",
    "右手", "左手", "両手", "右腕", "左腕", "両腕", "右足", "左足", "両足",
    "頭上", "頭の上", "顔の前", "胸の前", "胸に", "肩の高さ", "横に",
    "前方", "後方", "水平", "垂直", "ゆっくり", "速く", "早く", "小さく", "大きく",
)

# This detects a request to perform a motion, rather than a discussion of it.
# Keep it conservative: quoted examples, explanations, and speech style
# requests continue through the normal conversational gesture path.
_ACTION_WORDS = re.compile(
    r"(?:手|腕|足|脚|首|頭|腰|肩|胸|身体|体|全身|指差|指さ|指し示|お辞儀|礼|会釈|拍手|敬礼|"
    r"うなず|頷|挙手|ガッツポーズ|ピース|サムズ|歩|走|ランニング|ジャンプ|跳|"
    r"スクワット|屈伸|座|すわ|立ち上が|起立|しゃが|踊|ダンス|回転)"
)
_PERFORMABLE_ACTION = re.compile(
    r"(?:振|上げ|あげ|挙げ|伸ば|下げ|下ろ|動か|曲げ|広げ|開|閉|組|張|指差|指さ|指し示|"
    r"歩|走|ランニング|ジャンプ|跳|スクワット|屈伸|座|すわ|立ち上が|"
    r"立って|立つ|起立|しゃが|踊|ダンス|回|ひね|捻|傾|かしげ|"
    r"お辞儀|礼|会釈|拍手|敬礼|うなず|頷|挙手|ガッツポーズ|ピース|サムズ)"
)
_REQUEST_VERB_ENDINGS = (
    ("指差して", "指差す"), ("指さして", "指さす"), ("指し示して", "指し示す"),
    ("立ち上がって", "立ち上がる"), ("うなずいて", "うなずく"), ("頷いて", "頷く"),
    ("伸ばして", "伸ばす"), ("下ろして", "下ろす"), ("動かして", "動かす"),
    ("曲げて", "曲げる"), ("広げて", "広げる"), ("上げて", "上げる"),
    ("あげて", "あげる"), ("挙げて", "挙げる"), ("下げて", "下げる"), ("振って", "振る"),
    ("歩いて", "歩く"), ("走って", "走る"), ("跳んで", "跳ぶ"),
    ("座って", "座る"), ("立って", "立つ"), ("踊って", "踊る"),
    ("回って", "回る"), ("して", "する"),
)

_ACTION_REQUEST_END = re.compile(
    r"(?:ください|下さい|くれる|くれますか|もらえますか|もらえませんか|"
    r"お願い(?:します)?|やって|してみて|して|振って|上げて|"
    r"あげて|挙げて|伸ばして|下げて|下ろして|"
    r"動かして|曲げて|広げて|指差して|指さして|指し示して|歩いて|走って|"
    r"跳んで|座って|立って|立ち上がって|"
    r"うなずいて|頷いて|踊って|回って)(?:ね|よ)?[？?]?$"
)
_NON_ACTION_REQUEST = re.compile(
    r"(?:とは|って何|意味|説明|解説|教えて|調べて|検索|ニュース|画像|動画|"
    r"例文|引用|について|どうやって|なぜ|どうして|できますか|できる[？?])"
)


def extract_explicit_motion_request(text: str) -> Optional[MotionCatalogMatch]:
    """Extract a direct physical-action command without guessing from reply intent."""
    for clause in re.split(r"[。！!\n]", text):
        command = clause.strip().strip("？?")
        if not command or re.search(r'[「」『』"“”]', command):
            continue
        if (
            _NON_ACTION_REQUEST.search(command)
            or not _ACTION_WORDS.search(command)
            or not _PERFORMABLE_ACTION.search(command)
        ):
            continue
        # Avoid speech/body idioms such as 「頭を使って答えて」「足りない」.
        if re.search(r"(?:話して|答えて|返答|回答|声|口調|考えて|頭を使|足り)", command):
            continue
        if not _ACTION_REQUEST_END.search(command):
            continue
        command = re.sub(r"^(?:(?:学長|あなた|アバター|キャラクター|キャラ)[はがに、, ]*)", "", command)
        command = re.sub(r"^(?:では|それでは|じゃあ|まず|次は|今度は)[、, ]*", "", command)
        # Remove only request politeness.  Body, direction, speed and sequences
        # are intentionally retained and sent to the T2M model unchanged.
        command = re.sub(r"(?:ください|下さい|もらえませんか|もらえますか|くれますか|くれる|お願い(?:します)?)(?:ね|よ)?$", "", command).rstrip("、, ")
        for request_ending, caption_ending in _REQUEST_VERB_ENDINGS:
            if command.endswith(request_ending):
                command = command[:-len(request_ending)] + caption_ending
                break
        if not command:
            continue
        match = normalize_motion_prompt(command)
        return MotionCatalogMatch(
            original_prompt=clause.strip(), prompt=match.prompt,
            catalog_id=match.catalog_id, reason="explicit_request",
        )
    return None


def normalize_motion_prompt(motion_description: str) -> MotionCatalogMatch:
    original = motion_description.strip()
    normalized = _normalize(original)
    for rule in _RULES:
        if normalized == _normalize(rule.prompt):
            return MotionCatalogMatch(
                original_prompt=original, prompt=rule.prompt,
                catalog_id=rule.catalog_id, reason="catalog_exact",
            )
    # A few unconstrained, compatible commands benefit from the exact
    # training caption.  Match the entire short action: left/both hands,
    # exclusive limbs, speed, height, location and sequences must not match.
    short_action = normalized
    for prefix in (
        "立っている状態から", "立っている状態で", "立った状態から",
        "立った状態で", "立ったまま",
    ):
        if short_action.startswith(prefix):
            short_action = short_action[len(prefix):]
            break
    if short_action == "歩く":
        return MotionCatalogMatch(
            original_prompt=original, prompt="立っている状態から、歩く。",
            reason="verified_basic_caption",
        )
    simple_catalog_id = None
    if short_action in (
        "手を振る", "右手を振る", "手を振って挨拶する", "右手を振って挨拶する",
    ):
        simple_catalog_id = "MOT_701"
    elif short_action in (
        "手を上げる", "右手を上げる", "手をあげる", "右手をあげる",
        "手を挙げる", "右手を挙げる", "挙手", "挙手する",
    ):
        simple_catalog_id = "MOT_783"
    if simple_catalog_id:
        rule = next(rule for rule in _RULES if rule.catalog_id == simple_catalog_id)
        return MotionCatalogMatch(
            original_prompt=original, prompt=rule.prompt,
            catalog_id=rule.catalog_id, reason="compatible_catalog_caption",
        )

    # Keep detailed requests intact instead of substituting a broader gesture.
    # This also protects seated left/both-hand motions before seated rewriting.
    if _contains_any(normalized, _PHYSICAL_DETAILS) or re.search(r"[0-9０-９]", normalized):
        return MotionCatalogMatch(original_prompt=original, prompt=original)

    without_standing_prefix = normalized.replace("立っている状態から", "")

    seated_context = _contains_any(
        normalized,
        ("座った状態", "座っている状態", "座位", "腰掛けた状態"),
    )

    # Posture transitions should stay explicit because the frontend also uses
    # these words to decide whether to keep or release the final pose.
    if _contains_any(without_standing_prefix, ("立ち上が", "起き上が", "起立", "立って", "立つ")):
        return MotionCatalogMatch(
            original_prompt=original,
            prompt="座った状態から、立ち上がる",
            reason="posture_stand",
        )

    seated_prompt = _with_seated_context(original, normalized)
    if seated_prompt != original:
        return MotionCatalogMatch(
            original_prompt=original,
            prompt=seated_prompt,
            reason="seated_catalog_keyword",
        )

    if (
        not seated_context
        and _is_chair_seated(normalized)
        and _contains_any(normalized, ("座", "すわ", "腰掛"))
    ):
        return MotionCatalogMatch(
            original_prompt=original,
            prompt="立っている状態から、椅子に座る",
            reason="posture_chair_sit",
        )

    if (
        not seated_context
        and _is_floor_seated(normalized)
        and _contains_any(normalized, ("座", "すわ"))
    ):
        return MotionCatalogMatch(
            original_prompt=original,
            prompt="立っている状態から、床に座る",
            reason="posture_floor_sit",
        )

    for rule in _RULES:
        if rule.all_keywords and not _contains_all(normalized, rule.all_keywords):
            continue
        if _contains_any(normalized, rule.any_keywords):
            return MotionCatalogMatch(
                original_prompt=original,
                prompt=rule.prompt,
                catalog_id=rule.catalog_id,
                reason=rule.reason,
            )

    return MotionCatalogMatch(original_prompt=original, prompt=original)
