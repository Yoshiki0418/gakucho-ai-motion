"""会話セッションごとの動作の状態と、意味の層の配置。

LLM は応答本文に行動タグを置くだけなので、そのまま再生すると
  - 同じ動作が続く(うなずきの連発)
  - 近すぎる動作が重なって中途半端になる
  - 文の区切りで姿勢が跳ぶ・動作が途中で切れる
が起きる。ここで「最近使った動作」「進行中の動作」「直前の姿勢」を持ち、配置を決める。
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from .composer import FADE_IN, FADE_OUT, Overlay, Pose
from .library import ClipLibrary

# ストロークを語の頭から少し遅らせる(身振りの山場は語の強勢にやや先行〜同時)
STROKE_DELAY = 0.12
# 同じ動作を繰り返さない間隔(発話時間の秒)
REPEAT_WINDOW = 8.0
# 同じ部位の動作どうしのストローク間隔の下限
MIN_STROKE_GAP = 1.1

# ストロークの後、どれだけ動作を続けるか(秒)。収録は同じ動作を何度も繰り返しているため切る。
HOLD = {
    "MOT_702": 0.9,  # うなずき
    "MOT_703": 1.1,  # 首を横に振る
    "MOT_701": 1.8,  # 手を振る
    "MOT_740": 1.8,  # さようなら
    "MOT_711": 1.4,  # お辞儀
    "MOT_709": 1.6,  # 拍手
}
DEFAULT_HOLD = 1.4
EXPLICIT_HOLD = 3.0

# 直前に使った動作の代わりに使える動作
ALTERNATIVES = {
    "MOT_705": ("MOT_720",),
    "MOT_720": ("MOT_705",),
    "MOT_748": ("MOT_733",),
    "MOT_733": ("MOT_748",),
    "MOT_738": ("MOT_711",),
}
NOD = "MOT_702"


@dataclass
class PlannedTag:
    name: str
    catalog_id: str | None
    position: float
    explicit: bool = False
    mirrored: bool = False
    auto: bool = False


@dataclass
class SegmentPlan:
    segment_id: int
    duration: float
    clock: float
    overlays: list[Overlay]
    start_pose: Pose | None
    skipped: list[dict] = field(default_factory=list)


def _parts(overlay: Overlay) -> set[str]:
    return {group for group, weight in overlay.clip.weights.items() if weight >= 0.5}


class MotionSession:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.clock = 0.0
        self.recent: deque[tuple[str, float]] = deque(maxlen=32)
        self.carry: list[Overlay] = []
        self.end_pose: Pose | None = None
        self.plans: dict[int, SegmentPlan] = {}
        self.touched = time.monotonic()

    def begin_turn(self) -> None:
        """ターンの間は待機に戻るので、姿勢と進行中の動作の引き継ぎは切る。反復の記録は残す。"""
        with self.lock:
            self.carry = []
            self.end_pose = None
            self.plans = {}
            self.touched = time.monotonic()

    def _recently_used(self, catalog_id: str, at: float) -> bool:
        return any(
            used == catalog_id and at - when < REPEAT_WINDOW
            for used, when in self.recent
        )

    def plan(
        self,
        library: ClipLibrary,
        segment_id: int,
        duration: float,
        tags: list[PlannedTag],
    ) -> SegmentPlan:
        with self.lock:
            overlays = list(self.carry)
            skipped = []
            for tag in sorted(tags, key=lambda t: (not t.explicit, t.position)):
                catalog_id = tag.catalog_id
                clip = library.get(catalog_id, mirrored=tag.mirrored)
                if clip is None:
                    skipped.append({"tag": tag.name, "reason": "no-clip"})
                    continue
                anchor = tag.position * duration
                stroke_at = anchor + STROKE_DELAY
                amplitude = 1.0
                if not tag.explicit and self._recently_used(
                    catalog_id, self.clock + stroke_at
                ):
                    alternative = next(
                        (
                            a
                            for a in ALTERNATIVES.get(catalog_id, ())
                            if library.get(a)
                            and not self._recently_used(a, self.clock + stroke_at)
                        ),
                        None,
                    )
                    if alternative:
                        catalog_id, clip = alternative, library.get(
                            alternative, mirrored=tag.mirrored
                        )
                    elif catalog_id == NOD:
                        amplitude = 0.6  # うなずきは続いても自然。浅くする
                    else:
                        skipped.append({"tag": tag.name, "reason": "repeated"})
                        continue
                hold = (
                    EXPLICIT_HOLD
                    if tag.explicit
                    else HOLD.get(catalog_id, DEFAULT_HOLD)
                )
                start = max(stroke_at - clip.stroke, 0.0)
                length = min(clip.duration, clip.stroke + hold + FADE_OUT)
                candidate = Overlay(
                    clip,
                    start,
                    length,
                    tag=tag.name,
                    amplitude=amplitude,
                    explicit=tag.explicit,
                )
                conflict = None
                for other in overlays:
                    if (
                        _parts(other) & _parts(candidate)
                        and other.end > candidate.start
                    ):
                        conflict = other
                        break
                if conflict is not None:
                    if (
                        conflict.explicit
                        or candidate.stroke_time - conflict.stroke_time < MIN_STROKE_GAP
                    ):
                        skipped.append(
                            {"tag": tag.name, "reason": "busy", "with": conflict.tag}
                        )
                        continue
                    # 前の動作は新しい動作の入りに合わせて抜く
                    conflict.length = max(
                        candidate.start + FADE_IN - conflict.start, FADE_IN + FADE_OUT
                    )
                overlays.append(candidate)
                self.recent.append((catalog_id, self.clock + candidate.stroke_time))
            plan = SegmentPlan(
                segment_id, duration, self.clock, overlays, self.end_pose, skipped
            )
            self.plans[segment_id] = plan
            # 次の発話へ: はみ出した動作は時刻をずらして持ち越す
            self.carry = [
                Overlay(
                    o.clip, o.start - duration, o.length, o.tag, o.amplitude, o.explicit
                )
                for o in overlays
                if o.end > duration + 0.05
            ]
            self.clock += duration
            self.touched = time.monotonic()
            return plan

    def update_end_pose(self, pose: Pose) -> None:
        with self.lock:
            self.end_pose = pose


_sessions: OrderedDict[str, MotionSession] = OrderedDict()
_sessions_lock = threading.Lock()
MAX_SESSIONS = 64


def get_session(key: str) -> MotionSession:
    with _sessions_lock:
        session = _sessions.pop(key, None) or MotionSession()
        _sessions[key] = session
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)
        return session
