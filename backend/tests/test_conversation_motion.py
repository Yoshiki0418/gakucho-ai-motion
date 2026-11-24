"""会話画面の層構成方式(conversation_motion)の回帰テスト。

アプリのモデル(LLM / TTS)は読み込まず、パッケージ単体で確かめる。numpy と scipy が要る。
Run: python3 -m unittest discover -s backend/tests -p test_conversation_motion.py
"""

import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path

import numpy as np

APP = Path(
    os.environ.get(
        "CONVERSATION_MOTION_APP", Path(__file__).resolve().parents[1] / "app"
    )
)
TOOLS = APP / "agent/general_conversation/tools"
PACKAGE_NAME = "conversation_motion_test_tools"
PACKAGE = types.ModuleType(PACKAGE_NAME)
PACKAGE.__path__ = [str(TOOLS)]
sys.modules[PACKAGE_NAME] = PACKAGE


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE_NAME}.{name}",
        (
            TOOLS / name / "__init__.py"
            if (TOOLS / name).is_dir()
            else TOOLS / f"{name}.py"
        ),
        submodule_search_locations=(
            [str(TOOLS / name)] if (TOOLS / name).is_dir() else None
        ),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load("conversation_motion")
from conversation_motion_test_tools.conversation_motion import bvh as B  # noqa: E402
from conversation_motion_test_tools.conversation_motion import (  # noqa: E402
    composer as C,
)
from conversation_motion_test_tools.conversation_motion import (  # noqa: E402
    library as L,
)
from conversation_motion_test_tools.conversation_motion import (  # noqa: E402
    pipeline as P,
)
from conversation_motion_test_tools.conversation_motion import (  # noqa: E402
    session as S,
)

TAGS = _load("motion_tags")
LIBRARY = L.ClipLibrary(APP / "assets/conversation_motion")
# SentiAvatar の出力(研究評価用・非商用ライセンスのため リポジトリには置かない)
SENTI_SAMPLE = Path(
    os.environ.get(
        "SENTIAVATAR_SAMPLE_BVH",
        Path(__file__).resolve().parents[2]
        / ".runtime/staging/senti-out/greet-s1.raw.bvh",
    )
)


def angle_from_neutral(motion, name):
    j = motion.skeleton.index(name)
    return np.degrees(
        B.qangle(B.qmul(motion.quats[:, j], B.qinv(LIBRARY.neutral[j])[None]))
    )


class FakeChannel:
    def __init__(self):
        self.items = []
        self.jobs = 0

    def put(self, item):
        self.items.append(item)

    def start_job(self):
        self.jobs += 1

    def finish_job(self):
        self.jobs -= 1


class BvhTest(unittest.TestCase):
    def test_round_trip_keeps_joint_positions(self):
        motion = C.compose(
            LIBRARY, 2.0, [C.Overlay(LIBRARY.get("MOT_701"), 0.0, 2.0)]
        ).motion
        again = B.parse_bvh(B.write_bvh(motion))
        p0, _ = B.forward_kinematics(motion)
        p1, _ = B.forward_kinematics(again)
        self.assertLess(np.abs(p0 - p1).max(), 1e-3)
        self.assertAlmostEqual(again.fps, LIBRARY.fps, places=3)


class LibraryTest(unittest.TestCase):
    def test_every_action_tag_has_a_prepared_clip(self):
        missing = [
            tag
            for tag, catalog_id in TAGS.ACTION_TAGS.items()
            if LIBRARY.get(catalog_id) is None
        ]
        self.assertEqual(missing, [])

    def test_mirrored_wave_moves_the_left_arm(self):
        right = C.compose(
            LIBRARY, 3.0, [C.Overlay(LIBRARY.get("MOT_701"), 0.0, 3.0)]
        ).motion
        left = C.compose(
            LIBRARY, 3.0, [C.Overlay(LIBRARY.get("MOT_701", mirrored=True), 0.0, 3.0)]
        ).motion
        self.assertGreater(angle_from_neutral(right, "RightArm").max(), 45)
        self.assertGreater(angle_from_neutral(left, "LeftArm").max(), 45)
        self.assertLess(angle_from_neutral(left, "RightArm").max(), 20)


class ComposerTest(unittest.TestCase):
    def test_overlay_extends_the_segment_and_fades_back(self):
        clip = LIBRARY.get("MOT_701")
        composition = C.compose(LIBRARY, 1.0, [C.Overlay(clip, 0.2, 2.5)])
        self.assertGreaterEqual(composition.motion.duration, 2.7 - 1e-6)
        arm = angle_from_neutral(composition.motion, "RightArm")
        self.assertLess(arm[0], 5)  # 入る前は立ち姿
        self.assertGreater(arm.max(), 45)

    def test_start_pose_is_continuous(self):
        first = C.compose(
            LIBRARY, 2.0, [C.Overlay(LIBRARY.get("MOT_714"), 0.0, 3.0)]
        ).motion
        pose = C.pose_at(first, 1.0)
        second = C.compose(LIBRARY, 2.0, [], start_pose=pose).motion
        self.assertLess(np.abs(second.quats[0] - pose.quats).max(), 1e-6)

    def test_head_gesture_keeps_arm_of_base(self):
        base = LIBRARY.neutral_motion(91)
        j = LIBRARY.skeleton.index("LeftForeArm")
        base.quats[:, j] = B.qmul(
            base.quats[:, j], np.array([0.0, 0.0, np.sin(0.4), np.cos(0.4)])
        )
        composition = C.compose(
            LIBRARY, 3.0, [C.Overlay(LIBRARY.get("MOT_702"), 0.0, 2.0)], base=base
        )
        self.assertLess(
            np.abs(composition.motion.quats[30:60, j] - base.quats[30:60, j]).max(),
            1e-6,
        )
        self.assertGreater(angle_from_neutral(composition.motion, "Head").max(), 4)


class SessionTest(unittest.TestCase):
    def test_repeated_gestures_are_varied_or_softened(self):
        session = S.MotionSession()
        session.plan(
            LIBRARY,
            1,
            3.0,
            [
                S.PlannedTag("point", "MOT_705", 0.3),
                S.PlannedTag("nod", "MOT_702", 0.8),
            ],
        )
        plan = session.plan(
            LIBRARY,
            2,
            3.0,
            [
                S.PlannedTag("point", "MOT_705", 0.3),
                S.PlannedTag("nod", "MOT_702", 0.8),
            ],
        )
        fresh = [o for o in plan.overlays if o.start >= 0]
        self.assertIn("MOT_720", [o.clip.id for o in fresh])  # 指差しの代わりにどうぞ
        nod = [o for o in fresh if o.clip.id == "MOT_702"]
        self.assertTrue(nod and nod[0].amplitude < 1.0)

    def test_close_gestures_on_the_same_arm_are_dropped(self):
        session = S.MotionSession()
        plan = session.plan(
            LIBRARY,
            1,
            4.0,
            [
                S.PlannedTag("wave", "MOT_701", 0.2),
                S.PlannedTag("thanks", "MOT_738", 0.25),
            ],
        )
        self.assertEqual([o.clip.id for o in plan.overlays], ["MOT_701"])
        self.assertEqual(plan.skipped[0]["reason"], "busy")

    def test_long_gesture_is_carried_into_the_next_sentence(self):
        session = S.MotionSession()
        session.plan(LIBRARY, 1, 1.0, [S.PlannedTag("wave", "MOT_701", 0.5)])
        plan = session.plan(LIBRARY, 2, 3.0, [])
        self.assertTrue(plan.overlays and plan.overlays[0].start < 0)


class PipelineTest(unittest.TestCase):
    def test_semantic_turn_sends_one_bvh_per_sentence(self):
        channel = FakeChannel()
        turn = P.LayeredTurn(channel, "test:semantic", rhythm=False)
        ack = turn.acknowledge()
        self.assertTrue(ack["layered"]["playNow"])
        payload = turn.on_sentence(
            1, 2500, [S.PlannedTag("wave", "MOT_701", 0.1)], None
        )
        motion = B.parse_bvh(payload["bvh"])
        self.assertGreaterEqual(motion.duration, 2.5 - 0.05)
        self.assertEqual(payload["layered"]["group"], "seg-1")
        self.assertEqual(payload["layered"]["revision"], 0)
        self.assertNotIn("立", payload["prompt"])  # フロントの起立判定に掛からない
        self.assertEqual(channel.jobs, 0)

    def test_free_text_requests_resolve_to_clips(self):
        self.assertEqual(P.resolve_layered_request("手を振る"), ("MOT_701", False))
        self.assertEqual(P.resolve_layered_request("左手を振る"), ("MOT_701", True))
        self.assertEqual(P.resolve_layered_request("お辞儀する"), ("MOT_711", False))
        self.assertIsNone(P.resolve_layered_request("立っている状態から、椅子に座る"))

    @unittest.skipUnless(SENTI_SAMPLE.exists(), "SentiAvatar の出力サンプルが無い")
    def test_rhythm_revision_replaces_the_semantic_version(self):
        import json
        import threading
        import time
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        sample = SENTI_SAMPLE.read_text()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                body = json.dumps(
                    {"bvh_raw": sample, "timing": {"total": 0.1}}
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        original = P.SENTIAVATAR_URL
        P.SENTIAVATAR_URL = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            channel = FakeChannel()
            turn = P.LayeredTurn(channel, "test:rhythm", rhythm=True)
            first = turn.on_sentence(
                1, 3460, [S.PlannedTag("wave", "MOT_701", 0.3)], b"RIFF"
            )
            second = turn.on_sentence(2, 1700, [], b"RIFF")
            self.assertEqual(first["layered"]["rhythm"], "pending")
            deadline = time.time() + 30
            while (channel.jobs or len(channel.items) < 2) and time.time() < deadline:
                time.sleep(0.05)
            self.assertEqual(channel.jobs, 0)
            revisions = sorted(
                (
                    item["layered"]["group"],
                    item["layered"]["revision"],
                    item["layered"]["base"],
                )
                for item in channel.items
            )
            self.assertEqual(
                revisions, [("seg-1", 1, "rhythm"), ("seg-2", 1, "rhythm")]
            )
            self.assertEqual(second["layered"]["revision"], 0)
        finally:
            P.SENTIAVATAR_URL = original
            server.shutdown()

    def test_mode_parsing_defaults_to_current(self):
        self.assertEqual(P.parse_mode(None), "current")
        self.assertEqual(P.parse_mode("layered"), "layered")
        self.assertEqual(P.parse_mode("unknown"), "current")


if __name__ == "__main__":
    unittest.main()
