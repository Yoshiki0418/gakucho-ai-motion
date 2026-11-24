"""Regression checks for preserving and prioritizing explicit user motions."""

import importlib.util
import json
import os
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / "app/agent/general_conversation/tools"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


catalog = load_module("_motion_priority_catalog", TOOLS / "motion_catalog.py")

# Isolate the real queue/worker module from the optional LLM SDK and model
# preloading.  Motion generation is replaced only at the external boundary.
package = types.ModuleType("_motion_priority_tools")
package.__path__ = [str(TOOLS)]
backend = types.ModuleType("_motion_priority_tools.motion_backends")
backend.MotionRequest = lambda **kwargs: types.SimpleNamespace(**kwargs)
backend.get_motion_backend = lambda: types.SimpleNamespace(requires_audio=False)
agents = types.ModuleType("agents")
agents.function_tool = lambda function: function
with patch.dict(
    sys.modules,
    {
        "_motion_priority_tools": package,
        "_motion_priority_tools.motion_catalog": catalog,
        "_motion_priority_tools.motion_backends": backend,
        "agents": agents,
    },
), patch.dict(os.environ, {"T2M_PRELOAD_MODEL": "false"}):
    motion_tool = load_module(
        "_motion_priority_tools.motion_tool", TOOLS / "motion_tool.py"
    )


class ExplicitRequestTests(unittest.TestCase):
    def test_bow_command_uses_bow_catalog(self):
        motion = catalog.extract_explicit_motion_request(
            "学長、お辞儀してください。返答は一文で。"
        )
        self.assertIsNotNone(motion)
        self.assertEqual(motion.catalog_id, "MOT_711")
        self.assertEqual(motion.prompt, "立っている状態から、お辞儀する。")

    def test_left_side_and_speed_are_not_replaced_by_generic_gesture(self):
        motion = catalog.extract_explicit_motion_request(
            "左腕を頭の上にゆっくり上げてください。"
        )
        self.assertIsNotNone(motion)
        self.assertIn("左腕", motion.prompt)
        self.assertIn("頭の上", motion.prompt)
        self.assertIn("ゆっくり", motion.prompt)
        self.assertNotIn("右", motion.prompt)

    def test_running_and_clapping_commands_are_detected(self):
        for text in [
            "走ってください",
            "拍手して",
            "右手を振ってもらえますか？",
            "お辞儀をお願いします",
        ]:
            with self.subTest(text=text):
                self.assertIsNotNone(catalog.extract_explicit_motion_request(text))

    def test_pointing_commands_are_explicit_and_have_correct_verb_captions(self):
        for text, expected in [
            ("右手で指差してください。", "右手で指差す"),
            ("右方向を指差して", "右方向を指差す"),
            ("左手で指さしてください。", "左手で指さす"),
            ("指さして", "指さす"),
            ("指し示してください。", "指し示す"),
        ]:
            with self.subTest(text=text):
                request = catalog.extract_explicit_motion_request(text)
                self.assertIsNotNone(request)
                self.assertEqual(request.prompt, expected)
        for text in ["指差してください。", "前を指差してください。"]:
            with self.subTest(text=text):
                request = catalog.extract_explicit_motion_request(text)
                self.assertIsNotNone(request)
                self.assertEqual(request.catalog_id, "MOT_705")
        self.assertIsNone(
            catalog.extract_explicit_motion_request("指差すとは何ですか？")
        )
        self.assertIsNone(
            catalog.extract_explicit_motion_request("指差す方法を説明してください。")
        )

    def test_ordered_request_keeps_the_sequence_as_one_caption(self):
        motion = catalog.extract_explicit_motion_request(
            "拍手をしてから、お辞儀をしてください。"
        )
        self.assertEqual(motion.prompt, "拍手をしてから、お辞儀をする")
        motion = catalog.extract_explicit_motion_request(
            "左腕を上げてから右腕を上げてください。"
        )
        self.assertEqual(motion.prompt, "左腕を上げてから右腕を上げる")

    def test_conversation_explanations_and_quoted_examples_do_not_trigger(self):
        for text in [
            "手を振るとは何ですか？",
            "手を振る方法を説明してください。",
            "短く話してください。",
            "頭を使って答えてください。",
            "走るニュースを調べてください。",
            "「お辞儀して」は例文です。",
            "お辞儀はできますか？",
            "大学について教えてください。",
            "今日は走る。",
            "いつも左手を振る。",
            "左腕をお願いします。",
        ]:
            with self.subTest(text=text):
                self.assertIsNone(catalog.extract_explicit_motion_request(text))

    def test_normalization_preserves_physical_constraints(self):
        for prompt in [
            "左手を振る",
            "両手を振る",
            "右腕を頭上に上げる",
            "椅子に座った状態で、左手を振る",
            "ゆっくりお辞儀をする",
            "右手を顔の前で小さく振る",
            "正面へ歩く",
            "右手だけを振る",
            "右手をゆっくり振る",
            "右手を頭上に上げる",
            "ゆっくり歩く",
            "右に歩く",
            "3歩だけ歩く",
            "歩いてからお辞儀する",
            "走る",
            "左方向を指差す",
            "右肩だけをすくめる",
            "拍手をしてからお辞儀をする",
        ]:
            with self.subTest(prompt=prompt):
                self.assertEqual(catalog.normalize_motion_prompt(prompt).prompt, prompt)

    def test_simple_right_wave_uses_exact_compatible_training_caption(self):
        expected = next(
            rule.prompt for rule in catalog._RULES if rule.catalog_id == "MOT_701"
        )
        for prompt in ["右手を振る", "手を振る", "立ったまま、右手を振る"]:
            with self.subTest(prompt=prompt):
                match = catalog.normalize_motion_prompt(prompt)
                self.assertEqual(match.catalog_id, "MOT_701")
                self.assertEqual(match.prompt, expected)
        request = catalog.extract_explicit_motion_request("右手を振ってください。")
        self.assertEqual(request.prompt, expected)

    def test_simple_hand_raise_uses_available_training_caption(self):
        for prompt in ["手を上げる", "右手を上げる", "右手を挙げる"]:
            with self.subTest(prompt=prompt):
                self.assertEqual(
                    catalog.normalize_motion_prompt(prompt).prompt,
                    "右手を上げる",
                )
        self.assertEqual(
            catalog.normalize_motion_prompt("左手を上げる").prompt,
            "左手を上げる",
        )

    def test_simple_walk_uses_verified_basic_caption_without_losing_constraints(self):
        for prompt in ["歩く", "立ったまま、歩く", "立っている状態から、歩く。"]:
            with self.subTest(prompt=prompt):
                match = catalog.normalize_motion_prompt(prompt)
                self.assertEqual(match.prompt, "立っている状態から、歩く。")
                self.assertIsNone(match.catalog_id)
                self.assertEqual(match.reason, "verified_basic_caption")
        request = catalog.extract_explicit_motion_request("歩いてください。")
        self.assertEqual(request.prompt, "立っている状態から、歩く。")

    def test_hand_raise_commands_cover_kanji_and_direct_noun_request(self):
        request = catalog.extract_explicit_motion_request("挙手してください。")
        self.assertIsNotNone(request)
        self.assertEqual(request.prompt, "右手を上げる")
        request = catalog.extract_explicit_motion_request("右手を挙げてください。")
        self.assertEqual(request.prompt, "右手を上げる")
        request = catalog.extract_explicit_motion_request("左手を挙げてください。")
        self.assertEqual(request.prompt, "左手を挙げる")

    def test_exact_catalog_prompt_still_has_identity(self):
        for rule in catalog._RULES:
            with self.subTest(catalog_id=rule.catalog_id):
                match = catalog.normalize_motion_prompt(rule.prompt)
                self.assertEqual(match.prompt, rule.prompt)
                self.assertEqual(match.catalog_id, rule.catalog_id)


class MotionPriorityTests(unittest.TestCase):
    def setUp(self):
        motion_tool._motion_cache.clear()

    def test_explicit_motion_runs_once_without_competing_or_repeated_jobs(self):
        channel = motion_tool.MotionJobChannel(explicit_prompt="左手を振る")
        completed = threading.Event()
        calls = []

        def generate(prompt, key, channel, segment, **kwargs):
            calls.append((prompt, segment))
            channel.put(
                {"prompt": prompt, "segment_index": segment, "bvh": "requested"}
            )
            channel.finish_job()
            completed.set()

        with patch.object(motion_tool, "_run_motion_job", generate):
            self.assertTrue(motion_tool.enqueue_motion_job(channel, "左手を振る", 1))
            self.assertFalse(
                motion_tool.enqueue_motion_job(channel, "右手を唇に当てる", 1)
            )
            self.assertFalse(motion_tool.enqueue_motion_job(channel, "左手を振る", 2))
            self.assertFalse(motion_tool.enqueue_motion_job(channel, "うなずく", 3))
            self.assertTrue(completed.wait(2), "motion worker did not complete")
        self.assertEqual(calls, [("左手を振る", 1)])
        self.assertEqual(len(channel.drain()), 1)
        self.assertTrue(channel.is_idle)

    def test_cache_hit_also_cannot_restart_on_later_sentence(self):
        motion_tool._cache_motion(
            "左手を振る", {"bvh": "cached", "prompt": "左手を振る"}
        )
        channel = motion_tool.MotionJobChannel(explicit_prompt="左手を振る")
        self.assertTrue(motion_tool.enqueue_motion_job(channel, "左手を振る", 1))
        self.assertFalse(motion_tool.enqueue_motion_job(channel, "左手を振る", 2))
        self.assertEqual(
            channel.drain(),
            [
                {
                    "bvh": "cached",
                    "prompt": "左手を振る",
                    "segment_index": 1,
                    "start_offset_ratio": 0.0,
                }
            ],
        )

    def test_audio_driven_explicit_motion_generates_for_each_turn(self):
        # Segment 1 in two requests can contain completely different speech.
        # Neither a text cache hit nor a segment suffix may reject/reuse it.
        prompt = "左手を振る"
        motion_tool._cache_motion(prompt, {"bvh": "old-text-motion", "prompt": prompt})
        calls = []

        def generate(request):
            calls.append(request.audio_wav)
            return types.SimpleNamespace(
                bvh=request.audio_wav.hex(),
                latency_seconds=0,
                backend="sentiavatar",
            )

        fake_backend = types.SimpleNamespace(requires_audio=True, generate=generate)
        with patch.object(motion_tool, "get_motion_backend", return_value=fake_backend):
            for audio in [b"first-speech", b"second-speech"]:
                channel = motion_tool.MotionJobChannel(explicit_prompt=prompt)
                channel.request_action(prompt, None)
                queued_prompt, catalog_id = channel.drain_actions()[0]
                self.assertEqual(channel.drain_actions(), [])
                self.assertTrue(
                    motion_tool.enqueue_motion_job(
                        channel,
                        queued_prompt,
                        1,
                        catalog_id=catalog_id,
                        audio_wav=audio,
                    )
                )
                result = channel.get(timeout=2)
                self.assertIsNotNone(result)
                self.assertEqual(result["bvh"], audio.hex())
                self.assertFalse(
                    motion_tool.enqueue_motion_job(
                        channel,
                        prompt,
                        2,
                        audio_wav=audio,
                    )
                )
        self.assertEqual(calls, [b"first-speech", b"second-speech"])
        self.assertEqual(
            motion_tool._get_cached_motion(prompt)["bvh"], "old-text-motion"
        )

    def test_preaccepted_tool_call_cannot_add_a_competing_motion(self):
        channel = motion_tool.MotionJobChannel(explicit_prompt="左手を振る")
        token = motion_tool.motion_data_queue_var.set(channel)
        try:
            with patch.object(motion_tool, "_run_motion_job") as worker:
                result = motion_tool.generate_motion("右手を唇に当てる")
            self.assertEqual(json.loads(result)["motion_description"], "左手を振る")
            worker.assert_not_called()
            self.assertTrue(channel.is_idle)
        finally:
            motion_tool.motion_data_queue_var.reset(token)

    def test_tool_fallback_schedules_commands_outside_deterministic_detector(self):
        self.assertIsNone(catalog.extract_explicit_motion_request("笑ってください"))
        channel = motion_tool.MotionJobChannel()
        token = motion_tool.motion_data_queue_var.set(channel)
        finished = threading.Event()
        calls = []

        def generate(prompt, key, channel, segment, **kwargs):
            calls.append(prompt)
            channel.finish_job()
            finished.set()

        try:
            with patch.object(motion_tool, "_run_motion_job", generate):
                motion_tool.generate_motion("笑顔で笑う")
                self.assertTrue(finished.wait(2))
            self.assertEqual(calls, ["笑顔で笑う"])
        finally:
            motion_tool.motion_data_queue_var.reset(token)

    def test_normal_conversation_keeps_per_sentence_gestures(self):
        channel = motion_tool.MotionJobChannel()
        self.assertTrue(channel.claim_prompt("うなずく", 1))
        self.assertFalse(channel.claim_prompt("うなずく", 1))
        self.assertTrue(channel.claim_prompt("うなずく", 2))


if __name__ == "__main__":
    unittest.main()
