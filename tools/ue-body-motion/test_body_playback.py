"""Exercise the UE adapter's time and asset calls without starting an editor."""
from importlib import util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch


def load_adapter():
    unreal = SimpleNamespace(SkeletalMeshComponent=object, AnimSequence=object,
                             SystemLibrary=Mock())
    spec = util.spec_from_file_location("body_playback", Path(__file__).with_name("body_playback.py"))
    module = util.module_from_spec(spec)
    with patch.dict(sys.modules, {"unreal": unreal}):
        spec.loader.exec_module(module)
    return module


class BodyPlaybackTest(unittest.TestCase):
    def test_native_playback_continues_between_sparse_clock_updates_and_changes_windows(self):
        adapter = load_adapter()
        player = adapter.BodyPlayback()
        self.addCleanup(player.workers.shutdown)
        player.body, player.instance = object(), Mock()
        clip = SimpleNamespace(frame_count=121, frame_time=1 / 30)
        first = adapter.Window(0, 45, 1 / 60, object())
        second = adapter.Window(45, 120, 1 / 45, object())
        motion = Mock(body=player.body, clip=clip, last_update=0, window=None,
                      pending=first, builder=None)
        # The native animation has its own sampling rate. Between audio clock
        # updates it must keep playing at source speed, including at a handoff.
        with patch.object(adapter, "_advance_builder", side_effect=[None, second]):
            with patch.object(adapter.time, "monotonic", side_effect=[1, 1.15, 2.6]):
                self.assertTrue(player.show(motion, 5))
                self.assertTrue(player.show(motion, 9))
                self.assertIs(motion.window, first)
                self.assertTrue(player.show(motion, 49))
                self.assertIs(motion.window, second)
        self.assertEqual(player.instance.set_playing.call_args_list, [call(True), call(True)])
        self.assertEqual(player.instance.set_animation_asset.call_args_list,
                         [call(first.animation, False, 0.5), call(second.animation, False, (1 / 45) / (1 / 30))])
        for actual, expected in zip(player.instance.set_position.call_args_list, (5 / 60, 9 / 60, 4 / 45)):
            self.assertAlmostEqual(actual.args[0], expected)
