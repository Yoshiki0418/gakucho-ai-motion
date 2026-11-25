"""Exercise the UE adapter's time and asset calls without starting an editor."""
from importlib import util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch


def load_adapter():
    unreal = SimpleNamespace(SkeletalMeshComponent=object, AnimSequence=object,
                             AnimSingleNodeInstance=Mock, SystemLibrary=Mock())
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

    def test_face_track_recreates_the_player_in_the_middle_of_a_window(self):
        adapter = load_adapter()
        player = adapter.BodyPlayback()
        self.addCleanup(player.workers.shutdown)
        body = player.body = Mock()
        old = player.instance = Mock()
        player.previous_override = original = object()
        asset = object()
        player._set_animation(asset, 0.5)
        player._set_position(0.4)
        fresh = body.get_post_process_instance.return_value
        self.assertTrue(player.refresh_instance())
        fresh.set_animation_asset.assert_called_once_with(asset, False, 0.5)
        fresh.set_position.assert_called_once_with(0.4, False)
        fresh.set_playing.assert_called_once_with(True)
        self.assertFalse(player.refresh_instance())
        player._set_position(0.45)
        fresh.set_position.assert_called_with(0.45, False)
        old.set_position.assert_called_once_with(0.4, False)
        body.set_override_post_process_anim_bp.assert_not_called()
        player.restore()
        body.set_override_post_process_anim_bp.assert_called_once_with(original)
        self.assertFalse(player.refresh_instance())

    def test_replacement_during_release_seeks_the_same_fade_and_restores_idle(self):
        adapter = load_adapter()
        player = adapter.BodyPlayback()
        self.addCleanup(player.workers.shutdown)
        body = player.body = Mock()
        player.instance = Mock()
        player.previous_override = original = object()
        body.get_post_process_instance.return_value = player.instance
        player.release_started = 9.0
        player.release_asset = release = object()
        player.release_length = 1.6
        with patch.object(adapter.time, "monotonic", side_effect=[10, 10.08, 10.24, 10.4]):
            player.advance_release()
            position = player.position
            replacement = body.get_post_process_instance.return_value = Mock()
            self.assertTrue(player.refresh_instance())
            replacement.set_animation_asset.assert_called_once_with(release, False, 0.0)
            replacement.set_playing.assert_called_once_with(False)
            replacement.set_position.assert_called_once_with(position, False)
            player.advance_release()
            self.assertGreater(player.position, position)
            player.advance_release()
        body.set_override_post_process_anim_bp.assert_called_once_with(original)
        self.assertIsNone(player.animation)

    def test_init_delegate_resumes_motion_immediately_and_is_removed_before_restore(self):
        adapter = load_adapter()
        player = adapter.BodyPlayback()
        self.addCleanup(player.workers.shutdown)
        body = Mock(disable_post_process_blueprint=True)
        previous, first, replacement = Mock(), Mock(), Mock()
        body.get_post_process_instance.side_effect = [previous, first, replacement, replacement]
        asset = object()
        motion = Mock(body=body, clip=SimpleNamespace(frame_count=21, frame_time=1 / 30),
                      window=None, pending=adapter.Window(0, 20, 1 / 30, asset), last_update=0)
        self.assertTrue(player.show(motion, 10))
        callback, = body.on_anim_initialized.add_callable.call_args.args
        callback()
        replacement.set_animation_asset.assert_called_once_with(asset, False, 1)
        replacement.set_position.assert_called_once_with(10 / 30, False)
        self.assertTrue(player.refresh_instance())
        replacement.set_animation_asset.assert_called_once()
        body.reset_mock()
        player.restore()
        self.assertEqual(body.mock_calls, [
            call.on_anim_initialized.remove_callable(callback),
            call.set_override_post_process_anim_bp(previous.get_class.return_value),
        ])
        self.assertTrue(body.disable_post_process_blueprint)
