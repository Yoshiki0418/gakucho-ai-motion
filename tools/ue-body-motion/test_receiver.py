"""Exercise UE's body-only playback contract and its local HTTP transport."""

import http.client
import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from receiver import Command, EditorServer, Scheduler


def play(client, sequence, received=0, revision=0, offset=0, remaining=2):
    return Command(
        client,
        sequence,
        received,
        "play",
        "segment",
        revision,
        offset,
        remaining,
        SimpleNamespace(frame_count=16, frame_time=0.125),
    )


class SchedulingTest(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        # The scheduling layer may only use Body playback. No face/Sequencer
        # start, stop, or seek operations are part of the adapter contract.
        self.body = Mock(spec_set=["prepare", "show", "clear"])
        self.scheduler = Scheduler(self.body, lambda: self.now)

    def test_delayed_play_starts_at_audio_position_and_expires_at_sentence_end(self):
        cmd = play("a", 1, offset=0.25, remaining=0.75)
        self.scheduler.reserve(cmd)
        self.now = 0.5
        self.scheduler.tick()
        self.body.show.assert_called_once_with(self.body.prepare.return_value, 6)
        self.scheduler.tick()
        self.body.show.assert_called_once()
        self.now = 0.75
        self.scheduler.tick()
        self.body.clear.assert_called_once_with()
        self.assertFalse(self.scheduler.health()["playing"])

    def test_waits_for_visible_window_before_acknowledging_play(self):
        cmd = play("a", 1)
        cmd.whole_body = True
        self.scheduler.reserve(cmd)
        self.body.show.side_effect = [False, True]
        self.scheduler.tick()
        self.assertFalse(cmd.done.is_set())
        self.assertEqual(self.scheduler.frame, -1)
        self.body.prepare.assert_called_once_with(cmd.clip, whole_body=True)
        self.scheduler.tick()
        self.assertTrue(cmd.done.is_set())
        self.assertEqual(cmd.result[1]["state"], "playing")

    def test_prepare_time_revalidates_audio_deadline(self):
        def delayed_prepare(_clip, whole_body=False):
            self.now = 1.0
            return Mock()

        self.body.prepare.side_effect = delayed_prepare
        cmd = play("a", 1, offset=1.5)
        self.scheduler.reserve(cmd)
        self.scheduler.tick()
        self.assertEqual(cmd.result[1]["state"], "expired")
        self.body.show.assert_not_called()
        self.body.clear.assert_called_once_with()

    def test_late_stop_cancels_upload_and_prior_sequence_cannot_resume(self):
        uploading = play("a", 9)
        uploading.clip = None
        self.scheduler.reserve(uploading)
        stop = Command("a", 11, 0, "stop")
        self.assertTrue(self.scheduler.reserve(stop))
        self.scheduler.tick()
        self.assertEqual(stop.result[1]["state"], "stopped")
        self.assertEqual(uploading.result[1]["state"], "stale")
        self.assertFalse(self.scheduler.reserve(play("a", 10)))
        self.scheduler.tick()
        self.body.prepare.assert_not_called()

    def test_visible_clip_advances_through_upload_and_preparation_of_revision(self):
        first, second = Mock(), Mock()
        self.body.prepare.side_effect = [first, second]
        old = play("a", 1)
        self.scheduler.reserve(old)
        self.scheduler.tick()
        new = play("a", 2, revision=1)
        clip, new.clip = new.clip, None
        self.scheduler.reserve(new)
        self.now = 0.25
        self.scheduler.tick()
        self.body.show.assert_called_with(first, 2)
        new.clip = clip
        self.now = 0.375
        self.body.show.side_effect = lambda handle, frame: handle is first
        self.scheduler.tick()
        self.body.show.assert_called_with(first, 3)
        self.assertFalse(new.done.is_set())
        self.assertIs(self.scheduler.active, old)
        self.body.show.side_effect = None
        self.now = 0.5
        self.scheduler.tick()
        self.body.show.assert_called_with(second, 4)
        first.close.assert_called_once()
        second.close.assert_not_called()
        self.body.clear.assert_not_called()
        self.assertEqual(new.result[1]["state"], "playing")

    def test_stop_closes_the_visible_and_the_unfinished_window(self):
        first, pending = Mock(), Mock()
        self.body.prepare.side_effect = [first, pending]
        self.scheduler.reserve(play("a", 1))
        self.scheduler.tick()
        self.body.show.side_effect = lambda handle, frame: handle is first
        self.scheduler.reserve(play("a", 2, revision=1))
        self.scheduler.tick()
        self.scheduler.reserve(Command("a", 3, 0.1, "stop"))
        self.scheduler.tick()
        first.close.assert_called_once()
        pending.close.assert_called_once()
        self.body.clear.assert_called_once()
        self.assertFalse(self.scheduler.health()["playing"])

    def test_an_old_revision_and_a_previous_clients_stop_do_not_rewind_body(self):
        self.scheduler.reserve(play("a", 10))
        self.scheduler.tick()
        self.now = 0.5
        b = play("b", 1, received=0.5, revision=2)
        self.scheduler.reserve(b)
        self.scheduler.tick()
        self.body.reset_mock()
        self.assertFalse(self.scheduler.reserve(play("b", 2, received=0.5, revision=1)))
        self.assertFalse(self.scheduler.reserve(Command("a", 11, 0.5, "stop")))
        self.scheduler.tick()
        self.body.clear.assert_not_called()
        self.assertIs(self.scheduler.active, b)

    def test_previous_clients_slow_upload_cannot_replace_newer_play(self):
        self.now = 0.5
        current = play("b", 1, received=0.5)
        self.scheduler.reserve(current)
        self.scheduler.tick()
        # A's request arrived first but its upload completed after B started.
        delayed = play("a", 1, received=0.25)
        self.assertFalse(self.scheduler.reserve(delayed))
        self.assertEqual(delayed.result[1]["state"], "stale")
        self.scheduler.tick()
        self.assertIs(self.scheduler.active, current)

    def test_editor_apply_failure_cleans_up_and_next_play_recovers(self):
        self.body.show.side_effect = [RuntimeError("Body unavailable"), None]
        failed = play("a", 1)
        self.scheduler.reserve(failed)
        with self.assertRaisesRegex(RuntimeError, "Body unavailable"):
            self.scheduler.tick()
        self.body.clear.assert_called_once_with()
        self.assertEqual(failed.result[0], 422)
        self.assertEqual(
            self.scheduler.health(),
            {
                "ok": False,
                "playing": False,
                "frame": -1,
                "error": True,
            },
        )
        new = play("a", 2)
        self.scheduler.reserve(new)
        self.scheduler.tick()
        self.assertEqual(new.result[1]["state"], "playing")
        self.assertTrue(self.scheduler.health()["ok"])


class EditorHTTPTest(unittest.TestCase):
    def test_local_socket_rejects_invalid_bvh_then_plays_and_stops(self):
        bvh = (
            Path(__file__).resolve().parents[2]
            / "backend/app/assets/conversation_motion/skeleton.bvh"
        ).read_text()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "body.sock"
            body = Mock(spec_set=["prepare", "show", "clear"])
            scheduler = Scheduler(body, lambda: 0.0)
            with EditorServer(path, scheduler) as server:
                self.addCleanup(server.guard.close)
                serving = threading.Thread(
                    target=server.serve_forever, kwargs={"poll_interval": 0.01}
                )
                serving.start()
                done = threading.Event()

                def tick():
                    while not done.wait(0.005):
                        scheduler.tick()

                clock = threading.Thread(target=tick)
                clock.start()

                def request(method, route, data=None):
                    with socket.socket(socket.AF_UNIX) as connection:
                        connection.settimeout(2)
                        connection.connect(str(path))
                        client = http.client.HTTPConnection("editor")
                        client.sock = connection
                        client.request(
                            method,
                            route,
                            body=json.dumps(data) if data else None,
                            headers={"Content-Type": "application/json"},
                        )
                        response = client.getresponse()
                        return response.status, json.loads(response.read())

                try:
                    self.assertTrue(request("GET", "/health")[1]["ok"])
                    play_data = dict(
                        client_id="smoke",
                        sequence=1,
                        clip_id="s",
                        revision=0,
                        offset_seconds=0,
                        bvh="not bvh",
                    )
                    self.assertEqual(request("POST", "/play", play_data)[0], 400)
                    play_data.update(sequence=2, bvh=bvh)
                    self.assertEqual(
                        request("POST", "/play", play_data)[1]["state"], "playing"
                    )
                    self.assertTrue(request("GET", "/health")[1]["playing"])
                    self.assertEqual(
                        request("POST", "/stop", {"client_id": "smoke", "sequence": 3})[
                            1
                        ]["state"],
                        "stopped",
                    )
                    self.assertFalse(request("GET", "/health")[1]["playing"])
                    self.assertEqual(request("POST", "/unknown", {})[0], 404)
                finally:
                    done.set()
                    clock.join()
                    server.shutdown()
                    serving.join()


if __name__ == "__main__":
    unittest.main()
