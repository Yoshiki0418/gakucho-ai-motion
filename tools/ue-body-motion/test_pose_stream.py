from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from pose_stream import PoseStream


class PoseStreamTest(unittest.TestCase):
    def setUp(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(self.executor.shutdown)
        self.clip = SimpleNamespace(frame_time=1 / 30, frame_count=36000)
        self.gesture = Mock()
        self.gesture.sample.side_effect = lambda frame, dt: (frame, dt)
        self.stream = PoseStream(self.executor, self.gesture, self.clip)
        self.addCleanup(self.stream.close)

    def test_late_start_and_shared_endpoint_use_source_time(self):
        end = self.stream.submit(1500, 0.125, first=True)
        first = self.stream.future.result(timeout=2)
        self.assertEqual(first[0], (1500, 0))
        self.assertEqual(first[-1][0], end)
        self.assertAlmostEqual(sum(dt for _, dt in first), (end - 1500) / 30)
        next_end = self.stream.submit(end, 0.125, first=False)
        following = self.stream.future.result(timeout=2)
        self.assertEqual(following[0], (end, 0))
        self.assertEqual(following[-1][0], next_end)
        self.assertGreaterEqual((next_end - end) / 30, 2.5)
        self.assertLessEqual(len(following), 1 + 24 * 2.5)

    def test_slow_ticks_extend_lookahead_without_baking_the_whole_clip(self):
        end = self.stream.submit(0, 1, first=False)
        poses = self.stream.future.result(timeout=2)
        self.assertEqual(end, 180)
        self.assertLessEqual(len(poses), 145)
        end = self.stream.submit(self.clip.frame_count - 3, 1, first=False)
        self.assertEqual([n for n, _ in self.stream.future.result(timeout=2)], list(range(end - 2, end + 1)))

    def test_replacing_a_clip_cancels_a_running_sampler_between_keys(self):
        entered, release = threading.Event(), threading.Event()
        caller = threading.get_ident()

        def sample(frame, dt):
            self.assertNotEqual(threading.get_ident(), caller)
            entered.set()
            self.assertTrue(release.wait(2))
            return frame

        self.gesture.sample.side_effect = sample
        self.stream.submit(0, 1 / 30, first=True)
        try:
            self.assertTrue(entered.wait(2))
            self.stream.close()
        finally:
            release.set()
        self.assertEqual(self.stream.future.result(timeout=2), [])
        self.gesture.sample.assert_called_once_with(0, 0)
