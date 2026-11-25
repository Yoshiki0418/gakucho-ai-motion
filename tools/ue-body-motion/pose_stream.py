"""Prepare a bounded span of numeric poses without calling the Unreal API."""
from math import ceil
from threading import Event


class PoseStream:
    def __init__(self, executor, gesture, clip):
        self.executor = executor
        self.gesture = gesture
        self.clip = clip
        self.cancelled = Event()
        self.previous = None
        self.future = None

    def submit(self, start, tick_interval, first):
        # Leave enough source time to publish the track even if the editor
        # evaluates at a low rate. Seek at the audio position after a long gap.
        tick_horizon = 8 * max(tick_interval, self.clip.frame_time)
        horizon = max(1.5 if first else 2.5, min(6.0, tick_horizon))
        end = min(self.clip.frame_count - 1, start + ceil(horizon / self.clip.frame_time))
        key_rate = 15 if first else 24
        steps = max(1, min(end - start, ceil((end - start) * self.clip.frame_time * key_rate)))
        self.future = self.executor.submit(self._sample, start, end, steps)
        return end

    def _sample(self, start, end, steps):
        poses = []
        for i in range(steps + 1):
            if self.cancelled.is_set():
                return []
            frame = start + round(i * (end - start) / steps)
            dt = 0 if self.previous is None else (frame - self.previous) * self.clip.frame_time
            poses.append(self.gesture.sample(frame, max(0, dt)))
            self.previous = frame
        return poses

    def close(self):
        self.cancelled.set()
        if self.future is not None:
            self.future.cancel()
