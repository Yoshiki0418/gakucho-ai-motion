"""Transport and audio-clock scheduling. Only tick() accesses Unreal objects."""
from collections import OrderedDict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
import json
import math
import os
import socketserver
import threading
import time

from retarget import parse_bvh

MAX_BODY = 8 * 1024 * 1024


@dataclass
class Command:
    client: str
    sequence: int
    received: float
    action: str
    clip_id: str = ""
    revision: int = 0
    offset: float = 0.0
    remaining: float = math.inf
    clip: object = None
    whole_body: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    result: tuple = (200, {"ok": True, "state": "stale"})

    def finish(self, status, result):
        if not self.done.is_set():
            self.result = (status, result)
            self.done.set()


def command_from_json(action, data, now):
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    client = data.get("client_id")
    sequence = data.get("sequence")
    if not isinstance(client, str) or not 0 < len(client) <= 128:
        raise ValueError("Invalid client_id")
    if type(sequence) is not int or not 0 <= sequence < 2**53:
        raise ValueError("Invalid sequence")
    cmd = Command(client, sequence, now, action)
    if action == "play":
        cmd.whole_body = data.get("whole_body", False)
        if type(cmd.whole_body) is not bool:
            raise ValueError("Invalid whole_body")
        cmd.clip_id = data.get("clip_id")
        cmd.revision = data.get("revision")
        if not isinstance(cmd.clip_id, str) or not 0 < len(cmd.clip_id) <= 256:
            raise ValueError("Invalid clip_id")
        if type(cmd.revision) is not int or cmd.revision < 0:
            raise ValueError("Invalid revision")
        if not isinstance(data.get("bvh"), str):
            raise ValueError("Expected BVH text")
        for key, attr in (("offset_seconds", "offset"), ("remaining_seconds", "remaining")):
            if key == "remaining_seconds" and key not in data:
                continue
            value = data.get(key)
            if type(value) not in (float, int) or not 0 <= value <= 3600:
                raise ValueError(f"Invalid {key}")
            setattr(cmd, attr, value)
    return cmd


def success(state):
    return {"ok": True, "state": state}


class Scheduler:
    def __init__(self, playback, clock=time.monotonic):
        self.playback = playback
        self.clock = clock
        self.lock = threading.Lock()
        self.recent = OrderedDict()
        self.last_play_received = -math.inf
        self.desired = self.active = self.handle = None
        self.frame = -1
        self.failure = None

    def reserve(self, cmd):
        with self.lock:
            previous = self.recent.get(cmd.client)
            if previous:
                sequence, clip_id, revision, _ = previous
                if cmd.sequence <= sequence or (
                    cmd.action == "play" and cmd.clip_id == clip_id and cmd.revision < revision
                ):
                    cmd.finish(200, success("stale"))
                    return False
            while self.recent and next(iter(self.recent.values()))[3] < cmd.received - 3600:
                self.recent.popitem(last=False)
            if not previous and len(self.recent) >= 1024:
                cmd.finish(429, {"detail": "Too many active editor clients"})
                return False
            self.recent[cmd.client] = (cmd.sequence, cmd.clip_id, cmd.revision, cmd.received)
            self.recent.move_to_end(cmd.client)
            if cmd.action == "play":
                # A slower upload from a different tab must not reclaim the body.
                if cmd.received < self.last_play_received:
                    cmd.finish(200, success("stale"))
                    return False
                self.last_play_received = cmd.received
            if cmd.action == "stop" and (not self.desired or self.desired.client != cmd.client):
                cmd.finish(200, success("stopped"))
                return False
            if self.desired:
                self.desired.finish(200, success("stale"))
            self.desired = cmd
            return True

    def finish(self, cmd, status, result):
        with self.lock:
            cmd.finish(status, result)
            if self.desired is cmd:
                self.desired = None

    def clear(self):
        if self.handle is not None:
            self.playback.clear()
        self.active = self.handle = None
        self.frame = -1

    def tick(self):
        with self.lock:
            cmd = self.desired
        try:
            if cmd is not self.active:
                self.clear()
            if not cmd:
                return
            if cmd.action == "stop":
                self.finish(cmd, 200, success("stopped"))
                return
            if cmd.clip is None:
                return
            elapsed = self.clock() - cmd.received
            deadline = min(cmd.remaining, cmd.clip.frame_count * cmd.clip.frame_time - cmd.offset)
            if elapsed < deadline and self.handle is None:
                self.handle = self.playback.prepare(cmd.clip, whole_body=cmd.whole_body)
                self.active = cmd
                elapsed = self.clock() - cmd.received
            with self.lock:
                superseded = cmd is not self.desired
            if superseded or elapsed >= deadline:
                self.clear()
                self.finish(cmd, 200, success("stale" if superseded else "expired"))
                return
            frame = int((cmd.offset + elapsed) / cmd.clip.frame_time)
            if frame != self.frame:
                if self.playback.show(self.handle, frame) is False:
                    return
                self.frame = frame
            self.failure = None
            with self.lock:
                cmd.finish(200, success("playing"))
        except Exception as exc:
            self.failure = str(exc)[:512]
            try:
                self.playback.clear()
            finally:
                self.active = self.handle = None
                self.frame = -1
                if cmd:
                    self.finish(cmd, 422, {"detail": "Unreal could not apply the body pose"})
            raise

    def health(self):
        return {"ok": self.failure is None, "playing": self.active is not None,
                "frame": self.frame, "error": bool(self.failure)}


class EditorServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def __init__(self, path, scheduler):
        # Never unlink the socket of a running editor, even across processes.
        import fcntl
        self.guard = open(path.parent / "body.lock", "a")
        try:
            fcntl.flock(self.guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if path.is_socket():
                path.unlink()
            elif os.path.lexists(path):
                raise RuntimeError(f"Expected a socket at {path}")
            self.scheduler = scheduler
            self.parsers = threading.BoundedSemaphore(2)
            super().__init__(str(path), Handler)
            os.chmod(path, 0o666)  # Private host parent, shared with the API container.
        except Exception:
            self.guard.close()
            raise


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(12)

    def log_message(self, *args):
        pass

    def reply(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.reply(200, self.server.scheduler.health())
        else:
            self.reply(404, {"detail": "Unknown route"})

    def do_POST(self):
        action = self.path.removeprefix("/")
        if action not in ("play", "stop"):
            self.reply(404, {"detail": "Unknown route"})
            return
        scheduler = self.server.scheduler
        received = scheduler.clock()
        cmd = None
        parsing = False
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= MAX_BODY:
                self.reply(413, {"detail": "Invalid request size"})
                return
            if action == "play":
                parsing = self.server.parsers.acquire(blocking=False)
                if not parsing:
                    self.reply(429, {"detail": "Editor motion queue is busy"})
                    return
            data = json.loads(self.rfile.read(size))
            cmd = command_from_json(action, data, received)
            if scheduler.reserve(cmd):
                if parsing:
                    cmd.clip = parse_bvh(data["bvh"])
                if not cmd.done.wait(12):
                    scheduler.finish(cmd, 504, {"detail": "Editor did not evaluate the body pose in time"})
        except (ValueError, UnicodeError, RecursionError, OverflowError):
            if cmd is not None:
                scheduler.finish(cmd, 400, {"detail": "Invalid body motion"})
            self.reply(400, {"detail": "Invalid body motion"})
            return
        finally:
            if parsing:
                self.server.parsers.release()
        self.reply(*cmd.result)
