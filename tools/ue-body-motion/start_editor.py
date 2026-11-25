"""Run this file from the UE editor Python console or project Startup Scripts."""
import builtins
from pathlib import Path
import sys
import threading

import unreal

module_dir = str(Path(__file__).resolve().parent)
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

from body_playback import BodyPlayback
from bridge_path import bridge_directory
from receiver import EditorServer, Scheduler
from portrait_camera import start_portrait_camera


def start():
    if hasattr(builtins, "_gakucho_body_receiver"):
        return
    path = bridge_directory() / "body.sock"
    scheduler = Scheduler(BodyPlayback())
    server = EditorServer(path, scheduler)

    def tick(_delta):
        try:
            scheduler.tick()
            scheduler.playback.advance_release()
            if scheduler.desired is None:
                scheduler.playback.warmup()
        except Exception as exc:
            unreal.log_error(f"Gakucho body playback: {exc}")

    handle = unreal.register_slate_post_tick_callback(tick)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    builtins._gakucho_body_receiver = (server, scheduler, handle)
    unreal.log("Gakucho body motion receiver is ready")


start()
start_portrait_camera()
