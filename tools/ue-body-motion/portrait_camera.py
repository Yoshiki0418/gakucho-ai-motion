"""Scene-only Pixel Streaming camera for the on-screen president (UE 5.7)."""
import builtins
import math
import os
import time

import unreal

from body_playback import _find_body

STREAM_ID = "GakuchoPortrait"
WIDTH, HEIGHT = 960, 640
FPS = 24


class PortraitCamera:
    def __init__(self):
        self.actor = self.body = self.streamer = None
        self.check_at = self.capture_at = 0.0
        self.watched = False

    def _place(self, body):
        head = body.get_socket_location("head")
        pelvis = body.get_socket_location("pelvis")
        # Face along the character's horizontal forward axis. Anchor the camera
        # once per spawned Body, so individual gestures cannot shake the view.
        left = body.get_socket_location("upperarm_l")
        right = body.get_socket_location("upperarm_r")
        dx, dy = left.x - right.x, left.y - right.y
        length = math.hypot(dx, dy)
        if length < 1:
            raise ValueError("Cannot determine the front of the Body")
        fx, fy = -dy / length, dx / length
        top, bottom = head.z + 23.0, pelvis.z + 6.0
        height = max(70.0, top - bottom)
        distance = height * 1.9
        pos = unreal.Vector(pelvis.x + distance * fx, pelvis.y + distance * fy, (top + bottom) / 2)
        rot = unreal.Rotator(0, 0, math.degrees(math.atan2(-fy, -fx)))
        self.actor.set_actor_location_and_rotation(pos, rot, False, True)
        self.capture.fov_angle = math.degrees(2 * math.atan(height * WIDTH / HEIGHT / (2 * distance)))
        self.body = body

    def _start(self, body):
        actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        selected = actors.get_selected_level_actors()
        try:
            self.actor = actors.spawn_actor_from_class(unreal.SceneCapture2D, body.get_socket_location("pelvis"), transient=True)
        finally:
            actors.set_selected_level_actors(selected)
        if not self.actor:
            raise RuntimeError("Could not create the portrait camera")
        self.actor.set_actor_label(STREAM_ID, mark_dirty=False)
        self.capture = self.actor.get_component_by_class(unreal.SceneCaptureComponent2D)
        self.capture.capture_every_frame = False
        self.capture.capture_on_movement = False
        self.capture.always_persist_rendering_state = True
        self.capture.capture_source = unreal.SceneCaptureSource.SCS_FINAL_COLOR_LDR
        lighting = self.capture.post_process_settings
        lighting.override_dynamic_global_illumination_method = True
        lighting.dynamic_global_illumination_method = unreal.DynamicGlobalIlluminationMethod.LUMEN
        self.capture.post_process_settings = lighting
        self._place(body)
        self.target = unreal.RenderingLibrary.create_render_target2d(
            self.actor, WIDTH, HEIGHT, unreal.TextureRenderTargetFormat.RTF_RGBA8_SRGB
        )
        self.target.target_gamma = 1.95
        self.capture.texture_target = self.target
        self.producer = unreal.new_object(unreal.PixelStreaming2VideoProducerRenderTarget)
        self.producer.target = self.target
        self.streamer = unreal.new_object(unreal.PixelStreaming2StreamerComponent, outer=self.actor)
        for key, value in {
            "streamer_id": STREAM_ID,
            "video_producer": self.producer,
            "signalling_server_url": os.getenv("GAKUCHO_UE_SIGNALLING_URL", "ws://127.0.0.1:8899"),
            "stream_fps_override": FPS,
        }.items():
            self.streamer.set_editor_property(key, value)
        self.streamer.start_streaming()
        self.capture.capture_scene()

    def stop(self):
        if unreal.SystemLibrary.is_valid(self.streamer):
            self.streamer.stop_streaming()
        if unreal.SystemLibrary.is_valid(self.actor):
            unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(self.actor)
        self.actor = self.body = self.capture = self.target = self.producer = self.streamer = None
        self.watched = False

    def tick(self, _dt):
        now = time.monotonic()
        if now >= self.check_at:
            self.check_at = now + 0.5
            if not unreal.SystemLibrary.is_valid(self.actor) or not unreal.SystemLibrary.is_valid(self.body):
                try:
                    body, _ = _find_body()
                except RuntimeError:
                    if self.streamer is not None:
                        self.stop()
                    return  # Sequencer has not spawned the Body yet.
                try:
                    if unreal.SystemLibrary.is_valid(self.actor):
                        self._place(body)
                    else:
                        self.stop()
                        self._start(body)
                except Exception as exc:
                    self.stop()
                    self.check_at = now + 10
                    unreal.log_error(f"Gakucho portrait camera: {exc}")
                    return
            self.watched = bool(unreal.PixelStreaming2Blueprints.streamer_get_connected_players(STREAM_ID))
        if (self.watched and now >= self.capture_at
                and unreal.SystemLibrary.is_valid(self.actor) and unreal.SystemLibrary.is_valid(self.body)):
            # Keep the 24 Hz cadence on a 30/60 Hz Slate clock. Scheduling
            # from each rounded tick would repeatedly drop an extra frame.
            self.capture_at += 1 / FPS
            if self.capture_at <= now:
                self.capture_at = now + 1 / FPS
            self.capture.capture_scene()


def start_portrait_camera():
    # The editor can run Startup Scripts again without restarting the process.
    if not hasattr(builtins, "_gakucho_portrait"):
        camera = PortraitCamera()
        builtins._gakucho_portrait = (camera, unreal.register_slate_post_tick_callback(camera.tick))
