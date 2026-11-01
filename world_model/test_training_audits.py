"""Actual seat support is required, rather than a nearby crouched pose."""
from .home_scene import create_seated_task_scene
from .training import evaluate


def test_near_chair_audit_requires_sustained_weight_on_actual_seat():
    scene = create_seated_task_scene(seat_distance=.30)
    result = evaluate(task="sit", scene=scene, seeds=[4521], seconds=12,
                      randomize=False, reference=True, reference_start=False, audit_sit=True)
    quality = result["episodes"][0]["seatSupportQuality"]
    assert quality["verified"]
    assert quality["minimumFinal2SecondsPelvisLoadNewtons"] >= .3*quality["bodyWeightNewtons"]
    assert result["seatSupportVerifiedRate"] == 1


def test_removing_seat_cannot_pass_a_nearby_crouching_body():
    scene = create_seated_task_scene(seat_distance=.30)
    scene["geoms"] = [geom for geom in scene["geoms"] if geom.get("semantic") not in ("seat", "chair-seat", "chair_seat")]
    result = evaluate(task="sit", scene=scene, seeds=[4521], seconds=12,
                      randomize=False, reference=True, reference_start=False, audit_sit=True)
    quality = result["episodes"][0]["seatSupportQuality"]
    assert not quality["verified"]
    assert quality["minimumFinal2SecondsPelvisLoadNewtons"] == 0
    assert result["seatSupportVerifiedRate"] == 0
