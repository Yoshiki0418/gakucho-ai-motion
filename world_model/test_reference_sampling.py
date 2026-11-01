"""Compare fast sampling with the previous interpolation and SciPy oracle.

This checks the physical trajectory's values and timing independently of the
vectorized sampler, rather than assuming a faster implementation is equivalent.
"""

import copy
import hashlib
import math

import numpy as np
import pytest

from .humanoid import JOINT_NAMES
from .references import MotionReference, ReferenceSample, _ASSETS, _parse_bvh


def original_sample(reference, time_seconds):
    """The pre-optimization implementation, including SciPy quaternion SLERP."""
    if not math.isfinite(time_seconds) or time_seconds < 0:
        raise ValueError("Reference time must be finite and nonnegative")
    if reference.kind == "walk":
        cycles = math.floor(time_seconds / reference.duration)
        local_time = time_seconds - cycles * reference.duration
    else:
        cycles = 0
        local_time = min(time_seconds, reference.duration)
    phase = local_time / reference.duration
    targets = np.asarray([np.interp(local_time, reference.times, reference.joint_targets[:, i])
                          for i in range(len(JOINT_NAMES))])
    position = np.asarray([np.interp(local_time, reference.times, reference.root_positions[:, i])
                           for i in range(3)])
    position[:2] += cycles * reference.stride
    velocity = np.asarray([np.interp(local_time, reference.times, reference.root_velocities[:, i])
                           for i in range(3)])
    if reference.kind == "sit" and time_seconds > reference.duration:
        velocity.fill(0)
    rotation = reference._root_slerp(local_time)
    quaternion_xyzw = rotation.as_quat()
    nearest = min(int(round(local_time / reference.frame_time)), len(reference.times) - 1)
    return ReferenceSample(targets, position, velocity, quaternion_xyzw[[3, 0, 1, 2]], phase,
                           tuple(bool(value) for value in reference.contacts[nearest]),
                           float(rotation.as_matrix()[2, 2]))


@pytest.fixture(scope="module", params=["walk", "sit"])
def reference(request):
    return MotionReference(request.param)


def sample_times(reference):
    rng = np.random.default_rng(20261014)
    times = [0., reference.duration, reference.duration + .1]
    # Native knots, half frames (Python's ties-to-even contact rule), and their
    # neighboring floats exercise frame selection without changing clip speed.
    for time in np.concatenate((reference.times, reference.times[:-1] + reference.frame_time / 2)):
        times.extend([float(time), float(np.nextafter(time, np.inf))])
        if time > 0:
            times.append(float(np.nextafter(time, -np.inf)))
    if reference.kind == "walk":
        for cycle in [1, 2, 9, 1024]:
            boundary = cycle * reference.duration
            times.extend([float(np.nextafter(boundary, -np.inf)), boundary,
                          float(np.nextafter(boundary, np.inf)), boundary + .37])
    else:
        times.extend([reference.duration + 100, 1e9])
    times.extend(rng.uniform(0, reference.duration * 15, 1000).tolist())
    return times


def assert_samples_equivalent(actual, expected):
    for name, shape in [("joint_targets", (20,)), ("root_position", (3,)),
                        ("root_velocity", (3,)), ("root_quaternion", (4,))]:
        value = getattr(actual, name)
        assert value.shape == shape
        np.testing.assert_allclose(value, getattr(expected, name), atol=1e-12, rtol=0)
    assert actual.phase == expected.phase
    assert actual.foot_contacts == expected.foot_contacts
    assert actual.upright == pytest.approx(expected.upright, abs=1e-12, rel=0)


def test_random_native_knots_loop_boundaries_and_sit_clamp_match_old_sample(reference):
    for time in sample_times(reference):
        try:
            expected = original_sample(reference, time)
        except ValueError:
            # The old SciPy path rejected sub-ulp endpoint overshoots. Sampling
            # now clamps only that arithmetic residue, without retiming a clip.
            cycles = math.floor(time / reference.duration)
            local = time - cycles * reference.duration
            assert -1e-10 < local < reference.duration + 1e-10
            assert np.isfinite(reference.sample(time).root_quaternion).all()
        else:
            assert_samples_equivalent(reference.sample(time), expected)


def test_sampling_keeps_source_provenance_period_and_native_arrays_unchanged(reference):
    names = ["times", "source_root_positions", "native_root_positions", "stride",
             "joint_targets", "root_positions", "root_velocities", "native_root_velocities",
             "foot_heights", "contacts", "_sample_frames", "_sample_slopes", "_root_quaternions",
             "_root_angles", "_root_inverse_sines"]
    arrays = {name: getattr(reference, name).copy() for name in names}
    diagnostics = copy.deepcopy(reference.diagnostics)
    timing = (reference.frame_time, reference.duration, reference.source_duration,
              reference.source_frame_range, reference.scale)
    for time in sample_times(reference):
        reference.sample(time)
    assert reference.diagnostics == diagnostics
    assert reference.diagnostics["sourceSha256"] == hashlib.sha256(reference.path.read_bytes()).hexdigest()
    assert (reference.frame_time, reference.duration, reference.source_duration,
            reference.source_frame_range, reference.scale) == timing
    for name, before in arrays.items():
        value = getattr(reference, name)
        assert not value.flags.writeable
        np.testing.assert_array_equal(value, before)
    if reference.kind == "walk":
        assert reference.duration == 1.12
        assert reference.source_frame_range == (260, 372)
        for cycle in [1, 2, 1024]:
            before, after = reference.sample(0), reference.sample(cycle * reference.duration)
            np.testing.assert_array_equal(after.joint_targets, before.joint_targets)
            np.testing.assert_allclose(after.root_position[:2], before.root_position[:2] + cycle * reference.stride,
                                       atol=1e-12, rtol=0)


def test_returned_arrays_do_not_alias_reference_caches_or_later_samples(reference):
    first, second = reference.sample(.173), reference.sample(.173)
    original = original_sample(reference, .173)
    source_arrays = [value for value in vars(reference).values() if isinstance(value, np.ndarray)]
    for name in ["joint_targets", "root_position", "root_velocity", "root_quaternion"]:
        value, other = getattr(first, name), getattr(second, name)
        assert all(not np.shares_memory(value, source) for source in source_arrays)
        assert not np.shares_memory(value, other)
        value.fill(-123)
    assert_samples_equivalent(second, original)
    assert_samples_equivalent(reference.sample(.173), original)


def test_quaternion_cache_has_unit_lengths_and_shortest_adjacent_arcs(reference):
    np.testing.assert_allclose(np.linalg.norm(reference._root_quaternions, axis=1), 1, atol=1e-14, rtol=0)
    assert np.all(np.sum(reference._root_quaternions[:-1] * reference._root_quaternions[1:], axis=1) >= 0)
    for time in np.linspace(0, reference.duration, 301):
        sample = reference.sample(float(time))
        assert np.linalg.norm(sample.root_quaternion) == pytest.approx(1, abs=1e-14)


@pytest.mark.parametrize("second_root_angle", [0., 2e-7])
def test_identical_and_tiny_root_rotations_match_scipy(tmp_path, second_root_angle):
    source = (_ASSETS / "sit.bvh").read_text()
    hierarchy = source.split("MOTION", 1)[0]
    bvh = _parse_bvh(source)
    frames = np.repeat(bvh.frames[:1], 2, axis=0)
    root = bvh.bones[0]
    for index, name in enumerate(root.channels):
        if name.endswith("rotation"):
            frames[:, index] = 0
    frames[1, root.channels.index("Zrotation")] = second_root_angle
    path = tmp_path / "small-root-angle.bvh"
    rows = "\n".join(" ".join(format(value, ".17g") for value in frame) for frame in frames)
    path.write_text(hierarchy + "MOTION\nFrames: 2\nFrame Time: 0.01\n" + rows + "\n")
    tiny_reference = MotionReference("sit", path)
    assert tiny_reference._root_angles[0] < 1e-8
    for time in [0, .001, .005, .009, .01, .02]:
        assert_samples_equivalent(tiny_reference.sample(time), original_sample(tiny_reference, time))
    assert tiny_reference.diagnostics["provenance"] == "provided_bvh_provenance_unverified"


def test_long_walk_endpoint_roundoff_clamps_without_changing_native_path():
    reference = MotionReference("walk")
    first = reference.sample(0)
    # This exact float previously raised in SciPy because cycle subtraction
    # landed just outside its interval. It is the ninth native cycle boundary.
    times = [(10.08, 9)]
    for cycle in [1024, 32768, 1_000_000]:
        boundary = cycle * reference.duration
        times.extend([(float(np.nextafter(boundary, -np.inf)), cycle), (boundary, cycle)])
    for time, cycle in times:
        sample = reference.sample(time)
        np.testing.assert_allclose(sample.root_position[:2], first.root_position[:2] + cycle * reference.stride,
                                   atol=1e-9, rtol=0)
        np.testing.assert_allclose(sample.joint_targets, first.joint_targets, atol=1e-9, rtol=0)
        np.testing.assert_allclose(sample.root_quaternion, first.root_quaternion, atol=1e-9, rtol=0)
        assert np.isfinite(sample.root_velocity).all()
        assert reference.duration == 1.12


@pytest.mark.parametrize("invalid_time", [-.001, math.nan, math.inf, -math.inf])
def test_invalid_times_still_rejected(reference, invalid_time):
    with pytest.raises(ValueError):
        reference.sample(invalid_time)
    with pytest.raises(ValueError):
        original_sample(reference, invalid_time)
