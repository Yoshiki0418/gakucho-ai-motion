import math

import numpy as np

from .motor_feasibility import summarize_samples


def _samples(speed=.25, yaw_rate=.125):
    return [{"time":float(t),"heading":math.atan2(math.sin(3.+yaw_rate*t),math.cos(3.+yaw_rate*t)),
             "speed":speed,"positionXY":[speed*t,0.],"upright":.99,"externalForce":0.}
            for t in np.arange(.1,8.01,.1)]


def test_radius_unwraps_heading_across_pi():
    result=summarize_samples(_samples())
    assert np.isclose(result["steadyYawRateRadiansPerSecond"],.125)
    assert np.isclose(result["steadyRadiusMeters"],2.)


def test_braking_cannot_claim_stop_from_one_low_speed_sample():
    samples=_samples()
    samples[-1]["speed"]=0.
    result=summarize_samples(samples,4.)
    assert result["sustainedBelowPointOneMetersPerSecondAt"] is None
    assert np.isclose(result["displacementAfterStopMeters"],1.)


def test_sustained_stop_measured_from_requested_stop_time():
    samples=_samples()
    for sample in samples:
        if sample["time"]>=4.5:
            sample["speed"]=0.
    result=summarize_samples(samples,4.)
    assert 4.9<=result["sustainedBelowPointOneMetersPerSecondAt"]<=5.1
