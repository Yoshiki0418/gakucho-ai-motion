"""Train-scene matched physical probe confirming episode-horizon invariance.

Only the ordinary episode duration differs. Observation58 is actual gait
phase: the imitation subclass overrides the base episode-time feature.
Actor-only substitutions are artificial phase perturbations, never actual
runtime normalization and never a source of physical control.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
from pathlib import Path

import numpy as np

from .motor_feasibility import summarize_samples


def run_case(spec):
    from .tracked_visual_navigation import make_visual_env
    name,motor,horizon=spec
    env=make_visual_env(motor_checkpoint=motor,mode="turn",split="train",base_seed=90301,
        difficulty_range=(0.,0.),randomize=False,max_episode_seconds=horizon)
    samples=[];sensitivity=[];controls=substeps=0
    try:
        env.reset(seed=91301,options={"scene_seed":90301,"physics_seed":91301})
        body=env.unwrapped;goal_object=body.goal;goal=body.goal.copy()
        done=truncated=False
        while body.data.time<8.-1e-8 and not(done or truncated):
            _,_,done,truncated,info=env.step(np.asarray([1.,-.6],dtype=np.float32))
            controls+=info['command_control_steps'];substeps+=info['command_physics_substeps']
            rotation=body.data.xmat[body.pelvis_body_id].reshape(3,3)
            observation=env.motor_env._observation().copy()
            if len(sensitivity)<5 and body.data.time>=(1.,2.,4.,6.,8.)[len(sensitivity)]-1e-8:
                eight=observation.copy();thirty=observation.copy()
                eight[58]=body._step_count/400.
                thirty[58]=body._step_count/1500.
                action8=env._motor.predict(eight,deterministic=True)[0]
                action30=env._motor.predict(thirty,deterministic=True)[0]
                sensitivity.append({'time':float(body.data.time),'artificialPhaseFrom8SecondFormula':float(eight[58]),
                    'artificialPhaseFrom30SecondFormula':float(thirty[58]),'actualGaitPhaseInput58':float(observation[58]),
                    'phaseSubstitutionIsNotActualRuntimeNormalization':True,
                    'allOther109InputValuesIdentical':bool(np.array_equal(np.delete(eight,58),np.delete(thirty,58))),
                    'maxActionDifference':float(np.max(np.abs(action8-action30))),
                    'meanActionDifference':float(np.mean(np.abs(action8-action30))),
                    'actions8':action8.tolist(),'actions30':action30.tolist()})
            samples.append({'time':float(body.data.time),'heading':math.atan2(rotation[1,0],rotation[0,0]),
                'positionXY':body.data.qpos[:2].tolist(),'speed':float(np.linalg.norm(body.data.qvel[:2])),
                'upright':float(info['upright']),'externalForce':max(float(np.max(np.abs(body.data.qfrc_applied))),float(np.max(np.abs(body.data.xfrc_applied)))),
                'actualInput58':float(observation[58])})
            if body.goal is not goal_object or not np.array_equal(body.goal,goal):
                raise RuntimeError('time-domain probe changed actual goal')
        env.assert_frozen_motor(check_checkpoint=True)
        return {'name':name,'motorSha256':env.motor_checkpoint_sha256,'ordinaryEpisodeHorizonSeconds':horizon,
            'physicalControls':controls,'physicsSubsteps':substeps,'actorSensitivity':sensitivity,
            'summary':summarize_samples(samples),'samples':samples,'actualGoalFixed':True,'learningSteps':0}
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True);parser.add_argument('--candidate',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():
        raise ValueError('preserve prior time-domain diagnostic')
    specs=[('source8',args.source,8.),('source30',args.source,30.),('candidate8',args.candidate,8.),('candidate30',args.candidate,30.)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        cases=list(pool.map(run_case,specs))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'version':1,'sceneSeed':90301,'physicsSeed':91301,'split':'train',
        'onlyOrdinaryEpisodeHorizonDiffers':True,
        'discardedHypothesis':'episode normalization differs; disproved by inherited imitation observation override',
        'index58Semantics':'actual immutable walking reference phase, independently of episode max_steps (imitation.py override)',
        'actorCounterfactualInputsNeverDrivePhysics':True,'cases':cases},indent=2,allow_nan=False)+'\n')
    for case in cases:
        print(json.dumps({'name':case['name'],**case['summary'],
            'actorMaxDifferenceAcrossSavedTimes':max(x['maxActionDifference'] for x in case['actorSensitivity'])}),flush=True)


if __name__=='__main__':
    main()
