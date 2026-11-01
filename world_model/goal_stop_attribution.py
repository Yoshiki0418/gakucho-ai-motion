"""Bounded held-out geometry-teacher diagnostic with an explicit goal STOP.

The full-map teacher differs only by its observable high command: after actual
goal distance first reaches 0.20 m it latches [0,-1], the already-defined STOP.
It never changes a physical state, the final goal, weights or command bounds.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np

from .body_contract import trace_body_metadata
from .command_navigation_evaluation import append_control_frames
from .contact_attribution import DiagnosticGeometryController, _measurement, summarize_measurements
from .generalization import validate_scene_seed
from .navigation_evaluation import audit_navigation
from .training import dump_json


GOAL_STOP_TEACHER_CONTRACT={"version":1,"role":"full actual geometry diagnostic oracle only",
    "routeMarginMeters":.5,"routeCarrotMeters":.55,"walkingPaceAction":-.6,
    "stopTrigger":"actual root XY distance to the immutable final goal <= .20 m",
    "stopLatchedUntilEpisodeEnds":True,"latchedStopAction":[0.,-1.],
    "decisionHz":10.,"physicalControlHz":50.,"physicsSubstepHz":500.,
    "commandRangeAndLowWeightsUnchanged":True,"heldoutDiagnosticUsedForTraining":False}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class LatchedStopGeometryController(DiagnosticGeometryController):
    def __init__(self,env):
        super().__init__(env,margin=.5,carrot=.55,pace=-.6)
        self.stop_latched=False
        self.stop_trigger_time=None

    def action(self):
        body=self.env.unwrapped
        if not self.stop_latched and np.linalg.norm(body.goal[:2]-body.data.qpos[:2])<=.20:
            self.stop_latched=True
            self.stop_trigger_time=float(body.data.time)
        return np.asarray([0.,-1.],dtype=np.float32) if self.stop_latched else super().action()


def evaluate_one(config):
    import torch
    from .tracked_visual_navigation import make_visual_env
    torch.set_num_threads(1)
    seed=config['sceneSeed'];validate_scene_seed(seed,'heldout')
    output=Path(config['output'])
    if output.exists():raise ValueError('preserve completed goal STOP diagnostic')
    env=make_visual_env(motor_checkpoint=config['motor'],avatar_profile=config['profile'],
        record_trajectory=True,mode='obstacle',split='heldout',base_seed=seed,
        difficulty_range=(0.,.5),randomize=True,max_episode_seconds=30.)
    try:
        _,_=env.reset(seed=seed+2_000_000,options={'scene_seed':seed,'physics_seed':seed+2_000_000})
        body=env.unwrapped;scene=copy.deepcopy(env.scene)
        initial={'qpos':body.data.qpos.copy().tolist(),'qvel':body.data.qvel.copy().tolist(),
            'qposSha256':hashlib.sha256(body.data.qpos.tobytes()).hexdigest(),
            'qvelSha256':hashlib.sha256(body.data.qvel.tobytes()).hexdigest(),
            'sceneSha256':hashlib.sha256(json.dumps(scene,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
        baseline=json.loads(Path(config['baseline']).read_text())
        if baseline['initialPhysicalState']!=initial:
            raise RuntimeError('goal STOP diagnostic changed actual initial qpos/qvel/scene')
        teacher=LatchedStopGeometryController(env)
        furniture=next(g for g in scene['geoms'] if g.get('entityId')=='navigation-furniture')
        trace={'version':1,'sizeConvention':'mujoco','geometryScope':'complete-world','dt':.02,
            'task':'walk','controller':'privileged geometry diagnostic with ordinary latched goal STOP',
            'scene':scene,'gravity':body.model.opt.gravity.tolist(),'physicalFootstepPrior':True,
            'motionReference':'cached T2M BVH','initialPhysicalState':initial,
            'navigation':{'mode':'obstacle','split':'heldout','sceneSeed':seed,'physicsSeed':seed+2_000_000,
                'evaluationRole':'development-diagnostic','privilegedTeacherUsed':True,
                'motorCheckpointSha256':_sha(config['motor']),'teacherContract':GOAL_STOP_TEACHER_CONTRACT,
                'rootPositionOverwrittenDuringStep':False,'heldoutDataUsedForTraining':False},
            'frames':[env.snapshot()]}
        trace.update(trace_body_metadata(body))
        done=False;controls=substeps=0;decisions=[];measurements=[]
        while not done:
            action=teacher.action()
            decisions.append({'timeSeconds':float(body.data.time),'action':action.tolist(),
                'actualGoalDistanceMeters':float(np.linalg.norm(body.goal[:2]-body.data.qpos[:2])),
                'stopLatched':teacher.stop_latched})
            _,_,terminated,truncated,info=env.step(action)
            count,native=info['command_control_steps'],info['command_physics_substeps']
            if type(count) is not int or native!=count*10:raise RuntimeError('goal STOP lost native physics accounting')
            append_control_frames(trace,env.last_control_frames,expected_count=count,control_dt=.02)
            measurements.extend(_measurement(f,furniture) for f in env.last_control_frames)
            controls+=count;substeps+=native;done=terminated or truncated
        env.assert_frozen_motor(check_checkpoint=True)
        quality=audit_navigation(trace);trace['navigationQuality']=quality
        final=trace['frames'][-1]['stats'];metrics=final['navigation']['metrics']
        if substeps!=final['command_motor']['observedPhysicsSubstepsThisEpisode']:
            raise RuntimeError('goal STOP native solver counter mismatch')
        output.mkdir(parents=True)
        dump_json(output/'trace.json',trace)
        dump_json(output/'measurements.json',{'decisions':decisions,'native20msMeasurements':measurements})
        result={'sceneSeed':seed,'physicsSeed':seed+2_000_000,'motor':config['name'],
            'motorCheckpointSha256':_sha(config['motor']),'initialPhysicalState':initial,
            'matchedInitialSceneFullQposQvelExact':True,'teacherContract':GOAL_STOP_TEACHER_CONTRACT,
            'baselineQuality':baseline['quality'],'quality':quality,'seconds':float(body.data.time),
            'physicalControls':controls,'physical2msSubsteps':substeps,
            'stopLatched':teacher.stop_latched,'stopTriggerTimeSeconds':teacher.stop_trigger_time,
            'finalGoalDistanceMeters':final['goal_distance'],'contactGeometryIds':metrics['obstacle_contact_geom_ids'],
            'attribution':summarize_measurements(measurements,metrics['first_collision_time_seconds']),
            'heldoutDataUsedForTraining':False,'traceSha256':_sha(output/'trace.json')}
        dump_json(output/'episode.json',result)
        print(json.dumps({'motor':config['name'],'sceneSeed':seed,'success':quality['navigationVerified'],
            'baselineSuccess':baseline['quality']['navigationVerified'],'stopTriggered':teacher.stop_latched,
            'contactGeoms':result['contactGeometryIds']}),flush=True)
        return result
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-motor',required=True);parser.add_argument('--cycle-motor',required=True)
    parser.add_argument('--avatar-profile',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise ValueError('preserve goal STOP experiment output')
    source_names=('goal_stop_attribution.py','contact_attribution.py','tracked_visual_navigation.py',
        'motor_tracking_training.py','motor_cycle_tracking_training.py')
    source_hashes={name:_sha(Path(__file__).parent/name) for name in source_names}
    jobs=[]
    for name,motor,baseline_root in [('5aa',args.original_motor,Path('.runtime/world-model/vision-navigation/contact-r5-diagnostic')),
                                    ('v5',args.cycle_motor,Path('.runtime/world-model/vision-navigation/contact-r5-cycle-tracking-final'))]:
        for seed in (1180100,1180102,1180104,1180106):
            baseline=(baseline_root/f'geometry-{seed}'/'episode.json') if name=='5aa' else (baseline_root/str(seed)/f'geometry-{seed}'/'episode.json')
            jobs.append({'name':name,'motor':motor,'profile':args.avatar_profile,'sceneSeed':seed,
                'baseline':str(baseline),'output':str(output/f'{name}-{seed}')})
    results=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(evaluate_one,job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            dump_json(output/'comparison-progress.json',{'episodes':results,'sourceSha256':source_hashes})
    for name,sha in source_hashes.items():
        if _sha(Path(__file__).parent/name)!=sha:raise RuntimeError('goal STOP runtime source changed')
    archive=output/'source';archive.mkdir(exist_ok=True)
    for name in source_names:(archive/name).write_bytes((Path(__file__).parent/name).read_bytes())
    dump_json(output/'comparison.json',{'complete':True,'episodes':sorted(results,key=lambda e:(e['motor'],e['sceneSeed'])),
        'sourceSha256':source_hashes,'teacherContract':GOAL_STOP_TEACHER_CONTRACT,
        'counts':{name:{'before':sum(e['baselineQuality']['navigationVerified'] for e in results if e['motor']==name),
                        'after':sum(e['quality']['navigationVerified'] for e in results if e['motor']==name)} for name in ('5aa','v5')},
        'heldoutDataUsedForTraining':False,'noAdditionalPpoTraining':True})


if __name__=='__main__':main()
