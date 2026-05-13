// T-Pose default BVH derived from 100.bvh hierarchy (single zero-rotation frame).
// Bone names/offsets match the AI motion generator's output so that the rest-pose
// rig and animated motions use the same skeleton.
export const DEFAULT_BVH = `HIERARCHY
ROOT Hips
{
    OFFSET 0.000000 0.000000 0.000000
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine
    {
        OFFSET -0.000000 0.033870 -0.000000
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT Spine1
        {
            OFFSET 0.000000 0.059367 0.000000
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT Spine2
            {
                OFFSET 0.000000 0.065741 0.000000
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT Spine3
                {
                    OFFSET -0.000000 0.076895 -0.000000
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    JOINT Spine4
                    {
                        OFFSET 0.000000 0.169320 0.000000
                        CHANNELS 3 Zrotation Xrotation Yrotation
                        JOINT Neck
                        {
                            OFFSET -0.000000 0.107042 0.010358
                            CHANNELS 3 Zrotation Xrotation Yrotation
                            JOINT Neck1
                            {
                                OFFSET -0.000000 0.075553 0.000000
                                CHANNELS 3 Zrotation Xrotation Yrotation
                                JOINT Head
                                {
                                    OFFSET 0.000000 0.077448 0.000000
                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                    End Site
                                    {
                                        OFFSET 0.000000 0.191009 0.000000
                                    }
                                }
                            }
                        }
                        JOINT LeftShoulder
                        {
                            OFFSET 0.038160 0.027166 -0.005603
                            CHANNELS 3 Zrotation Xrotation Yrotation
                            JOINT LeftArm
                            {
                                OFFSET 0.154868 0.000000 0.000000
                                CHANNELS 3 Zrotation Xrotation Yrotation
                                JOINT LeftForeArm
                                {
                                    OFFSET 0.252275 -0.000000 -0.000000
                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                    JOINT LeftHand
                                    {
                                        OFFSET 0.256047 -0.000000 -0.000000
                                        CHANNELS 3 Zrotation Xrotation Yrotation
                                        JOINT LeftHandThumb1
                                        {
                                            OFFSET 0.019960 -0.022682 0.031755
                                            CHANNELS 3 Zrotation Xrotation Yrotation
                                            JOINT LeftHandThumb2
                                            {
                                                OFFSET 0.030147 0.000000 -0.000000
                                                CHANNELS 3 Zrotation Xrotation Yrotation
                                                JOINT LeftHandThumb3
                                                {
                                                    OFFSET 0.025767 -0.000000 -0.000000
                                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                                    End Site
                                                    {
                                                        OFFSET 0.024609 0.000000 0.000000
                                                    }
                                                }
                                            }
                                        }
                                        JOINT LeftHandIndex1
                                        {
                                            OFFSET 0.081657 -0.000000 0.031755
                                            CHANNELS 3 Zrotation Xrotation Yrotation
                                            JOINT LeftHandIndex2
                                            {
                                                OFFSET 0.045365 -0.000000 -0.000000
                                                CHANNELS 3 Zrotation Xrotation Yrotation
                                                JOINT LeftHandIndex3
                                                {
                                                    OFFSET 0.022682 -0.000000 0.000000
                                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                                    End Site
                                                    {
                                                        OFFSET 0.021663 0.000000 0.000000
                                                    }
                                                }
                                            }
                                        }
                                        JOINT LeftHandMiddle1
                                        {
                                            OFFSET 0.081657 0.000000 0.010525
                                            CHANNELS 3 Zrotation Xrotation Yrotation
                                            JOINT LeftHandMiddle2
                                            {
                                                OFFSET 0.049901 -0.000000 -0.000000
                                                CHANNELS 3 Zrotation Xrotation Yrotation
                                                JOINT LeftHandMiddle3
                                                {
                                                    OFFSET 0.027219 -0.000000 -0.000000
                                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                                    End Site
                                                    {
                                                        OFFSET 0.025995 0.000000 0.000000
                                                    }
                                                }
                                            }
                                        }
                                        JOINT LeftHandRing1
                                        {
                                            OFFSET 0.077120 -0.000000 -0.010525
                                            CHANNELS 3 Zrotation Xrotation Yrotation
                                            JOINT LeftHandRing2
                                            {
                                                OFFSET 0.045365 0.000000 -0.000000
                                                CHANNELS 3 Zrotation Xrotation Yrotation
                                                JOINT LeftHandRing3
                                                {
                                                    OFFSET 0.022682 -0.000000 -0.000000
                                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                                    End Site
                                                    {
                                                        OFFSET 0.021663 0.000000 0.000000
                                                    }
                                                }
                                            }
                                        }
                                        JOINT LeftHandPinky1
                                        {
                                            OFFSET 0.072584 -0.000000 -0.031755
                                            CHANNELS 3 Zrotation Xrotation Yrotation
                                            JOINT LeftHandPinky2
                                            {
                                                OFFSET 0.036292 0.000000 0.000000
                                                CHANNELS 3 Zrotation Xrotation Yrotation
                                                JOINT LeftHandPinky3
                                                {
                                                    OFFSET 0.018146 0.000000 0.000000
                                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                                    End Site
                                                    {
                                                        OFFSET 0.017330 0.000000 0.000000
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        JOINT RightShoulder
                        {
                            OFFSET -0.038160 0.027166 -0.005603
                            CHANNELS 3 Zrotation Xrotation Yrotation
                            JOINT RightArm
                            {
                                OFFSET -0.154868 -0.000000 0.000000
                                CHANNELS 3 Zrotation Xrotation Yrotation
                                JOINT RightForeArm
                                {
                                    OFFSET -0.252275 -0.000000 -0.000000
                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                    JOINT RightHand
                                    {
                                        OFFSET -0.256047 -0.000000 0.000000
                                        CHANNELS 3 Zrotation Xrotation Yrotation
                                        JOINT RightHandThumb1
                                        {
                                            OFFSET -0.019960 -0.022682 0.031755
                                            CHANNELS 3 Zrotation Xrotation Yrotation
                                            JOINT RightHandThumb2
                                            {
                                                OFFSET -0.030147 0.000000 -0.000000
                                                CHANNELS 3 Zrotation Xrotation Yrotation
                                                JOINT RightHandThumb3
                                                {
                                                    OFFSET -0.025767 -0.000000 0.000000
                                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                                    End Site
                                                    {
                                                        OFFSET -0.024609 0.000000 0.000000
                                                    }
                                                }
                                            }
                                        }
                                        JOINT RightHandIndex1
                                        {
                                            OFFSET -0.081656 -0.000000 0.031755
                                            CHANNELS 3 Zrotation Xrotation Yrotation
                                            JOINT RightHandIndex2
                                            {
                                                OFFSET -0.045365 -0.000000 -0.000000
                                                CHANNELS 3 Zrotation Xrotation Yrotation
                                                JOINT RightHandIndex3
                                                {
                                                    OFFSET -0.022682 -0.000000 0.000000
                                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                                    End Site
                                                    {
                                                        OFFSET -0.021663 0.000000 0.000000
                                                    }
                                                }
                                            }
                                        }
                                        JOINT RightHandMiddle1
                                        {
                                            OFFSET -0.081656 -0.000000 0.010525
                                            CHANNELS 3 Zrotation Xrotation Yrotation
                                            JOINT RightHandMiddle2
                                            {
                                                OFFSET -0.049901 -0.000000 0.000000
                                                CHANNELS 3 Zrotation Xrotation Yrotation
                                                JOINT RightHandMiddle3
                                                {
                                                    OFFSET -0.027219 -0.000000 0.000000
                                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                                    End Site
                                                    {
                                                        OFFSET -0.025995 0.000000 0.000000
                                                    }
                                                }
                                            }
                                        }
                                        JOINT RightHandRing1
                                        {
                                            OFFSET -0.077120 -0.000000 -0.010525
                                            CHANNELS 3 Zrotation Xrotation Yrotation
                                            JOINT RightHandRing2
                                            {
                                                OFFSET -0.045365 0.000000 -0.000000
                                                CHANNELS 3 Zrotation Xrotation Yrotation
                                                JOINT RightHandRing3
                                                {
                                                    OFFSET -0.022682 -0.000000 -0.000000
                                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                                    End Site
                                                    {
                                                        OFFSET -0.021663 0.000000 0.000000
                                                    }
                                                }
                                            }
                                        }
                                        JOINT RightHandPinky1
                                        {
                                            OFFSET -0.072584 -0.000000 -0.031755
                                            CHANNELS 3 Zrotation Xrotation Yrotation
                                            JOINT RightHandPinky2
                                            {
                                                OFFSET -0.036292 -0.000000 -0.000000
                                                CHANNELS 3 Zrotation Xrotation Yrotation
                                                JOINT RightHandPinky3
                                                {
                                                    OFFSET -0.018146 -0.000000 -0.000000
                                                    CHANNELS 3 Zrotation Xrotation Yrotation
                                                    End Site
                                                    {
                                                        OFFSET -0.017330 0.000000 0.000000
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    JOINT LeftUpLeg
    {
        OFFSET 0.095505 0.000000 0.000000
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftLeg
        {
            OFFSET 0.000000 -0.473204 -0.000000
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftFoot
            {
                OFFSET -0.000000 -0.396560 -0.000000
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT LeftToeBase
                {
                    OFFSET -0.000000 -0.062078 0.143257
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    End Site
                    {
                        OFFSET 0.000000 0.000000 0.038202
                    }
                }
            }
        }
    }
    JOINT RightUpLeg
    {
        OFFSET -0.095505 -0.000000 -0.000000
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT RightLeg
        {
            OFFSET 0.000000 -0.473204 0.000000
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightFoot
            {
                OFFSET 0.000000 -0.396560 -0.000000
                CHANNELS 3 Zrotation Xrotation Yrotation
                JOINT RightToeBase
                {
                    OFFSET -0.000000 -0.062078 0.143257
                    CHANNELS 3 Zrotation Xrotation Yrotation
                    End Site
                    {
                        OFFSET 0.000000 0.000000 0.038202
                    }
                }
            }
        }
    }
}
MOTION
Frames: 1
Frame Time: 0.033333
0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
`
