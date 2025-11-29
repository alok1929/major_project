import numpy as np


def calculate_angle(p1, p2, p3):
    """Calculate angle between three points"""
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    v1, v2 = p1 - p2, p3 - p2
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle


def get_joint_angles(keypoints):
    """Extract joint angles from keypoints"""
    angles = {}
    try:
        if all(keypoints[[5, 7, 9], 2] > 0.5):  # left elbow
            angles["left_elbow"] = calculate_angle(keypoints[5,:2], keypoints[7,:2], keypoints[9,:2])
        if all(keypoints[[6, 8, 10], 2] > 0.5):  # right elbow
            angles["right_elbow"] = calculate_angle(keypoints[6,:2], keypoints[8,:2], keypoints[10,:2])
        if all(keypoints[[11, 13, 15], 2] > 0.5):  # left knee
            angles["left_knee"] = calculate_angle(keypoints[11,:2], keypoints[13,:2], keypoints[15,:2])
        if all(keypoints[[12, 14, 16], 2] > 0.5):  # right knee
            angles["right_knee"] = calculate_angle(keypoints[12,:2], keypoints[14,:2], keypoints[16,:2])
        if all(keypoints[[11, 5, 7], 2] > 0.5):  # left shoulder
            angles["left_shoulder"] = calculate_angle(keypoints[11,:2], keypoints[5,:2], keypoints[7,:2])
        if all(keypoints[[12, 6, 8], 2] > 0.5):  # right shoulder
            angles["right_shoulder"] = calculate_angle(keypoints[12,:2], keypoints[6,:2], keypoints[8,:2])
        if all(keypoints[[5, 11, 13], 2] > 0.5):  # left hip
            angles["left_hip"] = calculate_angle(keypoints[5,:2], keypoints[11,:2], keypoints[13,:2])
        if all(keypoints[[6, 12, 14], 2] > 0.5):  # right hip
            angles["right_hip"] = calculate_angle(keypoints[6,:2], keypoints[12,:2], keypoints[14,:2])
    except:
        pass
    return angles


def get_person_x_position(keypoints):
    """Get person's x-position for tracking/sorting"""
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    
    if left_hip[2] > 0.5 and right_hip[2] > 0.5:
        return (left_hip[0] + right_hip[0]) / 2
    elif left_hip[2] > 0.5:
        return left_hip[0]
    elif right_hip[2] > 0.5:
        return right_hip[0]
    else:
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        return (left_shoulder[0] + right_shoulder[0]) / 2


def get_detection_confidence(keypoints):
    """Calculate average confidence from key body points"""
    left_shoulder_conf = keypoints[5][2]
    right_shoulder_conf = keypoints[6][2]
    left_hip_conf = keypoints[11][2]
    right_hip_conf = keypoints[12][2]
    
    return (left_shoulder_conf + right_shoulder_conf + left_hip_conf + right_hip_conf) / 4
