import numpy as np
import pandas as pd
import joblib
import asyncio
from ultralytics import YOLO
from utils.utils import get_joint_angles, get_person_x_position, get_detection_confidence
from utils.rep_tracking import SessionState, initialize_person_state, update_rep_tracking_async

# ============== LOAD MODELS ==============
model = YOLO("yolov8n-pose.pt")
clf = joblib.load("models/model_ex1.pkl")

print("🔍 Model expects these features in this order:")
if hasattr(clf, 'feature_names_in_'):
    print(clf.feature_names_in_)
    expected_features = clf.feature_names_in_
else:
    print("Model doesn't have feature names stored")
    expected_features = ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder', 
                        'left_hip', 'right_hip', 'left_knee', 'right_knee']

# ============== CONFIGURATION ==============
CONFIDENCE_THRESHOLD = 0.5


# ============== FRAME PROCESSING ==============
async def process_frame(frame: np.ndarray, session: SessionState, websocket):
    """Process a single frame and send results via WebSocket"""
    
    session.frame_count += 1
    
    # Run YOLO pose estimation
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, lambda: model(frame, verbose=False))
    
    frame_results = {
        'type': 'frame_result',
        'frame_count': session.frame_count,
        'persons': []
    }
    
    if results[0].keypoints is not None:
        all_keypoints = results[0].keypoints.data.cpu().numpy()
        num_people = len(all_keypoints)
        
        if num_people > 0:
            # Sort people by x-position
            people_with_positions = []
            for i in range(num_people):
                keypoints = all_keypoints[i]
                
                avg_confidence = get_detection_confidence(keypoints)
                if avg_confidence < CONFIDENCE_THRESHOLD:
                    continue
                
                x_pos = get_person_x_position(keypoints)
                people_with_positions.append((x_pos, i, keypoints))
            
            people_with_positions.sort(key=lambda x: x[0])
            
            # Process each person
            for person_id, (x_pos, original_idx, keypoints) in enumerate(people_with_positions):
                angles = get_joint_angles(keypoints)
                
                if len(angles) >= 4:
                    if person_id not in session.person_states:
                        session.person_states[person_id] = initialize_person_state()
                    
                    state = session.person_states[person_id]
                    
                    # Get tracking angle based on current arm
                    if state['current_arm'] == 'right':
                        tracking_angle = angles.get('right_shoulder', 0)
                    else:
                        tracking_angle = angles.get('left_shoulder', 0)
                    
                    # Run classifier
                    features_dict = {feature: angles.get(feature, 0) for feature in expected_features}
                    features = pd.DataFrame([features_dict])
                    
                    pred, pred_proba = await loop.run_in_executor(
                        None,
                        lambda: (clf.predict(features)[0], clf.predict_proba(features)[0])
                    )
                    is_correct = pred == 1
                    
                    # Update rep tracking
                    rep_completed = await update_rep_tracking_async(
                        person_id, tracking_angle, is_correct, angles, 
                        session, websocket
                    )
                    
                    # Build person result
                    person_result = {
                        'person_id': person_id,
                        'is_correct': bool(is_correct),
                        'confidence': float(max(pred_proba)),
                        'current_angle': float(tracking_angle),
                        'state': state['state'],
                        'current_arm': state['current_arm'],
                        'rep_count': state['rep_count'],
                        'exercise_complete': state['exercise_complete'],
                        'grades': {
                            'right': state['right_arm_grades'].copy(),
                            'left': state['left_arm_grades'].copy()
                        }
                    }
                    
                    frame_results['persons'].append(person_result)
    
    # Send frame results
    await websocket.send_json(frame_results)

