import cv2
import numpy as np
import pandas as pd
import joblib
import json
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from utils.utils import get_joint_angles, calculate_angle
from llm_feedback import get_exercise_feedback_async

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

# ============== FASTAPI APP ==============
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== CONFIGURATION ==============
UP_THRESHOLD = 140
DOWN_THRESHOLD = 70
REPS_PER_ARM = 5
CONFIDENCE_THRESHOLD = 0.5
LLM_FEEDBACK_EVERY_N_REPS = 2

GRADE_EXCELLENT_MIN = 0.90
GRADE_GOOD_MIN = 0.70
GRADE_NEEDS_IMPROVEMENT_MIN = 0.50


# ============== HELPER FUNCTIONS ==============
def get_rep_grade(correct_ratio):
    if correct_ratio >= GRADE_EXCELLENT_MIN:
        return "Excellent"
    elif correct_ratio >= GRADE_GOOD_MIN:
        return "Good"
    elif correct_ratio >= GRADE_NEEDS_IMPROVEMENT_MIN:
        return "Needs Improvement"
    else:
        return "Poor"


def get_error_severity(phase_correctness):
    if phase_correctness >= 0.90:
        return "none"
    elif phase_correctness >= 0.70:
        return "mild"
    elif phase_correctness >= 0.50:
        return "moderate"
    else:
        return "severe"


def get_likely_issue(error_phases, phase_correctness):
    if not error_phases:
        return None
    
    issues = []
    for phase in error_phases:
        severity = get_error_severity(phase_correctness.get(phase, 1.0))
        if phase == "ascent":
            issues.append(f"{severity} form breakdown while raising arm")
        elif phase == "peak":
            issues.append(f"{severity} instability at full extension")
        elif phase == "descent":
            issues.append(f"{severity} form breakdown while lowering arm")
    
    return "; ".join(issues) if issues else None


def get_person_x_position(keypoints):
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
    left_shoulder_conf = keypoints[5][2]
    right_shoulder_conf = keypoints[6][2]
    left_hip_conf = keypoints[11][2]
    right_hip_conf = keypoints[12][2]
    
    return (left_shoulder_conf + right_shoulder_conf + left_hip_conf + right_hip_conf) / 4


def analyze_rep_errors(angle_details, correctness):
    if not angle_details or not correctness:
        return {
            'ascent_correctness': 1.0,
            'peak_correctness': 1.0,
            'descent_correctness': 1.0,
            'error_phases': [],
            'phase_correctness': {}
        }
    
    total_frames = len(correctness)
    if total_frames == 0:
        return {
            'ascent_correctness': 1.0,
            'peak_correctness': 1.0,
            'descent_correctness': 1.0,
            'error_phases': [],
            'phase_correctness': {}
        }
    
    third = max(1, total_frames // 3)
    
    ascent_correct = correctness[:third]
    peak_correct = correctness[third:2*third]
    descent_correct = correctness[2*third:]
    
    ascent_ratio = sum(ascent_correct) / len(ascent_correct) if ascent_correct else 1.0
    peak_ratio = sum(peak_correct) / len(peak_correct) if peak_correct else 1.0
    descent_ratio = sum(descent_correct) / len(descent_correct) if descent_correct else 1.0
    
    error_phases = []
    if ascent_ratio < 0.7:
        error_phases.append("ascent")
    if peak_ratio < 0.7:
        error_phases.append("peak")
    if descent_ratio < 0.7:
        error_phases.append("descent")
    
    return {
        'ascent_correctness': ascent_ratio,
        'peak_correctness': peak_ratio,
        'descent_correctness': descent_ratio,
        'error_phases': error_phases,
        'phase_correctness': {
            'ascent': ascent_ratio,
            'peak': peak_ratio,
            'descent': descent_ratio
        }
    }


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return convert_to_serializable(obj.tolist())
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


def calculate_arm_statistics(reps, grades):
    if not reps:
        return {
            'total_reps': 0,
            'correct_reps_count': 0,
            'incorrect_reps_count': 0,
            'form_ratio': 0.0,
            'quality_score': 0.0,
            'avg_peak_angle': 0.0,
            'peak_angle_consistency': 0.0,
            'avg_range_of_motion': 0.0,
            'common_error_phase': None,
            'fatigue_indicator': None,
            'grades_breakdown': grades
        }
    
    correct_reps = grades['Excellent'] + grades['Good']
    incorrect_reps = grades['Needs Improvement'] + grades['Poor']
    total_reps = correct_reps + incorrect_reps
    
    form_ratio = correct_reps / total_reps if total_reps > 0 else 0.0
    
    weights = {'Excellent': 1.0, 'Good': 0.8, 'Needs Improvement': 0.5, 'Poor': 0.2}
    weighted_sum = sum(grades[g] * weights[g] for g in grades)
    quality_score = weighted_sum / total_reps if total_reps > 0 else 0.0
    
    peak_angles = [rep['peak_angle'] for rep in reps]
    avg_peak_angle = sum(peak_angles) / len(peak_angles)
    
    if len(peak_angles) > 1:
        variance = sum((a - avg_peak_angle) ** 2 for a in peak_angles) / len(peak_angles)
        peak_angle_consistency = variance ** 0.5
    else:
        peak_angle_consistency = 0.0
    
    ranges = [rep['range_of_motion'] for rep in reps]
    avg_range_of_motion = sum(ranges) / len(ranges)
    
    error_phase_counts = {'ascent': 0, 'peak': 0, 'descent': 0}
    for rep in reps:
        for phase in rep.get('error_phase', []):
            error_phase_counts[phase] += 1
    
    if max(error_phase_counts.values()) > 0:
        common_error_phase = max(error_phase_counts, key=error_phase_counts.get)
    else:
        common_error_phase = None
    
    first_rep_quality = reps[0]['correct_ratio']
    last_rep_quality = reps[-1]['correct_ratio']
    quality_change = last_rep_quality - first_rep_quality
    
    if quality_change < -0.15:
        fatigue_indicator = "significant_decline"
    elif quality_change < -0.05:
        fatigue_indicator = "slight_decline"
    elif quality_change > 0.05:
        fatigue_indicator = "improved"
    else:
        fatigue_indicator = "consistent"
    
    return {
        'total_reps': total_reps,
        'correct_reps_count': correct_reps,
        'incorrect_reps_count': incorrect_reps,
        'form_ratio': form_ratio,
        'quality_score': quality_score,
        'avg_peak_angle': avg_peak_angle,
        'peak_angle_consistency': peak_angle_consistency,
        'avg_range_of_motion': avg_range_of_motion,
        'common_error_phase': common_error_phase,
        'fatigue_indicator': fatigue_indicator,
        'grades_breakdown': grades
    }


def get_overall_assessment(right_stats, left_stats, exercise_complete):
    total_correct = right_stats['correct_reps_count'] + left_stats['correct_reps_count']
    total_incorrect = right_stats['incorrect_reps_count'] + left_stats['incorrect_reps_count']
    total_reps = total_correct + total_incorrect
    
    if total_reps == 0:
        return {
            'exercise_completion_status': 'not_started',
            'overall_form_rating': 'N/A',
            'overall_form_ratio': 0.0,
            'primary_recommendation': 'No exercise data recorded'
        }
    
    overall_form_ratio = total_correct / total_reps if total_reps > 0 else 0.0
    
    if overall_form_ratio >= 0.90:
        overall_rating = "Excellent"
    elif overall_form_ratio >= 0.70:
        overall_rating = "Good"
    elif overall_form_ratio >= 0.50:
        overall_rating = "Needs Improvement"
    else:
        overall_rating = "Poor"
    
    if exercise_complete:
        completion_status = "complete"
    elif total_reps > 0:
        completion_status = "partial"
    else:
        completion_status = "not_started"
    
    recommendations = []
    
    common_errors = []
    if right_stats['common_error_phase']:
        common_errors.append(right_stats['common_error_phase'])
    if left_stats['common_error_phase']:
        common_errors.append(left_stats['common_error_phase'])
    
    if 'descent' in common_errors:
        recommendations.append("Focus on controlled lowering of the arm")
    if 'ascent' in common_errors:
        recommendations.append("Work on smooth, controlled arm raising")
    if 'peak' in common_errors:
        recommendations.append("Improve stability at full arm extension")
    
    if not recommendations:
        recommendations.append("Great work! Maintain current form")
    
    return {
        'exercise_completion_status': completion_status,
        'overall_form_rating': overall_rating,
        'overall_form_ratio': overall_form_ratio,
        'total_correct_reps': total_correct,
        'total_incorrect_reps': total_incorrect,
        'total_reps': total_reps,
        'primary_recommendation': recommendations[0],
        'all_recommendations': recommendations
    }


def initialize_person_state():
    return {
        'current_arm': 'right',
        'state': 'down',
        'rep_count': 0,
        'current_rep_angles': [],
        'current_rep_correctness': [],
        'current_rep_angle_details': [],
        'right_arm_reps': [],
        'left_arm_reps': [],
        'exercise_complete': False,
        'right_arm_grades': {
            'Excellent': 0,
            'Good': 0,
            'Needs Improvement': 0,
            'Poor': 0
        },
        'left_arm_grades': {
            'Excellent': 0,
            'Good': 0,
            'Needs Improvement': 0,
            'Poor': 0
        },
        'total_reps_for_feedback': 0  # Track reps for LLM feedback trigger
    }


# ============== SESSION STATE CLASS ==============
class SessionState:
    """Holds state for a single WebSocket session"""
    def __init__(self):
        self.person_states = {}
        self.frame_count = 0
        self.pending_feedback = {}  # person_id -> feedback text
    
    def reset(self):
        self.person_states = {}
        self.frame_count = 0
        self.pending_feedback = {}


# ============== FRAME PROCESSING ==============
async def process_frame(frame: np.ndarray, session: SessionState, websocket: WebSocket):
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


async def update_rep_tracking_async(person_id, angle, is_correct, all_angles, session: SessionState, websocket: WebSocket):
    """Update rep counting and trigger events via WebSocket"""
    
    state = session.person_states[person_id]
    
    if state['exercise_complete']:
        return False
    
    current_state = state['state']
    rep_completed = False
    
    if current_state == 'down':
        if angle > UP_THRESHOLD:
            state['state'] = 'up'
            state['current_rep_angles'].append(angle)
            state['current_rep_correctness'].append(is_correct)
            state['current_rep_angle_details'].append(all_angles.copy())
            
    elif current_state == 'up':
        state['current_rep_angles'].append(angle)
        state['current_rep_correctness'].append(is_correct)
        state['current_rep_angle_details'].append(all_angles.copy())
        
        if angle < DOWN_THRESHOLD:
            state['state'] = 'down'
            state['rep_count'] += 1
            state['total_reps_for_feedback'] += 1
            rep_completed = True
            
            # Calculate metrics
            angles_list = state['current_rep_angles']
            correct_ratio = sum(state['current_rep_correctness']) / len(state['current_rep_correctness']) if state['current_rep_correctness'] else 0
            grade = get_rep_grade(correct_ratio)
            
            error_analysis = analyze_rep_errors(state['current_rep_angle_details'], state['current_rep_correctness'])
            
            rep_data = {
                'rep_number': state['rep_count'],
                'grade': grade,
                'correct_ratio': correct_ratio,
                'starting_angle': angles_list[0] if angles_list else 0,
                'peak_angle': max(angles_list) if angles_list else 0,
                'ending_angle': angles_list[-1] if angles_list else 0,
                'avg_angle': sum(angles_list) / len(angles_list) if angles_list else 0,
                'range_of_motion': max(angles_list) - min(angles_list) if angles_list else 0,
                'error_phase': error_analysis['error_phases'],
                'error_severity': {
                    'ascent': get_error_severity(error_analysis['ascent_correctness']),
                    'peak': get_error_severity(error_analysis['peak_correctness']),
                    'descent': get_error_severity(error_analysis['descent_correctness'])
                },
                'likely_issue': get_likely_issue(error_analysis['error_phases'], error_analysis['phase_correctness']),
                'phase_correctness': {
                    'ascent': error_analysis['ascent_correctness'],
                    'peak': error_analysis['peak_correctness'],
                    'descent': error_analysis['descent_correctness']
                }
            }
            
            # Store rep data
            if state['current_arm'] == 'right':
                state['right_arm_reps'].append(rep_data)
                state['right_arm_grades'][grade] += 1
            else:
                state['left_arm_reps'].append(rep_data)
                state['left_arm_grades'][grade] += 1
            
            # Send rep complete event
            rep_complete_msg = {
                'type': 'rep_complete',
                'person_id': person_id,
                'arm': state['current_arm'],
                'rep_number': state['rep_count'],
                'grade': grade,
                'correct_ratio': float(correct_ratio),
                'peak_angle': float(rep_data['peak_angle']),
                'likely_issue': rep_data['likely_issue']
            }
            await websocket.send_json(rep_complete_msg)
            
            print(f"🏋️ Person {person_id}: {state['current_arm'].upper()} arm Rep {state['rep_count']} - {grade} ({correct_ratio*100:.0f}%)")
            
            # Reset for next rep
            state['current_rep_angles'] = []
            state['current_rep_correctness'] = []
            state['current_rep_angle_details'] = []
            
            # Check for LLM feedback trigger
            if state['total_reps_for_feedback'] % LLM_FEEDBACK_EVERY_N_REPS == 0:
                asyncio.create_task(
                    trigger_llm_feedback(person_id, session, websocket)
                )
            
            # Check if need to switch arms or exercise complete
            if state['rep_count'] >= REPS_PER_ARM:
                if state['current_arm'] == 'right':
                    state['current_arm'] = 'left'
                    state['rep_count'] = 0
                    state['state'] = 'down'
                    
                    arm_switch_msg = {
                        'type': 'arm_switch',
                        'person_id': person_id,
                        'new_arm': 'left'
                    }
                    await websocket.send_json(arm_switch_msg)
                    print(f"✅ Person {person_id}: Right arm complete! Switching to left arm.")
                else:
                    state['exercise_complete'] = True
                    
                    exercise_complete_msg = {
                        'type': 'exercise_complete',
                        'person_id': person_id
                    }
                    await websocket.send_json(exercise_complete_msg)
                    print(f"🎉 Person {person_id}: Exercise complete!")
                    
                    # Trigger final LLM feedback
                    asyncio.create_task(
                        trigger_llm_feedback(person_id, session, websocket, final=True)
                    )
    
    return rep_completed


async def trigger_llm_feedback(person_id: int, session: SessionState, websocket: WebSocket, final: bool = False):
    """Trigger LLM feedback for a person"""
    
    state = session.person_states.get(person_id)
    if not state:
        return
    
    total_reps = sum(state['right_arm_grades'].values()) + sum(state['left_arm_grades'].values())
    if total_reps == 0:
        return
    
    # Build data for LLM
    right_stats = calculate_arm_statistics(state['right_arm_reps'], state['right_arm_grades'])
    left_stats = calculate_arm_statistics(state['left_arm_reps'], state['left_arm_grades'])
    assessment = get_overall_assessment(right_stats, left_stats, state['exercise_complete'])
    
    person_data = {
        'exercise_info': {
            'exercise_name': 'Shoulder Raise',
            'target_reps_per_arm': REPS_PER_ARM,
            'exercise_complete': state['exercise_complete']
        },
        'right_arm': {
            'statistics': convert_to_serializable(right_stats),
            'reps_detail': convert_to_serializable(state['right_arm_reps'])
        },
        'left_arm': {
            'statistics': convert_to_serializable(left_stats),
            'reps_detail': convert_to_serializable(state['left_arm_reps'])
        },
        'overall_assessment': convert_to_serializable(assessment)
    }
    
    print(f"🤖 Triggering {'final ' if final else ''}LLM feedback for Person {person_id}...")
    
    try:
        result = await get_exercise_feedback_async(
            person_data,
            person_id=person_id,
        )
        
        feedback_msg = {
            'type': 'llm_feedback',
            'person_id': person_id,
            'is_final': final,
            'feedback': result.get('feedback_text', 'No feedback available'),
            'feedback_json': result.get('feedback_json', {})
        }
        await websocket.send_json(feedback_msg)
        
    except Exception as e:
        print(f"❌ Error getting LLM feedback: {e}")
        error_msg = {
            'type': 'llm_feedback_error',
            'person_id': person_id,
            'error': str(e)
        }
        await websocket.send_json(error_msg)


def build_session_summary(session: SessionState):
    """Build final session summary"""
    summary = {
        'type': 'session_summary',
        'total_frames': session.frame_count,
        'persons': []
    }
    
    for person_id, state in session.person_states.items():
        total_reps = sum(state['right_arm_grades'].values()) + sum(state['left_arm_grades'].values())
        if total_reps == 0:
            continue
        
        right_stats = calculate_arm_statistics(state['right_arm_reps'], state['right_arm_grades'])
        left_stats = calculate_arm_statistics(state['left_arm_reps'], state['left_arm_grades'])
        assessment = get_overall_assessment(right_stats, left_stats, state['exercise_complete'])
        
        person_summary = {
            'person_id': person_id,
            'exercise_complete': state['exercise_complete'],
            'right_arm': convert_to_serializable(right_stats),
            'left_arm': convert_to_serializable(left_stats),
            'overall_assessment': convert_to_serializable(assessment)
        }
        summary['persons'].append(person_summary)
    
    return summary


# ============== WEBSOCKET ENDPOINT ==============
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connection established")
    
    session = SessionState()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get('type') == 'frame':
                # Decode base64 frame
                frame_data = base64.b64decode(data['data'])
                np_arr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    await process_frame(frame, session, websocket)
                else:
                    await websocket.send_json({'type': 'error', 'message': 'Failed to decode frame'})
            
            elif data.get('type') == 'reset':
                session.reset()
                await websocket.send_json({'type': 'session_reset'})
                print("🔄 Session reset")
            
            elif data.get('type') == 'get_summary':
                summary = build_session_summary(session)
                await websocket.send_json(summary)
            
            elif data.get('type') == 'stop':
                summary = build_session_summary(session)
                await websocket.send_json(summary)
                break
                
    except WebSocketDisconnect:
        print("❌ WebSocket disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        try:
            await websocket.send_json({'type': 'error', 'message': str(e)})
        except:
            pass


# ============== HEALTH CHECK ==============
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Exercise tracker server running"}


# ============== RUN ==============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)