import asyncio
from utils.analysis import (
    get_rep_grade, analyze_rep_errors, get_error_severity, get_likely_issue
)

# ============== CONFIGURATION ==============
UP_THRESHOLD = 140
DOWN_THRESHOLD = 70
REPS_PER_ARM = 5
LLM_FEEDBACK_EVERY_N_REPS = 2


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


async def update_rep_tracking_async(person_id, angle, is_correct, all_angles, session: SessionState, websocket):
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
                from utils.feedback import trigger_llm_feedback
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
                    from utils.feedback import trigger_llm_feedback
                    asyncio.create_task(
                        trigger_llm_feedback(person_id, session, websocket, final=True)
                    )
    
    return rep_completed

