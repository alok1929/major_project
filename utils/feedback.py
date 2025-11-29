from llm_feedback import get_exercise_feedback_async
from utils.analysis import calculate_arm_statistics, get_overall_assessment
from utils.serialization import convert_to_serializable
from utils.rep_tracking import REPS_PER_ARM


async def trigger_llm_feedback(person_id: int, session, websocket, final: bool = False):
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
        
        # Get structured JSON feedback (already parsed from LLM JSON response)
        feedback_json = result.get('feedback_json', {})
        
        # Clean up feedback_json - remove person_id and raw_feedback if present
        # (person_id is already in the message, raw_feedback not needed for frontend)
        clean_feedback_json = {
            'session_summary': feedback_json.get('session_summary', ''),
            'what_went_well': feedback_json.get('what_went_well', ''),
            'areas_to_improve': feedback_json.get('areas_to_improve', ''),
            'tips_for_next_session': feedback_json.get('tips_for_next_session', ''),
            'focus_point': feedback_json.get('focus_point', '')
        }
        
        feedback_msg = {
            'type': 'llm_feedback',
            'person_id': person_id,
            'is_final': final,
            'feedback_type': 'final' if final else 'periodic',
            'rep_count': total_reps,
            'feedback': result.get('feedback_text', 'No feedback available'),  # Formatted text for fallback
            'feedback_json': clean_feedback_json  # Clean structured JSON for frontend
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


def build_session_summary(session):
    """Build final session summary"""
    from utils.analysis import calculate_arm_statistics, get_overall_assessment
    
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

