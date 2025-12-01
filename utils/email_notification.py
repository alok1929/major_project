import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# ============== EMAIL CONFIGURATION ==============
SENDER_EMAIL = "alokhegde221@gmail.com"
DOCTOR_EMAIL = "hegdealok0@gmail.com"
EMAIL_PASSWORD = "axjs hhcf qdhe etsr"  # App password


def format_progress_email(session, person_id, llm_feedback_json=None):
    """Format the progress email body with session summary and LLM feedback"""
    
    state = session.person_states.get(person_id)
    if not state:
        return "No progress data available."
    
    # Get summary data
    from utils.analysis import calculate_arm_statistics, get_overall_assessment
    
    right_stats = calculate_arm_statistics(state['right_arm_reps'], state['right_arm_grades'])
    left_stats = calculate_arm_statistics(state['left_arm_reps'], state['left_arm_grades'])
    assessment = get_overall_assessment(right_stats, left_stats, state['exercise_complete'])
    
    # Calculate totals
    total_reps = sum(state['right_arm_grades'].values()) + sum(state['left_arm_grades'].values())
    right_reps = sum(state['right_arm_grades'].values())
    left_reps = sum(state['left_arm_grades'].values())
    
    # Build email body with correct keys from statistics functions
    email_body = f"""Patient Exercise Progress Report

Exercise: Shoulder Raise
Total Reps Completed: {total_reps}
Right Arm Reps: {right_reps}
Left Arm Reps: {left_reps}

--- RIGHT ARM STATISTICS ---
Total Reps: {right_stats.get('total_reps', 0)}
Form Quality: {right_stats.get('form_ratio', 0.0):.1%}
Quality Score: {right_stats.get('quality_score', 0.0):.2f}
Average Peak Angle: {right_stats.get('avg_peak_angle', 0.0):.1f}°
Average Range of Motion: {right_stats.get('avg_range_of_motion', 0.0):.1f}°

Grade Distribution:
  - Excellent: {state['right_arm_grades'].get('Excellent', 0)}
  - Good: {state['right_arm_grades'].get('Good', 0)}
  - Needs Improvement: {state['right_arm_grades'].get('Needs Improvement', 0)}
  - Poor: {state['right_arm_grades'].get('Poor', 0)}

--- LEFT ARM STATISTICS ---
Total Reps: {left_stats.get('total_reps', 0)}
Form Quality: {left_stats.get('form_ratio', 0.0):.1%}
Quality Score: {left_stats.get('quality_score', 0.0):.2f}
Average Peak Angle: {left_stats.get('avg_peak_angle', 0.0):.1f}°
Average Range of Motion: {left_stats.get('avg_range_of_motion', 0.0):.1f}°

Grade Distribution:
  - Excellent: {state['left_arm_grades'].get('Excellent', 0)}
  - Good: {state['left_arm_grades'].get('Good', 0)}
  - Needs Improvement: {state['left_arm_grades'].get('Needs Improvement', 0)}
  - Poor: {state['left_arm_grades'].get('Poor', 0)}

--- OVERALL ASSESSMENT ---
Overall Form Rating: {assessment.get('overall_form_rating', 'N/A')}
Overall Form Ratio: {assessment.get('overall_form_ratio', 0.0):.1%}
Completion Status: {assessment.get('exercise_completion_status', 'unknown').replace('_', ' ').title()}
Primary Recommendation: {assessment.get('primary_recommendation', 'N/A')}
"""
    
    # Add LLM feedback section if available
    if llm_feedback_json:
        email_body += f"""

--- AI-GENERATED FEEDBACK ---

📊 Session Summary:
{llm_feedback_json.get('session_summary', 'N/A')}

✅ What Went Well:
{llm_feedback_json.get('what_went_well', 'N/A')}

⚠️ Areas to Improve:
{llm_feedback_json.get('areas_to_improve', 'N/A')}

💡 Tips for Next Session:
{llm_feedback_json.get('tips_for_next_session', 'N/A')}

🎯 Focus Point:
{llm_feedback_json.get('focus_point', 'N/A')}
"""
    
    email_body += "\nThis is an automated progress report from the Exercise Tracker system."
    
    return email_body


async def send_progress_email(session, person_id):
    """Send progress email to doctor with LLM feedback"""
    
    try:
        # Generate LLM feedback first
        from llm_feedback import get_exercise_feedback_async
        from utils.analysis import calculate_arm_statistics, get_overall_assessment
        from utils.serialization import convert_to_serializable
        from utils.rep_tracking import REPS_PER_ARM
        
        state = session.person_states.get(person_id)
        if not state:
            print("❌ No state found for person_id")
            return False
        
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
        
        # Get LLM feedback
        print("🤖 Generating LLM feedback for email...")
        llm_result = await get_exercise_feedback_async(
            person_data,
            person_id=person_id,
        )
        llm_feedback_json = llm_result.get('feedback_json', {})
        
        # Create message
        message = MIMEMultipart()
        message["From"] = SENDER_EMAIL
        message["To"] = DOCTOR_EMAIL
        
        # Calculate total reps for subject
        total_reps = 0
        if state:
            total_reps = sum(state.get('right_arm_grades', {}).values()) + sum(state.get('left_arm_grades', {}).values())
        
        message["Subject"] = f"Patient Exercise Progress Report - {total_reps} reps completed"
        
        # Format email body with LLM feedback
        body = format_progress_email(session, person_id, llm_feedback_json)
        message.attach(MIMEText(body, "plain"))
        
        # Send email
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()  # Enable TLS encryption
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.sendmail(SENDER_EMAIL, DOCTOR_EMAIL, message.as_string())
        
        print(f"✅ Progress email with LLM feedback sent successfully to doctor!")
        return True
        
    except Exception as e:
        print(f"❌ Error sending progress email: {e}")
        import traceback
        traceback.print_exc()
        return False

