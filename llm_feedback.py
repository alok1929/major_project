import json
import os
import sys
import openai
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in .env file or as an environment variable.")
client = AsyncOpenAI(api_key=api_key)


def get_exercise_feedback(json_filepath):
    """
    Get personalized exercise feedback from GPT based on exercise tracking data.
    
    Args:
        json_filepath: Path to the JSON file containing exercise data
    
    Returns:
        str: Formatted feedback from GPT
    """
    
    with open(json_filepath, "r") as f:
        exercise_data = json.load(f)

    system_prompt = """You are a friendly and knowledgeable physical therapy assistant specializing in rehabilitation exercises.

Your role is to:
- Analyze exercise performance data and provide encouraging, actionable feedback
- Use simple, patient-friendly language (avoid overly technical jargon)
- Be supportive while being honest about areas needing improvement
- Prioritize safety - if form is consistently poor, recommend consulting their therapist

EXERCISE CONTEXT:
- Exercise: Shoulder Raise (arm elevation for shoulder rehabilitation)
- Movement: Patient raises arm from resting position (at side) to overhead, then lowers back down
- Phases:
  * Ascent = raising the arm up
  * Peak = arm fully extended overhead
  * Descent = lowering the arm back down
- Target: 5 reps per arm, right arm first, then left arm

GRADING SYSTEM:
- Excellent (90-100% correct): Near-perfect form throughout the rep
- Good (70-89% correct): Minor deviations, still acceptable form
- Needs Improvement (50-69% correct): Noticeable issues that should be corrected
- Poor (below 50% correct): Significant form problems, potential injury risk

KEY METRICS TO CONSIDER:
- correct_reps_count / incorrect_reps_count: How many reps had good vs poor form
- form_ratio: Percentage of reps with acceptable form (Excellent + Good)
- peak_angle: Higher is better (ideal: 160-180°), indicates full range of motion
- peak_angle_consistency: Lower is better (in degrees), indicates controlled movement
- phase_correctness: 1.0 = perfect, lower values indicate issues in that phase
- fatigue_indicator: Shows if quality declined over reps (consistent/improved/slight_decline/significant_decline)
- common_error_phase: The movement phase where most mistakes occurred
- error_severity: none/mild/moderate/severe for each phase"""

    user_prompt = f"""Please analyze this exercise session and provide personalized feedback for the patient:
```json
{json.dumps(exercise_data, indent=2)}
```

Provide your feedback in this exact format:

## 📊 Session Summary
(Brief overview: how many reps completed out of target, overall form rating, whether exercise was completed)

## ✅ What You Did Well
(2-3 specific positive points based on the actual data - mention specific metrics like peak angles, consistency, or phases that were strong. Be encouraging!)

## ⚠️ Areas to Improve
(2-3 specific issues found in the data. Reference actual numbers like phase correctness scores, error phases, or specific rep issues. Be constructive, not critical. If form was mostly good, acknowledge that while noting minor improvements.)

## 💡 Tips for Next Session
(2-3 actionable tips they can apply immediately. Make these specific to their identified issues - e.g., if descent was the problem, give tips for controlled lowering.)

## 🎯 Focus Point
(ONE single, clear thing to focus on in their next session. Keep it simple and memorable - something they can repeat to themselves while exercising.)

Keep the total response between 250-400 words. Use a warm, encouraging tone throughout - remember this is for rehabilitation patients who may be frustrated or in pain."""

    try:
        response = openai.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )

        feedback = response.choices[0].message.content
        return feedback
    
    except openai.APIError as e:
        return f"❌ OpenAI API Error: {str(e)}"
    except Exception as e:
        return f"❌ Error generating feedback: {str(e)}"


def get_exercise_feedback_with_history(json_filepath, previous_sessions=None):
    """
    Get feedback that considers previous session history for progress tracking.
    
    Args:
        json_filepath: Path to current session JSON
        previous_sessions: List of previous session summaries (optional)
    
    Returns:
        str: Formatted feedback with progress comparison
    """
    
    with open(json_filepath, "r") as f:
        exercise_data = json.load(f)

    system_prompt = """You are a friendly physical therapy assistant specializing in rehabilitation.

Your role is to:
- Analyze exercise performance and provide encouraging, actionable feedback
- Compare with previous sessions to highlight progress or concerns
- Use simple, patient-friendly language
- Prioritize safety - recommend consulting therapist if needed

EXERCISE: Shoulder Raise (arm from rest to overhead and back)
PHASES: Ascent (raising), Peak (full extension), Descent (lowering)
TARGET: 5 reps per arm

GRADING: Excellent (90%+), Good (70-89%), Needs Improvement (50-69%), Poor (<50%)"""

    history_context = ""
    if previous_sessions:
        history_context = f"\n\nPREVIOUS SESSIONS:\n{json.dumps(previous_sessions, indent=2)}"
    else:
        history_context = "\n\n(This is the patient's first tracked session)"

    user_prompt = f"""Analyze this exercise session and provide personalized feedback:

CURRENT SESSION:
```json
{json.dumps(exercise_data, indent=2)}
```
{history_context}

Provide feedback in this format:

## 📊 Session Summary
(Reps completed, overall rating, completion status)

## 📈 Progress Check
(Compare to previous sessions if available, or welcome them to their first tracked session)

## ✅ What You Did Well
(2-3 specific positives with data references)

## ⚠️ Areas to Improve
(2-3 specific issues - be constructive)

## 💡 Tips for Next Session
(2-3 actionable tips specific to their issues)

## 🎯 Focus Point
(ONE main thing to focus on next time)

Keep response 300-450 words. Be warm and encouraging."""

    try:
        response = openai.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=900,
            temperature=0.7
        )

        feedback = response.choices[0].message.content
        return feedback
    
    except openai.APIError as e:
        return f"❌ OpenAI API Error: {str(e)}"
    except Exception as e:
        return f"❌ Error generating feedback: {str(e)}"


def print_feedback(feedback):
    """Pretty print the feedback"""
    print("\n" + "="*70)
    print("🤖 AI EXERCISE FEEDBACK")
    print("="*70)
    print(feedback)
    print("="*70 + "\n")


async def get_exercise_feedback_async(exercise_data_dict, person_id=None, output_dir="."):
    """
    Async version: Get personalized exercise feedback from GPT based on exercise data dict.
    
    Args:
        exercise_data_dict: Dictionary containing exercise data (not file path)
        person_id: Optional person ID for file naming
        output_dir: Directory to save JSON output
    
    Returns:
        dict: Contains 'feedback_text' (str) and 'feedback_json' (dict) with structured data
    """
    
    system_prompt = """You are a friendly and knowledgeable physical therapy assistant specializing in rehabilitation exercises.

Your role is to:
- Analyze exercise performance data and provide encouraging, actionable feedback
- Use simple, patient-friendly language (avoid overly technical jargon)
- Be supportive while being honest about areas needing improvement
- Prioritize safety - if form is consistently poor, recommend consulting their therapist

EXERCISE CONTEXT:
- Exercise: Shoulder Raise (arm elevation for shoulder rehabilitation)
- Movement: Patient raises arm from resting position (at side) to overhead, then lowers back down
- Phases:
  * Ascent = raising the arm up
  * Peak = arm fully extended overhead
  * Descent = lowering the arm back down
- Target: 5 reps per arm, right arm first, then left arm

GRADING SYSTEM:
- Excellent (90-100% correct): Near-perfect form throughout the rep
- Good (70-89% correct): Minor deviations, still acceptable form
- Needs Improvement (50-69% correct): Noticeable issues that should be corrected
- Poor (below 50% correct): Significant form problems, potential injury risk

KEY METRICS TO CONSIDER:
- correct_reps_count / incorrect_reps_count: How many reps had good vs poor form
- form_ratio: Percentage of reps with acceptable form (Excellent + Good)
- peak_angle: Higher is better (ideal: 160-180°), indicates full range of motion
- peak_angle_consistency: Lower is better (in degrees), indicates controlled movement
- phase_correctness: 1.0 = perfect, lower values indicate issues in that phase
- fatigue_indicator: Shows if quality declined over reps (consistent/improved/slight_decline/significant_decline)
- common_error_phase: The movement phase where most mistakes occurred
- error_severity: none/mild/moderate/severe for each phase"""

    user_prompt = f"""Please analyze this exercise session and provide personalized feedback for the patient:
```json
{json.dumps(exercise_data_dict, indent=2)}
```

Provide your feedback in this exact format:

## 📊 Session Summary
(Brief overview: how many reps completed out of target, overall form rating, whether exercise was completed)

## ✅ What You Did Well
(2-3 specific positive points based on the actual data - mention specific metrics like peak angles, consistency, or phases that were strong. Be encouraging!)

## ⚠️ Areas to Improve
(2-3 specific issues found in the data. Reference actual numbers like phase correctness scores, error phases, or specific rep issues. Be constructive, not critical. If form was mostly good, acknowledge that while noting minor improvements.)

## 💡 Tips for Next Session
(2-3 actionable tips they can apply immediately. Make these specific to their identified issues - e.g., if descent was the problem, give tips for controlled lowering.)

## 🎯 Focus Point
(ONE single, clear thing to focus on in their next session. Keep it simple and memorable - something they can repeat to themselves while exercising.)

Keep the total response between 250-400 words. Use a warm, encouraging tone throughout - remember this is for rehabilitation patients who may be frustrated or in pain."""

    try:
        response = await client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )

        feedback_text = response.choices[0].message.content
        
        # Parse feedback into structured JSON
        feedback_json = {
            "person_id": person_id,
            "feedback_text": feedback_text,
            "session_summary": extract_section(feedback_text, "Session Summary"),
            "what_went_well": extract_section(feedback_text, "What You Did Well"),
            "areas_to_improve": extract_section(feedback_text, "Areas to Improve"),
            "tips_for_next_session": extract_section(feedback_text, "Tips for Next Session"),
            "focus_point": extract_section(feedback_text, "Focus Point"),
            "raw_feedback": feedback_text
        }
        
        # Save JSON file
        if person_id is not None:
            json_filename = os.path.join(output_dir, f"person_{person_id}_feedback.json")
        else:
            json_filename = os.path.join(output_dir, "exercise_feedback.json")
        
        with open(json_filename, 'w') as f:
            json.dump(feedback_json, f, indent=2)
        
        print(f"✅ LLM feedback saved to: {json_filename}")
        
        return {
            'feedback_text': feedback_text,
            'feedback_json': feedback_json,
            'json_file': json_filename
        }
    
    except Exception as e:
        error_msg = f"❌ Error generating feedback: {str(e)}"
        print(error_msg)
        return {
            'feedback_text': error_msg,
            'feedback_json': {'error': str(e)},
            'json_file': None
        }


def extract_section(text, section_name):
    """Extract a section from markdown-formatted feedback text"""
    import re
    # Look for ## Section Name followed by content until next ##
    pattern = rf"##\s*{re.escape(section_name)}\s*\n(.*?)(?=\n##|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


# ============== MAIN ==============
if __name__ == "__main__":
    # Default file path
    json_file = "exercise_data.json"
    
    # Allow command line argument for file path
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(json_file):
        print(f"❌ File not found: {json_file}")
        print("Usage: python llm_feedback.py [exercise_data.json]")
        sys.exit(1)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    print(f"📂 Loading exercise data from: {json_file}")
    
    feedback = get_exercise_feedback(json_file)
    
    # Print feedback
    print_feedback(feedback)
    
    # Optionally save feedback to file
    feedback_file = json_file.replace(".json", "_feedback.txt")
    with open(feedback_file, "w") as f:
        f.write(feedback)
    print(f"✅ Feedback saved to: {feedback_file}")