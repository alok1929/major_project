# ============== CONFIGURATION ==============
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

