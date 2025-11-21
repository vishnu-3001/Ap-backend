import re
import json
from typing import Dict, List, Tuple, Any, Optional
from fastapi import HTTPException, Response

def validate_response_consistency(problem: str, disability: str, student_attempt: Dict, expected_answer: str) -> Dict[str, Any]:
    """
    Validates the consistency of a student's response across multiple dimensions.
    
    Args:
        problem: The original math problem
        disability: The disability being simulated
        student_attempt: The student's attempt JSON response
        expected_answer: The correct answer to the problem
        
    Returns:
        Dictionary containing validation results and scores
    """
    
    validation_results = {
        "overall_consistency_score": 0.0,
        "checks": {},
        "recommendations": [],
        "flags": []
    }
    
    # Extract student's final answer
    student_answer = extract_final_answer(student_attempt)
    
    # Check 1: Answer consistency with steps
    step_consistency = check_step_answer_consistency(student_attempt, student_answer)
    validation_results["checks"]["step_answer_consistency"] = step_consistency
    
    # Check 2: Disability-specific behavior validation
    disability_validation = validate_disability_behavior(disability, student_attempt, problem)
    validation_results["checks"]["disability_behavior"] = disability_validation
    
    # Check 3: Mathematical reasoning consistency
    math_consistency = validate_mathematical_reasoning(student_attempt, problem, expected_answer)
    validation_results["checks"]["mathematical_reasoning"] = math_consistency
    
    # Check 4: Error pattern consistency
    error_consistency = validate_error_patterns(disability, student_attempt)
    validation_results["checks"]["error_patterns"] = error_consistency
    
    # Check 5: Response completeness
    completeness = validate_response_completeness(student_attempt)
    validation_results["checks"]["completeness"] = completeness
    
    # Calculate overall consistency score
    scores = [check["score"] for check in validation_results["checks"].values()]
    validation_results["overall_consistency_score"] = sum(scores) / len(scores) if scores else 0.0
    
    # Generate recommendations
    validation_results["recommendations"] = generate_recommendations(validation_results["checks"])
    
    # Flag critical issues
    validation_results["flags"] = identify_critical_flags(validation_results["checks"])
    
    return validation_results

def _parse_fraction(s: str) -> Optional[float]:
    m = re.match(r"^\s*(-?\d+)\s*\/\s*(-?\d+)\s*$", str(s).strip())
    if not m:
        return None
    try:
        num = float(m.group(1)); den = float(m.group(2))
        if den == 0:
            return None
        return num / den
    except Exception:
        return None


def _parse_percent(s: str) -> Optional[float]:
    st = str(s).strip()
    if not st.endswith('%'):
        return None
    try:
        return float(re.sub(r"[^0-9.-]", "", st[:-1])) / 100.0
    except Exception:
        return None


def _parse_numeric_like(s: Any) -> Optional[float]:
    if s is None:
        return None
    # direct numeric
    try:
        return float(s)
    except Exception:
        pass
    # fraction
    frac = _parse_fraction(s)
    if frac is not None:
        return frac
    # percent
    perc = _parse_percent(s)
    if perc is not None:
        return perc
    # find last number in text
    nums = re.findall(r'[-+]?[0-9]*\.?[0-9]+', str(s))
    if nums:
        try:
            return float(nums[-1])
        except Exception:
            return None
    return None


def extract_final_answer(student_attempt: Dict) -> str:
    """Extract the final answer from student attempt, returning a numeric-like string when possible."""
    if not student_attempt:
        return ""

    # Try final_answer field first (sanitize to numeric-like if possible)
    if "final_answer" in student_attempt:
        raw = str(student_attempt["final_answer"]).strip()
        num = _parse_numeric_like(raw)
        return str(num) if num is not None else raw

    # Try to extract from steps (last number wins)
    steps = student_attempt.get("steps_to_solve", [])
    if steps and len(steps) > 0:
        last_step = str(steps[-1])
        num = _parse_numeric_like(last_step)
        if num is not None:
            return str(num)

    # Try thoughtprocess
    thoughtprocess = student_attempt.get("thoughtprocess", "")
    if thoughtprocess:
        num = _parse_numeric_like(thoughtprocess)
        if num is not None:
            return str(num)

    return ""

def check_step_answer_consistency(student_attempt: Dict, student_answer: str) -> Dict[str, Any]:
    """Check if the student's final answer matches their step-by-step work."""
    steps = student_attempt.get("steps_to_solve", [])
    if not steps or not student_answer:
        return {
            "score": 0.0,
            "status": "incomplete",
            "details": "Missing steps or final answer"
        }
    # Student final answer numeric value (if possible)
    student_val = _parse_numeric_like(student_answer)
    # Look for numerical answers in steps
    step_answers: List[str] = []
    for step in steps:
        numbers = re.findall(r'[-+]?[0-9]*\.?[0-9]+', str(step))
        if numbers:
            step_answers.extend(numbers)

    final_answer_found = False
    if student_val is not None and step_answers:
        for step_answer in step_answers:
            try:
                if abs(student_val - float(step_answer)) < 0.01:
                    final_answer_found = True
                    break
            except Exception:
                continue

    score = 1.0 if final_answer_found else 0.3
    status = "consistent" if final_answer_found else "inconsistent"
    
    return {
        "score": score,
        "status": status,
        "details": f"Final answer '{student_answer}' {'found' if final_answer_found else 'not found'} in step-by-step work",
        "step_answers": step_answers
    }

def validate_disability_behavior(disability: str, student_attempt: Dict, problem: str) -> Dict[str, Any]:
    """Validate that the student's behavior matches the expected disability characteristics."""
    
    disability_patterns = {
        "Dyslexia": {
            "expected_behaviors": ["confusion", "re-reading", "number reversal", "mixing up", "difficulty reading", "reversed", "transposed", "b/d", "p/q", "6/9"],
            "unexpected_behaviors": ["clear understanding", "no confusion", "perfect reading", "easily understood"]
        },
        "Dyscalculia": {
            "expected_behaviors": ["number confusion", "operation confusion", "calculation errors", "number sense issues", "confused", "mistake", "wrong operation"],
            "unexpected_behaviors": ["perfect calculations", "no number confusion", "clear calculations"]
        },
        "Attention Deficit Hyperactivity Disorder": {
            "expected_behaviors": ["rushing", "skipping steps", "careless errors", "impulsive", "losing focus", "quickly", "fast", "skip"],
            "unexpected_behaviors": ["careful work", "no rushing", "complete steps", "thoroughly"]
        },
        "Dysgraphia": {
            "expected_behaviors": ["handwriting", "writing", "difficulty writing", "messy", "unclear", "backwards"],
            "unexpected_behaviors": ["clear writing", "neat", "perfect handwriting"]
        },
        "Auditory Processing Disorder": {
            "expected_behaviors": ["misunderstood", "confused instructions", "hearing", "listening", "misheard"],
            "unexpected_behaviors": ["clear understanding", "perfect hearing"]
        },
        "Non verbal Learning Disorder": {
            "expected_behaviors": ["visual", "spatial", "diagram", "chart", "graph", "confused"],
            "unexpected_behaviors": ["clear visual understanding", "perfect spatial"]
        },
        "Language Processing Disorder": {
            "expected_behaviors": ["language", "words", "vocabulary", "confused", "misunderstood"],
            "unexpected_behaviors": ["clear language", "perfect understanding"]
        },
        "No disability": {
            "expected_behaviors": ["clear thinking", "logical steps", "careful work", "methodical", "systematic"],
            "unexpected_behaviors": ["excessive confusion", "major errors", "disability-like patterns", "very confused", "completely wrong"]
        }
    }
    
    patterns = disability_patterns.get(disability, disability_patterns["No disability"])
    
    # Analyze the text content
    text_content = " ".join([
        str(student_attempt.get("thoughtprocess", "")),
        " ".join(str(step) for step in student_attempt.get("steps_to_solve", [])),
        str(student_attempt.get("disability_impact", ""))
    ]).lower()
    
    expected_matches = sum(1 for behavior in patterns["expected_behaviors"] if behavior in text_content)
    unexpected_matches = sum(1 for behavior in patterns["unexpected_behaviors"] if behavior in text_content)
    
    # Calculate score based on expected vs unexpected behaviors
    total_expected = len(patterns["expected_behaviors"])
    total_unexpected = len(patterns["unexpected_behaviors"])
    
    # More nuanced scoring
    expected_score = min(1.0, expected_matches / max(1, total_expected * 0.3))  # Need at least 30% of expected behaviors
    unexpected_penalty = min(0.5, unexpected_matches / max(1, total_unexpected * 0.5))  # Penalty for unexpected behaviors
    
    # Base score for having some expected behaviors
    base_score = 0.3 if expected_matches > 0 else 0.1
    
    score = min(1.0, base_score + expected_score - unexpected_penalty)
    
    return {
        "score": score,
        "status": "realistic" if score > 0.5 else "unrealistic",
        "details": f"Found {expected_matches}/{total_expected} expected behaviors, {unexpected_matches}/{total_unexpected} unexpected behaviors",
        "expected_found": [behavior for behavior in patterns["expected_behaviors"] if behavior in text_content],
        "unexpected_found": [behavior for behavior in patterns["unexpected_behaviors"] if behavior in text_content]
    }

def validate_mathematical_reasoning(student_attempt: Dict, problem: str, expected_answer: str) -> Dict[str, Any]:
    """Validate the mathematical reasoning in the student's response."""
    steps = student_attempt.get("steps_to_solve", [])
    student_answer = extract_final_answer(student_attempt)

    if not steps or not student_answer:
        return {
            "score": 0.0,
            "status": "incomplete",
            "details": "Missing mathematical work"
        }
    # Parse numeric-like values
    student_num = _parse_numeric_like(student_answer)
    expected_num = _parse_numeric_like(expected_answer)

    # Check for mathematical operations
    has_operations = any(
        any(op in str(step).lower() for op in ["+", "-", "×", "*", "÷", "/", "=", "equals"])
        for step in steps
    )
    
    # Check for logical progression
    has_progression = len(steps) >= 2
    
    # Check if answer is reasonable (within 50% of expected), only if expected is numeric-like
    if expected_num is not None and student_num is not None:
        try:
            reasonable_answer = abs(student_num - expected_num) / expected_num < 0.5 if expected_num != 0 else True
        except ZeroDivisionError:
            reasonable_answer = True
    else:
        reasonable_answer = False
    
    score = sum([has_operations, has_progression, reasonable_answer]) / 3
    
    return {
        "score": score,
        "status": "good" if score > 0.7 else "needs_improvement",
        "details": f"Operations: {has_operations}, Progression: {has_progression}, Reasonable: {reasonable_answer}",
        "has_operations": has_operations,
        "has_progression": has_progression,
        "reasonable_answer": reasonable_answer
    }

def validate_error_patterns(disability: str, student_attempt: Dict) -> Dict[str, Any]:
    """Validate that error patterns are consistent with the disability."""
    
    error_patterns = {
        "Dyslexia": ["6/9", "b/d", "p/q", "reversed", "transposed"],
        "Dyscalculia": ["operation confusion", "number confusion", "place value"],
        "Attention Deficit Hyperactivity Disorder": ["rushed", "skipped", "careless"],
        "No disability": []
    }
    
    expected_errors = error_patterns.get(disability, [])
    
    text_content = " ".join([
        str(student_attempt.get("thoughtprocess", "")),
        " ".join(str(step) for step in student_attempt.get("steps_to_solve", [])),
        str(student_attempt.get("disability_impact", ""))
    ]).lower()
    
    if disability == "No disability":
        # For no disability, we expect minimal errors
        error_indicators = ["confusion", "mistake", "error", "wrong", "difficult"]
        error_count = sum(1 for indicator in error_indicators if indicator in text_content)
        score = max(0, 1.0 - (error_count * 0.2))
        status = "appropriate" if score > 0.7 else "too_many_errors"
    else:
        # For disabilities, we expect some relevant error patterns
        relevant_errors = sum(1 for pattern in expected_errors if pattern in text_content)
        score = min(1.0, relevant_errors / max(1, len(expected_errors))) if expected_errors else 0.5
        status = "realistic" if score > 0.3 else "unrealistic"
    
    return {
        "score": score,
        "status": status,
        "details": f"Found {relevant_errors if disability != 'No disability' else 'minimal'} error patterns",
        "expected_patterns": expected_errors,
        "found_patterns": [pattern for pattern in expected_errors if pattern in text_content]
    }

def validate_response_completeness(student_attempt: Dict) -> Dict[str, Any]:
    """Validate that the response is complete and well-structured."""
    
    required_fields = ["thoughtprocess", "steps_to_solve", "disability_impact"]
    present_fields = [field for field in required_fields if field in student_attempt and student_attempt[field]]
    
    completeness_score = len(present_fields) / len(required_fields)
    
    # Check if steps have meaningful content
    steps = student_attempt.get("steps_to_solve", [])
    meaningful_steps = sum(1 for step in steps if len(str(step).strip()) > 10)
    
    # Check if thoughtprocess has meaningful content
    thoughtprocess = student_attempt.get("thoughtprocess", "")
    meaningful_thoughts = len(str(thoughtprocess).strip()) > 20
    
    structure_score = (meaningful_steps / max(1, len(steps)) + meaningful_thoughts) / 2
    
    overall_score = (completeness_score + structure_score) / 2
    
    return {
        "score": overall_score,
        "status": "complete" if overall_score > 0.7 else "incomplete",
        "details": f"Fields present: {len(present_fields)}/{len(required_fields)}, Meaningful content: {structure_score:.2f}",
        "present_fields": present_fields,
        "missing_fields": [field for field in required_fields if field not in present_fields]
    }

def generate_recommendations(checks: Dict[str, Dict]) -> List[str]:
    """Generate recommendations based on validation results."""
    
    recommendations = []
    
    for check_name, result in checks.items():
        if result["score"] < 0.5:
            if check_name == "step_answer_consistency":
                recommendations.append("Ensure the final answer matches the step-by-step calculations")
            elif check_name == "disability_behavior":
                recommendations.append("Review disability characteristics to make the simulation more realistic")
            elif check_name == "mathematical_reasoning":
                recommendations.append("Include more detailed mathematical work and logical progression")
            elif check_name == "error_patterns":
                recommendations.append("Add more disability-specific error patterns to the response")
            elif check_name == "completeness":
                recommendations.append("Ensure all required fields are present and meaningful")
    
    return recommendations

def identify_critical_flags(checks: Dict[str, Dict]) -> List[str]:
    """Identify critical issues that need immediate attention."""
    
    flags = []
    
    if checks.get("step_answer_consistency", {}).get("score", 1) < 0.3:
        flags.append("CRITICAL: Final answer doesn't match step-by-step work")
    
    if checks.get("disability_behavior", {}).get("score", 1) < 0.2:
        flags.append("CRITICAL: Response doesn't match expected disability behavior")
    
    if checks.get("completeness", {}).get("score", 1) < 0.3:
        flags.append("CRITICAL: Response is severely incomplete")
    
    return flags

async def validate_consistency(problem: str, disability: str, student_attempt: str, expected_answer: str):
    """Main validation function for the API endpoint."""
    
    try:
        # Parse student attempt if it's a string
        if isinstance(student_attempt, str):
            student_attempt = json.loads(student_attempt)
        
        # Run validation
        results = validate_response_consistency(problem, disability, student_attempt, expected_answer)
        
        return Response(content=json.dumps(results), media_type="application/json")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error validating consistency: {str(e)}")
