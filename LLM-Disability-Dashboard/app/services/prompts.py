"""Centralized prompt templates for LangGraph workflow nodes."""

from typing import Dict, Any


class WorkflowPrompts:
    """Centralized prompt templates for all workflow nodes."""
    
    @staticmethod
    def get_problem_generation_prompt(grade_level: str, difficulty: str) -> str:
        """Generate a math problem prompt tailored to grade level and difficulty."""
        return f"""
You are an expert mathematics educator specializing in creating age-appropriate word problems for students with learning disabilities. You understand the cognitive development stages and can create problems that are challenging yet accessible.

Generate a well-structured mathematics word problem suitable for a {grade_level} grade student with {difficulty} difficulty level. The problem should:

1. Be age-appropriate and engaging
2. Use clear, simple language
3. Include real-world context that students can relate to
4. Have a single, clear solution path
5. Be solvable in 3-5 steps
6. Include numbers that are manageable for the grade level
7. Match the specified difficulty level

For 2nd grade: Focus on basic addition/subtraction, simple counting, basic shapes
For 5th grade: Include fractions, decimals, basic geometry, multi-step problems
For 7th grade: Include algebra basics, ratios, percentages, more complex word problems

Difficulty levels:
- Easy: Simple operations, small numbers, 2-3 steps
- Medium: Moderate complexity, medium numbers, 3-4 steps
- Hard: Complex reasoning, larger numbers, 4-5 steps

CRITICAL: The "answer" field must contain the EXACT final numerical answer that matches the solution steps. Double-check that your answer is consistent with your solution approach.

Format your output as JSON in the following structure:
{{
  "problem": "<Word problem>",
  "answer": "<Final numerical answer - must match solution>",
  "solution": "<Detailed step-by-step approach to solve the problem>",
  "grade_level": "{grade_level}",
  "concepts": ["<list of math concepts covered>"],
  "difficulty": "{difficulty}"
}}
"""

    @staticmethod
    def get_student_attempt_prompt(disability: str, problem: str, target_correctness: str = "", 
                                 expected_answer: str = "", error_style: str = "") -> Dict[str, str]:
        """Generate prompts for simulating student attempts with disability-specific errors."""
        
        disability_data = {
            "Dyslexia": {
                "description": "Difficulty with reading, writing, and processing written information",
                "characteristics": ["letter/number reversals", "difficulty with sequences", "slow processing"],
                "math_impact": "May reverse digits (6→9), struggle with multi-step problems, confuse similar-looking numbers"
            },
            "Dyscalculia": {
                "description": "Difficulty with number sense, calculation, and mathematical reasoning",
                "characteristics": ["poor number sense", "difficulty with basic operations", "trouble with math facts"],
                "math_impact": "May confuse operations (+ vs ×), struggle with place value, have difficulty with mental math"
            },
            "Attention Deficit Hyperactivity Disorder": {
                "description": "Difficulty with attention, focus, and executive functioning",
                "characteristics": ["inattention", "impulsivity", "difficulty with sustained focus"],
                "math_impact": "May skip steps, rush through problems, make careless errors, lose track of multi-step processes"
            },
            "Dysgraphia": {
                "description": "Difficulty with writing and fine motor skills",
                "characteristics": ["poor handwriting", "difficulty with written expression", "motor coordination issues"],
                "math_impact": "May miscopy numbers, struggle with written calculations, have difficulty organizing work on paper"
            },
            "Auditory Processing Disorder": {
                "description": "Difficulty processing and interpreting auditory information",
                "characteristics": ["trouble following verbal instructions", "difficulty with auditory memory"],
                "math_impact": "May mishear numbers in word problems, struggle with verbal math instructions"
            },
            "Non verbal Learning Disorder": {
                "description": "Difficulty with visual-spatial processing and nonverbal reasoning",
                "characteristics": ["poor visual-spatial skills", "difficulty with patterns", "trouble with visual organization"],
                "math_impact": "May struggle with geometry, visual representations, spatial relationships in math"
            },
            "Language Processing Disorder": {
                "description": "Difficulty understanding and using language effectively",
                "characteristics": ["trouble with complex language", "difficulty with abstract concepts"],
                "math_impact": "May misinterpret word problems, struggle with mathematical vocabulary, have trouble with abstract reasoning"
            }
        }
        
        disability_info = disability_data.get(disability, disability_data["Dyslexia"])
        
        system_prompt = f"""
You are simulating how a student with {disability} would approach a math problem. 

{disability_info['description']}
Key characteristics: {', '.join(disability_info['characteristics'])}
Math impact: {disability_info['math_impact']}

Your task is to show realistic thinking and mistakes that align with this disability. Be authentic - show the genuine struggles and thought processes.

Output format: JSON with these exact fields:
- studentAnswer: which is wrong
- thoughtprocess: "Detailed step-by-step thinking showing realistic mistakes"
- steps_to_solve: ["Step 1", "Step 2", "Step 3", "Step 4"] (4 strings showing the approach)
- disability_impact: "How the disability specifically affected this problem"
- final_answer: "The student's final answer (should be incorrect)"
- is_final_answer_intentionally_incorrect: true
- error_pattern: "Type of error made (e.g., 'digit_reversal', 'operation_confusion')"


CRITICAL: The final answer must be incorrect and realistic for this disability. Do not provide the correct answer.
"""
        
        user_prompt = f"""
Problem: {problem}

Target correctness: {target_correctness or 'likely_incorrect'}
Expected correct answer (if known): {expected_answer or '[not provided]'}
Preferred error pattern: {error_style or 'realistic_for_disability'}

Instructions:
- Think aloud in steps and show realistic mistakes aligned with {disability}
- End with an incorrect final answer (do not output the exact expected correct answer)
- Use compact JSON with the specified fields only
- Make the errors authentic to how this disability would manifest in math
"""
        
        return {
            "system": system_prompt,
            "user": user_prompt
        }

    @staticmethod
    def get_thought_analysis_prompt(disability: str, problem: str, attempt_json: str) -> str:
        """Generate prompt for analyzing student thought processes."""
        return f"""
You are an expert educational psychologist specializing in learning disabilities and mathematical cognition. Your task is to analyze a student's attempt at solving a math problem and provide insights into their thinking process.

Problem: {problem}
Student's attempt: {attempt_json}
Disability context: {disability}

Analyze the student's approach and provide insights on:

1. **Cognitive Patterns**: What thinking patterns do you observe?
2. **Error Analysis**: What specific errors were made and why?
3. **Disability Impact**: How did {disability} specifically influence their approach?
4. **Strengths**: What did the student do well or show understanding of?
5. **Areas for Growth**: What concepts need reinforcement?
6. **Emotional State**: What emotions might the student be experiencing?

Format as JSON:
{{
  "cognitive_patterns": "Analysis of thinking approach",
  "error_analysis": "Detailed breakdown of mistakes",
  "disability_impact": "How {disability} affected performance",
  "strengths": "What the student did well",
  "growth_areas": "Concepts needing work",
  "emotional_indicators": "Inferred emotional state",
  "confidence_level": "low/medium/high",
  "recommendations": "Specific next steps for support"
}}
"""

    @staticmethod
    def get_teaching_strategies_prompt(disability: str, problem: str, attempt_json: str, thought_json: str) -> str:
        """Generate prompt for creating teaching strategies."""
        return f"""
You are a master teacher and learning disability specialist with expertise in {disability}. Create targeted teaching strategies based on the student's attempt and analysis.

Problem: {problem}
Student's attempt: {attempt_json}
Thought analysis: {thought_json}

Develop comprehensive teaching strategies that:

1. **Address Specific Challenges**: Target the exact difficulties shown
2. **Leverage Strengths**: Build on what the student does well
3. **Use Evidence-Based Methods**: Apply proven techniques for {disability}
4. **Provide Multiple Pathways**: Offer different ways to understand the concept
5. **Include Scaffolding**: Break down complex concepts into manageable steps
6. **Consider Emotional Support**: Address confidence and motivation

Format as JSON:
{{
  "primary_strategies": [
    {{
      "name": "Strategy name",
      "description": "How to implement",
      "rationale": "Why this works for {disability}",
      "implementation": "Step-by-step instructions"
    }}
  ],
  "alternative_approaches": [
    {{
      "name": "Alternative method",
      "description": "Different way to teach the concept",
      "when_to_use": "When primary strategies don't work"
    }}
  ],
  "scaffolding_sequence": [
    "Step 1: Start with...",
    "Step 2: Then introduce...",
    "Step 3: Gradually add..."
  ],
  "accommodations": [
    "Specific accommodations for {disability}",
    "Tools or resources needed"
  ],
  "assessment_methods": [
    "How to check understanding",
    "Alternative ways to demonstrate learning"
  ]
}}
"""

    @staticmethod
    def get_tutor_session_prompt(disability: str, problem: str, attempt_json: str, thought_json: str) -> str:
        """Generate prompt for creating tutor conversation."""
        return f"""
You are an experienced, patient, and skilled tutor who specializes in working with students with {disability}. You have deep expertise in evidence-based teaching methods and understand how to support students with learning differences.

Problem: {problem}
Student's approach: {attempt_json}
Teacher analysis: {thought_json}

Create a realistic 10-12 exchange tutoring conversation that:

1. **Builds Rapport**: Start with understanding and empathy
2. **Addresses Challenges**: Gently work through specific difficulties
3. **Uses Scaffolding**: Guide step-by-step without giving answers
4. **Provides Multiple Perspectives**: Offer different ways to understand
5. **Checks Understanding**: Regularly assess comprehension
6. **Maintains Encouragement**: Keep the student motivated and confident
7. **Adapts to Disability**: Use techniques specific to {disability}

The student's responses should be realistic - showing initial confusion, gradual understanding, and occasional setbacks, but overall progress with guidance.

Format as JSON:
{{
  "conversation": [
    {{
      "speaker": "Tutor",
      "text": "Tutor's message",
      "strategy": "Teaching strategy being used",
      "purpose": "Why this approach"
    }},
    {{
      "speaker": "Student", 
      "text": "Student's response",
      "emotion": "Student's emotional state",
      "understanding_level": "low/medium/high"
    }}
  ],
  "test_question": {{
    "question": "Follow-up question to check understanding",
    "expected_answer": "Correct answer",
    "context": "Same real-world context as original problem"
  }},
  "session_summary": {{
    "key_breakthroughs": "What the student learned",
    "remaining_challenges": "What still needs work",
    "next_steps": "Recommended follow-up"
  }}
}}
"""

    @staticmethod
    def get_consistency_validation_prompt(problem: str, disability: str, attempt_json: str, expected_answer: str) -> str:
        """Generate prompt for consistency validation."""
        return f"""
You are an expert educational assessor specializing in learning disability evaluation. Analyze the consistency between a student's attempt and the expected solution, considering the context of {disability}.

Problem: {problem}
Student's attempt: {attempt_json}
Expected answer: {expected_answer}
Disability context: {disability}

Evaluate:

1. **Solution Consistency**: How well does the student's approach align with valid solution methods?
2. **Error Patterns**: Are the errors consistent with {disability} characteristics?
3. **Reasoning Quality**: Is the underlying mathematical reasoning sound despite errors?
4. **Disability Alignment**: Do the mistakes match expected patterns for {disability}?
5. **Learning Progress**: What does this attempt reveal about the student's understanding?

Format as JSON:
{{
  "consistency_score": 0.0-1.0,
  "error_analysis": {{
    "primary_errors": ["List of main mistakes"],
    "error_categories": ["Types of errors made"],
    "disability_related": true/false,
    "severity": "low/medium/high"
  }},
  "reasoning_quality": {{
    "logical_steps": "Assessment of reasoning",
    "concept_understanding": "Level of concept grasp",
    "method_appropriateness": "Suitability of approach"
  }},
  "disability_considerations": {{
    "typical_patterns": "Expected patterns for {disability}",
    "atypical_elements": "Unexpected aspects",
    "accommodation_effectiveness": "How well accommodations worked"
  }},
  "recommendations": {{
    "immediate_support": "What to address now",
    "long_term_goals": "Broader learning objectives",
    "strategy_adjustments": "Changes to teaching approach"
  }}
}}
"""

    @staticmethod
    def get_adaptive_difficulty_prompt(history: list, current_difficulty: str) -> str:
        """Generate prompt for adaptive difficulty adjustment."""
        return f"""
You are an expert educational data analyst specializing in adaptive learning systems. Analyze the student's learning history to recommend appropriate difficulty adjustments.

Student History: {history}
Current Difficulty: {current_difficulty}

Analyze patterns in:

1. **Performance Trends**: How has the student's performance changed over time?
2. **Error Patterns**: What types of errors are most common?
3. **Learning Velocity**: How quickly does the student master new concepts?
4. **Engagement Levels**: What difficulty levels maintain optimal engagement?
5. **Struggle Points**: Where does the student consistently struggle?

Recommend adjustments that:

- Maintain appropriate challenge level
- Build on strengths
- Address persistent difficulties
- Keep the student engaged and motivated
- Follow evidence-based progression patterns

Format as JSON:
{{
  "recommended_difficulty": "easy/medium/hard",
  "confidence_level": 0.0-1.0,
  "analysis": {{
    "performance_trend": "improving/stable/declining",
    "mastery_level": "beginner/intermediate/advanced",
    "error_frequency": "high/medium/low",
    "engagement_indicators": "high/medium/low"
  }},
  "reasoning": {{
    "strengths_observed": "What the student does well",
    "challenges_identified": "Areas needing support",
    "learning_patterns": "How the student learns best"
  }},
  "recommendations": {{
    "immediate_adjustments": "Changes to make now",
    "gradual_progression": "How to advance over time",
    "monitoring_points": "What to watch for"
  }},
  "alternative_paths": [
    "Different learning approaches to try",
    "Alternative difficulty progressions"
  ]
}}
"""

    @staticmethod
    def get_disability_identification_prompt(problem: str, student_response: str) -> str:
        """Generate prompt for disability identification."""
        return f"""
You are an expert educational diagnostician specializing in learning disability identification. Analyze a student's response to identify potential learning differences.

Problem: {problem}
Student Response: {student_response}

Look for patterns that might indicate:

1. **Dyslexia**: Letter/number reversals, sequencing difficulties, reading comprehension issues
2. **Dyscalculia**: Number sense problems, operation confusion, calculation difficulties
3. **ADHD**: Attention issues, impulsivity, executive function challenges
4. **Dysgraphia**: Writing difficulties, fine motor issues, organization problems
5. **Auditory Processing**: Difficulty following verbal instructions, memory issues
6. **Non-verbal Learning**: Visual-spatial difficulties, pattern recognition problems
7. **Language Processing**: Vocabulary issues, abstract concept difficulties

Analyze the response for:
- Error patterns
- Processing characteristics
- Cognitive strengths and weaknesses
- Behavioral indicators
- Learning style preferences

Format as JSON:
{{
  "potential_disabilities": [
    {{
      "disability": "Name of potential disability",
      "confidence": 0.0-1.0,
      "indicators": ["Specific signs observed"],
      "severity": "mild/moderate/severe"
    }}
  ],
  "primary_concern": "Most likely disability based on evidence",
  "error_analysis": {{
    "pattern_type": "Type of error pattern",
    "frequency": "How often this pattern appears",
    "consistency": "How consistent the pattern is"
  }},
  "strengths_observed": [
    "Cognitive strengths shown",
    "Learning preferences evident"
  ],
  "recommendations": {{
    "immediate_support": "What to do right away",
    "assessment_needs": "Further evaluation recommended",
    "accommodations": "Supports to implement"
  }},
  "confidence_level": 0.0-1.0,
  "notes": "Additional observations and context"
}}
"""


# Convenience function to get all prompts
def get_workflow_prompts() -> WorkflowPrompts:
    """Get the workflow prompts instance."""
    return WorkflowPrompts()
