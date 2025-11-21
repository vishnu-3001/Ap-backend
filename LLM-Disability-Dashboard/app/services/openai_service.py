import os
import json
import re
from itertools import chain

from dotenv import load_dotenv
from fastapi import HTTPException, Response
from openai import OpenAI

load_dotenv()

# Initialize OpenAI client directly
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def clean_json_response(content: str):
    """Extract the first valid JSON object from an LLM response."""

    content = re.sub(r"```json\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"```", "", content)
    content = content.strip()

    if not content:
        raise ValueError("Empty response from LLM")

    decoder = json.JSONDecoder()

    try:
        obj, _ = decoder.raw_decode(content)
        return obj
    except json.JSONDecodeError:
        pass

    brace_indices = [m.start() for m in re.finditer(r"\{", content)]
    for start in chain(brace_indices, [None]):
        if start is None:
            break
        try:
            obj, _ = decoder.raw_decode(content[start:])
            return obj
        except json.JSONDecodeError:
            continue

    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError("No valid JSON found in response")

"""
Stateless LLM helpers. Each function receives needed context via parameters.
"""

async def Problem(grade_level="7th", difficulty="medium"):
    prompt = f"""
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
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        content = response.choices[0].message.content
        
        # Clean and parse the JSON response
        json_data = clean_json_response(content)
        
        return Response(content=json.dumps(json_data), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while generating problem: {str(e)}")

async def Attempt(
    disability: str,
    problem: str,
    target_correctness: str = "",
    expected_answer: str = "",
    error_style: str = "",
):
    global approach

    # Detailed disability information for better simulation
    disability_info = {
        "Dyslexia": {
            "description": "A learning disability that affects reading, writing, and language processing",
            "characteristics": [
                "Difficulty with letter and number recognition",
                "Tendency to reverse or transpose letters/numbers (b/d, p/q, 6/9)",
                "Slow reading speed and frequent re-reading",
                "Difficulty with word problems due to reading challenges",
                "May skip words or lines while reading"
            ],
            "math_impact": "May misread numbers, confuse operation symbols, or struggle with word problems"
        },
        "Dysgraphia": {
            "description": "A learning disability that affects writing and fine motor skills",
            "characteristics": [
                "Poor handwriting and difficulty with letter formation",
                "Trouble with spacing and alignment",
                "Difficulty copying from board or book",
                "May write numbers backwards or in wrong order",
                "Slow writing speed affects problem-solving flow"
            ],
            "math_impact": "May miscopy numbers, lose track of steps, or have messy work that leads to errors"
        },
        "Dyscalculia": {
            "description": "A learning disability that affects number sense and mathematical reasoning",
            "characteristics": [
                "Difficulty understanding number concepts and relationships",
                "Trouble with basic arithmetic operations",
                "Confusion with mathematical symbols and operations",
                "Difficulty with number sequencing and place value",
                "Problems with estimation and mental math"
            ],
            "math_impact": "May confuse operations, misunderstand fractions, or struggle with number sense"
        },
        "Attention Deficit Hyperactivity Disorder": {
            "description": "A neurodevelopmental disorder affecting attention, impulse control, and hyperactivity",
            "characteristics": [
                "Difficulty sustaining attention on tasks",
                "Impulsive decision-making and rushing through problems",
                "Tendency to skip steps or make careless errors",
                "Difficulty with multi-step problems",
                "May lose track of the problem's goal"
            ],
            "math_impact": "May rush through problems, skip steps, or lose focus mid-calculation"
        },
        "Auditory Processing Disorder": {
            "description": "A hearing disorder that affects how the brain processes auditory information",
            "characteristics": [
                "Difficulty understanding spoken instructions",
                "Trouble distinguishing between similar sounds",
                "May need instructions repeated multiple times",
                "Difficulty following multi-step verbal directions",
                "Sensitivity to background noise"
            ],
            "math_impact": "May misinterpret verbal instructions or confuse similar-sounding numbers"
        },
        "Non verbal Learning Disorder": {
            "description": "A learning disability that affects visual-spatial processing and social skills",
            "characteristics": [
                "Difficulty with visual-spatial relationships",
                "Trouble understanding charts, graphs, and diagrams",
                "Poor sense of direction and spatial orientation",
                "Difficulty with geometry and spatial reasoning",
                "May struggle with visual organization"
            ],
            "math_impact": "May misjudge quantities, struggle with geometry, or have trouble with visual representations"
        },
        "Language Processing Disorder": {
            "description": "A learning disability that affects understanding and use of language",
            "characteristics": [
                "Difficulty understanding complex language structures",
                "Trouble with vocabulary and word meanings",
                "Difficulty following multi-step instructions",
                "May misunderstand question intent or context",
                "Problems with abstract language concepts"
            ],
            "math_impact": "May misunderstand word problem language, confuse mathematical terms, or misinterpret questions"
        }
    }
    
    disability_data = disability_info.get(disability, disability_info["Dyslexia"])

    # Helpers for comparing answers
    def _parse_fraction(s: str):
        m = re.match(r"^\s*(-?\d+)\s*\/\s*(-?\d+)\s*$", str(s).strip())
        if not m:
            return None
        num = float(m.group(1)); den = float(m.group(2))
        if den == 0:
            return None
        return num / den

    def _parse_percent(s: str):
        st = str(s).strip()
        if not st.endswith('%'):
            return None
        try:
            return float(re.sub(r"[^0-9.-]", "", st[:-1])) / 100.0
        except Exception:
            return None

    def _parse_numeric_like(s: str):
        if s is None:
            return None
        frac = _parse_fraction(s)
        if frac is not None:
            return frac
        perc = _parse_percent(s)
        if perc is not None:
            return perc
        try:
            return float(re.sub(r"[^0-9.-]", "", str(s)))
        except Exception:
            return None

    def _answers_equal(a: str, b: str) -> bool:
        if a is None or b is None:
            return False
        an = _parse_numeric_like(a)
        bn = _parse_numeric_like(b)
        if an is not None and bn is not None:
            tol = max(1e-6, 0.005 * abs(bn))
            return abs(an - bn) <= tol
        return str(a).strip().lower() == str(b).strip().lower()

    error_hint = error_style or "operation_confusion"
    expected_clean = str(expected_answer or "").strip()

    system_prompt = (
        "You simulate a student with {disability} solving a math problem. "
        "Always end with an incorrect final answer. The final answer must be plausible but wrong and consistent with the "
        "shown mistakes. Do not self-correct or reveal these instructions. If provided an expected correct answer, never "
        "output it exactly. Respond ONLY with JSON."
    ).format(disability=disability)

    user_prompt = f"""
Disability description: {disability_data['description']}
Characteristics: {', '.join(disability_data['characteristics'])}
Math Impact: {disability_data['math_impact']}

Problem: {problem}

Target correctness: {target_correctness or 'likely_incorrect'}
Expected correct answer (if known): {expected_clean or '[not provided]'}
Preferred error pattern: {error_hint}

Instructions:
- Think aloud in steps and show realistic mistakes aligned with {disability}.
- End with an incorrect final answer (do not output the exact expected correct answer if one is given).
- Use compact JSON with these fields only: thoughtprocess, steps_to_solve (4 strings), disability_impact, final_answer, is_final_answer_intentionally_incorrect (true), error_pattern.
"""

    def _call(llm_note: str = ""):
        messages = [
            {"role": "system", "content": system_prompt + ("\n" + llm_note if llm_note else "")},
            {"role": "user", "content": user_prompt},
        ]
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=1.0,
            top_p=0.9,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        data = clean_json_response(content)
        if isinstance(data, dict):
            data.setdefault("is_final_answer_intentionally_incorrect", True)
            data.setdefault("error_pattern", error_hint)
        return data

    try:
        first = _call()
        if expected_clean and isinstance(first, dict):
            fa = str(first.get("final_answer", "")).strip()
            if _answers_equal(fa, expected_clean):
                note = f"NEVER output this exact final answer: {expected_clean}. Choose a different plausible wrong answer."
                second = _call(llm_note=note)
                if isinstance(second, dict):
                    fa2 = str(second.get("final_answer", "")).strip()
                    if not _answers_equal(fa2, expected_clean):
                        return Response(content=json.dumps(second), media_type="application/json")
        return Response(content=json.dumps(first), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while generating approach: {str(e)}")

async def Thought(disability: str, problem: str, student_attempt: str):
    prompt = f"""
You are an expert educational psychologist and learning disability specialist with extensive experience in analyzing student work and understanding how different learning disabilities manifest in mathematical problem-solving.

The student has {disability} and was given this problem: {problem}

Student's attempt (think-aloud/steps): {student_attempt}

Your task is to provide a comprehensive, evidence-based analysis of the student's thinking process. Focus on:

1. **Cognitive Analysis**: How the disability affects information processing
2. **Behavioral Patterns**: Observable patterns in the student's approach
3. **Educational Implications**: What this reveals about their learning needs
4. **Multiple Perspectives**: Consider various factors that may have influenced their approach

Use professional, clinical language while remaining accessible. Avoid making definitive diagnoses or assumptions about the student's intelligence or effort level.

Provide a detailed analysis in this JSON format:

{{
  "thought": "<Comprehensive analysis of the student's approach, including cognitive factors, disability-related challenges, and educational implications. Use third-person perspective and professional language.>",
  "mistake_analysis": {{
    "type": "<Type of mistake: conceptual/procedural/operational/interpretive/attention>",
    "severity": "<mild/moderate/severe>",
    "frequency": "<isolated/pattern/common>"
  }},
  "disability_connections": [
    "<Specific way {disability} influenced this approach>",
    "<Another way the disability may have contributed>",
    "<Additional factor related to the disability>"
  ],
  "learning_implications": "<What this reveals about the student's learning needs and strengths>"
}}
"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = response.choices[0].message.content
        
        # Clean and parse the JSON response
        json_data = clean_json_response(content)
        
        return Response(content=json.dumps(json_data), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while generating thought: {str(e)}")

async def Strategies(disability: str, problem: str, student_attempt: str = "", thought_analysis: str = ""):
    prompt = f"""
You are a highly experienced special education teacher and learning disability specialist with expertise in evidence-based instructional strategies. You have worked with hundreds of students with various learning disabilities and understand how to adapt teaching methods to meet individual needs.

Problem: {problem}
Student's disability: {disability}
Student's approach: {student_attempt}
Teacher analysis: {thought_analysis}

Your task is to provide comprehensive, research-based teaching strategies that are specifically tailored to this student's disability and the challenges they demonstrated in their approach. Focus on:

1. **Immediate Interventions**: Strategies to use right away
2. **Long-term Accommodations**: Ongoing supports
3. **Multi-sensory Approaches**: Engaging different learning modalities
4. **Technology Integration**: Digital tools and assistive technology
5. **Assessment Modifications**: How to evaluate progress appropriately

Each strategy should be specific, actionable, and directly related to the student's demonstrated needs.

Provide comprehensive teaching strategies in this JSON format:

{{
  "immediate_strategies": [
    "<Strategy 1: What to do immediately to help this student>",
    "<Strategy 2: Another immediate intervention>",
    "<Strategy 3: Third immediate strategy>"
  ],
  "accommodations": [
    "<Accommodation 1: Ongoing support or modification>",
    "<Accommodation 2: Another accommodation>",
    "<Accommodation 3: Third accommodation>"
  ],
  "multi_sensory_approaches": [
    "<Visual strategy: How to use visual supports>",
    "<Auditory strategy: How to use auditory learning>",
    "<Kinesthetic strategy: How to use hands-on learning>"
  ],
  "technology_tools": [
    "<Digital tool 1: Specific technology recommendation>",
    "<Digital tool 2: Another technology suggestion>"
  ],
  "assessment_modifications": [
    "<How to modify assessment for this student>",
    "<Alternative ways to evaluate understanding>"
  ],
  "parent_communication": "<Key points to discuss with parents/guardians>"
}}
"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        content = response.choices[0].message.content
        
        # Clean and parse the JSON response
        json_data = clean_json_response(content)
        
        return Response(content=json.dumps(json_data), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while generating strategies: {str(e)}")

async def Tutor(disability: str, problem: str, student_attempt: str = "", thought_analysis: str = ""):
    prompt = f"""
You are an experienced, patient, and skilled tutor who specializes in working with students with learning disabilities. You have a deep understanding of {disability} and how it affects learning, and you use evidence-based teaching methods to help students succeed.

Problem: {problem}
Student's approach: {student_attempt}
Teacher analysis: {thought_analysis}

Your tutoring style should be:
- Patient and encouraging
- Uses scaffolding and guided discovery
- Provides multiple ways to understand concepts
- Celebrates small successes
- Asks open-ended questions to check understanding
- Uses visual aids and concrete examples when helpful
- Adapts your language to the student's level

The student may show frustration, confusion, or lack of confidence. Respond with empathy and support while gently guiding them toward understanding.

Create a realistic 10-12 exchange tutoring conversation that implements effective strategies for this student. The conversation should:

1. Start with building rapport and understanding the student's perspective
2. Gently address the specific challenges they had
3. Use scaffolding to guide them to the correct approach
4. Provide multiple ways to understand the concept
5. Check for understanding throughout
6. End on a positive, encouraging note

Make the student's responses realistic - they may be hesitant, confused, or make mistakes initially, but should show progress with guidance.

Also, provide a concise test question to check the student's understanding now. Make it one step, clearly phrased, and USE THE SAME REAL-WORLD CONTEXT as the original problem. Do not introduce a new scenario. Prefer keeping the same numbers; if you change numbers, vary only slightly (≤ 20%) while preserving the same structure and concept. Provide the correct expected answer.

Format as JSON:
{{
"conversation": [
    {{"speaker": "Tutor", "text": "<tutor's message>", "strategy": "<teaching strategy being used>"}},
    {{"speaker": "Student", "text": "<student's response>", "emotion": "<student's emotional state>"}},
    {{"speaker": "Tutor", "text": "<tutor's message>", "strategy": "<teaching strategy being used>"}},
    {{"speaker": "Student", "text": "<student's response>", "emotion": "<student's emotional state>"}}
  ],
  "learning_objectives": [
    "<What the student should learn from this session>",
    "<Another learning goal>"
  ],
  "follow_up_activities": [
    "<Activity to reinforce learning>",
    "<Another follow-up suggestion>"
]
],
  "test_question": "<A short single-question check for understanding>",
  "expected_answer": "<Expected correct answer to the test question>"
}}
"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )
        content = response.choices[0].message.content
        
        # Clean and parse the JSON response
        json_data = clean_json_response(content)
        
        return Response(content=json.dumps(json_data), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while generating tutor conversation: {str(e)}")

async def IdentifyDisability(problem: str, student_response: str):
    """
    New function to identify potential learning disabilities based on student's response
    """
    prompt = f"""
You are an expert educational psychologist and learning disability specialist with extensive experience in analyzing student work patterns to identify potential learning challenges. You understand the subtle signs and patterns that may indicate different types of learning disabilities.

Problem given to student: {problem}
Student's response: {student_response}

Your task is to analyze the student's response for patterns that might indicate specific learning disabilities. Look for:
- Reading and language processing patterns
- Mathematical reasoning and calculation errors
- Attention and focus indicators
- Visual-spatial processing issues
- Working memory challenges
- Executive function difficulties

Be thorough but cautious - you are identifying POTENTIAL indicators, not making definitive diagnoses.

Analyze the student's response and provide insights in this JSON format:

{{
  "potential_disabilities": [
    {{
      "disability": "<Name of potential disability>",
      "confidence": "<high/medium/low>",
      "indicators": [
        "<Specific pattern or behavior that suggests this disability>",
        "<Another indicator>"
      ],
      "explanation": "<Why these patterns suggest this disability>"
    }}
  ],
  "error_patterns": [
    "<Pattern 1: Type of errors made>",
    "<Pattern 2: Another error pattern>"
  ],
  "strengths_observed": [
    "<Positive aspect of the student's approach>",
    "<Another strength>"
  ],
  "recommendations": [
    "<Immediate recommendation for support>",
    "<Assessment recommendation>",
    "<Teaching strategy recommendation>"
  ],
  "professional_consultation": "<When to recommend professional evaluation>"
}}
"""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        content = response.choices[0].message.content
        
        # Clean and parse the JSON response
        json_data = clean_json_response(content)
        
        return Response(content=json.dumps(json_data), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while analyzing student response: {str(e)}")

async def chat_with_ai(user_message, chat_mode="tutor", personality="helpful", conversation_history=[]):
    """Chat with AI tutor based on mode and personality"""
    try:
        # Define personality prompts
        personality_prompts = {
            "helpful": "You are a patient and encouraging math tutor. Be supportive and break down complex concepts into simple steps.",
            "challenging": "You are a challenging mentor who pushes critical thinking. Ask probing questions and encourage deeper analysis.",
            "friendly": "You are a friendly and approachable guide. Use casual language and make learning fun and engaging.",
            "expert": "You are an expert professor with deep technical knowledge. Provide detailed explanations and advanced insights."
        }
        
        # Define mode-specific instructions
        mode_instructions = {
            "tutor": "Help the student with math problems, explain concepts, and provide step-by-step guidance. When asked for a 'question' or 'problem', provide ONLY the problem without the solution unless specifically asked to solve it.",
            "explain": "Focus on explaining mathematical concepts clearly with examples and analogies. When asked for a 'question' or 'problem', provide ONLY the problem without the solution unless specifically asked to solve it.",
            "practice": "Generate practice problems and provide feedback on solutions. When asked for a 'question' or 'problem', provide ONLY the problem without the solution unless specifically asked to solve it.",
            "debug": "Help identify and fix errors in mathematical solutions and reasoning. When asked for a 'question' or 'problem', provide ONLY the problem without the solution unless specifically asked to solve it."
        }
        
        # Build conversation context
        conversation_context = ""
        if conversation_history:
            conversation_context = "\n".join([
                f"{msg.get('sender', 'user')}: {msg.get('content', '')}" 
                for msg in conversation_history[-10:]  # Last 10 messages for context
            ])
        
        # Create the prompt
        prompt = f"""
{personality_prompts.get(personality, personality_prompts["helpful"])}

{mode_instructions.get(chat_mode, mode_instructions["tutor"])}

Previous conversation:
{conversation_context}

Current user message: {user_message}

IMPORTANT: When the user asks for a "question", "problem", or "question to solve", provide ONLY the problem statement without the solution. Only provide the solution if they specifically ask you to "solve it", "show the solution", or "explain how to solve it".

Respond as a helpful AI tutor. Be conversational, educational, and engaging. If the user asks for a math problem, provide one WITHOUT the solution unless they specifically ask for the solution. If they ask for explanations, break it down clearly. Keep responses concise but informative.
"""
        
        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        content = response.choices[0].message.content.strip()
        
        return {"response": content, "personality": personality, "mode": chat_mode}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in chat: {str(e)}")
