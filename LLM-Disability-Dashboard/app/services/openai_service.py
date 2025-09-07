import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage
from fastapi import HTTPException, Response
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"),temperature=0.5)

# Global variables to store intermediate results
approach = ""
thought = ""
strategies = ""
problem=""

async def Problem():
    global approach, thought, strategies
    prompt_template = PromptTemplate(
        input_variables=[],
        template="""
<system>
You are an educator teaching 7th standard students.
</system>
<task>
Generate a simple, well-structured mathematics word problem suitable for a 7th standard student.
The problem should be self-explanatory and solvable by a student.
Provide the answer and the correct approach to solve the problem.
Format your output as JSON in the following structure:
{{
  "problem": "<Word problem>",
  "answer": "<Answer>",
  "solution": "<Approach to solve the problem>"
}}
</task>
"""
    )
    try:
        chain = prompt_template | llm
        response = await chain.ainvoke({})
        if isinstance(response, AIMessage):
            content = response.content
        else:
            raise HTTPException(status_code=500, detail=f"Unexpected response type from OpenAI {str(response)}")
        # Reset globals for new problem
        approach = ""
        thought = ""
        strategies = ""
        return Response(content=content, media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while generating problem: {str(e)}")

async def Attempt(disability: str, problem: str):
    global approach
    prompt_template = PromptTemplate(
        input_variables=["disability", "problem"],
        template="""
<system>
You are a middle school student who has {disability}. You are trying to solve the following math word problem:

{problem}

Because of your {disability}, you will make a realistic mistake while solving this problem. You believe you are solving it correctly, but your disability affects how you read, write, process, or calculate.

Below are examples of how your disability might affect your reasoning:
- Dyslexia: You may misread numbers or words (e.g., 24 becomes 42, or 4 becomes 6).
- Dysgraphia: You may write numbers incorrectly, miscopy steps, or mix up number placement.
- Dyscalculia: You may confuse operations like addition and multiplication, or misinterpret fractions.
- Attention Deficit Hyperactivity Disorder: You may skip steps, rush, change operations mid-way, or lose track of your goal.
- Auditory Processing Disorder: You may misinterpret instructions or confuse similar-sounding numbers.
- Non-verbal Learning Disorder: You may misjudge visual or spatial relationships, like quantities, layout, or “equal” distribution.
- Language Processing Disorder: You may misunderstand keywords (like “per,” “each,” or “total”) or the intent of the question.

</system>

<task>
Think aloud as you try to solve the problem step-by-step. You must include **at least 3 steps** and simulate a mistake that would naturally result from your disability.

Output your reasoning in this JSON format:

{{
  "thoughtprocess": "<What you're thinking internally — include any confusion or assumptions>",
  "steps_to_solve": [
    "Step 1: <Your first step>",
    "Step 2: <What you do next>",
    "Step 3: <What you do after that>",
    "Final Step: <Your answer — even if it’s wrong>"
  ],
}}
</task>

        """
    )
    try:
        chain = prompt_template | llm
        response = await chain.ainvoke({"disability": disability, "problem": problem})
        if isinstance(response, AIMessage):
            approach = response.content
        else:
            raise HTTPException(status_code=500, detail=f"Unexpected response type from OpenAI {str(response)}")
        return Response(content=approach, media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while generating approach: {str(e)}")

async def Thought(disability: str, problem: str):
    global approach, thought
    prompt_template = PromptTemplate(
        input_variables=["disability", "problem", "approach"],
        template="""
<system>
You are an expert in learning disabilities analyzing student responses. The student has {disability} and has been given the following problem:

{problem}

They attempted the following approach:

{approach}

Your task is to provide a detailed explanation of why the student might have chosen this approach, even if it led to a mistake. You are not confirming a single cause — instead, you are suggesting **multiple plausible cognitive, emotional, or perceptual reasons** that may have contributed to this behavior, especially as they relate to the student's disability.

Do NOT write from the student’s point of view. Write as an outside observer, using third-person analysis. Your output should describe:
- The type of mistake made (e.g., conceptual, procedural, operational, interpretive)
- At least 3 plausible reasons for this mistake, based on how {disability} typically affects math learning
- Avoid asserting only one explanation — explain multiple possible contributing factors

</system>

<task>
Return your explanation in the following JSON format:

{{
  "thought": "<A detailed, third-person analysis explaining why the student might have used this incorrect approach, including multiple possible disability-related contributing factors.>"
}}
</task>

        """
    )
    try:
        chain = prompt_template | llm
        response = await chain.ainvoke({"disability": disability, "problem": problem, "approach": approach})
        if isinstance(response, AIMessage):
            thought = response.content
        else:
            raise HTTPException(status_code=500, detail=f"Unexpected response type from OpenAI {str(response)}")
        return Response(content=thought, media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while generating thought: {str(e)}")

async def Strategies(disability: str, problem: str):
    global approach, thought, strategies
    prompt_template = PromptTemplate(
        input_variables=["disability", "problem", "approach", "thought"],
        template="""
<system>
You are an expert in educational strategies for students with disabilities.
Problem: {problem}
Student's disability: {disability}
Student's approach: {approach}
Student's thought process: {thought}
</system>
<task>
Provide teaching strategies to help this student. Format as JSON:
{{
"strategies":["<list of strategies>"]
}}
</task>
        """
    )
    try:
        chain = prompt_template | llm
        response = await chain.ainvoke({
            "disability": disability,
            "problem": problem,
            "approach": approach,
            "thought": thought
        })
        if isinstance(response, AIMessage):
            strategies = response.content
        else:
            raise HTTPException(status_code=500, detail=f"Unexpected response type from OpenAI {str(response)}")
        return Response(content=strategies, media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while generating strategies: {str(e)}")

async def Tutor(disability: str, problem: str):
    global approach, thought, strategies
    prompt_template = PromptTemplate(
        input_variables=["disability", "problem", "approach", "thought", "strategies"],
        template="""
<system>
You are a tutor helping a student with {disability}.
Problem: {problem}
Student's approach: {approach}
Student's thoughts: {thought}
Available strategies: {strategies}
</system>
<task>
Create a 8-10 exchange conversation implementing one strategy. Format as JSON:
{{
"conversation": [
    {{"speaker":"Tutor", "text":"<text>"}},
    {{"speaker":"Student", "text":"<text>"}}
]
}}
</task>
        """
    )
    try:
        chain = prompt_template | llm
        response = await chain.ainvoke({
            "disability": disability,
            "problem": problem,
            "approach": approach,
            "thought": thought,
            "strategies": strategies
        })
        if isinstance(response, AIMessage):
            content = response.content
        else:
            raise HTTPException(status_code=500, detail=f"Unexpected response type from OpenAI {str(response)}")
        return Response(content=content, media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while generating tutor conversation: {str(e)}")
