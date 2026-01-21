from typing import TypedDict,List
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()


llm=ChatOpenAI(model="gpt-4o-mini",temperature=0,api_key=os.getenv("OPENAI_API_KEY"))
parser=StrOutputParser()

class StudentState(TypedDict):
    past_attempts: str  
    student_summary:str
    generated_problem:str
    student_attempt:str
    improvement_analysis:str
    practice_problems:List[str]

async def generate_summary(state:StudentState)->StudentState:
    chain = (
        ChatPromptTemplate.from_template(
            """
            You are an educational psychologist.
            Given the student's past performance and responses below,
            summarize the key struggles, misconceptions, and strengths
            in 4-5 sentences.

            Past Attempts / Logs:
            {attempts}
            """
        )
        | llm
        | parser
    )
    state["student_summary"] = await chain.ainvoke({"attempts": state["past_attempts"]})
    return state

async def generate_problem(state: StudentState) -> StudentState:
    chain = (
        ChatPromptTemplate.from_template(
            """
            The student previously struggled with:
            {summary}

            Generate one math problem of the SAME type and difficulty
            to test if the student has improved.
            dont include just the problem and output should be like this in json format
            {{"problem": ""}}

            """
        )
        | llm
        | parser
    )
    state["generated_problem"] = await chain.ainvoke({"summary": state["student_summary"]})
    return state

async def simulate_student(state: StudentState) -> StudentState:
    chain = (
        ChatPromptTemplate.from_template(
            """
            Simulate how this same student would attempt the new problem. what are students thoughts while solving this new problem based on
            old learnings
            Include reasoning steps and a final answer.

            Student Profile Summary:
            {summary}

            Problem:
            {problem}

            Output strictly in this JSON format (no code block):
{{"thoughts": "<student's reasoning>", "steps": ["step1", "step2", ...], "final_answer": "<answer>"}}
            """
        )
        | llm
        | parser
    )
    state["student_attempt"] = await chain.ainvoke({
        "summary": state["student_summary"],
        "problem": state["generated_problem"]
    })
    return state

async def analyze_improvement(state: StudentState) -> StudentState:
    chain = (
        ChatPromptTemplate.from_template(
            """
            Compare the student's previous weaknesses and their latest attempt.

            Weaknesses Summary:
            {summary}

            Latest Attempt:
            {attempt}

            Write a concise analysis (4-6 sentences) highlighting:
            - Conceptual improvements
            - Remaining issues
            - Emotional/behavioral changes if any
            """
        )
        | llm
        | parser
    )
    state["improvement_analysis"] = await chain.ainvoke({
        "summary": state["student_summary"],
        "attempt": state["student_attempt"]
    })
    return state

async def generate_practice_problems(state: StudentState) -> StudentState:
    chain = (
        ChatPromptTemplate.from_template(
            """
            The student showed the following improvement:
            {improvement}

            Generate 3 practice problems of similar type and difficulty
            to reinforce learning. Format as a numbered list.
            """
        )
        | llm
        | parser
    )
    text = await chain.ainvoke({"improvement": state["improvement_analysis"]})
    state["practice_problems"] = [
        line.strip() for line in text.split("\n") if line.strip()
    ]
    return state

def build_student_learning_graph():
    graph=StateGraph(StudentState)
    graph.add_node("generate summary",generate_summary)
    graph.add_node("generate problem",generate_problem)
    graph.add_node("simulate student",simulate_student)
    graph.add_node("analyze improvement",analyze_improvement)
    graph.add_node("generate practice problems",generate_practice_problems)

    graph.set_entry_point("generate summary")

    graph.add_edge("generate summary","generate problem")
    graph.add_edge("generate problem","simulate student")
    graph.add_edge("simulate student","analyze improvement")
    graph.add_edge("analyze improvement","generate practice problems")

    return graph.compile()

def improvement_graph():
    graph=build_student_learning_graph()
    return graph
