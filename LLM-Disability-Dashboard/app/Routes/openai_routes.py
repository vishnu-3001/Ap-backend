from fastapi import APIRouter,HTTPException,Request
from app.services import Problem,Thought,Attempt,Strategies,Tutor,IdentifyDisability
from app.services.consistency_validator import validate_consistency
from app.services.adaptive_difficulty import get_adaptive_difficulty
openai_router=APIRouter()

# @openai_router.get("/generate_conversation")
# async def generateConversation(disability:str):
#     try:
#         response=await Conversation(disability)
#         return response
#     except Exception as e:
#         raise HTTPException(status_code=500,details=str(e))

@openai_router.get("/generate_problem")
async def generateProblem(grade_level: str = "7th", difficulty: str = "medium"):
    try:
        response=await Problem(grade_level, difficulty)
        return response
    except Exception as e:
        raise HTTPException(status_code=500,details=str(e))

@openai_router.post("/generate_thought")
async def generateThought(request:Request):
    try:
        data=await request.json()
        disability=data.get("disability")
        problem=data.get("problem")
        student_attempt=data.get("student_attempt", "")  # Make optional with empty default
        if not disability or not problem:
            raise HTTPException(status_code=400,detail="disability and problem are required")
        response=await Thought(disability,problem,student_attempt)
        return response
    except Exception as e:
        raise HTTPException(status_code=500,details=str(e))

@openai_router.post("/generate_strategies")
async def generateThought(request:Request):
    try:
        data=await request.json()
        disability=data.get("disability")
        problem=data.get("problem")
        student_attempt=data.get("student_attempt", "")
        thought_analysis=data.get("thought_analysis", "")
        if not disability or not problem:
            raise HTTPException(status_code=400,detail="disability and problem are required")
        response=await Strategies(disability,problem,student_attempt,thought_analysis)
        return response
    except Exception as e:
        raise HTTPException(status_code=500,details=str(e))
    
@openai_router.post("/generate_attempt")
async def generateThought(request:Request):
    try:
        data=await request.json()
        disability=data.get("disability")
        problem=data.get("problem")
        if not disability or not problem:
            raise HTTPException(status_code=400,detail="disability and problem are required")
        response=await Attempt(disability,problem)
        return response
    except Exception as e:
        raise HTTPException(status_code=500,details=str(e))

@openai_router.post("/generate_tutor")
async def generateTutor(request:Request):
    try:
        data=await request.json()
        disability=data.get("disability")
        problem=data.get("problem")
        student_attempt=data.get("student_attempt", "")
        thought_analysis=data.get("thought_analysis", "")
        if not disability or not problem:
            raise HTTPException(status_code=400,detail="disability and problem are required")
        response=await Tutor(disability,problem, student_attempt, thought_analysis)
        return response
    except Exception as e:
        raise HTTPException(status_code=500,details=str(e))

@openai_router.post("/identify_disability")
async def identifyDisability(request:Request):
    try:
        data=await request.json()
        problem=data.get("problem")
        student_response=data.get("student_response")
        if not problem or not student_response:
            raise HTTPException(status_code=400,detail="problem and student_response are required")
        response=await IdentifyDisability(problem,student_response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500,details=str(e))

@openai_router.post("/validate_consistency")
async def validateConsistency(request:Request):
    try:
        data=await request.json()
        problem=data.get("problem")
        disability=data.get("disability")
        student_attempt=data.get("student_attempt")
        expected_answer=data.get("expected_answer")
        if not all([problem, disability, student_attempt, expected_answer]):
            raise HTTPException(status_code=400,detail="problem, disability, student_attempt, and expected_answer are required")
        response=await validate_consistency(problem, disability, student_attempt, expected_answer)
        return response
    except Exception as e:
        raise HTTPException(status_code=500,details=str(e))

@openai_router.post("/adaptive_difficulty")
async def adaptiveDifficulty(request:Request):
    try:
        data=await request.json()
        student_history=data.get("student_history", [])
        current_difficulty=data.get("current_difficulty", "medium")
        if not current_difficulty:
            raise HTTPException(status_code=400,detail="current_difficulty is required")
        response=await get_adaptive_difficulty(student_history, current_difficulty)
        return response
    except Exception as e:
        raise HTTPException(status_code=500,details=str(e))

@openai_router.post("/chat")
async def chatWithAI(request:Request):
    try:
        data=await request.json()
        user_message=data.get("message", "")
        chat_mode=data.get("chat_mode", "tutor")
        personality=data.get("personality", "helpful")
        conversation_history=data.get("conversation_history", [])
        
        if not user_message:
            raise HTTPException(status_code=400,detail="message is required")
        
        # Import the chat service
        from app.services.openai_service import chat_with_ai
        response=await chat_with_ai(user_message, chat_mode, personality, conversation_history)
        return response
    except Exception as e:
        raise HTTPException(status_code=500,details=str(e))

