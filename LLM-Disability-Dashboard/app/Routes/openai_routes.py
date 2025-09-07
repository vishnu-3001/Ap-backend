from fastapi import APIRouter,HTTPException,Request
from app.services import Problem,Thought,Attempt,Strategies,Tutor
openai_router=APIRouter()

# @openai_router.get("/generate_conversation")
# async def generateConversation(disability:str):
#     try:
#         response=await Conversation(disability)
#         return response
#     except Exception as e:
#         raise HTTPException(status_code=500,details=str(e))

@openai_router.get("/generate_problem")
async def generateProblem():
    try:
        response=await Problem()
        return response
    except Exception as e:
        raise HTTPException(status_code=500,details=str(e))

@openai_router.post("/generate_thought")
async def generateThought(request:Request):
    try:
        data=await request.json()
        disability=data.get("disability")
        problem=data.get("problem")
        if not disability or not problem:
            raise HTTPException(status_code=400,detail="disability and problem are required")
        response=await Thought(disability,problem)
        return response
    except Exception as e:
        raise HTTPException(status_code=500,details=str(e))

@openai_router.post("/generate_strategies")
async def generateThought(request:Request):
    try:
        data=await request.json()
        disability=data.get("disability")
        problem=data.get("problem")
        if not disability or not problem:
            raise HTTPException(status_code=400,detail="disability and problem are required")
        response=await Strategies(disability,problem)
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
async def generateThought(request:Request):
    try:
        data=await request.json()
        disability=data.get("disability")
        problem=data.get("problem")
        if not disability or not problem:
            raise HTTPException(status_code=400,detail="disability and problem are required")
        response=await Tutor(disability,problem)
        return response
    except Exception as e:
        raise HTTPException(status_code=500,details=str(e))

