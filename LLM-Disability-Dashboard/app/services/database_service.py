import os
import json
import sqlite3
import aiosqlite
from typing import Dict, Any, List
from pathlib import Path

# Database path
DB_PATH = Path(__file__).parent.parent / "database.sqlite"

async def init_database():
    """Initialize the database and create tables if they don't exist"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                grade TEXT,
                age INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        await db.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                student_info TEXT,
                generated_questions TEXT,
                session_type TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students (id)
            )
        ''')
        
        await db.execute('''
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER,
                responses TEXT,
                teacher_feedback TEXT,
                ai_analysis TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (student_id) REFERENCES students (id)
            )
        ''')
        
        await db.commit()

async def save_user_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save user data to the database
    
    Args:
        data: Dictionary containing studentInfo, generatedQuestions, sessionType
        
    Returns:
        Dict with inserted record ID
    """
    try:
        await init_database()
        
        student_info = data.get("studentInfo", {})
        generated_questions = data.get("generatedQuestions", {})
        session_type = data.get("sessionType", "")
        
        async with aiosqlite.connect(DB_PATH) as db:
            cursor = await db.execute(
                '''INSERT INTO sessions 
                   (student_id, student_info, generated_questions, session_type) 
                   VALUES (?, ?, ?, ?)''',
                (
                    student_info.get("studentId"),
                    json.dumps(student_info),
                    json.dumps(generated_questions),
                    session_type
                )
            )
            await db.commit()
            
            return {"id": cursor.lastrowid}
    except Exception as error:
        print(f"Error saving user data: {str(error)}")
        raise Exception(f"Failed to save user data: {str(error)}")

async def save_feedback(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save feedback data to the database
    
    Args:
        data: Dictionary containing studentId, responses, teacherFeedback, aiAnalysis
        
    Returns:
        Dict with inserted record ID
    """
    try:
        await init_database()
        
        student_id = data.get("studentId")
        responses = data.get("responses", [])
        teacher_feedback = data.get("teacherFeedback", "")
        ai_analysis = data.get("aiAnalysis", {})
        
        async with aiosqlite.connect(DB_PATH) as db:
            cursor = await db.execute(
                '''INSERT INTO responses 
                   (student_id, responses, teacher_feedback, ai_analysis) 
                   VALUES (?, ?, ?, ?)''',
                (
                    student_id,
                    json.dumps(responses),
                    teacher_feedback,
                    json.dumps(ai_analysis)
                )
            )
            await db.commit()
            
            return {"id": cursor.lastrowid}
    except Exception as error:
        print(f"Error saving feedback: {str(error)}")
        raise Exception(f"Failed to save feedback: {str(error)}")

async def get_student_history(student_id: str) -> Dict[str, Any]:
    """
    Get student history from the database
    
    Args:
        student_id: ID of the student
        
    Returns:
        Dict containing student history data
    """
    try:
        await init_database()
        
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = sqlite3.Row
            
            # Get student info
            cursor = await db.execute('SELECT * FROM students WHERE id = ?', (student_id,))
            student = await cursor.fetchone()
            student = dict(student) if student else None
            
            # Get sessions
            cursor = await db.execute(
                'SELECT * FROM sessions WHERE student_id = ? ORDER BY timestamp DESC', 
                (student_id,)
            )
            sessions = await cursor.fetchall()
            sessions = [dict(row) for row in sessions]
            
            # Get responses
            cursor = await db.execute(
                'SELECT * FROM responses WHERE student_id = ? ORDER BY timestamp DESC', 
                (student_id,)
            )
            responses = await cursor.fetchall()
            responses = [dict(row) for row in responses]
            
            # Parse JSON data
            parsed_sessions = []
            for s in sessions:
                session_dict = dict(s)
                session_dict["student_info"] = json.loads(session_dict["student_info"])
                session_dict["generated_questions"] = json.loads(session_dict["generated_questions"])
                parsed_sessions.append(session_dict)
            
            parsed_responses = []
            for r in responses:
                response_dict = dict(r)
                response_dict["responses"] = json.loads(response_dict["responses"])
                response_dict["ai_analysis"] = json.loads(response_dict["ai_analysis"])
                parsed_responses.append(response_dict)
            
            # Compile learning progress
            learning_progress = []
            for r in parsed_responses:
                analysis = r["ai_analysis"]
                progress_item = {
                    "date": r["timestamp"],
                    "strengths": analysis.get("strengths", []),
                    "weaknesses": analysis.get("weaknesses", []),
                    "approaches": [a.get("area") for a in analysis.get("suggestedApproaches", [])]
                }
                learning_progress.append(progress_item)
            
            return {
                "studentInfo": student,
                "sessions": len(sessions),
                "responses": len(responses),
                "learningProgress": learning_progress,
                "recentSessions": parsed_sessions[:5],
                "recentResponses": parsed_responses[:5]
            }
    except Exception as error:
        print(f"Error getting student history: {str(error)}")
        raise Exception(f"Failed to retrieve student history: {str(error)}")

async def create_student(student_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new student in the database
    
    Args:
        student_data: Dictionary containing name, grade, age
        
    Returns:
        Dict with created student info including ID
    """
    try:
        await init_database()
        
        async with aiosqlite.connect(DB_PATH) as db:
            cursor = await db.execute(
                'INSERT INTO students (name, grade, age) VALUES (?, ?, ?)',
                (student_data.get("name"), student_data.get("grade"), student_data.get("age"))
            )
            await db.commit()
            
            return {"id": cursor.lastrowid, **student_data}
    except Exception as error:
        print(f"Error creating student: {str(error)}")
        raise Exception(f"Failed to create student: {str(error)}")