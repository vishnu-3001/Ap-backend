import os
import asyncio
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def generate_questions(student_data, num_questions=3):
    """
    Generate multiple personalized questions based on student information
    
    Args:
        student_data: Dictionary containing student information
        num_questions: Number of questions to generate (default: 3)
        
    Returns:
        Dict containing generated questions with potential mistakes, reasons and approaches
    """
    try:
        # Create a prompt based on the student info
        prompt = f"""Based on the following student information, generate {num_questions} personalized math questions:

Student Name: {student_data.get('Name', '')}
Age: {student_data.get('Age', '')}
Grade: {student_data.get('Grade', '')}
Subject: {student_data.get('Subject', 'Mathematics')}
Previous questions given: {student_data.get('Given-questions', 0)}
Previous correct answers: {student_data.get('Correct-answered', 0)}
Known disability: {student_data.get('Known-disability', False)}
Previously given question: {student_data.get('Given-question', '')}
Previous mistakes: {student_data.get('Mistake-made', '')}
Time taken previously: {student_data.get('Time-taken', '')}
Additional observations: {student_data.get('Additional-observation', '')}

Generate a response in JSON format with the following structure:
{{
  "questions": [
    {{
      "question": "The complete question text",
      "mistakes": ["potential mistake 1", "potential mistake 2"],
      "reasons": ["reason for potential mistake 1", "reason for potential mistake 2"],
      "approaches": ["suggested approach 1", "suggested approach 2"]
    }},
    ... (repeat for {num_questions} questions)
  ]
}}
"""
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # You can use gpt-4 if available for better results
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert math educator who creates personalized questions for students based on their profile, learning history, and specific needs."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        content = json.loads(response.choices[0].message.content)
        
        return content
    except Exception as error:
        print(f"Error in OpenAI question generation: {str(error)}")
        raise Exception(f"Failed to generate questions with OpenAI: {str(error)}")

async def get_user_input():
    """Get student information from user input"""
    student_data = {}
    
    print("\n--- Student Information Input ---")
    student_data["Name"] = input("Student Name: ")
    student_data["Age"] = input("Age: ")
    student_data["Grade"] = input("Grade (e.g., 5): ")
    student_data["Subject"] = input("Subject (e.g., Fractions, Decimals): ")
    
    # Optional fields with defaults
    given_q = input("Number of previous questions given [0]: ")
    student_data["Given-questions"] = int(given_q) if given_q else 0
    
    correct_q = input("Number of previous correct answers [0]: ")
    student_data["Correct-answered"] = int(correct_q) if correct_q else 0
    
    disability = input("Known disability (yes/no) [no]: ").lower()
    student_data["Known-disability"] = disability == "yes"
    
    student_data["Given-question"] = input("Last question given to student [none]: ") or ""
    student_data["Mistake-made"] = input("Mistakes made by student [none]: ") or ""
    student_data["Time-taken"] = input("Time taken by student [none]: ") or ""
    student_data["Additional-observation"] = input("Additional observations [none]: ") or ""
    
    return student_data

async def main():
    """Main function to run the enhanced test"""
    print("--------------------------------------")
    
    # Get student data from user input
    student_data = await get_user_input()
    
    print("\nGenerating questions based on the provided information...")
    
    try:
        # Generate questions
        result = await generate_questions(student_data, num_questions=3)
        
        # Display the results in a more readable format
        print("\n=== Generated Questions ===\n")
        
        for i, q in enumerate(result["questions"], 1):
            print(f"Question {i}: {q['question']}")
            
            print("\nPotential Mistakes:")
            for mistake in q["mistakes"]:
                print(f"- {mistake}")
                
            print("\nReasons:")
            for reason in q["reasons"]:
                print(f"- {reason}")
                
            print("\nSuggested Approaches:")
            for approach in q["approaches"]:
                print(f"- {approach}")
                
            print("\n" + "-" * 50 + "\n")
        
        # Save to file option
        save = input("Would you like to save these questions to a file? (yes/no): ").lower()
        if save == "yes":
            filename = input("Enter filename [questions.json]: ") or "questions.json"
            with open(filename, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Questions saved to {filename}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())