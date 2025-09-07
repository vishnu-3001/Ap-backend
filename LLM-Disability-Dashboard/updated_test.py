import os
import asyncio
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Set API key
api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key (first few chars): {api_key[:5]}***" if api_key else "No API key found!")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

def test_openai():
    """Test OpenAI connection with the latest API client"""
    try:
        print("Testing OpenAI connection with the latest client...")
        
        # Use chat completions API with GPT-3.5-turbo
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Generate a simple math question for a 5th grader."}
            ]
        )
        
        # Print the response
        message_content = response.choices[0].message.content
        print(f"Success! Response: {message_content}")
        
        # Get model information
        print(f"Model used: {response.model}")
        print(f"Response created at: {response.created}")
        
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_generate_question():
    """Test the question generation API with a sample student profile"""
    try:
        print("\nTesting question generation...")
        
        # Import the generate_question function
        from services.openai_service import generate_question
        
        # Sample student data
        student_data = {
            "Name": "Alex",
            "Age": "10",
            "Grade": 5,
            "Subject": "Fractions",
            "Given-questions": 3,
            "Correct-answered": 2,
            "Known-disability": False,
            "Given-question": "What is 1/4 + 1/2?",
            "Mistake-made": "Added numerators and denominators directly: 1/4 + 1/2 = 2/6",
            "Time-taken": "2 minutes",
            "Additional-observation": "Student seems to struggle with finding common denominators"
        }
        
        # Generate a question
        result = await generate_question(student_data)
        
        # Print the results
        print("✅ Question generated successfully!")
        print(f"\nGenerated Question: {result['Question']}")
        print(f"\nPotential Mistakes:")
        for mistake in result['Mistakes']:
            print(f"- {mistake}")
        print(f"\nReasons:")
        for reason in result['Reasons']:
            print(f"- {reason}")
        print(f"\nSuggested Approaches:")
        for approach in result['Approaches']:
            print(f"- {approach}")
            
        return True
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test direct OpenAI connection
    if test_openai():
        print("\nBasic OpenAI connection test passed.")
    else:
        print("\nBasic OpenAI connection test failed.")
    
    # Test the question generation function
    if asyncio.run(test_generate_question()):
        print("\nQuestion generation test passed.")
    else:
        print("\nQuestion generation test failed.")