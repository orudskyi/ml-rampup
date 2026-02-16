import os

from dotenv import load_dotenv
from google import genai

# Load environment variables from .env
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("API Key not found. Check your .env file.")

client = genai.Client(api_key=api_key)

# Using 'flash' model as it is fast and cost-effective
MODEL_ID = "gemini-2.5-flash-lite"


def count_tokens(text: str) -> int:
    """
    Calculates tokens using the new client.models.count_tokens method.
    """
    response = client.models.count_tokens(model=MODEL_ID, contents=text)
    return response.total_tokens


def get_chat_response(prompt: str):
    """
    Generates content using the new client.models.generate_content method.
    """
    response = client.models.generate_content(model=MODEL_ID, contents=prompt)
    return response


def main():
    user_prompt = (
        "Explain RAG (Retrieval-Augmented Generation) in one sentence to a beginner."
    )

    print(f"--- Analysis for model: {MODEL_ID} (New SDK) ---")

    # 1. Estimate tokens
    estimated_tokens = count_tokens(user_prompt)
    print(f"🧮 Estimated tokens: {estimated_tokens}")

    # 2. Generate response
    print("⏳ Sending request to Gemini...")
    response = get_chat_response(user_prompt)

    # Extract text
    answer_text = response.text

    # Extract usage metadata
    usage = response.usage_metadata

    print("\n🤖 Model Response:")
    print(f"> {answer_text}")

    print("\n📊 Real Usage Statistics:")
    print(f"   - Input Tokens (Prompt): {usage.prompt_token_count}")
    print(f"   - Output Tokens (Response): {usage.candidates_token_count}")
    print(f"   - Total Tokens: {usage.total_token_count}")

    # Validation
    if estimated_tokens == usage.prompt_token_count:
        print("\n✅ Success! Estimated count matches actual usage.")
    else:
        diff = usage.prompt_token_count - estimated_tokens
        print(f"\n⚠️ Mismatch detected: {diff} tokens difference.")


if __name__ == "__main__":
    main()
