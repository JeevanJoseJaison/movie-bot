import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-12-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

def get_movie_recommendations(preferences: dict) -> str:
    prompt = f"""
You are an intelligent movie recommendation assistant.

User preferences:
{preferences}

Recommend exactly 3 movies.

For each movie provide:
1. Title
2. Year
3. Why it matches the user
4. Whether it is family-friendly or not

Keep the response short and Telegram-friendly.
"""

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You recommend movies clearly and accurately."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )

    return response.choices[0].message.content