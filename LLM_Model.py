import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load variables from .env into the environment
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY not found. Make sure you have a .env file "
        "with OPENROUTER_API_KEY=your-key-here in the same folder."
    )

llm = ChatOpenAI(
    model="openrouter/free",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.6,
)


