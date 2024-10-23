import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.getcwd(), '.env')
load_dotenv(dotenv_path)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Debug: Print to verify they are loaded correctly
print(f"PINECONE_API_KEY: {PINECONE_API_KEY}")
print(f"PINECONE_ENVIRONMENT: {PINECONE_ENVIRONMENT}")