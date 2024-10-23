import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from pinecone import Pinecone, ServerlessSpec
from src.utils.config import PINECONE_API_KEY, PINECONE_ENVIRONMENT

pc = Pinecone(api_key=PINECONE_API_KEY)

# Function to create or connect to the index
def initialize_pinecone_index(index_name="recipe-embeddings", dimension=384, metric="cosine"):
    existing_indexes = pc.list_indexes()
    if index_name in existing_indexes:
        return pc.Index(index_name)
    else:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(
                cloud='aws', 
                region=PINECONE_ENVIRONMENT
            )
        )
        return pc.Index(index_name)

index = initialize_pinecone_index()
