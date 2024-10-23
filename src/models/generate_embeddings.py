import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import concurrent.futures
import asyncio
from tqdm import tqdm
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from src.utils.config import PINECONE_API_KEY

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to the existing index directly
index = pc.Index("recipe-embeddings")

# Path to the cleaned recipes data
recipes_cleaned_path = "data/processed/recipes_cleaned.parquet"
embeddings_output_path = "data/processed/recipes_with_embeddings.parquet"

def load_cleaned_data():
    """
    Load the cleaned recipes data from parquet file.
    """
    return pd.read_parquet(recipes_cleaned_path).head(100)  # Limit to 100 rows

def generate_ingredient_embeddings_parallel(recipes, model, num_workers=4):
    """
    Generate embeddings for the cleaned ingredients in parallel.
    """
    def encode_ingredients(ingredients):
        return model.encode(ingredients[:100]).astype(np.float32)  # Limit to 100 embeddings

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        embeddings = list(tqdm(executor.map(encode_ingredients, recipes['ingredients_cleaned'].head(100)), total=len(recipes)))
    
    recipes['ingredient_embeddings'] = embeddings
    return recipes

def save_new_embeddings_data(recipes):
    """
    Save the updated recipes data with new embeddings.
    """
    recipes[['RecipeId', 'Name', 'ingredients_cleaned', 'RecipeInstructions', 'ingredient_embeddings']].to_parquet(embeddings_output_path, index=False, engine='pyarrow')
    print("New embeddings data saved successfully.")

async def update_metadata_in_pinecone_async(index, recipes, batch_size=100, namespace="recipes"):
    """
    Update Pinecone with new vectors based on ingredients asynchronously and include metadata like name, instructions, ingredients, and total_time.
    """
    total_recipes = len(recipes)

    for i in range(0, total_recipes, batch_size):
        batch = recipes.iloc[i:i+batch_size]
        upserts = []

        for idx, row in batch.iterrows():
            # Generate new vector based on cleaned ingredients
            ingredient_vector = row['ingredient_embeddings']

            # Extract and format ingredients
            ingredients = row["RecipeIngredientParts"]
            if isinstance(ingredients, np.ndarray):
                ingredients = ingredients.tolist()  
            if isinstance(ingredients, list):
                ingredients = ", ".join(ingredients) 

            # Extract instructions
            instructions = row.get("RecipeInstructions", "")

            # Prepare metadata update (keeping existing name, ingredients, total_time, and adding instructions)
            metadata = {
                "name": row["Name"],
                "ingredients": ingredients,  
                "instructions": instructions,  
                "total_time": row.get("TotalTimeMinutes", 0)  
            }

            upserts.append({
                "id": str(row['RecipeId']),
                "values": ingredient_vector,  
                "metadata": metadata
            })

            # Log each ID being uploaded
            print(f"Subiendo RecipeId: {row['RecipeId']} con metadatos {metadata}")

        # Upsert batch to Pinecone asynchronously
        try:
            await asyncio.to_thread(index.upsert, vectors=upserts, namespace=namespace)
            print(f"Batch {i//batch_size + 1} subido correctamente con {len(upserts)} recetas.")
        except Exception as e:
            print(f"Error subiendo el batch {i//batch_size + 1}: {e}")

    print("Metadata y vectores han sido actualizados en Pinecone")

if __name__ == "__main__":
    # Load the cleaned data
    recipes = load_cleaned_data()

    # Generate embeddings based only on the cleaned ingredients in parallel
    model = SentenceTransformer('all-MiniLM-L6-v2')
    recipes = generate_ingredient_embeddings_parallel(recipes, model)

    # Save the new embeddings data
    save_new_embeddings_data(recipes)

    # Update Pinecone with new vectors and metadata
    asyncio.run(update_metadata_in_pinecone_async(index, recipes, batch_size=100))
