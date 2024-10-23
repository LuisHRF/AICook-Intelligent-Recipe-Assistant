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
    return pd.read_parquet(recipes_cleaned_path)

def combine_ingredients(recipes):
    """
    Combina los ingredientes limpios en una sola cadena.
    """
    combined = []
    for ingredients in recipes['ingredients_cleaned']:
        combined.append(' '.join(ingredients))  
    return combined

def generate_ingredient_embeddings_parallel(recipes, num_workers=4):
    """
    Generate embeddings for recipe ingredients in parallel, focusing only on cleaned ingredients.
    Show progress using tqdm.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Combine the ingredients into a single string for each recipe
    recipes['combined_ingredients'] = combine_ingredients(recipes)

    # Function to encode a single recipe's ingredients
    def encode_ingredients(ingredients):
        return model.encode(ingredients).astype(np.float32).tolist()

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Map the function to the ingredients and show progress with tqdm
        embeddings = list(tqdm(executor.map(encode_ingredients, recipes['combined_ingredients']), total=len(recipes)))
    
    # Add the embeddings to the DataFrame
    recipes['ingredient_embeddings'] = embeddings
    
    return recipes
    
def save_new_embeddings_data(recipes):
    """
    Save the updated recipes data with new embeddings.
    Ensure embeddings are in list format for parquet compatibility.
    """
    # Convert any numpy arrays in 'ingredient_embeddings' to lists
    recipes['ingredient_embeddings'] = recipes['ingredient_embeddings'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    # Save to parquet
    recipes[['RecipeId', 'Name', 'ingredients_cleaned', 'RecipeInstructions', 'ingredient_embeddings']].to_parquet(embeddings_output_path, index=False, engine='pyarrow')
    print("New embeddings data saved successfully.")

async def update_metadata_in_pinecone_async(index, recipes, batch_size=1000, namespace="recipes"):
    """
    Update Pinecone with new vectors based on ingredients asynchronously and include metadata like name, instructions, ingredients, and total_time.
    """
    total_recipes = len(recipes)

    for i in range(0, total_recipes, batch_size):
        batch = recipes.iloc[i:i+batch_size]
        upserts = []

        for idx, row in batch.iterrows():
            # Use the embedding vector (already in list format)
            ingredient_vector = row['ingredient_embeddings']

            # Extract and format ingredients
            ingredients = row["RecipeIngredientParts"]
            if isinstance(ingredients, np.ndarray):
                ingredients = ingredients.tolist()  # Convert to list if it's an ndarray
            if isinstance(ingredients, list):
                ingredients = ", ".join(ingredients)  # Convert list to comma-separated string

            # Extract and convert instructions to string
            instructions = row.get("RecipeInstructions", "")
            if isinstance(instructions, np.ndarray):  # Convert ndarray to string if needed
                instructions = " ".join(instructions)
            elif not isinstance(instructions, str):
                instructions = str(instructions)

            # Prepare metadata update (keeping existing name, ingredients, total_time, and adding instructions)
            metadata = {
                "name": row["Name"],
                "ingredients": ingredients,  # Ensure it's a string
                "instructions": instructions,  # Add the instructions to the metadata
                "total_time": row.get("TotalTimeMinutes", 0)  # Handle missing total_time
            }

            upserts.append({
                "id": str(row['RecipeId']),
                "values": ingredient_vector,  # Use the new ingredient vector
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
    #model = SentenceTransformer('all-MiniLM-L6-v2')
    recipes = generate_ingredient_embeddings_parallel(recipes)

    # Save the new embeddings data
    save_new_embeddings_data(recipes)

    # Update Pinecone with new vectors and metadata
    asyncio.run(update_metadata_in_pinecone_async(index, recipes, batch_size=1000))


