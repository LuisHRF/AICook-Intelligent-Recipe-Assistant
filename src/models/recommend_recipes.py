import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from dotenv import load_dotenv
from pinecone import Pinecone
from src.utils.config import PINECONE_API_KEY
from scipy.spatial.distance import cosine

# Load the cleaned recipes
recipes_cleaned_path = "data/processed/recipes_cleaned.parquet"
recipes_cleaned = pd.read_parquet(recipes_cleaned_path)

# Convert RecipeId to string to match Pinecone format
recipes_cleaned['RecipeId'] = recipes_cleaned['RecipeId'].astype(str)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("recipe-embeddings")

# Load the sentence transformer model (same as used for recipe embeddings)
model = SentenceTransformer('all-MiniLM-L6-v2')  # Example model

def vectorize_ingredients(ingredients):
    """
    Generates an embedding for a list of ingredients.
    """
    return model.encode(", ".join(ingredients))  # Joining ingredients for a single embedding

def search_recipes(user_embedding, top_n=15):
    """
    Searches for recipes in Pinecone using the ingredient embeddings.
    """
    try:
        query_response = index.query(
            vector=user_embedding.tolist(),
            top_k=top_n,
            include_metadata=True,
            namespace="recipes"
        )
        return query_response['matches']
    except Exception as e:
        print(f"Error searching for recipes: {e}")
        return []

def fetch_recipe_vector(recipe_id):
    """
    Retrieves the vector and metadata of a specific recipe in Pinecone.
    """
    try:
        recipe_data = index.fetch([recipe_id], namespace="recipes")
        return recipe_data
    except Exception as e:
        print(f"Error retrieving vector for recipe {recipe_id}: {e}")
        return None

def find_most_similar_recipe(user_ingredients):
    """
    Finds the most similar recipe based on user-provided ingredients.
    """
    user_embedding = vectorize_ingredients(user_ingredients)
    similar_recipes = search_recipes(user_embedding)

    if similar_recipes:
        max_similarity = -1
        best_recipe = None

        for match in similar_recipes:
            recipe_id = match['id']
            recipe_metadata = match['metadata']

            recipe_vector_data = fetch_recipe_vector(recipe_id)
            if recipe_vector_data and recipe_id in recipe_vector_data['vectors']:
                recipe_embedding = np.array(recipe_vector_data['vectors'][recipe_id]['values'])
            else:
                print(f"Warning: Recipe ID {recipe_id} has an empty embedding.")
                continue
            
            similarity = 1 - cosine(user_embedding, recipe_embedding)

            if similarity > max_similarity:
                max_similarity = similarity
                best_recipe = {
                    "id": recipe_id,
                    "title": recipe_metadata.get('name', 'Untitled Recipe'),
                    "ingredients": recipe_metadata.get('ingredients', 'Not available'),
                    "instructions": recipe_metadata.get('instructions', 'Not available'),
                    "similarity": similarity
                }

        # Generate a detailed recipe using GPT after finding the best match
        if best_recipe:
            best_recipe['gpt_recipe'] = generate_gpt_recipe(
                best_recipe['title'],
                best_recipe['ingredients'].split(', '),
                best_recipe['instructions']
            )

        # Validate the generated recipe instructions
        best_recipe['gpt_recipe'] = validate_gpt_instructions(best_recipe['gpt_recipe'])

        return best_recipe
    else:
        return None

def generate_gpt_recipe(title, ingredients, instructions):
    """
    Generates a detailed recipe using GPT.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates detailed recipes."},
        {"role": "user", "content": f"Generate a detailed recipe using the following ingredients: {', '.join(ingredients)}. "
                                      "Please format the response with a clear section for 'Ingredients' and another for 'Instructions'."}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=1500,
            temperature=0.4
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error generating recipe with GPT: {e}")
        return "Error generating recipe."

def validate_gpt_instructions(instructions):
    """
    Validates the generated recipe instructions to ensure they are appropriate.
    """
    inappropriate_phrases = [
        "take off your clothes", "stand on your head", "chanting", "singing weird", "sacrifice"
    ]
    for phrase in inappropriate_phrases:
        if phrase in instructions.lower():
            return "Error: Inappropriate instructions detected. Please regenerate the recipe."
    return instructions

# # Example usage
# if __name__ == "__main__":
#     user_ingredients = ["salmon", "light cream cheese", "cracked pepper"]
    
#     # Find the most similar recipe
#     best_recipe = find_most_similar_recipe(user_ingredients)

#     if best_recipe:
#         print(f"Best Recipe ID: {best_recipe['id']}")
#         print(f"Ingredients: {best_recipe['ingredients']}")
#         print(f"Instructions: {best_recipe['instructions']}")
#         print(f"Cosine Similarity: {best_recipe['similarity']:.4f}")

#         # Generate a detailed recipe using GPT
#         gpt_recipe = generate_gpt_recipe(best_recipe['ingredients'].split(', '), best_recipe['instructions'])
#         print("\nGenerated Recipe from GPT:")
#         print(gpt_recipe)
#     else:
#         print("No matching recipes found.")


def main(user_ingredients):
    
    print(f"Ingredients provided: {user_ingredients}")

if __name__ == "__main__":
    ingredients = sys.argv[1] 
    main(ingredients)