import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import re
import openai
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from src.utils.config import PINECONE_API_KEY

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone and Sentence Transformer
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("recipe-embeddings")
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_ingredient_embedding(ingredients_list):
    """
    Generate an embedding vector for a list of ingredients.
    """
    # Verificar cuántos ingredientes se están pasando realmente
    print(f"Generating embedding for ingredients: {ingredients_list}")
    
    # Join all the ingredients into a single string to generate a combined embedding
    ingredients_text = ", ".join(ingredients_list)  
    embedding = model.encode(ingredients_text).tolist()
    return embedding

def search_similar_recipes(ingredient_embedding, top_n=20):
    """
    Query Pinecone to find similar recipes based on ingredient embeddings.
    """
    query_response = index.query(
        vector=ingredient_embedding,
        top_k=top_n,
        include_metadata=True,
        namespace="recipes"
    )

    # Imprimir las recetas similares encontradas para depurar
    print(f"Similar recipes found: {query_response['matches']}")
    
    return query_response['matches']

def filter_by_ingredient_match(similar_recipes, user_ingredients, threshold=0.5):
    """
    Filter the recipes based on the percentage of matching ingredients.
    """
    filtered_recipes = []
    for match in similar_recipes:
        recipe_ingredients = match['metadata']['ingredients']
        matched_ingredients = set(user_ingredients).intersection(set(recipe_ingredients.split(", ")))
        if len(matched_ingredients) / len(user_ingredients) >= threshold:
            filtered_recipes.append(match)
    return filtered_recipes

def generate_recipe_with_gpt(user_ingredients, closest_recipe):
    """
    Use GPT to generate a recipe based on the user's ingredients, while using
    the closest matching recipe from Pinecone for inspiration.
    """
    recipe_name = closest_recipe['metadata'].get('name', 'Recipe')
    ingredients = ", ".join(user_ingredients)

    # Construct the prompt for GPT
    prompt = (
        f"Create a recipe called '{recipe_name}' using the following ingredients: {ingredients}. "
        f"Please provide the recipe title, ingredients, and instructions in separate sections."
    )

    # Call GPT to generate the recipe
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant of a famous chef that generates recipes."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.7
    )

    recipe_text = response['choices'][0]['message']['content']

    # Process the response to extract title, ingredients, and instructions using regex
    title = re.search(r"Title:(.*)", recipe_text)
    ingredients = re.search(r"Ingredients:(.*?)(Instructions|$)", recipe_text, re.DOTALL)
    instructions = re.search(r"Instructions:(.*)", recipe_text, re.DOTALL)

    # Extract and clean the values
    title = title.group(1).strip() if title else recipe_name
    ingredients = ingredients.group(1).strip() if ingredients else "Ingredients not provided"
    instructions = instructions.group(1).strip() if instructions else "Instructions not provided"

    return {
        "title": title,
        "ingredients": ingredients,
        "instructions": instructions
    }


def create_recipe_from_ingredients(user_ingredients):
    """
    Main function that handles the full process of generating a recipe.
    - Takes a list of ingredients.
    - Generates embeddings.
    - Searches for similar recipes.
    - Filters recipes by ingredient match.
    - Uses GPT to generate a new recipe based on the closest match.
    """
    # Step 1: Generate embeddings for the ingredients
    ingredient_embedding = generate_ingredient_embedding(user_ingredients)

    # Step 2: Search for similar recipes
    similar_recipes = search_similar_recipes(ingredient_embedding)
    if not similar_recipes:
        return {"error": "No similar recipes found."}

    # Step 3: Filter recipes based on ingredient match
    filtered_recipes = filter_by_ingredient_match(similar_recipes, user_ingredients)
    if not filtered_recipes:
        return {"error": "No recipes match the given ingredients after filtering."}

    # Step 4: Generate a new recipe based on the user's ingredients, using the closest recipe as inspiration
    closest_recipe = filtered_recipes[0]
    generated_recipe = generate_recipe_with_gpt(user_ingredients, closest_recipe)

    # Use the generated title, ingredients, and instructions
    return {
        "title": generated_recipe['title'],  # Title from GPT
        "ingredients": generated_recipe['ingredients'],  # Ingredients from GPT
        "instructions": generated_recipe['instructions']  # Instructions from GPT
    }


# # Example usage
# if __name__ == "__main__":
#     user_ingredients = ["salmon", "cracked pepper"]

#     # Step 1: Generate an embedding for the ingredients
#     ingredient_embedding = generate_ingredient_embedding(user_ingredients)

#     # Step 2: Search for similar recipes
#     similar_recipes = search_similar_recipes(ingredient_embedding)

#     # Step 3: Filter recipes based on ingredient match
#     filtered_recipes = filter_by_ingredient_match(similar_recipes, user_ingredients)

#     if filtered_recipes:
#         # Step 4: Generate a new recipe based on the closest match
#         closest_recipe = filtered_recipes[0]  # Assuming the first match is the closest
#         generated_recipe = generate_recipe_with_gpt(closest_recipe)

#         print("Generated Recipe:")
#         print(generated_recipe)
#     else:
#         print("No matching recipes found.")