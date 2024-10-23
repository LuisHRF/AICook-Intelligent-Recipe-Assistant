import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pinecone import Pinecone
from src.utils.config import PINECONE_API_KEY

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("recipe-embeddings")
openai.api_key = openai_api_key

recipes_cleaned_path = "data/processed/recipes_cleaned.parquet"
recipes_cleaned = pd.read_parquet(recipes_cleaned_path)

def find_recipe_by_name(recipe_name, recipes_cleaned):
    """
    Search for the recipe by name or similar ingredients in the cleaned recipes dataframe.
    """
    # First, try finding the recipe by name similarity
    recipe_found = recipes_cleaned[recipes_cleaned['Name'].str.lower().str.contains(recipe_name.lower())]
    
    # If no exact match found by name, try finding recipes with similar ingredients
    if recipe_found.empty:
        key_ingredients = recipe_name.split()  # Assuming the recipe name contains key ingredients
        ingredient_matches = recipes_cleaned[recipes_cleaned['RecipeIngredientParts'].apply(
            lambda x: any(ingredient.lower() in " ".join(x).lower() for ingredient in key_ingredients)
        )]
        if not ingredient_matches.empty:
            return ingredient_matches.iloc[0]  
    
    return recipe_found.iloc[0] if not recipe_found.empty else None

def validate_recipe_ingredients(recipe, user_ingredients, threshold=0.5):
    """
    Validates that the recipe ingredients match the user-provided ingredients by a certain threshold.
    """
    recipe_ingredients = recipe['RecipeIngredientParts']
    matched_ingredients = set(user_ingredients).intersection(set(recipe_ingredients))
    return len(matched_ingredients) / len(user_ingredients) >= threshold

def generate_recipe_embedding(recipe):
    """
    Generate embeddings for the recipe based on ingredients using SentenceTransformer.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')  
    ingredients_text = ' '.join(recipe['RecipeIngredientParts'])  
    recipe_embedding = model.encode(ingredients_text)  
    return recipe_embedding 

def search_similar_recipes_in_pinecone(recipe_embedding, top_n=5):
    """
    Search for similar recipes in Pinecone using the recipe embedding.
    """
    # Convert the NumPy array to a list
    recipe_embedding_list = recipe_embedding.tolist()
    
    query_response = index.query(
        vector=recipe_embedding_list,  
        top_k=top_n,
        include_metadata=True,
        namespace="recipes"
    )
    
    return query_response

def explain_similarity_in_english(original_recipe_name, similar_recipe_name):
    """
    Generate a GPT response in English explaining why the two recipes are similar.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that explains why two recipes are similar."},
        {"role": "user", "content": f"Explain why the recipe '{original_recipe_name}' is similar to '{similar_recipe_name}'."}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=1500
    )
    
    explanation = response['choices'][0]['message']['content'].strip()
    return explanation

def find_similar_recipe_flow(user_recipe_name, recipes_cleaned):
    """
    Main flow to find, embed, and search similar recipes in Pinecone.
    """
    # Find the recipe in the database
    recipe = find_recipe_by_name(user_recipe_name, recipes_cleaned)
    
    if recipe is not None:
        print(f"Recipe found: {recipe['Name']}")
        
        # Generate the recipe embedding
        recipe_embedding = generate_recipe_embedding(recipe)
        
        # Search for similar recipes in Pinecone
        similar_recipes = search_similar_recipes_in_pinecone(recipe_embedding)
        
        if similar_recipes['matches']:
            closest_match = similar_recipes['matches'][0]  # Take the closest recipe
            similar_recipe_name = closest_match['metadata']['name']
            
            print(f"Similar recipe found: {similar_recipe_name}")
            
            # Use GPT to explain the similarity in English
            explanation = explain_similarity_in_english(user_recipe_name, similar_recipe_name)
            print(f"GPT Explanation: {explanation}")
        
        return similar_recipes
    else:
        print(f"No recipe found for '{user_recipe_name}' in the database.")
        return None

# Example usage
if __name__ == "__main__":
    user_recipe_name = "Pork Tenderloin"  # Example recipe provided by user
    find_similar_recipe_flow(user_recipe_name, recipes_cleaned)