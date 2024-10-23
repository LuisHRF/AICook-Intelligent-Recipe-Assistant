import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_recipe_text(recipe_name, ingredients):
    """
    Use GPT to generate a recipe text based on the recipe name and ingredients.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates detailed recipes."},
        {"role": "user", "content": f"Create a detailed recipe for {recipe_name} using the following ingredients: {ingredients}."}
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4o",  
        messages=messages,
        max_tokens=500
    )
    
    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    recipe_name = "Garlic Chicken"
    ingredients = "chicken, garlic, onion, olive oil"
    recipe_text = generate_recipe_text(recipe_name, ingredients)
    print(recipe_text)