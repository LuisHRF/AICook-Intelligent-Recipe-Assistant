import pandas as pd
import random
import re
import ast

recipes_path = "data/raw/recipes.parquet"

def load_data():

    recipes_raw = pd.read_parquet(recipes_path)
    return recipes_raw

def convert_iso8601_duration_to_minutes(duration):
    """
    Convert an ISO 8601 duration string (e.g., PT24H45M) to total minutes.
    """
    hours = re.search(r'(\d+)H', duration)  
    minutes = re.search(r'(\d+)M', duration)  
    
    hours = int(hours.group(1)) if hours else 0
    minutes = int(minutes.group(1)) if minutes else 0
    
    total_minutes = hours * 60 + minutes

    if total_minutes == 0:
        total_minutes = random.randint(10, 35)
    
    return total_minutes

def clean_ingredients_and_quantities(recipes): #Preprocess the columns

    recipes = load_data()

    #Clean ingredients and nutrition
    recipes['ingredients_cleaned'] = recipes['RecipeIngredientParts'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    recipes['quantities_cleaned'] = recipes['RecipeIngredientQuantities'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    return recipes

def save_cleaned_data(recipes):
    
    columns_to_exclude = ['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 
                          'SodiumContent', 'CarbohydrateContent', 'FiberContent', 
                          'SugarContent', 'ProteinContent']
    
    cleaned_data = recipes.drop(columns=columns_to_exclude)
    cleaned_data.to_parquet("data/processed/recipes_cleaned.parquet")
    print("Data cleaned and saved in 'data/processed/'")

def save_nutrition_data(recipes):
    nutrition_columns = ['RecipeId', 'Calories', 'FatContent', 'SaturatedFatContent', 
                         'CholesterolContent', 'SodiumContent', 'CarbohydrateContent', 
                         'FiberContent', 'SugarContent', 'ProteinContent']
    
    nutrition_data = recipes[nutrition_columns]
    nutrition_data.to_parquet("data/processed/nutrition_data.parquet")
    print("Nutrition data saved in 'data/processed/'")


if __name__ == "__main__":
    recipes = load_data()
    
    recipes_cleaned = clean_ingredients_and_quantities(recipes)

    recipes_cleaned['TotalTimeMinutes'] = recipes_cleaned['TotalTime'].apply(convert_iso8601_duration_to_minutes)
    
    save_cleaned_data(recipes_cleaned)
    
    save_nutrition_data(recipes_cleaned)